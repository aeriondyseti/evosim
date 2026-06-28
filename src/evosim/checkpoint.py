"""Checkpointing: full deterministic save & resume (per SPEC).

A checkpoint captures everything needed to resume a run bit-identically on the same backend:
the complete :class:`~evosim.state.State` (components, fields, tick, next_id), the schema
needed to reconstruct it, and optionally the RNG root key. Because randomness is derived from
``root_key + tick`` (counter-based), saving the root key + tick is sufficient for exact resume.

Format: a single ``.npz`` file holding all arrays plus small JSON blobs (schema spec, meta).
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import rng as _rng
from .schema import Field, Schema
from .state import State

__all__ = ["CheckpointData", "save", "load"]


class CheckpointData(NamedTuple):
    state: State
    rng_key: jax.Array | None
    meta: dict[str, Any]


def _schema_spec(schema: Schema) -> list[dict[str, Any]]:
    return [
        {"name": f.name, "dtype": f.dtype, "shape": list(f.shape), "default": f.default}
        for f in schema
    ]


def _schema_from_spec(spec: list[dict[str, Any]]) -> Schema:
    fields = [
        Field(name=s["name"], dtype=s["dtype"], shape=tuple(s["shape"]), default=s["default"])
        for s in spec
    ]
    return Schema(*fields)


def save(path: str, state: State, rng_key: jax.Array | None = None,
         meta: dict[str, Any] | None = None) -> None:
    """Write a checkpoint to ``path`` (``.npz`` appended if missing)."""
    data: dict[str, np.ndarray] = {}
    for name, arr in state.components.items():
        data[f"comp::{name}"] = np.asarray(arr)
    for name, arr in state.fields.items():
        data[f"field::{name}"] = np.asarray(arr)
    data["__tick__"] = np.asarray(state.tick)
    data["__next_id__"] = np.asarray(state.next_id)
    data["__capacity__"] = np.asarray(state.capacity, dtype=np.int64)
    data["__schema__"] = np.array(json.dumps(_schema_spec(state.schema)))
    data["__meta__"] = np.array(json.dumps(meta or {}))
    if rng_key is not None:
        data["__rng__"] = np.asarray(_rng.key_bits(rng_key))
    np.savez(path, **data)


def load(path: str) -> CheckpointData:
    """Load a checkpoint written by :func:`save`."""
    npz = np.load(path, allow_pickle=False)
    schema = _schema_from_spec(json.loads(npz["__schema__"].item()))
    capacity = int(npz["__capacity__"])

    components: dict[str, jax.Array] = {}
    fields: dict[str, jax.Array] = {}
    for key in npz.files:
        if key.startswith("comp::"):
            components[key[len("comp::"):]] = jnp.asarray(npz[key])
        elif key.startswith("field::"):
            fields[key[len("field::"):]] = jnp.asarray(npz[key])

    state = State(
        components=components,
        tick=jnp.asarray(npz["__tick__"]),
        next_id=jnp.asarray(npz["__next_id__"]),
        schema=schema,
        capacity=capacity,
        fields=fields,
    )

    rng_key = None
    if "__rng__" in npz.files:
        rng_key = jax.random.wrap_key_data(jnp.asarray(npz["__rng__"]))
    meta = json.loads(npz["__meta__"].item())
    return CheckpointData(state=state, rng_key=rng_key, meta=meta)
