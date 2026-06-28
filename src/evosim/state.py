"""Simulation state: an immutable, JAX-registered PyTree.

:class:`State` is the single value threaded through the simulation
(``state -> state'``), which makes the whole tick amenable to ``jit``, ``scan`` and
``vmap``. It bundles the SoA component buffers (from a :class:`~evosim.schema.Schema`)
plus the framework-managed ``tick`` and ``next_id`` counters.

The RNG root key is intentionally *not* stored here — randomness is derived from a root
key plus ``tick`` (see ``rng.py``), so the scheduler/simulation owns the root key and only
``tick`` needs to live in the state.

Static (non-traced) metadata — the schema and the buffer ``capacity`` — is carried as PyTree
aux data so it survives jit/scan without becoming a traced array.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .schema import Schema

__all__ = ["State", "state_fingerprint"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State:
    """Immutable container of the full simulation state.

    Attributes
    ----------
    components:
        ``{field_name: array}`` SoA buffers, each of shape ``(capacity, *field.shape)``.
    tick:
        Scalar int32 array, the current tick (starts at 0).
    next_id:
        Scalar int32 array, the next unique id to assign on birth (monotonic).
    schema:
        The :class:`Schema` (static aux data).
    capacity:
        Number of slots in every buffer (static aux data).
    """

    components: dict[str, jax.Array]
    tick: jax.Array
    next_id: jax.Array
    schema: Schema
    capacity: int

    # -- construction --------------------------------------------------------
    @classmethod
    def create(cls, schema: Schema, capacity: int) -> "State":
        """Allocate an empty state: all slots dead (``alive=False``, ``id=-1``)."""
        comps = schema.allocate(capacity)
        return cls(
            components=comps,
            tick=jnp.asarray(0, dtype=jnp.int32),
            next_id=jnp.asarray(0, dtype=jnp.int32),
            schema=schema,
            capacity=capacity,
        )

    # -- PyTree protocol -----------------------------------------------------
    def tree_flatten(self):
        children = (self.components, self.tick, self.next_id)
        aux = (self.schema, self.capacity)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        components, tick, next_id = children
        schema, capacity = aux
        return cls(components, tick, next_id, schema, capacity)

    # -- convenient accessors ------------------------------------------------
    @property
    def alive(self) -> jax.Array:
        return self.components["alive"]

    @property
    def ids(self) -> jax.Array:
        return self.components["id"]

    @property
    def n_alive(self) -> jax.Array:
        """Number of alive slots (traced int32 scalar)."""
        return jnp.sum(self.alive.astype(jnp.int32))

    @property
    def free_mask(self) -> jax.Array:
        return jnp.logical_not(self.alive)

    def __getitem__(self, name: str) -> jax.Array:
        return self.components[name]

    def __contains__(self, name: object) -> bool:
        return name in self.components

    def get(self, name: str) -> jax.Array:
        return self.components[name]

    # -- immutable updates ---------------------------------------------------
    def replace(self, **changes) -> "State":
        """Return a copy with top-level attributes replaced."""
        return dataclasses.replace(self, **changes)

    def set(self, name: str, value: jax.Array) -> "State":
        """Return a copy with component ``name`` replaced by ``value``."""
        if name not in self.components:
            raise KeyError(f"unknown component {name!r}")
        new = dict(self.components)
        new[name] = value
        return self.replace(components=new)

    def set_many(self, updates: dict[str, jax.Array]) -> "State":
        """Return a copy with several components replaced."""
        new = dict(self.components)
        for k, v in updates.items():
            if k not in new:
                raise KeyError(f"unknown component {k!r}")
            new[k] = v
        return self.replace(components=new)

    def increment_tick(self) -> "State":
        return self.replace(tick=self.tick + 1)

    # -- determinism helpers -------------------------------------------------
    def fingerprint(self) -> str:
        """Stable host-side hash of the full state (for determinism golden-masters)."""
        return state_fingerprint(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"State(capacity={self.capacity}, tick={int(self.tick)}, "
            f"n_alive={int(self.n_alive)}, fields={self.schema.names})"
        )


def state_fingerprint(state: State) -> str:
    """SHA-256 over all component buffers (in schema order) + tick + next_id.

    Pulls arrays to host; intended for tests, not the hot loop.
    """
    h = hashlib.sha256()
    for name in state.schema.names:
        arr = np.asarray(state.components[name])
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(b"tick")
    h.update(np.asarray(state.tick).tobytes())
    h.update(b"next_id")
    h.update(np.asarray(state.next_id).tobytes())
    return h.hexdigest()
