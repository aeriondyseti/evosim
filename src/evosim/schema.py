"""Component schema: typed fields backed by structure-of-arrays (SoA) buffers.

A :class:`Schema` declares the named, typed, shaped fields that make up an agent's
heritable + transient state. The framework owns the SoA storage: for a population of
``capacity`` slots, each :class:`Field` becomes one array of shape ``(capacity, *field.shape)``
with the field's dtype. Per-field dtypes are supported (e.g. ``int8`` traits, ``float16``
positions, ``float32`` genomes) per SPEC mixed-precision decision.

Two fields are reserved and auto-added by the framework:

- ``alive`` (bool): liveness mask for the capacity-based population (see ``population.py``).
- ``id`` (int32): stable unique id; ``-1`` marks an empty slot.

Determinism note: JAX silently downcasts 64-bit dtypes to 32-bit unless x64 mode is enabled.
To avoid surprising precision loss we *raise* when a 64-bit field is allocated without x64,
pointing the caller at :func:`enable_x64`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["Field", "Schema", "enable_x64", "RESERVED_FIELDS"]

#: Names managed by the framework; auto-added to every schema if absent.
RESERVED_FIELDS: frozenset[str] = frozenset({"alive", "id"})

_DTYPES_64 = frozenset({"float64", "complex128", "int64", "uint64"})


def enable_x64() -> None:
    """Enable JAX 64-bit precision globally (needed for float64/int64 fields)."""
    jax.config.update("jax_enable_x64", True)


def _x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


@dataclass(frozen=True)
class Field:
    """A single typed component field.

    Parameters
    ----------
    name:
        Field name. Optional when the field is passed as a keyword to :class:`Schema`
        (the keyword supplies the name). Must be a valid Python identifier otherwise.
    dtype:
        Any numpy-parseable dtype string (e.g. ``"float32"``, ``"int8"``, ``"bool"``).
        Canonicalized to its numpy name.
    shape:
        Per-entity trailing shape. ``()`` (scalar) by default; ``(2,)`` for a 2-vector, etc.
        An ``int`` is promoted to a 1-tuple.
    default:
        Fill value for freshly allocated / reset slots.
    doc:
        Optional human description.
    """

    name: str = ""
    dtype: str = "float32"
    shape: tuple[int, ...] = ()
    default: float | int | bool = 0
    doc: str = ""

    def __post_init__(self) -> None:
        # Normalize shape (accept int or iterable).
        if isinstance(self.shape, int):
            object.__setattr__(self, "shape", (self.shape,))
        else:
            object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        for d in self.shape:
            if d < 0:
                raise ValueError(f"Field {self.name!r} has negative shape dim: {self.shape}")
        # Canonicalize dtype via numpy.
        try:
            np_dtype = np.dtype(self.dtype)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Field {self.name!r}: invalid dtype {self.dtype!r}") from exc
        object.__setattr__(self, "dtype", np_dtype.name)
        # Validate name only when present (it may be filled in later by Schema).
        if self.name and not self.name.isidentifier():
            raise ValueError(f"Field name {self.name!r} is not a valid identifier")

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def is_64bit(self) -> bool:
        return self.dtype in _DTYPES_64

    def rename(self, name: str) -> "Field":
        return dataclasses.replace(self, name=name)

    def buffer_shape(self, capacity: int) -> tuple[int, ...]:
        return (capacity, *self.shape)

    def allocate(self, capacity: int) -> jax.Array:
        """Allocate this field's SoA buffer of ``capacity`` slots, filled with ``default``."""
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        if self.is_64bit and not _x64_enabled():
            raise ValueError(
                f"Field {self.name!r} uses 64-bit dtype {self.dtype!r} but JAX x64 mode is "
                f"disabled (values would be silently downcast). Call evosim.enable_x64() first."
            )
        return jnp.full(self.buffer_shape(capacity), self.default, dtype=self.np_dtype)


class Schema:
    """An ordered collection of typed component :class:`Field` s.

    Construct from named keyword fields (the keyword becomes the field name) and/or
    positional :class:`Field` s that already carry a name::

        schema = Schema(
            position=Field(dtype="int16", shape=(2,)),
            energy=Field(dtype="float32"),
            genome=Field(dtype="float32", shape=(8,)),
        )

    Reserved fields ``alive`` (bool) and ``id`` (int32) are appended automatically unless
    the caller supplies them explicitly.
    """

    def __init__(self, *positional: Field, **named: Field) -> None:
        fields: dict[str, Field] = {}
        for f in positional:
            if not isinstance(f, Field):
                raise TypeError(f"positional args must be Field, got {type(f).__name__}")
            if not f.name:
                raise ValueError("positional Field must have a name")
            if f.name in fields:
                raise ValueError(f"duplicate field name {f.name!r}")
            fields[f.name] = f
        for name, f in named.items():
            if not isinstance(f, Field):
                raise TypeError(f"field {name!r} must be a Field, got {type(f).__name__}")
            if name in fields:
                raise ValueError(f"duplicate field name {name!r}")
            fields[name] = f.rename(name)

        # Auto-add reserved fields if absent.
        if "alive" not in fields:
            fields["alive"] = Field(name="alive", dtype="bool", default=False,
                                    doc="liveness mask (framework-managed)")
        if "id" not in fields:
            fields["id"] = Field(name="id", dtype="int32", default=-1,
                                 doc="stable unique id; -1 == empty slot (framework-managed)")

        self._fields: dict[str, Field] = fields

    # -- introspection -------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._fields.keys())

    @property
    def user_names(self) -> tuple[str, ...]:
        return tuple(n for n in self._fields if n not in RESERVED_FIELDS)

    def user_fields(self) -> tuple[Field, ...]:
        return tuple(f for n, f in self._fields.items() if n not in RESERVED_FIELDS)

    def __iter__(self) -> Iterator[Field]:
        return iter(self._fields.values())

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __getitem__(self, name: str) -> Field:
        return self._fields[name]

    # -- allocation ----------------------------------------------------------
    def allocate(self, capacity: int) -> dict[str, jax.Array]:
        """Allocate all SoA buffers for ``capacity`` slots as a ``{name: array}`` dict."""
        return {name: f.allocate(capacity) for name, f in self._fields.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        body = ", ".join(
            f"{n}:{f.dtype}{('' if not f.shape else f.shape)}" for n, f in self._fields.items()
        )
        return f"Schema({body})"
