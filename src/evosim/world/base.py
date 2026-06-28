"""World interface.

A *world* defines the spatial topology agents live in and the spatial-query primitives
available to systems. Per SPEC the world is pluggable: the first concrete module is a
discrete toric 2D grid (:class:`~evosim.world.grid.ToricGrid2D`); continuous 2D/3D and
graph worlds can implement the same surface later.

The interface is intentionally small. A world provides:

- ``ndim`` / ``shape`` — topology description (static),
- ``wrap(pos)`` — apply boundary conditions to positions,
- spatial-query helpers appropriate to the topology (grids expose cell/Moore-neighborhood
  operations; a continuous world would expose radius / kNN instead).

Worlds are *static* objects (no per-tick mutable state). Dynamic environment data — resource
or pheromone grids, the Conway cell grid — lives in ``State.fields`` so it flows through
jit/scan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax

__all__ = ["World"]


class World(ABC):
    """Abstract spatial world."""

    #: Number of spatial dimensions.
    ndim: int

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Topology shape (e.g. ``(H, W)`` for a 2D grid)."""

    @abstractmethod
    def wrap(self, pos: jax.Array) -> jax.Array:
        """Apply boundary conditions to integer/float positions ``(..., ndim)``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} shape={self.shape}>"
