"""Discrete toric 2D grid world (first world module per SPEC).

Positions are integer ``(row, col)`` pairs with component 0 = row in ``[0, H)`` and
component 1 = col in ``[0, W)``. The grid is *toric*: it wraps on both axes.

Provides cell-binning and neighborhood primitives that vectorize cleanly:

- :meth:`cell_id` / :meth:`wrap` / :meth:`move` — coordinate helpers.
- :meth:`cell_counts` — agents-per-cell occupancy via scatter-add.
- :meth:`scatter_field` / :meth:`gather_field` — deposit per-agent values into a grid /
  read a grid value at each agent's cell.
- :meth:`moore_sum` / :meth:`von_neumann_sum` — toric neighborhood sums (the 8- and
  4-neighborhoods). ``moore_sum(grid, include_center=False)`` is exactly the live-neighbor
  count used by Conway's Game of Life.
- :meth:`accumulate_neighborhood` — per-agent aggregate over the neighborhood (compose of
  scatter -> neighborhood-sum -> gather).
- :meth:`random_positions` — uniform random cell positions (deterministic given a key).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import World

__all__ = ["ToricGrid2D", "MOORE_OFFSETS", "VON_NEUMANN_OFFSETS"]

#: 8 Moore-neighborhood offsets (row, col), excluding the center.
MOORE_OFFSETS = jnp.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=jnp.int32
)

#: 4 von-Neumann-neighborhood offsets (row, col), excluding the center.
VON_NEUMANN_OFFSETS = jnp.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=jnp.int32)


class ToricGrid2D(World):
    """A wrap-around (toric) 2D grid of shape ``(H, W)``."""

    ndim = 2

    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def n_cells(self) -> int:
        return self.height * self.width

    # -- coordinate helpers --------------------------------------------------
    def wrap(self, pos: jax.Array) -> jax.Array:
        """Wrap positions ``(..., 2)`` into the grid (toric mod)."""
        hw = jnp.array([self.height, self.width], dtype=pos.dtype if jnp.issubdtype(pos.dtype, jnp.integer) else jnp.int32)
        return jnp.mod(pos.astype(jnp.int32), hw).astype(jnp.int32)

    def move(self, pos: jax.Array, delta: jax.Array) -> jax.Array:
        """Move by integer ``delta`` and wrap."""
        return self.wrap(pos.astype(jnp.int32) + delta.astype(jnp.int32))

    def cell_id(self, pos: jax.Array) -> jax.Array:
        """Linear cell id ``row * W + col`` for (wrapped) positions ``(..., 2)``."""
        p = self.wrap(pos)
        return (p[..., 0] * self.width + p[..., 1]).astype(jnp.int32)

    def coords_of(self, cell_id: jax.Array) -> jax.Array:
        """Inverse of :meth:`cell_id`: linear id -> ``(row, col)``."""
        cid = cell_id.astype(jnp.int32)
        return jnp.stack([cid // self.width, cid % self.width], axis=-1)

    # -- binning / scatter-gather --------------------------------------------
    def cell_counts(self, positions: jax.Array, alive: jax.Array | None = None) -> jax.Array:
        """Count agents per cell -> grid ``(H, W)`` (dead agents excluded)."""
        ids = self.cell_id(positions)
        w = jnp.ones((ids.shape[0],), dtype=jnp.int32) if alive is None else alive.astype(jnp.int32)
        flat = jnp.zeros((self.n_cells,), dtype=jnp.int32).at[ids].add(w)
        return flat.reshape(self.shape)

    def scatter_field(self, positions: jax.Array, values: jax.Array,
                      alive: jax.Array | None = None) -> jax.Array:
        """Sum per-agent ``values`` into the grid cell each agent occupies.

        ``values`` is ``(N, *tail)``; result is ``(H, W, *tail)``. Dead agents contribute 0.
        """
        ids = self.cell_id(positions)
        vals = values
        if alive is not None:
            mask = alive.astype(values.dtype)
            vals = values * mask.reshape((-1,) + (1,) * (values.ndim - 1))
        tail = values.shape[1:]
        flat = jnp.zeros((self.n_cells, *tail), dtype=values.dtype).at[ids].add(vals)
        return flat.reshape((*self.shape, *tail))

    def gather_field(self, positions: jax.Array, grid: jax.Array) -> jax.Array:
        """Read the grid value at each agent's cell. ``grid`` is ``(H, W, *tail)``."""
        ids = self.cell_id(positions)
        tail = grid.shape[2:]
        flat = grid.reshape((self.n_cells, *tail))
        return flat[ids]

    # -- neighborhood sums (toric) -------------------------------------------
    def moore_sum(self, grid: jax.Array, include_center: bool = False) -> jax.Array:
        """Sum over the 8-neighborhood (toric) for a ``(H, W[, ...])`` grid."""
        total = grid.astype(grid.dtype) if include_center else jnp.zeros_like(grid)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                total = total + jnp.roll(jnp.roll(grid, dy, axis=0), dx, axis=1)
        return total

    def von_neumann_sum(self, grid: jax.Array, include_center: bool = False) -> jax.Array:
        """Sum over the 4-neighborhood (toric)."""
        total = grid.astype(grid.dtype) if include_center else jnp.zeros_like(grid)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            total = total + jnp.roll(jnp.roll(grid, dy, axis=0), dx, axis=1)
        return total

    def accumulate_neighborhood(self, positions: jax.Array, values: jax.Array,
                                alive: jax.Array | None = None,
                                include_center: bool = True) -> jax.Array:
        """Per-agent aggregate of ``values`` over each agent's Moore neighborhood.

        Equivalent to scatter -> moore_sum -> gather. Returns ``(N, *tail)``.
        """
        grid = self.scatter_field(positions, values, alive)
        summed = self.moore_sum(grid, include_center=include_center)
        return self.gather_field(positions, summed)

    # -- sampling ------------------------------------------------------------
    def random_positions(self, key: jax.Array, n: int) -> jax.Array:
        """Sample ``n`` uniform random integer positions ``(n, 2)``."""
        k1, k2 = jax.random.split(key)
        rows = jax.random.randint(k1, (n,), 0, self.height, dtype=jnp.int32)
        cols = jax.random.randint(k2, (n,), 0, self.width, dtype=jnp.int32)
        return jnp.stack([rows, cols], axis=-1)
