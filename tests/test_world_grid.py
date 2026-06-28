"""Unit tests for evosim.world.grid (ToricGrid2D)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim.world import ToricGrid2D


def test_shape_and_ncells():
    g = ToricGrid2D(4, 5)
    assert g.shape == (4, 5)
    assert g.n_cells == 20
    assert g.ndim == 2


def test_wrap_toric():
    g = ToricGrid2D(4, 5)
    pos = jnp.array([[4, 5], [-1, -1], [7, 12]])
    w = np.asarray(g.wrap(pos))
    assert np.array_equal(w, [[0, 0], [3, 4], [3, 2]])


def test_cell_id_and_coords_roundtrip():
    g = ToricGrid2D(4, 5)
    pos = jnp.array([[0, 0], [1, 2], [3, 4]])
    ids = g.cell_id(pos)
    assert np.array_equal(np.asarray(ids), [0, 7, 19])
    back = g.coords_of(ids)
    assert np.array_equal(np.asarray(back), np.asarray(pos))


def test_move_wraps():
    g = ToricGrid2D(3, 3)
    pos = jnp.array([[2, 2]])
    moved = g.move(pos, jnp.array([[1, 1]]))
    assert np.array_equal(np.asarray(moved), [[0, 0]])


def test_cell_counts():
    g = ToricGrid2D(3, 3)
    pos = jnp.array([[0, 0], [0, 0], [1, 2]])
    counts = g.cell_counts(pos)
    assert int(counts[0, 0]) == 2
    assert int(counts[1, 2]) == 1
    assert int(counts.sum()) == 3


def test_cell_counts_respects_alive():
    g = ToricGrid2D(3, 3)
    pos = jnp.array([[0, 0], [0, 0], [1, 2]])
    alive = jnp.array([True, False, True])
    counts = g.cell_counts(pos, alive)
    assert int(counts[0, 0]) == 1
    assert int(counts.sum()) == 2


def test_scatter_gather_roundtrip():
    g = ToricGrid2D(3, 3)
    pos = jnp.array([[0, 0], [1, 1]])
    vals = jnp.array([5.0, 7.0])
    grid = g.scatter_field(pos, vals)
    assert grid.shape == (3, 3)
    assert float(grid[0, 0]) == 5.0
    assert float(grid[1, 1]) == 7.0
    got = g.gather_field(pos, grid)
    assert np.allclose(np.asarray(got), [5.0, 7.0])


def test_moore_sum_single_seed():
    g = ToricGrid2D(5, 5)
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[2, 2].set(1)
    nb = g.moore_sum(grid, include_center=False)
    assert int(nb.sum()) == 8
    assert int(nb[1, 1]) == 1
    assert int(nb[2, 2]) == 0
    nb_c = g.moore_sum(grid, include_center=True)
    assert int(nb_c[2, 2]) == 1


def test_von_neumann_sum():
    g = ToricGrid2D(5, 5)
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[2, 2].set(1)
    nb = g.von_neumann_sum(grid)
    assert int(nb.sum()) == 4
    assert int(nb[2, 1]) == 1
    assert int(nb[1, 1]) == 0  # diagonal not counted


def test_moore_sum_toric_wrap():
    g = ToricGrid2D(3, 3)
    grid = jnp.zeros((3, 3), dtype=jnp.int32).at[0, 0].set(1)
    nb = g.moore_sum(grid)
    # on a 3x3 toric grid, every other cell is a Moore neighbor of (0,0)
    assert int(nb[2, 2]) == 1
    assert int(nb[0, 1]) == 1


def test_accumulate_neighborhood():
    g = ToricGrid2D(4, 4)
    pos = jnp.array([[0, 0], [0, 1]])
    vals = jnp.array([1.0, 1.0])
    out = g.accumulate_neighborhood(pos, vals, include_center=True)
    assert np.allclose(np.asarray(out), [2.0, 2.0])


def test_random_positions_bounds_and_determinism():
    g = ToricGrid2D(6, 8)
    k = jax.random.key(0)
    p1 = g.random_positions(k, 100)
    p2 = g.random_positions(k, 100)
    assert p1.shape == (100, 2)
    assert np.array_equal(np.asarray(p1), np.asarray(p2))  # deterministic
    assert np.all(np.asarray(p1)[:, 0] < 6) and np.all(np.asarray(p1)[:, 0] >= 0)
    assert np.all(np.asarray(p1)[:, 1] < 8) and np.all(np.asarray(p1)[:, 1] >= 0)


def test_grid_ops_jit_able():
    g = ToricGrid2D(5, 5)

    @jax.jit
    def step(grid):
        return g.moore_sum(grid)

    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[2, 2].set(1)
    out = step(grid)
    assert int(out.sum()) == 8
