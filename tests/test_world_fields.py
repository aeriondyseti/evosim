"""Unit tests for evosim.world.fields (field-layer dynamics systems)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from evosim import rng
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.state import State
from evosim.system import Context
from evosim.world import ToricGrid2D, decay, diffuse, life_like, regrow


def make_state(grid, world_shape=None):
    schema = Schema(dummy=Field(dtype="float32"))
    return State.create(schema, 0, fields={"f": grid})


def ctx(world):
    return Context(tick=jnp.asarray(0, jnp.int32), key=rng.root_key(0), world=world)


def test_decay():
    s = make_state(jnp.full((3, 3), 10.0))
    sys = decay("f", 0.1)
    out = sys(s, ctx(ToricGrid2D(3, 3)))
    assert np.allclose(np.asarray(out.get_field("f")), 9.0)


def test_regrow_with_cap():
    s = make_state(jnp.full((2, 2), 4.0))
    out = regrow("f", 3.0, max_value=5.0)(s, ctx(ToricGrid2D(2, 2)))
    assert np.allclose(np.asarray(out.get_field("f")), 5.0)


def test_diffuse_conserves_mass_and_spreads():
    g = ToricGrid2D(5, 5)
    grid = jnp.zeros((5, 5), dtype=jnp.float32).at[2, 2].set(100.0)
    s = make_state(grid)
    out = diffuse("f", 0.2)(s, ctx(g))
    f = np.asarray(out.get_field("f"))
    assert np.isclose(f.sum(), 100.0, atol=1e-3)  # mass conserved
    assert f[2, 2] < 100.0  # spike spread out
    assert f[2, 1] > 0.0    # neighbor gained mass


def test_life_like_blinker_oscillates():
    g = ToricGrid2D(5, 5)
    # vertical blinker (3 cells in a column)
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[1, 2].set(1).at[2, 2].set(1).at[3, 2].set(1)
    s = make_state(grid)
    conway = life_like("f")  # born {3}, survive {2,3}
    s1 = conway(s, ctx(g))
    f1 = np.asarray(s1.get_field("f"))
    # after one step: horizontal blinker (row 2, cols 1,2,3)
    assert int(f1[2, 1]) == 1 and int(f1[2, 2]) == 1 and int(f1[2, 3]) == 1
    assert int(f1[1, 2]) == 0 and int(f1[3, 2]) == 0
    # after two steps: back to vertical
    s2 = conway(s1, ctx(g))
    f2 = np.asarray(s2.get_field("f"))
    assert np.array_equal(f2, np.asarray(grid))


def test_life_like_block_is_still():
    g = ToricGrid2D(6, 6)
    grid = jnp.zeros((6, 6), dtype=jnp.int32)
    for r in (2, 3):
        for c in (2, 3):
            grid = grid.at[r, c].set(1)
    s = make_state(grid)
    out = life_like("f")(s, ctx(g))
    assert np.array_equal(np.asarray(out.get_field("f")), np.asarray(grid))


def test_field_system_runs_in_scheduler_scan():
    g = ToricGrid2D(5, 5)
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[1, 2].set(1).at[2, 2].set(1).at[3, 2].set(1)
    s = make_state(grid)
    sched = Scheduler(stages=("environment",))
    sched.add(life_like("f"))
    # 2 ticks -> blinker returns to start
    final = sched.run(s, n_steps=2, root_key=rng.root_key(0), world=g)
    assert np.array_equal(np.asarray(final.get_field("f")), np.asarray(grid))
