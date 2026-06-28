"""Unit + integration tests for evosim.sim.Simulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim import population, rng
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.sim import Simulation
from evosim.state import State, state_fingerprint
from evosim.system import System


def inc_sched():
    sched = Scheduler(stages=("act",))
    sched.add(System("inc", "act", lambda s, c: s.set("energy", s["energy"] + 1.0)))
    return sched


def base_state(cap=4):
    schema = Schema(energy=Field(dtype="float32"))
    return State.create(schema, cap).set("alive", jnp.array([True] * cap))


def test_run_matches_scheduler():
    sched = inc_sched()
    s = base_state()
    sim = Simulation(sched, seed=0)
    final = sim.run(s, 5)
    assert np.allclose(np.asarray(final["energy"]), 5.0)
    assert int(final.tick) == 5


def test_new_state_uses_schema():
    schema = Schema(energy=Field(dtype="float32", default=2.0))
    sim = Simulation(inc_sched(), seed=0, schema=schema)
    s = sim.new_state(3)
    assert s.capacity == 3
    assert np.allclose(np.asarray(s["energy"]), 2.0)


def test_run_with_record():
    sim = Simulation(inc_sched(), seed=0)
    final, recs = sim.run(base_state(), 3, record=lambda s: jnp.sum(s["energy"]))
    assert np.allclose(np.asarray(recs), [4, 8, 12])


def test_run_with_growth_increases_capacity():
    schema = Schema(energy=Field(dtype="float32"))

    def spawner(s, c):
        # spawn one clone each tick
        return population.spawn(s, {"energy": s["energy"][:1]}).state

    sched = Scheduler(stages=("spawn",))
    sched.add(System("spawn1", "spawn", spawner))
    sim = Simulation(sched, seed=0)

    s0 = State.create(schema, 1).set_many({"alive": jnp.array([True]),
                                           "energy": jnp.array([1.0])})
    final = sim.run_with_growth(s0, n_steps=3, min_free=1)
    assert int(final.n_alive) == 4         # started 1, +1 per tick for 3 ticks
    assert final.capacity >= 4             # grew from 1


def test_run_ensemble_parallel_worlds():
    schema = Schema(energy=Field(dtype="float32"))

    def init_fn(key):
        s = State.create(schema, 4).set("alive", jnp.array([True] * 4))
        return s.set("energy", jax.random.normal(key, (4,)))

    def noise(s, c):
        return s.set("energy", s["energy"] + jax.random.normal(c.key, (s.capacity,)))

    sched = Scheduler(stages=("act",))
    sched.add(System("noise", "act", noise))
    sim = Simulation(sched, seed=7)

    batched = sim.run_ensemble(init_fn, n_worlds=5, n_steps=3)
    assert batched["energy"].shape == (5, 4)        # leading world axis
    e = np.asarray(batched["energy"])
    # worlds differ (independent keys)
    assert not np.allclose(e[0], e[1])


def test_run_ensemble_deterministic():
    schema = Schema(energy=Field(dtype="float32"))

    def init_fn(key):
        return State.create(schema, 3).set_many(
            {"alive": jnp.array([True] * 3), "energy": jax.random.normal(key, (3,))})

    sched = Scheduler(stages=("act",))
    sched.add(System("noise", "act",
                     lambda s, c: s.set("energy", s["energy"] + jax.random.normal(c.key, (s.capacity,)))))
    a = Simulation(sched, seed=1).run_ensemble(init_fn, 4, 2)
    b = Simulation(sched, seed=1).run_ensemble(init_fn, 4, 2)
    assert np.allclose(np.asarray(a["energy"]), np.asarray(b["energy"]))


def test_run_ensemble_with_record():
    schema = Schema(energy=Field(dtype="float32"))

    def init_fn(key):
        return State.create(schema, 4).set("alive", jnp.array([True] * 4))

    sim = Simulation(inc_sched(), seed=0)
    _, recs = sim.run_ensemble(init_fn, n_worlds=3, n_steps=2,
                               record=lambda s: jnp.sum(s["energy"]))
    # shape (n_worlds, n_steps)
    assert np.asarray(recs).shape == (3, 2)
