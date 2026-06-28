"""Consolidated determinism golden-masters across the engine (per SPEC: same-device bit-exact).

Exercises a mixed simulation (agents + RNG + environment field + spawn + kill) and asserts:
- same seed -> identical final fingerprint (reproducible),
- different seed -> different fingerprint (RNG actually drives the sim),
- scan-based run == host-loop run (the two run paths agree bit-for-bit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from evosim import population, rng
from evosim.recorders import run_recorded
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.sim import Simulation
from evosim.state import State, state_fingerprint
from evosim.system import System
from evosim.world import ToricGrid2D, diffuse


def _build():
    world = ToricGrid2D(16, 16)
    schema = Schema(position=Field(dtype="int16", shape=(2,)), energy=Field(dtype="float32"))

    def wander(s, c):
        step = jax.random.randint(c.key, (s.capacity, 2), -1, 2)
        return s.set("position", c.world.move(s["position"], step).astype(jnp.int16))

    def deposit(s, c):
        grid = c.world.scatter_field(s["position"], jnp.ones((s.capacity,)), s.alive)
        return s.set_field("pher", s.get_field("pher") + grid)

    def churn(s, c):
        # kill ~half by a random mask, spawn a couple of clones
        kmask = jnp.logical_and(s.alive, jax.random.uniform(c.key, (s.capacity,)) < 0.1)
        s = population.kill(s, kmask)
        return population.spawn(s, {"position": s["position"][:2], "energy": s["energy"][:2]}).state

    sched = Scheduler()
    sched.add(System("wander", "act", wander))
    sched.add(System("deposit", "interact", deposit))
    sched.add(diffuse("pher", 0.1))           # environment-stage field dynamics
    sched.add(System("churn", "death", churn))
    return sched, world, schema


def _initial(schema, world):
    s = State.create(schema, 64, fields={"pher": jnp.zeros((16, 16), dtype=jnp.float32)})
    idx = jnp.arange(64)
    alive = idx < 32
    pos = world.random_positions(jax.random.key(0), 64).astype(jnp.int16)
    return s.set_many({"alive": alive, "position": pos,
                       "id": jnp.where(alive, idx, -1).astype(jnp.int32)}).replace(
        next_id=jnp.asarray(32, dtype=jnp.int32))


def test_same_seed_identical():
    sched, world, schema = _build()
    s0 = _initial(schema, world)
    a = Simulation(sched, world=world, seed=7).run(s0, 40)
    b = Simulation(sched, world=world, seed=7).run(s0, 40)
    assert state_fingerprint(a) == state_fingerprint(b)


def test_different_seed_differs():
    sched, world, schema = _build()
    s0 = _initial(schema, world)
    a = Simulation(sched, world=world, seed=7).run(s0, 40)
    b = Simulation(sched, world=world, seed=8).run(s0, 40)
    assert state_fingerprint(a) != state_fingerprint(b)


def test_scan_equals_host_loop():
    sched, world, schema = _build()
    s0 = _initial(schema, world)
    scan_final = Simulation(sched, world=world, seed=7).run(s0, 40)
    host_final = run_recorded(sched, s0, 40, rng.root_key(7), world=world)
    assert state_fingerprint(scan_final) == state_fingerprint(host_final)
