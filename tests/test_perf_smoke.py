"""Performance smoke test — sanity-check throughput (agent-ticks/sec) on the CPU backend.

This is a regression guard, not the real benchmark (GPU targets in SPEC are validated on
Linux/WSL2). It asserts a conservative lower bound so it won't flake on slow machines, and
prints the measured rate (visible with ``pytest -s``).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from evosim import rng
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.sim import Simulation
from evosim.state import State
from evosim.system import System


def _build_sim():
    sched = Scheduler(stages=("act", "environment"))
    sched.add(System("metabolize", "act",
                     lambda s, c: s.set("energy", s["energy"] * 0.999 + 0.001)))
    sched.add(System("drift", "environment",
                     lambda s, c: s.set("x", s["x"] + jax.random.normal(c.key, (s.capacity,)) * 0.01)))
    return Simulation(sched, seed=0)


def test_throughput_smoke(capsys):
    n_agents = 50_000
    steps = 100
    schema = Schema(energy=Field(dtype="float32", default=1.0), x=Field(dtype="float32"))
    s = State.create(schema, n_agents).set("alive", jnp.ones((n_agents,), dtype=bool))
    sim = _build_sim()

    # Warmup: trigger compilation for this (capacity, steps) shape.
    warm = sim.run(s, steps)
    jax.block_until_ready(warm)

    t0 = time.perf_counter()
    out = sim.run(s, steps)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t0

    agent_ticks = n_agents * steps
    rate = agent_ticks / dt
    with capsys.disabled():
        print(f"\n[perf] {n_agents} agents x {steps} steps in {dt*1000:.1f} ms "
              f"-> {rate/1e6:.1f}M agent-ticks/s (CPU)")

    assert int(out.tick) == steps
    # Conservative floor (CPU). Real GPU targets per SPEC are validated separately.
    assert rate > 1e5
