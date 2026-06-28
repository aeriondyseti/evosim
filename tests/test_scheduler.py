"""Unit + integration tests for evosim.system and evosim.scheduler."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evosim import rng
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.state import State
from evosim.system import Context, System, system


def make_state(capacity=4):
    schema = Schema(energy=Field(dtype="float32", default=0.0))
    return State.create(schema, capacity)


# -- system / registration ---------------------------------------------------

def test_system_decorator():
    @system("act", name="mover")
    def move(state, ctx):
        return state

    assert isinstance(move, System)
    assert move.stage == "act"
    assert move.name == "mover"


def test_ordered_respects_stage_then_registration():
    sched = Scheduler(stages=("sense", "act", "cleanup"))
    calls = []
    sched.add(System("a", "act", lambda s, c: (calls.append("a") or s)))
    sched.add(System("s", "sense", lambda s, c: (calls.append("s") or s)))
    sched.add(System("a2", "act", lambda s, c: (calls.append("a2") or s)))
    sched.add(System("c", "cleanup", lambda s, c: (calls.append("c") or s)))
    names = [sys.name for sys in sched.ordered()]
    assert names == ["s", "a", "a2", "c"]


def test_unknown_stage_raises():
    sched = Scheduler(stages=("act",))
    with pytest.raises(ValueError):
        sched.add(System("x", "nope", lambda s, c: s))


def test_add_raw_fn_requires_stage():
    sched = Scheduler()
    with pytest.raises(ValueError):
        sched.add(lambda s, c: s)


# -- execution ---------------------------------------------------------------

def test_make_tick_runs_and_increments_tick():
    sched = Scheduler(stages=("act",))
    sched.add(System("inc", "act", lambda s, c: s.set("energy", s["energy"] + 1.0)))
    tick = sched.make_tick(rng.root_key(0))
    out = tick(make_state())
    assert int(out.tick) == 1
    assert np.allclose(np.asarray(out["energy"]), 1.0)


def test_step_jit():
    sched = Scheduler(stages=("act",))
    sched.add(System("inc", "act", lambda s, c: s.set("energy", s["energy"] + 2.0)))
    out = sched.step(make_state(), rng.root_key(1))
    assert np.allclose(np.asarray(out["energy"]), 2.0)


def test_run_accumulates_over_steps():
    sched = Scheduler(stages=("act",))
    sched.add(System("inc", "act", lambda s, c: s.set("energy", s["energy"] + 1.0)))
    final = sched.run(make_state(), n_steps=10, root_key=rng.root_key(0))
    assert int(final.tick) == 10
    assert np.allclose(np.asarray(final["energy"]), 10.0)


def test_run_with_record():
    sched = Scheduler(stages=("act",))
    sched.add(System("inc", "act", lambda s, c: s.set("energy", s["energy"] + 1.0)))
    final, recs = sched.run(make_state(), n_steps=5, root_key=rng.root_key(0),
                            record=lambda s: jnp.sum(s["energy"]))
    # 4 slots, energy grows 1 per tick: sums are 4,8,12,16,20
    assert np.allclose(np.asarray(recs), [4, 8, 12, 16, 20])


def test_per_system_keys_distinct():
    sched = Scheduler(stages=("act",))
    sched.add(System("A", "act", lambda s, c: s.set_field("a", rng.key_bits(c.key))))
    sched.add(System("B", "act", lambda s, c: s.set_field("b", rng.key_bits(c.key))))
    out = sched.step(make_state(), rng.root_key(7), jit=False)
    assert not np.array_equal(np.asarray(out.get_field("a")), np.asarray(out.get_field("b")))


def test_rng_determinism_same_and_different_seed():
    def build():
        sched = Scheduler(stages=("act",))

        def add_noise(s, c):
            return s.set("energy", s["energy"] + jax.random.uniform(c.key, (s.capacity,)))

        sched.add(System("noise", "act", add_noise))
        return sched

    a = build().run(make_state(), 8, rng.root_key(123))
    b = build().run(make_state(), 8, rng.root_key(123))
    c = build().run(make_state(), 8, rng.root_key(999))
    assert np.allclose(np.asarray(a["energy"]), np.asarray(b["energy"]))
    assert not np.allclose(np.asarray(a["energy"]), np.asarray(c["energy"]))


def test_fields_update_through_scan():
    # Conway-like: a single system that increments a grid field each tick.
    schema = Schema(energy=Field(dtype="float32"))
    grid = jnp.zeros((3, 3), dtype=jnp.int32)
    s0 = State.create(schema, 0, fields={"cells": grid})  # no agents

    sched = Scheduler(stages=("environment",))
    sched.add(System("grow", "environment",
                     lambda s, c: s.set_field("cells", s.get_field("cells") + 1)))
    final = sched.run(s0, n_steps=4, root_key=rng.root_key(0))
    assert np.all(np.asarray(final.get_field("cells")) == 4)
    assert int(final.tick) == 4
