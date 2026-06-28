"""Unit tests for evosim.checkpoint (save/load + deterministic resume)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim import checkpoint, rng
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.state import State, state_fingerprint
from evosim.system import System


def make_state():
    schema = Schema(
        energy=Field(dtype="float32", default=1.0),
        genome=Field(dtype="float32", shape=(3,)),
        kind=Field(dtype="int8", default=2),
    )
    s = State.create(schema, 5, fields={"resource": jnp.ones((4, 4), dtype=jnp.float32)})
    s = s.set("alive", jnp.array([True, True, False, True, False]))
    return s


def test_save_load_roundtrip(tmp_path):
    s = make_state()
    path = str(tmp_path / "ckpt.npz")
    checkpoint.save(path, s)
    cp = checkpoint.load(path)
    assert state_fingerprint(cp.state) == state_fingerprint(s)
    assert cp.state.capacity == s.capacity
    assert cp.state.schema.names == s.schema.names
    assert cp.state.has_field("resource")


def test_rng_key_roundtrip(tmp_path):
    s = make_state()
    key = rng.root_key(42)
    path = str(tmp_path / "ckpt.npz")
    checkpoint.save(path, s, rng_key=key)
    cp = checkpoint.load(path)
    assert cp.rng_key is not None
    assert rng.keys_equal(cp.rng_key, key)


def test_meta_roundtrip(tmp_path):
    s = make_state()
    path = str(tmp_path / "ckpt.npz")
    checkpoint.save(path, s, meta={"run": "test", "step": 7})
    cp = checkpoint.load(path)
    assert cp.meta == {"run": "test", "step": 7}


def test_deterministic_resume(tmp_path):
    def build():
        schema = Schema(energy=Field(dtype="float32"))
        st = State.create(schema, 4).set("alive", jnp.array([True] * 4))
        sched = Scheduler(stages=("act",))
        sched.add(System("noise", "act",
                         lambda s, c: s.set("energy", s["energy"] + jax.random.normal(c.key, (s.capacity,)))))
        return sched, st

    root = rng.root_key(123)
    # full run of 12 ticks
    full_sched, s0 = build()
    full = full_sched.run(s0, 12, root)

    # run 7, checkpoint, resume 5
    mid_sched, s0b = build()
    mid = mid_sched.run(s0b, 7, root)
    path = str(tmp_path / "resume.npz")
    checkpoint.save(path, mid, rng_key=root)
    cp = checkpoint.load(path)
    assert int(cp.state.tick) == 7
    cont_sched, _ = build()
    cont = cont_sched.run(cp.state, 5, cp.rng_key)

    assert state_fingerprint(cont) == state_fingerprint(full)
