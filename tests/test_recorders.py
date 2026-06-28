"""Unit tests for evosim.recorders."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim import metrics, rng
from evosim.recorders import MetricsRecorder, SnapshotRecorder, run_recorded
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.state import State, state_fingerprint
from evosim.system import System


def build():
    schema = Schema(energy=Field(dtype="float32"))
    s = State.create(schema, 4)
    s = s.set("alive", jnp.array([True, True, True, True]))
    sched = Scheduler(stages=("act",))
    sched.add(System("inc", "act", lambda st, c: st.set("energy", st["energy"] + 1.0)))
    return sched, s


def test_metrics_recorder_collects_per_tick():
    sched, s = build()
    rec = MetricsRecorder(metrics.standard(scalar_fields=["energy"]))
    final = run_recorded(sched, s, n_steps=5, root_key=rng.root_key(0), recorders=[rec])
    out = rec.result()
    assert np.array_equal(out["tick"], [1, 2, 3, 4, 5])
    assert np.allclose(out["energy_mean"], [1, 2, 3, 4, 5])
    assert int(final.tick) == 5


def test_metrics_recorder_subsampling():
    sched, s = build()
    rec = MetricsRecorder(metrics.standard(scalar_fields=["energy"]), every=2)
    run_recorded(sched, s, n_steps=6, root_key=rng.root_key(0), recorders=[rec])
    out = rec.result()
    # records at the 1st, 3rd, 5th invocation -> ticks 1,3,5
    assert np.array_equal(out["tick"], [1, 3, 5])


def test_snapshot_recorder_and_save(tmp_path):
    sched, s = build()
    rec = SnapshotRecorder(components=["energy"], every=2)
    run_recorded(sched, s, n_steps=4, root_key=rng.root_key(0), recorders=[rec])
    snaps = rec.result()
    # records on the 1st and 3rd invocation -> ticks 1 and 3 (consistent with `every`)
    assert [t for t, _ in snaps] == [1, 3]
    assert np.allclose(snaps[0][1]["energy"], 1.0)

    path = str(tmp_path / "snaps.npz")
    rec.save_npz(path)
    loaded = np.load(path)
    assert np.array_equal(loaded["ticks"], [1, 3])
    assert loaded["energy"].shape == (2, 4)


def test_run_recorded_matches_scan_run():
    # Host-loop run_recorded must match scan-based run bit-for-bit (determinism).
    sched, s = build()
    final_host = run_recorded(sched, s, n_steps=7, root_key=rng.root_key(3))
    sched2, s2 = build()
    final_scan = sched2.run(s2, n_steps=7, root_key=rng.root_key(3))
    assert state_fingerprint(final_host) == state_fingerprint(final_scan)


def test_run_recorded_with_rng_system_determinism():
    def build_noise():
        schema = Schema(energy=Field(dtype="float32"))
        st = State.create(schema, 4).set("alive", jnp.array([True] * 4))
        sched = Scheduler(stages=("act",))
        sched.add(System("noise", "act",
                         lambda s, c: s.set("energy", s["energy"] + jax.random.normal(c.key, (s.capacity,)))))
        return sched, st

    a_sched, a_s = build_noise()
    b_sched, b_s = build_noise()
    fa = run_recorded(a_sched, a_s, 6, rng.root_key(9))
    fb = b_sched.run(b_s, 6, rng.root_key(9))
    assert state_fingerprint(fa) == state_fingerprint(fb)
