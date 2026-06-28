"""Unit tests for evosim.metrics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from evosim import metrics, rng
from evosim.scheduler import Scheduler
from evosim.schema import Field, Schema
from evosim.state import State
from evosim.system import System


def test_masked_mean_scalar():
    vals = jnp.array([1.0, 2.0, 3.0, 100.0])
    mask = jnp.array([True, True, True, False])
    assert float(metrics.masked_mean(vals, mask)) == 2.0


def test_masked_mean_vector():
    vals = jnp.array([[1.0, 10.0], [3.0, 30.0], [0.0, 0.0]])
    mask = jnp.array([True, True, False])
    out = np.asarray(metrics.masked_mean(vals, mask))
    assert np.allclose(out, [2.0, 20.0])


def test_masked_mean_no_alive_is_zero():
    vals = jnp.array([1.0, 2.0])
    mask = jnp.array([False, False])
    assert float(metrics.masked_mean(vals, mask)) == 0.0


def test_masked_var():
    vals = jnp.array([2.0, 4.0, 6.0, 999.0])
    mask = jnp.array([True, True, True, False])
    # variance of [2,4,6] = 2.667
    assert np.isclose(float(metrics.masked_var(vals, mask)), 8.0 / 3.0)


def test_population_and_compute():
    schema = Schema(energy=Field(dtype="float32"))
    s = State.create(schema, 4)
    s = s.set_many({"alive": jnp.array([True, True, False, False]),
                    "energy": jnp.array([5.0, 7.0, 0.0, 0.0])})
    ms = metrics.standard(scalar_fields=["energy"])
    out = ms.compute(s)
    assert int(out["population"]) == 2
    assert float(out["energy_mean"]) == 6.0
    assert float(out["energy_var"]) == 1.0


def test_record_fn_in_scheduler_run():
    schema = Schema(energy=Field(dtype="float32"))
    s = State.create(schema, 4)
    s = s.set("alive", jnp.array([True, True, True, True]))
    sched = Scheduler(stages=("act",))
    sched.add(System("inc", "act", lambda st, c: st.set("energy", st["energy"] + 1.0)))
    ms = metrics.standard(scalar_fields=["energy"])
    final, recs = sched.run(s, n_steps=3, root_key=rng.root_key(0), record=ms.record_fn())
    assert np.array_equal(np.asarray(recs["population"]), [4, 4, 4])
    assert np.allclose(np.asarray(recs["energy_mean"]), [1.0, 2.0, 3.0])


def test_genetic_diversity():
    schema = Schema(genome=Field(dtype="float32", shape=(2,)))
    s = State.create(schema, 3)
    s = s.set_many({"alive": jnp.array([True, True, True]),
                    "genome": jnp.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])})
    div = metrics.genetic_diversity("genome")(s)
    # per-gene var of [0,2,4] = 8/3 for both genes; mean = 8/3
    assert np.isclose(float(div), 8.0 / 3.0)
