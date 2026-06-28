"""Integration tests for the evolving-foragers demo.

These assert the framework's full agent-based loop works end-to-end: the population survives,
reproduction/death occur, and emergent natural selection produces a directional shift in the
heritable efficiency gene. Runs are deterministic (fixed seed), so the signal is reproducible.
"""

from __future__ import annotations

import jax
import numpy as np

from evosim import metrics
from evosim.examples import foragers as F
from evosim.state import state_fingerprint

CFG = F.ForagerConfig(height=24, width=24, capacity=1500, n_initial=150)
STEPS = 180


def _run(seed=0):
    sim = F.build(CFG, seed=seed)
    s = F.initial_state(sim, CFG, jax.random.key(0))
    final, recs = sim.run(
        s, STEPS,
        record=lambda st: {"pop": st.n_alive,
                           "eff": metrics.masked_mean(st["genome"][:, 0], st.alive)},
    )
    return sim, final, recs


def test_population_survives_and_is_bounded():
    _, final, recs = _run()
    pop = np.asarray(recs["pop"])
    assert pop[-1] > 0            # not extinct
    assert pop.min() > 0          # never went extinct mid-run
    assert pop.max() <= CFG.capacity  # never exceeds capacity


def test_reproduction_and_death_occur():
    _, final, recs = _run()
    # ids were assigned beyond the initial cohort -> births happened
    assert int(final.next_id) > CFG.n_initial


def test_emergent_selection_increases_efficiency_gene():
    _, _, recs = _run()
    eff = np.asarray(recs["eff"])
    # directional selection: efficient agents pay less to live -> mean gene rises
    assert eff[-1] > eff[0] + 0.004
    # and the trend is generally upward (end well above the early average)
    assert eff[-1] > eff[:20].mean()


def test_determinism_same_seed():
    _, final_a, _ = _run(seed=0)
    _, final_b, _ = _run(seed=0)
    assert state_fingerprint(final_a) == state_fingerprint(final_b)


def test_different_seed_differs():
    _, final_a, _ = _run(seed=0)
    _, final_b, _ = _run(seed=1)
    assert state_fingerprint(final_a) != state_fingerprint(final_b)
