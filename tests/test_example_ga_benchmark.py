"""Integration tests for the GA benchmark demo (explicit-fitness path)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim.examples import ga_benchmark as GA
from evosim.state import state_fingerprint


def _run(cfg, gens, seed=0):
    sim = GA.build(cfg, seed=seed)
    s = GA.initial_state(sim, cfg, jax.random.key(0))
    objective = GA.FITNESS_FUNCTIONS[cfg.objective]
    final, best = sim.run(s, gens, record=lambda st: jnp.min(objective(st["genome"])))
    return final, np.asarray(best)


def test_fitness_functions_min_at_zero():
    z = jnp.zeros((1, 5))
    assert float(GA.sphere(z)[0]) == 0.0
    assert np.isclose(float(GA.rastrigin(z)[0]), 0.0)


def test_sphere_converges():
    cfg = GA.GAConfig(dim=10, pop_size=128, objective="sphere")
    _, best = _run(cfg, gens=120)
    assert best[-1] < best[0]           # improved
    assert best[-1] < 0.2               # converged near optimum (0)


def test_elitism_monotonic_non_increasing():
    cfg = GA.GAConfig(dim=8, pop_size=128, elite=2, objective="sphere")
    _, best = _run(cfg, gens=60)
    # with elitism the best objective never gets worse generation-to-generation
    diffs = np.diff(best)
    assert np.all(diffs <= 1e-5)


def test_rastrigin_improves():
    cfg = GA.GAConfig(dim=5, pop_size=200, mut_sigma=0.2, objective="rastrigin")
    _, best = _run(cfg, gens=120)
    assert best[-1] < best[0]           # improves on the hard multimodal landscape


def test_determinism():
    cfg = GA.GAConfig(dim=6, pop_size=64, objective="sphere")
    a, ba = _run(cfg, gens=30, seed=0)
    b, bb = _run(cfg, gens=30, seed=0)
    assert state_fingerprint(a) == state_fingerprint(b)
    assert np.array_equal(ba, bb)


def test_different_seed_differs():
    cfg = GA.GAConfig(dim=6, pop_size=64, objective="sphere")
    a, _ = _run(cfg, gens=30, seed=0)
    b, _ = _run(cfg, gens=30, seed=1)
    assert state_fingerprint(a) != state_fingerprint(b)
