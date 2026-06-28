"""Unit tests for evosim.operators (mutation / crossover / selection)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim import rng
from evosim.operators import crossover, mutation, selection


# --- mutation ---------------------------------------------------------------

def test_gaussian_changes_and_preserves_shape_dtype():
    g = jnp.zeros((4, 8), dtype=jnp.float32)
    out = mutation.gaussian(rng.root_key(0), g, sigma=0.5)
    assert out.shape == g.shape and out.dtype == g.dtype
    assert not np.allclose(np.asarray(out), 0.0)


def test_gaussian_rate_zero_is_noop():
    g = jnp.ones((3, 5), dtype=jnp.float32)
    out = mutation.gaussian(rng.root_key(1), g, sigma=1.0, rate=0.0)
    assert np.allclose(np.asarray(out), 1.0)


def test_gaussian_deterministic():
    g = jnp.zeros((4, 4), dtype=jnp.float32)
    a = mutation.gaussian(rng.root_key(2), g, sigma=0.3)
    b = mutation.gaussian(rng.root_key(2), g, sigma=0.3)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_gaussian_clip():
    g = jnp.zeros((100,), dtype=jnp.float32)
    out = mutation.gaussian(rng.root_key(3), g, sigma=5.0, clip=(-1.0, 1.0))
    assert np.asarray(out).min() >= -1.0 and np.asarray(out).max() <= 1.0


def test_uniform_mutation_in_range():
    g = jnp.zeros((1000,), dtype=jnp.float32)
    out = np.asarray(mutation.uniform(rng.root_key(4), g, low=2.0, high=3.0, rate=1.0))
    assert out.min() >= 2.0 and out.max() < 3.0


def test_bitflip_binary_int():
    g = jnp.zeros((1000,), dtype=jnp.int32)
    out = mutation.bitflip(rng.root_key(5), g, rate=0.3)
    vals = np.asarray(out)
    assert set(np.unique(vals)).issubset({0, 1})
    assert 0.2 < vals.mean() < 0.4  # roughly the flip rate


def test_bitflip_bool():
    g = jnp.zeros((100,), dtype=jnp.bool_)
    out = mutation.bitflip(rng.root_key(6), g, rate=1.0)
    assert np.all(np.asarray(out))  # all flipped to True


# --- crossover --------------------------------------------------------------

def test_uniform_crossover_genes_from_parents():
    p1 = jnp.full((50,), 1.0)
    p2 = jnp.full((50,), 2.0)
    child = np.asarray(crossover.uniform(rng.root_key(0), p1, p2))
    assert set(np.unique(child)).issubset({1.0, 2.0})
    assert 1.0 in child and 2.0 in child  # mixed


def test_one_point_prefix_suffix():
    p1 = jnp.arange(10.0)
    p2 = jnp.full((10,), -1.0)
    child = np.asarray(crossover.one_point(rng.root_key(1), p1, p2))
    # there is exactly one transition from p1-values to p2-values
    is_p2 = child == -1.0
    # once it switches to p2 it stays p2
    first_p2 = np.argmax(is_p2) if is_p2.any() else len(child)
    assert np.all(is_p2[first_p2:]) if is_p2.any() else True
    assert 1 <= first_p2 < 10  # cut in [1, G)


def test_n_point_batched_shape():
    p1 = jnp.ones((8, 12))
    p2 = jnp.zeros((8, 12))
    child = crossover.n_point(rng.root_key(2), p1, p2, n=3)
    assert child.shape == (8, 12)
    assert set(np.unique(np.asarray(child))).issubset({0.0, 1.0})


def test_blend_midpoint():
    p1 = jnp.full((5,), 0.0)
    p2 = jnp.full((5,), 10.0)
    child = crossover.blend(rng.root_key(3), p1, p2, alpha=0.5)
    assert np.allclose(np.asarray(child), 5.0)


def test_clone():
    p1 = jnp.arange(5.0)
    assert np.array_equal(np.asarray(crossover.clone(p1)), np.asarray(p1))


# --- selection --------------------------------------------------------------

def test_tournament_favors_best_majority():
    # Sampling is with replacement, so the best isn't guaranteed every tournament, but a
    # large tournament size makes it the majority winner.
    fit = jnp.array([0.1, 0.5, 0.9, 0.3])
    sel = np.asarray(selection.tournament(rng.root_key(0), fit, num_selected=2000,
                                          tournament_size=4))
    frac_best = (sel == 2).mean()
    assert frac_best > 0.5  # index 2 (max) wins most tournaments


def test_tournament_bias_toward_fit():
    fit = jnp.arange(20.0)
    sel = np.asarray(selection.tournament(rng.root_key(1), fit, 2000, tournament_size=3))
    # mean selected fitness should beat the population mean (9.5)
    assert fit[sel].mean() > 12.0


def test_roulette_concentrates_on_high_fitness():
    fit = jnp.array([0.0, 0.0, 0.0, 1.0])
    sel = selection.roulette(rng.root_key(2), fit, num_selected=50)
    assert np.all(np.asarray(sel) == 3)


def test_truncation_within_top():
    fit = jnp.arange(10.0)  # top 50% = indices 5..9
    sel = np.asarray(selection.truncation(rng.root_key(3), fit, num_selected=100, frac=0.5))
    assert set(np.unique(sel)).issubset({5, 6, 7, 8, 9})


def test_elitism_returns_top():
    fit = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
    elite = np.asarray(selection.elitism(fit, 2))
    assert np.array_equal(elite, [4, 2])  # indices of 5.0, 4.0


def test_selection_deterministic():
    fit = jnp.arange(30.0)
    a = selection.tournament(rng.root_key(7), fit, 50)
    b = selection.tournament(rng.root_key(7), fit, 50)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_operators_jit_able():
    @jax.jit
    def breed(key, p1, p2, fit):
        sel = selection.tournament(key, fit, p1.shape[0])
        child = crossover.uniform(key, p1[sel], p2[sel])
        return mutation.gaussian(key, child, sigma=0.1)

    p1 = jnp.ones((6, 4))
    p2 = jnp.zeros((6, 4))
    fit = jnp.arange(6.0)
    out = breed(rng.root_key(0), p1, p2, fit)
    assert out.shape == (6, 4)
