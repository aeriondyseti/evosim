"""Integration tests for the Conway's Life demo (golden-master patterns)."""

from __future__ import annotations

import jax
import numpy as np

from evosim.examples import conway


def test_glider_translates_by_one_diagonal_after_4_steps():
    sim = conway.build(10, 10)
    g0 = conway.stamp(conway.empty_grid(10, 10), conway.GLIDER, top=2, left=2)
    state = conway.initial_state(sim, g0)
    final, _ = conway.run_history(sim, state, 4)
    expected = conway.stamp(conway.empty_grid(10, 10), conway.GLIDER, top=3, left=3)
    assert np.array_equal(np.asarray(final.get_field("cells")), np.asarray(expected))


def test_blinker_period_two():
    sim = conway.build(7, 7)
    g0 = conway.stamp(conway.empty_grid(7, 7), conway.BLINKER, top=3, left=2)
    state = conway.initial_state(sim, g0)
    final, _ = conway.run_history(sim, state, 2)
    assert np.array_equal(np.asarray(final.get_field("cells")), np.asarray(g0))


def test_block_is_still_life():
    sim = conway.build(6, 6)
    g0 = conway.stamp(conway.empty_grid(6, 6), conway.BLOCK, top=2, left=2)
    state = conway.initial_state(sim, g0)
    final, _ = conway.run_history(sim, state, 10)
    assert np.array_equal(np.asarray(final.get_field("cells")), np.asarray(g0))


def test_glider_population_constant():
    sim = conway.build(12, 12)
    g0 = conway.stamp(conway.empty_grid(12, 12), conway.GLIDER, top=1, left=1)
    state = conway.initial_state(sim, g0)
    _, hist = conway.run_history(sim, state, 16)
    pops = conway.population_series(hist)
    assert np.all(pops == 5)  # a glider always has 5 live cells


def test_history_shape_and_determinism():
    sim = conway.build(8, 8)
    g0 = conway.random_grid(jax.random.key(1), 8, 8, density=0.3)
    s = conway.initial_state(sim, g0)
    final_a, hist_a = conway.run_history(sim, s, 20)
    final_b, hist_b = conway.run_history(sim, s, 20)
    assert hist_a.shape == (20, 8, 8)
    assert np.array_equal(np.asarray(hist_a), np.asarray(hist_b))  # deterministic


def test_render_smoke():
    g = conway.stamp(conway.empty_grid(3, 3), conway.BLOCK)
    out = conway.render(g)
    assert out.count("#") == 4
