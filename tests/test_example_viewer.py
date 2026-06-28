"""Tests for the example PyGame viewer (evosim.examples.pygame_viewer).

Exercised headlessly via SDL's dummy video driver, so they run in CI without a display.
Skipped entirely if pygame (the optional ``viz`` extra) is not installed.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import jax  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("pygame")

from evosim import rng  # noqa: E402
from evosim.recorders import run_recorded  # noqa: E402
from evosim.viz import AgentRenderer, GridRenderer  # noqa: E402
from evosim.examples import conway, foragers, ga_benchmark  # noqa: E402
from evosim.viz import compose  # noqa: E402
from evosim.examples.pygame_viewer import PygameViewer, agent_overlay, run_live  # noqa: E402


def test_run_live_conway_headless():
    sim = conway.build(8, 8)
    state = conway.initial_state(sim, conway.stamp(conway.empty_grid(8, 8), conway.GLIDER, 1, 1))
    final = run_live(sim, state, n_steps=5,
                     layers=[GridRenderer("cells", cmap="green", vmin=0, vmax=1)],
                     px_per_cell=4, fps=1000)
    assert int(final.tick) == 5


def test_run_live_foragers_headless():
    cfg = foragers.ForagerConfig(height=12, width=12, capacity=128, n_initial=30)
    sim = foragers.build(cfg)
    state = foragers.initial_state(sim, cfg, jax.random.key(0))
    layers = [GridRenderer("food", cmap="fire", vmin=0, vmax=cfg.food_max),
              AgentRenderer("position", color_by="energy", cmap="viridis")]
    final = run_live(sim, state, n_steps=4, layers=layers, px_per_cell=4, fps=1000)
    assert int(final.tick) == 4


def test_foragers_field_background_with_agent_dots_overlay():
    # Food field as frame_fn background; agents drawn as dots via agent_overlay.
    cfg = foragers.ForagerConfig(height=12, width=12, capacity=128, n_initial=30)
    sim = foragers.build(cfg)
    state = foragers.initial_state(sim, cfg, jax.random.key(0))
    field = GridRenderer("food", cmap="fire", vmin=0, vmax=cfg.food_max)
    final = run_live(
        sim, state, n_steps=4,
        frame_fn=lambda s: compose([field], s, sim.world),
        overlay_fn=agent_overlay(sim.world, color_by="energy", cmap="viridis", vmin=0, vmax=2),
        px_per_cell=8, fps=1000)
    assert int(final.tick) == 4


def test_run_live_default_layers():
    sim = conway.build(6, 6)
    state = conway.initial_state(sim, conway.random_grid(jax.random.key(0), 6, 6, 0.3))
    final = run_live(sim, state, n_steps=3, px_per_cell=4, fps=1000)
    assert int(final.tick) == 3


def test_run_live_legend_toggle_runs():
    sim = conway.build(10, 10)
    state = conway.initial_state(sim, conway.stamp(conway.empty_grid(10, 10), conway.GLIDER, 1, 1))
    a = run_live(sim, state, n_steps=2, px_per_cell=6, fps=1000, show_legend=True)
    b = run_live(sim, state, n_steps=2, px_per_cell=6, fps=1000, show_legend=False)
    assert int(a.tick) == 2 and int(b.tick) == 2


def test_ga_run_view_headless():
    # Non-grid sim visualized via run_live(frame_fn=...) + ScatterRenderer.
    cfg = ga_benchmark.GAConfig(dim=4, pop_size=64, objective="sphere")
    final = ga_benchmark.run_view(cfg, seed=0, steps=5)
    assert int(final.tick) == 5


def test_run_live_frame_fn_no_world():
    # run_live works without a world when a frame_fn is supplied.
    import numpy as np
    cfg = ga_benchmark.GAConfig(dim=3, pop_size=16, objective="sphere")
    sim = ga_benchmark.build(cfg)               # sim.world is None
    state = ga_benchmark.initial_state(sim, cfg, jax.random.key(0))
    final = run_live(sim, state, n_steps=3,
                     frame_fn=lambda s: np.zeros((32, 32, 3), dtype=np.uint8),
                     fps=1000)
    assert int(final.tick) == 3


def test_pygame_viewer_as_recorder():
    sim = conway.build(8, 8)
    state = conway.initial_state(sim, conway.stamp(conway.empty_grid(8, 8), conway.BLINKER, 3, 2))
    viewer = PygameViewer(sim.world, layers=[GridRenderer("cells", cmap="green", vmin=0, vmax=1)],
                          px_per_cell=4, fps=1000)
    try:
        final = run_recorded(sim.scheduler, state, 3, rng.root_key(0),
                             recorders=[viewer], world=sim.world)
        assert int(final.tick) == 3
        assert viewer.closed is False
    finally:
        viewer.close()
