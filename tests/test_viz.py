"""Tests for the visualization layer.

Renderers are pure-numpy; the pygame window code is exercised headlessly via SDL's dummy
video driver, so these run in CI without a display.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("pygame")

from evosim.recorders import run_recorded  # noqa: E402
from evosim import rng  # noqa: E402
from evosim.viz import (AgentRenderer, GridRenderer, PygameViewer, apply_colormap,  # noqa: E402
                        compose, run_live)
from evosim.examples import conway, foragers  # noqa: E402


# --- renderers (pure numpy) -------------------------------------------------

def test_apply_colormap_shape_and_dtype():
    norm = np.linspace(0, 1, 12).reshape(3, 4)
    img = apply_colormap(norm, "viridis")
    assert img.shape == (3, 4, 3)
    assert img.dtype == np.uint8


def test_apply_colormap_unknown_raises():
    with pytest.raises(KeyError):
        apply_colormap(np.zeros((2, 2)), "nope")


def test_grid_renderer_on_conway():
    sim = conway.build(6, 8)
    g = conway.stamp(conway.empty_grid(6, 8), conway.BLOCK, 1, 1)
    state = conway.initial_state(sim, g)
    img, mask = GridRenderer("cells", cmap="green", vmin=0, vmax=1).render(state, sim.world)
    assert img.shape == (6, 8, 3)
    assert mask.shape == (6, 8) and mask.all()  # grid layer fills everything
    # live cells are brighter than dead background
    assert img[1, 1].sum() > img[0, 0].sum()


def test_agent_renderer_marks_agent_cells():
    cfg = foragers.ForagerConfig(height=10, width=10, capacity=8, n_initial=3)
    sim = foragers.build(cfg)
    s = foragers.initial_state(sim, cfg, jax.random.key(0))
    img, mask = AgentRenderer("position", color_by="energy").render(s, sim.world)
    assert mask.shape == (10, 10)
    assert int(mask.sum()) == int(np.asarray(s.n_alive))  # one marked cell per alive agent (distinct positions)


def test_compose_layers_paint_on_top():
    cfg = foragers.ForagerConfig(height=8, width=8, capacity=4, n_initial=2)
    sim = foragers.build(cfg)
    s = foragers.initial_state(sim, cfg, jax.random.key(1))
    layers = [GridRenderer("food", cmap="fire"), AgentRenderer("position", color_by=None,
                                                               color=(255, 255, 255))]
    img = compose(layers, s, sim.world)
    assert img.shape == (8, 8, 3)


# --- pygame window (headless) ----------------------------------------------

def test_run_live_conway_headless():
    sim = conway.build(8, 8)
    g = conway.stamp(conway.empty_grid(8, 8), conway.GLIDER, 1, 1)
    state = conway.initial_state(sim, g)
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


def test_run_live_default_layers():
    sim = conway.build(6, 6)
    state = conway.initial_state(sim, conway.random_grid(jax.random.key(0), 6, 6, 0.3))
    final = run_live(sim, state, n_steps=3, px_per_cell=4, fps=1000)
    assert int(final.tick) == 3


def test_run_live_legend_toggle_runs():
    # Both legend states should render without error (the panel draws the controls + status).
    sim = conway.build(10, 10)
    g = conway.stamp(conway.empty_grid(10, 10), conway.GLIDER, 1, 1)
    state = conway.initial_state(sim, g)
    a = run_live(sim, state, n_steps=2, px_per_cell=6, fps=1000, show_legend=True)
    b = run_live(sim, state, n_steps=2, px_per_cell=6, fps=1000, show_legend=False)
    assert int(a.tick) == 2 and int(b.tick) == 2


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
