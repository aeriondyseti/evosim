"""Tests for the visualization toolkit (pure-numpy renderers; no GUI dependency).

The PyGame viewer is an example, tested separately in ``test_example_viewer.py``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from evosim.viz import AgentRenderer, GridRenderer, apply_colormap, compose
from evosim.examples import conway, foragers


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
    assert img[1, 1].sum() > img[0, 0].sum()    # live cell brighter than dead background


def test_agent_renderer_marks_agent_cells():
    cfg = foragers.ForagerConfig(height=10, width=10, capacity=8, n_initial=3)
    sim = foragers.build(cfg)
    s = foragers.initial_state(sim, cfg, jax.random.key(0))
    img, mask = AgentRenderer("position", color_by="energy").render(s, sim.world)
    assert mask.shape == (10, 10)
    assert int(mask.sum()) == int(np.asarray(s.n_alive))  # distinct positions -> one cell each


def test_compose_layers():
    cfg = foragers.ForagerConfig(height=8, width=8, capacity=4, n_initial=2)
    sim = foragers.build(cfg)
    s = foragers.initial_state(sim, cfg, jax.random.key(1))
    layers = [GridRenderer("food", cmap="fire"),
              AgentRenderer("position", color_by=None, color=(255, 255, 255))]
    img = compose(layers, s, sim.world)
    assert img.shape == (8, 8, 3)


def test_viz_import_has_no_pygame_dependency():
    # evosim.viz must import without any GUI toolkit installed.
    import importlib
    import evosim.viz as v
    importlib.reload(v)
    assert hasattr(v, "GridRenderer")
