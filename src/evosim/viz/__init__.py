"""Visualization toolkit — framework-agnostic, dependency-free renderers.

These turn a :class:`~evosim.state.State` into RGB images, independent of *what* eventually
paints them (PyGame, matplotlib, a web canvas, moderngl, ...). They depend only on numpy, so
``import evosim.viz`` never pulls a GUI dependency.

A concrete live viewer built on these renderers ships as an example, not as part of the core:
see :mod:`evosim.examples.pygame_viewer` (``run_live`` / ``PygameViewer``), which requires the
optional ``viz`` extra. This mirrors how ``evosim.examples.conway`` demonstrates the core engine.

    from evosim.viz import GridRenderer, AgentRenderer, compose, apply_colormap
"""

from __future__ import annotations

from .render import COLORMAPS, AgentRenderer, GridRenderer, apply_colormap, compose

__all__ = [
    "GridRenderer",
    "AgentRenderer",
    "compose",
    "apply_colormap",
    "COLORMAPS",
]
