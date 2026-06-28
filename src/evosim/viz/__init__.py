"""Visualization (optional).

Pure-numpy renderers (:class:`GridRenderer`, :class:`AgentRenderer`, :func:`compose`,
:func:`apply_colormap`) import without any extra dependencies. The PyGame window helpers
(:func:`run_live`, :class:`PygameViewer`) require ``evosim[viz]`` and raise a clear error if
pygame is missing.

    from evosim.viz import run_live, GridRenderer, AgentRenderer
"""

from __future__ import annotations

from .pygame_viewer import PygameViewer, run_live
from .render import COLORMAPS, AgentRenderer, GridRenderer, apply_colormap, compose

__all__ = [
    "GridRenderer",
    "AgentRenderer",
    "compose",
    "apply_colormap",
    "COLORMAPS",
    "run_live",
    "PygameViewer",
]
