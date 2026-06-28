"""Pure-numpy rendering primitives for the viewer (no pygame dependency).

Renderers turn a :class:`~evosim.state.State` into an ``(H, W, 3)`` uint8 RGB image plus a
boolean ``(H, W)`` mask of which cells they actually drew. :func:`compose` stacks layers
(later layers paint over earlier ones where their mask is set). These are framework-only and
testable without a display; the pygame window code lives in ``pygame_viewer.py``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["COLORMAPS", "apply_colormap", "GridRenderer", "AgentRenderer", "compose"]

# Colormaps as (stop, (r, g, b)) control points, interpolated over [0, 1].
COLORMAPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "gray": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "green": [(0.0, (0, 12, 0)), (1.0, (50, 255, 90))],
    "blue": [(0.0, (0, 0, 16)), (1.0, (90, 170, 255))],
    "fire": [(0.0, (0, 0, 0)), (0.4, (140, 0, 0)), (0.75, (255, 150, 0)), (1.0, (255, 255, 210))],
    "viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
                (0.75, (94, 201, 98)), (1.0, (253, 231, 37))],
}


def apply_colormap(norm: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Map normalized values in [0, 1] (any shape) to RGB uint8 (shape + (3,))."""
    if cmap not in COLORMAPS:
        raise KeyError(f"unknown colormap {cmap!r}; available: {sorted(COLORMAPS)}")
    stops = COLORMAPS[cmap]
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    n = np.clip(np.asarray(norm, dtype=np.float64), 0.0, 1.0)
    out = np.stack([np.interp(n, xs, cols[:, ch]) for ch in range(3)], axis=-1)
    return out.astype(np.uint8)


def _normalize(arr: np.ndarray, vmin, vmax) -> np.ndarray:
    lo = float(arr.min()) if vmin is None else float(vmin)
    hi = float(arr.max()) if vmax is None else float(vmax)
    denom = (hi - lo) or 1.0
    return np.clip((arr - lo) / denom, 0.0, 1.0)


class GridRenderer:
    """Render an environment field (``state.fields[field]``) as a heatmap."""

    def __init__(self, field: str, cmap: str = "viridis", vmin=None, vmax=None):
        self.field = field
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    def render(self, state, world) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(state.fields[self.field]).astype(np.float64)
        if arr.ndim == 3:  # vector field -> use first channel
            arr = arr[..., 0]
        norm = _normalize(arr, self.vmin, self.vmax)
        img = apply_colormap(norm, self.cmap)
        mask = np.ones(arr.shape, dtype=bool)
        return img, mask


class AgentRenderer:
    """Rasterize agents to their grid cells, optionally colored by a component."""

    def __init__(self, position: str = "position", color_by: str | None = None,
                 cmap: str = "fire", color: tuple[int, int, int] = (255, 255, 255),
                 vmin=None, vmax=None):
        self.position = position
        self.color_by = color_by
        self.cmap = cmap
        self.color = color
        self.vmin = vmin
        self.vmax = vmax

    def render(self, state, world) -> tuple[np.ndarray, np.ndarray]:
        h, w = world.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=bool)
        pos = np.asarray(state[self.position])
        alive = np.asarray(state.alive)
        if not alive.any():
            return img, mask
        rows = (pos[alive, 0] % h).astype(np.intp)
        cols = (pos[alive, 1] % w).astype(np.intp)
        if self.color_by is not None:
            vals = np.asarray(state[self.color_by])
            if vals.ndim > 1:
                vals = vals[:, 0]
            vals = vals[alive].astype(np.float64)
            norm = _normalize(vals, self.vmin, self.vmax)
            colors = apply_colormap(norm, self.cmap)
            img[rows, cols] = colors
        else:
            img[rows, cols] = np.array(self.color, dtype=np.uint8)
        mask[rows, cols] = True
        return img, mask


def compose(layers: Sequence, state, world) -> np.ndarray:
    """Composite layers into one ``(H, W, 3)`` uint8 image (later layers paint on top)."""
    h, w = world.shape
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for layer in layers:
        img, mask = layer.render(state, world)
        base[mask] = img[mask]
    return base
