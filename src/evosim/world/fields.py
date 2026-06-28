"""Environment field-layer dynamics as reusable systems.

Field layers (resource, pheromone, terrain, the Conway cell grid, ...) live in
``State.fields``. This module provides factory functions that return :class:`~evosim.system.System`
objects implementing common field dynamics, registered (by default) into the ``environment``
stage:

- :func:`decay` — exponential decay / evaporation.
- :func:`regrow` — additive (optionally capped) regrowth.
- :func:`diffuse` — discrete-Laplacian diffusion (toric; conserves total mass).
- :func:`map_field` — apply an arbitrary elementwise function.
- :func:`life_like` — a life-like cellular automaton (Conway's Life = born {3}, survive {2,3}).

Diffusion and life-like rules read neighborhood sums from ``ctx.world`` (a grid world).
"""

from __future__ import annotations

from typing import Callable, Iterable

import jax
import jax.numpy as jnp

from ..system import System

__all__ = ["decay", "regrow", "diffuse", "map_field", "life_like"]


def decay(field: str, rate: float, *, stage: str = "environment", name: str | None = None) -> System:
    """Multiply ``field`` by ``(1 - rate)`` each tick (evaporation)."""
    factor = 1.0 - float(rate)

    def fn(state, ctx):
        return state.set_field(field, state.get_field(field) * factor)

    return System(name or f"decay[{field}]", stage, fn)


def regrow(field: str, amount: float, *, max_value: float | None = None,
           stage: str = "environment", name: str | None = None) -> System:
    """Add ``amount`` to ``field`` each tick, optionally clipped to ``max_value``."""

    def fn(state, ctx):
        g = state.get_field(field) + amount
        if max_value is not None:
            g = jnp.minimum(g, max_value)
        return state.set_field(field, g)

    return System(name or f"regrow[{field}]", stage, fn)


def diffuse(field: str, rate: float, *, kind: str = "von_neumann",
            stage: str = "environment", name: str | None = None) -> System:
    """Discrete-Laplacian diffusion of ``field`` (toric, mass-conserving).

    ``kind`` selects the 4- ("von_neumann") or 8- ("moore") neighborhood. ``rate`` should be
    small for stability (<= 0.25 for von_neumann).
    """

    def fn(state, ctx):
        w = ctx.world
        g = state.get_field(field)
        if kind == "moore":
            lap = w.moore_sum(g, include_center=False) - 8.0 * g
        else:
            lap = w.von_neumann_sum(g, include_center=False) - 4.0 * g
        return state.set_field(field, g + rate * lap)

    return System(name or f"diffuse[{field}]", stage, fn)


def map_field(field: str, func: Callable[[jax.Array], jax.Array], *,
              stage: str = "environment", name: str | None = None) -> System:
    """Apply an arbitrary elementwise ``func`` to ``field``."""

    def fn(state, ctx):
        return state.set_field(field, func(state.get_field(field)))

    return System(name or f"map[{field}]", stage, fn)


def life_like(field: str, born: Iterable[int] = (3,), survive: Iterable[int] = (2, 3), *,
              stage: str = "environment", name: str | None = None) -> System:
    """A life-like cellular automaton over a 0/1 ``field`` (Conway's Life by default).

    A dead cell becomes alive if its live-neighbor count is in ``born``; a live cell stays
    alive if its count is in ``survive``. Uses the toric Moore neighborhood from ``ctx.world``.
    """
    born_t = tuple(int(b) for b in born)
    survive_t = tuple(int(s) for s in survive)

    def fn(state, ctx):
        w = ctx.world
        g = state.get_field(field)
        alive = g > 0
        n = w.moore_sum(alive.astype(jnp.int32), include_center=False)
        born_mask = jnp.zeros_like(alive)
        for b in born_t:
            born_mask = born_mask | (n == b)
        surv_mask = jnp.zeros_like(alive)
        for s in survive_t:
            surv_mask = surv_mask | (n == s)
        new = (born_mask & jnp.logical_not(alive)) | (surv_mask & alive)
        return state.set_field(field, new.astype(g.dtype))

    return System(name or f"life_like[{field}]", stage, fn)
