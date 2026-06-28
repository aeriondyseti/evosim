"""On-device aggregate metrics (cheap per-tick reducers).

Per SPEC, metrics are computed on-device as reductions over the population so they're cheap
even at millions of agents. All reducers respect the ``alive`` mask.

Use a :class:`MetricSet` to bundle named reducers. It has two modes:

- :meth:`MetricSet.compute` — call on a state to get a ``{name: value}`` dict (host/eager).
- :meth:`MetricSet.record_fn` — returns a ``state -> dict`` function suitable for
  ``Scheduler.run(record=...)``, where the per-tick dicts are stacked on-device by ``scan``.
"""

from __future__ import annotations

from typing import Callable, Iterable

import jax
import jax.numpy as jnp

from .state import State

__all__ = [
    "masked_mean",
    "masked_var",
    "population",
    "mean_of",
    "var_of",
    "sum_of",
    "genetic_diversity",
    "MetricSet",
    "standard",
]


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean of ``values`` over rows where ``mask`` is True (0 if none alive).

    For vector-valued fields ``(N, *tail)`` the mean is taken over axis 0 (per-component).
    """
    w = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(w), 1.0)
    if values.ndim == 1:
        return jnp.sum(values.astype(jnp.float32) * w) / denom
    wr = w.reshape((-1,) + (1,) * (values.ndim - 1))
    return jnp.sum(values.astype(jnp.float32) * wr, axis=0) / denom


def masked_var(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Population variance over alive rows (per-component for vector fields)."""
    mean = masked_mean(values, mask)
    sq = masked_mean(values.astype(jnp.float32) ** 2, mask)
    return jnp.maximum(sq - mean ** 2, 0.0)


# -- reducer factories (state -> value) --------------------------------------

def population(state: State) -> jax.Array:
    """Number of alive agents."""
    return state.n_alive


def mean_of(field: str) -> Callable[[State], jax.Array]:
    """Reducer: alive-masked mean of ``field``."""
    return lambda s: masked_mean(s[field], s.alive)


def var_of(field: str) -> Callable[[State], jax.Array]:
    """Reducer: alive-masked variance of ``field``."""
    return lambda s: masked_var(s[field], s.alive)


def sum_of(field: str) -> Callable[[State], jax.Array]:
    """Reducer: sum of ``field`` over alive agents."""
    def fn(s: State) -> jax.Array:
        w = s.alive.astype(jnp.float32)
        v = s[field].astype(jnp.float32)
        if v.ndim == 1:
            return jnp.sum(v * w)
        return jnp.sum(v * w.reshape((-1,) + (1,) * (v.ndim - 1)), axis=0)
    return fn


def genetic_diversity(field: str) -> Callable[[State], jax.Array]:
    """Reducer: mean per-gene variance of ``field`` over alive agents (scalar)."""
    return lambda s: jnp.mean(masked_var(s[field], s.alive))


class MetricSet:
    """A named collection of reducers ``state -> value``."""

    def __init__(self) -> None:
        self._fns: dict[str, Callable[[State], jax.Array]] = {}

    def add(self, name: str, fn: Callable[[State], jax.Array]) -> "MetricSet":
        self._fns[name] = fn
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._fns)

    def compute(self, state: State) -> dict[str, jax.Array]:
        return {name: fn(state) for name, fn in self._fns.items()}

    def record_fn(self) -> Callable[[State], dict[str, jax.Array]]:
        """Return a ``state -> dict`` function for ``Scheduler.run(record=...)``."""
        fns = dict(self._fns)

        def rec(state: State) -> dict[str, jax.Array]:
            return {name: fn(state) for name, fn in fns.items()}

        return rec


def standard(scalar_fields: Iterable[str] = (), diversity_fields: Iterable[str] = ()) -> MetricSet:
    """Build a standard metric set: population + mean/var of each scalar field + diversity."""
    ms = MetricSet().add("population", population)
    for f in scalar_fields:
        ms.add(f"{f}_mean", mean_of(f)).add(f"{f}_var", var_of(f))
    for f in diversity_fields:
        ms.add(f"{f}_diversity", genetic_diversity(f))
    return ms
