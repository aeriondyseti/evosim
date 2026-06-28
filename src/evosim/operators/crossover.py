"""Crossover (recombination) operators.

Operators combine two parent genomes ``p1, p2`` of shape ``(..., G)`` into a child of the
same shape. They vectorize over leading batch dimensions; cut points are drawn per pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

__all__ = ["clone", "uniform", "one_point", "n_point", "blend"]


def clone(p1: jax.Array, p2: jax.Array | None = None) -> jax.Array:
    """Asexual reproduction: return ``p1`` unchanged (``p2`` ignored)."""
    return p1


def uniform(key: jax.Array, p1: jax.Array, p2: jax.Array, rate: float = 0.5) -> jax.Array:
    """Each gene is taken from ``p1`` with probability ``rate``, else from ``p2``."""
    mask = random.uniform(key, p1.shape) < rate
    return jnp.where(mask, p1, p2)


def one_point(key: jax.Array, p1: jax.Array, p2: jax.Array) -> jax.Array:
    """Single crossover point per pair: genes before the cut from ``p1``, rest from ``p2``."""
    g = p1.shape[-1]
    cut = random.randint(key, p1.shape[:-1], 1, g)  # (..,) cut in [1, G)
    idx = jnp.arange(g)
    mask = idx < cut[..., None]
    return jnp.where(mask, p1, p2)


def n_point(key: jax.Array, p1: jax.Array, p2: jax.Array, n: int = 2) -> jax.Array:
    """``n``-point crossover: alternate segments between cut points."""
    g = p1.shape[-1]
    # Choose n cut positions in [1, G) per pair, sorted; segment parity selects parent.
    cuts = jnp.sort(random.randint(key, (*p1.shape[:-1], n), 1, g), axis=-1)
    idx = jnp.arange(g)
    # number of cuts <= each index -> parity decides which parent
    cnt = jnp.sum(idx[..., None, :] >= cuts[..., :, None], axis=-2)  # (.., G)
    from_p1 = (cnt % 2) == 0
    return jnp.where(from_p1, p1, p2)


def blend(key: jax.Array, p1: jax.Array, p2: jax.Array, alpha: float = 0.5,
          random_mix: bool = False) -> jax.Array:
    """Arithmetic/BLX blend for real genomes: ``a*p1 + (1-a)*p2``.

    If ``random_mix`` is True, ``a`` is drawn per-gene from ``U[0, 1)``; otherwise the fixed
    weight ``alpha`` is used.
    """
    if random_mix:
        a = random.uniform(key, p1.shape)
    else:
        a = alpha
    return (a * p1 + (1.0 - a) * p2).astype(p1.dtype)
