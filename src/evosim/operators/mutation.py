"""Mutation operators (library; users may supply their own).

All operators are pure functions ``(key, genome, ...) -> genome`` that preserve the genome's
shape and dtype, and vectorize over any leading batch dimensions. ``key`` is a JAX PRNG key.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

__all__ = ["gaussian", "uniform", "bitflip"]


def gaussian(key: jax.Array, genome: jax.Array, sigma: float = 0.1, rate: float = 1.0,
             clip: tuple[float, float] | None = None) -> jax.Array:
    """Add Gaussian noise ``N(0, sigma)`` to (a ``rate`` fraction of) genes.

    Intended for real-valued genomes. With ``rate < 1`` each gene is perturbed independently
    with probability ``rate``. ``clip=(lo, hi)`` optionally bounds the result.
    """
    k1, k2 = random.split(key)
    noise = sigma * random.normal(k1, genome.shape)
    if rate < 1.0:
        mask = random.uniform(k2, genome.shape) < rate
        noise = jnp.where(mask, noise, 0.0)
    out = genome + noise.astype(genome.dtype)
    if clip is not None:
        out = jnp.clip(out, clip[0], clip[1])
    return out


def uniform(key: jax.Array, genome: jax.Array, low: float = 0.0, high: float = 1.0,
            rate: float = 0.1) -> jax.Array:
    """Replace a ``rate`` fraction of genes with fresh uniform draws in ``[low, high)``."""
    k1, k2 = random.split(key)
    draws = random.uniform(k1, genome.shape, minval=low, maxval=high).astype(genome.dtype)
    mask = random.uniform(k2, genome.shape) < rate
    return jnp.where(mask, draws, genome)


def bitflip(key: jax.Array, genome: jax.Array, rate: float = 0.01) -> jax.Array:
    """Flip binary genes (0<->1) with probability ``rate``. For 0/1 int or bool genomes."""
    mask = random.uniform(key, genome.shape) < rate
    if genome.dtype == jnp.bool_:
        return jnp.where(mask, jnp.logical_not(genome), genome)
    return jnp.where(mask, 1 - genome, genome)
