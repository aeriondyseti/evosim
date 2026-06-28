"""Selection operators (explicit-fitness path).

Each returns an int array of *parent indices* into the population, given a 1-D ``fitness``
array. To exclude dead agents, set their fitness to ``-inf`` before calling. Operators are
vectorized and jit-able (selection counts / fractions are static).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

__all__ = ["tournament", "roulette", "truncation", "elitism"]


def tournament(key: jax.Array, fitness: jax.Array, num_selected: int,
               tournament_size: int = 2) -> jax.Array:
    """Run ``num_selected`` tournaments of ``tournament_size`` random contenders each.

    Returns the index of the fittest contender in each tournament.
    """
    n = fitness.shape[0]
    contenders = random.randint(key, (num_selected, tournament_size), 0, n)
    fits = fitness[contenders]  # (num_selected, tournament_size)
    best = jnp.argmax(fits, axis=1)
    return contenders[jnp.arange(num_selected), best]


def roulette(key: jax.Array, fitness: jax.Array, num_selected: int) -> jax.Array:
    """Fitness-proportional (roulette-wheel) selection. Negative fitness is clipped to 0."""
    f = jnp.clip(fitness, 0.0, None)
    total = jnp.sum(f)
    n = fitness.shape[0]
    p = jnp.where(total > 0, f / jnp.where(total > 0, total, 1.0), jnp.full((n,), 1.0 / n))
    return random.choice(key, n, shape=(num_selected,), p=p, replace=True)


def truncation(key: jax.Array, fitness: jax.Array, num_selected: int,
               frac: float = 0.5) -> jax.Array:
    """Sample uniformly from the top ``frac`` of the population by fitness."""
    n = fitness.shape[0]
    n_top = max(1, int(n * frac))
    top = jnp.argsort(-fitness)[:n_top]
    pick = random.randint(key, (num_selected,), 0, n_top)
    return top[pick]


def elitism(fitness: jax.Array, n_elite: int) -> jax.Array:
    """Return indices of the ``n_elite`` fittest individuals (descending fitness)."""
    return jnp.argsort(-fitness)[:n_elite]
