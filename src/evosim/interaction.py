"""Agent-agent interaction primitives: deterministic contention resolution + pairing.

Per SPEC, interactions (eat, move-into-cell, fight, mate) are resolved with a deterministic
arbitration pass so the result is same-device bit-exact regardless of parallel evaluation
order. The workhorse is :func:`resolve_claims` (scatter-min / segmented reduction):

    many claimants -> one winner per contested target (by priority, ties broken by index).

A *stochastic lottery* is just arbitration with random priorities — see
:func:`lottery_priorities`. Convenience wrappers cover the common cases:

- :func:`resolve_cell_claims` — contention over grid cells (move/eat).
- :func:`mutual_match` — symmetric pairing for mating (i picks j and j picks i).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ClaimResult",
    "resolve_claims",
    "resolve_cell_claims",
    "mutual_match",
    "lottery_priorities",
]


class ClaimResult(NamedTuple):
    """Result of :func:`resolve_claims`.

    Attributes
    ----------
    winner:
        ``(n_targets,)`` int32: winning claimant index per target, ``-1`` if uncontested/empty.
    won:
        ``(N,)`` bool: whether each claimant won the target it claimed.
    """

    winner: jax.Array
    won: jax.Array


def _max_sentinel(dtype) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.asarray(np.inf, dtype=dtype)
    return jnp.asarray(np.iinfo(np.dtype(dtype)).max, dtype=dtype)


def _min_sentinel(dtype) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.asarray(-np.inf, dtype=dtype)
    return jnp.asarray(np.iinfo(np.dtype(dtype)).min, dtype=dtype)


def resolve_claims(targets: jax.Array, priorities: jax.Array, n_targets: int,
                   valid: jax.Array | None = None, lower_wins: bool = True) -> ClaimResult:
    """Resolve contention: pick one winning claimant per target.

    Parameters
    ----------
    targets:
        ``(N,)`` int target id each claimant claims (e.g. cell id, resource/partner index).
    priorities:
        ``(N,)`` comparable values. By default the *lowest* priority wins; ties are always
        broken by the smallest claimant index (deterministic).
    n_targets:
        Number of distinct targets (static).
    valid:
        Optional ``(N,)`` bool mask of real claims (others never win).
    lower_wins:
        If False, the *highest* priority wins instead.
    """
    n = targets.shape[0]
    if valid is None:
        valid = jnp.ones((n,), dtype=bool)
    idx = jnp.arange(n, dtype=jnp.int32)

    dtype = jnp.asarray(priorities).dtype
    safe_tgt = jnp.where(valid, targets.astype(jnp.int32), n_targets)  # sink slot = n_targets

    if lower_wins:
        sentinel = _max_sentinel(dtype)
        safe_pr = jnp.where(valid, priorities, sentinel)
        best = jnp.full((n_targets + 1,), sentinel, dtype=dtype).at[safe_tgt].min(safe_pr)
    else:
        sentinel = _min_sentinel(dtype)
        safe_pr = jnp.where(valid, priorities, sentinel)
        best = jnp.full((n_targets + 1,), sentinel, dtype=dtype).at[safe_tgt].max(safe_pr)

    # Among claimants matching the best priority for their target, the smallest index wins.
    is_best = jnp.logical_and(valid, safe_pr == best[safe_tgt])
    sentinel_i = jnp.asarray(n, dtype=jnp.int32)
    cand = jnp.where(is_best, idx, sentinel_i)
    winner_full = jnp.full((n_targets + 1,), sentinel_i, dtype=jnp.int32).at[safe_tgt].min(cand)

    won = jnp.logical_and(valid, winner_full[safe_tgt] == idx)
    winner = winner_full[:n_targets]
    winner = jnp.where(winner == sentinel_i, jnp.asarray(-1, jnp.int32), winner)
    return ClaimResult(winner=winner, won=won)


def resolve_cell_claims(world, positions: jax.Array, priorities: jax.Array,
                        valid: jax.Array | None = None, lower_wins: bool = True) -> ClaimResult:
    """Resolve contention over grid cells. ``winner`` is indexed by linear cell id."""
    targets = world.cell_id(positions)
    return resolve_claims(targets, priorities, world.n_cells, valid, lower_wins)


def mutual_match(proposal: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Symmetric pairing: agent ``i`` matches ``j`` iff ``proposal[i]==j`` and ``proposal[j]==i``.

    ``proposal`` is ``(N,)`` int with ``-1`` meaning "no proposal".

    Returns ``(matched, partner)``: bool ``(N,)`` and int ``(N,)`` (partner index or ``-1``).
    """
    n = proposal.shape[0]
    p = proposal.astype(jnp.int32)
    has = p >= 0
    safe = jnp.where(has, p, 0)
    reciprocated = p[safe] == jnp.arange(n, dtype=jnp.int32)
    matched = jnp.logical_and(has, reciprocated)
    partner = jnp.where(matched, p, jnp.asarray(-1, jnp.int32))
    return matched, partner


def lottery_priorities(key: jax.Array, n: int) -> jax.Array:
    """Random priorities in ``[0, 1)`` for stochastic (lottery) arbitration."""
    return jax.random.uniform(key, (n,))
