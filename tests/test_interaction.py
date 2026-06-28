"""Unit tests for evosim.interaction (deterministic arbitration + pairing)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim import interaction as ix
from evosim import rng
from evosim.world import ToricGrid2D


def test_resolve_claims_lowest_priority_wins():
    targets = jnp.array([0, 0, 1])
    pris = jnp.array([5.0, 3.0, 9.0])
    res = ix.resolve_claims(targets, pris, n_targets=2)
    assert np.array_equal(np.asarray(res.winner), [1, 2])
    assert np.array_equal(np.asarray(res.won), [False, True, True])


def test_resolve_claims_tie_broken_by_index():
    targets = jnp.array([0, 0])
    pris = jnp.array([5.0, 5.0])
    res = ix.resolve_claims(targets, pris, n_targets=1)
    assert int(res.winner[0]) == 0
    assert np.array_equal(np.asarray(res.won), [True, False])


def test_resolve_claims_higher_wins():
    targets = jnp.array([0, 0, 0])
    pris = jnp.array([1.0, 7.0, 4.0])
    res = ix.resolve_claims(targets, pris, n_targets=1, lower_wins=False)
    assert int(res.winner[0]) == 1  # priority 7 wins


def test_resolve_claims_valid_mask():
    targets = jnp.array([0, 0, 0])
    pris = jnp.array([1.0, 2.0, 3.0])
    valid = jnp.array([False, True, True])
    res = ix.resolve_claims(targets, pris, n_targets=1, valid=valid)
    assert int(res.winner[0]) == 1  # claimant 0 invalid, so 1 (pri 2) wins
    assert np.array_equal(np.asarray(res.won), [False, True, False])


def test_resolve_claims_empty_target_is_minus_one():
    targets = jnp.array([0, 0])
    pris = jnp.array([1.0, 2.0])
    res = ix.resolve_claims(targets, pris, n_targets=3)
    assert int(res.winner[0]) == 0
    assert int(res.winner[1]) == -1  # no claimant
    assert int(res.winner[2]) == -1


def test_resolve_claims_integer_priorities():
    targets = jnp.array([0, 0])
    pris = jnp.array([10, 4], dtype=jnp.int32)
    res = ix.resolve_claims(targets, pris, n_targets=1)
    assert int(res.winner[0]) == 1


def test_resolve_cell_claims_grid():
    g = ToricGrid2D(3, 3)
    # two agents in cell (1,1), one in (0,0)
    pos = jnp.array([[1, 1], [1, 1], [0, 0]])
    pris = jnp.array([2.0, 1.0, 5.0])
    res = ix.resolve_cell_claims(g, pos, pris)
    # agent 1 wins cell (1,1), agent 2 wins (0,0)
    assert np.array_equal(np.asarray(res.won), [False, True, True])


def test_mutual_match_pairs():
    proposal = jnp.array([1, 0, 3, 2])
    matched, partner = ix.mutual_match(proposal)
    assert np.all(np.asarray(matched))
    assert np.array_equal(np.asarray(partner), [1, 0, 3, 2])


def test_mutual_match_cycle_not_matched():
    proposal = jnp.array([1, 2, 0])  # 0->1->2->0 cycle, no reciprocation
    matched, partner = ix.mutual_match(proposal)
    assert not np.any(np.asarray(matched))
    assert np.all(np.asarray(partner) == -1)


def test_mutual_match_with_no_proposal():
    proposal = jnp.array([1, 0, -1])
    matched, partner = ix.mutual_match(proposal)
    assert np.array_equal(np.asarray(matched), [True, True, False])


def test_lottery_arbitration_single_winner_and_deterministic():
    targets = jnp.array([0, 0, 0, 0])
    pris = ix.lottery_priorities(rng.root_key(3), 4)
    res = ix.resolve_claims(targets, pris, n_targets=1)
    # exactly one winner
    assert int(np.asarray(res.won).sum()) == 1
    # deterministic
    pris2 = ix.lottery_priorities(rng.root_key(3), 4)
    res2 = ix.resolve_claims(targets, pris2, n_targets=1)
    assert np.array_equal(np.asarray(res.won), np.asarray(res2.won))


def test_resolve_claims_jit_able():
    @jax.jit
    def run(targets, pris):
        return ix.resolve_claims(targets, pris, n_targets=4).won

    won = run(jnp.array([0, 1, 1, 2]), jnp.array([1.0, 2.0, 0.5, 3.0]))
    assert np.array_equal(np.asarray(won), [True, False, True, True])
