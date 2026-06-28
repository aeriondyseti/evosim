"""Unit tests for evosim.population (spawn/kill/compact/grow)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim import population as pop
from evosim.schema import Field, Schema
from evosim.state import State, state_fingerprint


def make_state(capacity=8):
    schema = Schema(
        energy=Field(dtype="float32", default=0.0),
        genome=Field(dtype="float32", shape=(2,), default=0.0),
    )
    return State.create(schema, capacity)


def test_spawn_into_empty():
    s = make_state(8)
    child = {"energy": jnp.array([1.0, 2.0, 3.0]),
             "genome": jnp.arange(6.0).reshape(3, 2)}
    res = pop.spawn(s, child)
    assert int(res.n_placed) == 3
    assert int(res.n_overflow) == 0
    ns = res.state
    assert int(ns.n_alive) == 3
    # first 3 slots alive with contiguous ids 0,1,2
    assert np.array_equal(np.asarray(ns.alive)[:3], [True, True, True])
    assert np.array_equal(np.asarray(ns.ids)[:3], [0, 1, 2])
    assert np.allclose(np.asarray(ns["energy"])[:3], [1.0, 2.0, 3.0])
    assert int(ns.next_id) == 3


def test_spawn_birth_mask_partial():
    s = make_state(8)
    child = {"energy": jnp.array([10.0, 20.0, 30.0, 40.0])}
    mask = jnp.array([True, False, True, False])
    res = pop.spawn(s, child, birth_mask=mask)
    assert int(res.n_placed) == 2
    ns = res.state
    # accepted values 10.0 and 30.0 placed into first two slots in order
    assert np.allclose(np.asarray(ns["energy"])[:2], [10.0, 30.0])
    assert int(ns.n_alive) == 2


def test_spawn_overflow_does_not_clobber():
    s = make_state(4)
    # pre-fill 3 alive
    res = pop.spawn(s, {"energy": jnp.array([1.0, 2.0, 3.0])})
    s = res.state
    assert int(pop.n_free(s)) == 1
    # request 3 more, only 1 fits
    res2 = pop.spawn(s, {"energy": jnp.array([7.0, 8.0, 9.0])})
    assert int(res2.n_placed) == 1
    assert int(res2.n_overflow) == 2
    ns = res2.state
    assert int(ns.n_alive) == 4
    # original 3 untouched, the single placed birth is the first accepted (7.0)
    assert np.allclose(np.asarray(ns["energy"]), [1.0, 2.0, 3.0, 7.0])


def test_spawn_ids_monotonic_across_calls():
    s = make_state(8)
    s = pop.spawn(s, {"energy": jnp.array([1.0, 2.0])}).state
    s = pop.spawn(s, {"energy": jnp.array([3.0, 4.0])}).state
    ids = np.asarray(s.ids)[:4]
    assert np.array_equal(ids, [0, 1, 2, 3])
    assert int(s.next_id) == 4


def test_kill():
    s = make_state(6)
    s = pop.spawn(s, {"energy": jnp.array([1.0, 2.0, 3.0, 4.0])}).state
    s2 = pop.kill(s, jnp.array([False, True, False, True, False, False]))
    assert int(s2.n_alive) == 2
    assert np.array_equal(np.asarray(s2.alive)[:4], [True, False, True, False])
    # killed slots get id -1
    assert np.asarray(s2.ids)[1] == -1
    assert np.asarray(s2.ids)[3] == -1


def test_compact_moves_alive_front_preserving_order():
    s = make_state(6)
    s = pop.spawn(s, {"energy": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])}).state
    s = pop.kill(s, jnp.array([False, True, False, True, False, False]))  # keep 1,3,5
    sc = pop.compact(s)
    assert int(sc.n_alive) == 3
    assert np.array_equal(np.asarray(sc.alive)[:3], [True, True, True])
    assert not np.any(np.asarray(sc.alive)[3:])
    # surviving energies (1.0,3.0,5.0) moved to front in original order
    assert np.allclose(np.asarray(sc["energy"])[:3], [1.0, 3.0, 5.0])


def test_grow_preserves_data_and_adds_dead_slots():
    s = make_state(4)
    s = pop.spawn(s, {"energy": jnp.array([1.0, 2.0])}).state
    g = pop.grow(s, 8)
    assert g.capacity == 8
    assert g["energy"].shape == (8,)
    assert np.allclose(np.asarray(g["energy"])[:2], [1.0, 2.0])
    # new region dead, ids -1
    assert not np.any(np.asarray(g.alive)[4:])
    assert np.all(np.asarray(g.ids)[4:] == -1)
    assert int(g.n_alive) == 2


def test_grow_to_fit_doubles_until_enough():
    s = make_state(4)
    s = pop.spawn(s, {"energy": jnp.array([1.0, 2.0, 3.0, 4.0])}).state  # full
    g = pop.grow_to_fit(s, 5)  # need 5 free; 4->8 gives 4 free, 8->16? alive=4 -> 16-4=12>=5
    assert g.capacity >= 16 or (g.capacity - int(g.n_alive)) >= 5
    assert (g.capacity - int(g.n_alive)) >= 5


def test_spawn_is_jit_able():
    s = make_state(8)

    @jax.jit
    def do_spawn(state, vals):
        return pop.spawn(state, {"energy": vals}).state

    ns = do_spawn(s, jnp.array([5.0, 6.0]))
    assert int(ns.n_alive) == 2
    assert np.allclose(np.asarray(ns["energy"])[:2], [5.0, 6.0])


def test_spawn_deterministic():
    s = make_state(8)
    child = {"energy": jnp.array([1.0, 2.0, 3.0])}
    a = pop.spawn(s, child).state
    b = pop.spawn(s, child).state
    assert state_fingerprint(a) == state_fingerprint(b)
