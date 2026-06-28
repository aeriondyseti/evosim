"""Unit tests for evosim.state (State PyTree)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evosim.schema import Field, Schema
from evosim.state import State, state_fingerprint


def make_schema():
    return Schema(
        position=Field(dtype="int16", shape=(2,)),
        energy=Field(dtype="float32", default=1.0),
        genome=Field(dtype="float32", shape=(3,)),
    )


def test_create_empty_state():
    s = State.create(make_schema(), capacity=8)
    assert s.capacity == 8
    assert int(s.tick) == 0
    assert int(s.next_id) == 0
    assert int(s.n_alive) == 0
    assert not np.any(np.asarray(s.alive))
    assert np.all(np.asarray(s.ids) == -1)
    assert s["energy"].shape == (8,)
    assert s["position"].shape == (8, 2)


def test_accessors_and_contains():
    s = State.create(make_schema(), capacity=4)
    assert "energy" in s
    assert "nope" not in s
    assert s.get("genome").shape == (4, 3)
    assert np.allclose(np.asarray(s["energy"]), 1.0)


def test_set_is_immutable():
    s = State.create(make_schema(), capacity=4)
    s2 = s.set("energy", jnp.full((4,), 5.0, dtype=jnp.float32))
    assert np.allclose(np.asarray(s2["energy"]), 5.0)
    # original unchanged
    assert np.allclose(np.asarray(s["energy"]), 1.0)


def test_set_unknown_component_raises():
    s = State.create(make_schema(), capacity=2)
    with pytest.raises(KeyError):
        s.set("ghost", jnp.zeros((2,)))


def test_set_many():
    s = State.create(make_schema(), capacity=3)
    alive = jnp.array([True, False, True])
    s2 = s.set_many({"alive": alive, "energy": jnp.array([2.0, 0.0, 3.0], dtype=jnp.float32)})
    assert int(s2.n_alive) == 2
    assert np.array_equal(np.asarray(s2.free_mask), [False, True, False])


def test_increment_tick():
    s = State.create(make_schema(), capacity=2)
    s2 = s.increment_tick().increment_tick()
    assert int(s2.tick) == 2
    assert int(s.tick) == 0  # immutability


def test_pytree_roundtrip():
    s = State.create(make_schema(), capacity=5)
    leaves, treedef = jax.tree_util.tree_flatten(s)
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert s2.capacity == s.capacity
    assert s2.schema.names == s.schema.names
    assert state_fingerprint(s) == state_fingerprint(s2)


def test_pytree_works_under_jit():
    s = State.create(make_schema(), capacity=4)

    @jax.jit
    def bump(state: State) -> State:
        return state.set("energy", state["energy"] + 1.0).increment_tick()

    out = bump(s)
    assert int(out.tick) == 1
    assert np.allclose(np.asarray(out["energy"]), 2.0)


def test_pytree_works_under_vmap():
    # Batch of independent worlds (vmap-over-worlds primitive).
    schema = make_schema()

    def make(seed):
        return State.create(schema, capacity=4)

    states = jax.vmap(lambda i: State.create(schema, 4))(jnp.arange(3))
    # leading batch dim added to every leaf
    assert states["energy"].shape == (3, 4)
    assert states.tick.shape == (3,)


def test_fingerprint_stable_and_sensitive():
    s = State.create(make_schema(), capacity=4)
    f1 = state_fingerprint(s)
    f2 = state_fingerprint(State.create(make_schema(), capacity=4))
    assert f1 == f2  # identical states -> identical fingerprint
    s3 = s.set("energy", s["energy"] + 0.001)
    assert state_fingerprint(s3) != f1  # any change -> different fingerprint


def test_fingerprint_detects_tick_change():
    s = State.create(make_schema(), capacity=4)
    assert state_fingerprint(s) != state_fingerprint(s.increment_tick())


# --- fields (environment layers) --------------------------------------------

def test_create_with_fields():
    grid = jnp.zeros((3, 3), dtype=jnp.int32)
    s = State.create(make_schema(), capacity=4, fields={"cells": grid})
    assert s.has_field("cells")
    assert s.get_field("cells").shape == (3, 3)


def test_set_field_immutable_and_added():
    s = State.create(make_schema(), capacity=2)
    assert not s.has_field("res")
    s2 = s.set_field("res", jnp.ones((2, 2)))
    assert s2.has_field("res")
    assert not s.has_field("res")  # original unchanged


def test_fields_pytree_roundtrip_and_jit():
    grid = jnp.arange(9).reshape(3, 3)
    s = State.create(make_schema(), 4, fields={"g": grid})
    leaves, treedef = jax.tree_util.tree_flatten(s)
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.array_equal(np.asarray(s2.get_field("g")), np.asarray(grid))

    @jax.jit
    def bump(state):
        return state.set_field("g", state.get_field("g") + 1)

    out = bump(s)
    assert np.array_equal(np.asarray(out.get_field("g")), np.asarray(grid) + 1)


def test_fingerprint_sensitive_to_fields():
    s = State.create(make_schema(), 4, fields={"g": jnp.zeros((2, 2))})
    f1 = state_fingerprint(s)
    s2 = s.set_field("g", jnp.ones((2, 2)))
    assert state_fingerprint(s2) != f1
