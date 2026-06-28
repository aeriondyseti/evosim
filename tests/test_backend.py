"""Unit tests for evosim.backend (backend abstraction)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from evosim import backend as be


def test_default_backend_is_jax():
    b = be.get_backend()
    assert b.name == "jax"
    assert b.xp is jnp


def test_jit_compiles_and_runs():
    b = be.get_backend()
    f = b.jit(lambda x: x * 2 + 1)
    out = f(jnp.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.asarray(out), [3.0, 5.0, 7.0])


def test_jit_static_argnums():
    b = be.get_backend()

    def repeat(x, n):
        return jnp.tile(x, n)

    f = b.jit(repeat, static_argnums=(1,))
    out = f(jnp.array([1, 2]), 3)
    assert np.array_equal(np.asarray(out), [1, 2, 1, 2, 1, 2])


def test_scan_cumsum():
    b = be.get_backend()

    def step(carry, x):
        carry = carry + x
        return carry, carry

    final, ys = b.scan(step, jnp.array(0.0), jnp.arange(5.0))
    assert float(final) == 10.0
    assert np.allclose(np.asarray(ys), [0, 1, 3, 6, 10])


def test_vmap():
    b = be.get_backend()
    f = b.vmap(lambda v: jnp.sum(v))
    out = f(jnp.arange(6).reshape(3, 2))
    assert np.array_equal(np.asarray(out), [1, 5, 9])


def test_devices_nonempty():
    b = be.get_backend()
    assert len(b.devices()) >= 1
    assert b.default_device() is not None


def test_registry_and_default_switch():
    class Dummy(be.Backend):
        name = "dummy"

        @property
        def xp(self):
            return jnp

    be.register_backend("dummy", Dummy)
    try:
        assert be.get_backend("dummy").name == "dummy"
        with be.use_backend("dummy"):
            assert be.get_backend().name == "dummy"
        # restored after context
        assert be.get_backend().name == "jax"
    finally:
        be.set_default_backend("jax")


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        be.get_backend("does-not-exist")
