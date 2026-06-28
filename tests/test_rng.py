"""Unit tests for evosim.rng (counter-based deterministic RNG)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evosim import rng


def test_root_key_reproducible():
    assert rng.keys_equal(rng.root_key(0), rng.root_key(0))
    assert not rng.keys_equal(rng.root_key(0), rng.root_key(1))


def test_derive_deterministic_and_order_sensitive():
    k = rng.root_key(42)
    assert rng.keys_equal(rng.derive(k, 1, 2), rng.derive(k, 1, 2))
    assert not rng.keys_equal(rng.derive(k, 1, 2), rng.derive(k, 2, 1))


def test_derive_distinct_per_coordinate():
    k = rng.root_key(7)
    a = rng.derive(k, 0, 0)  # tick 0, system 0
    b = rng.derive(k, 0, 1)  # tick 0, system 1
    c = rng.derive(k, 1, 0)  # tick 1, system 0
    assert not rng.keys_equal(a, b)
    assert not rng.keys_equal(a, c)
    assert not rng.keys_equal(b, c)


def test_per_slot_shape_and_distinct():
    k = rng.root_key(1)
    ks = rng.per_slot(k, 16)
    bits = np.asarray(rng.key_bits(ks))
    assert bits.shape[0] == 16
    # all 16 slot keys are distinct
    uniq = {tuple(row) for row in bits.reshape(16, -1)}
    assert len(uniq) == 16


def test_per_slot_prefix_stable_under_growth():
    k = rng.root_key(123)
    small = rng.key_bits(rng.per_slot(k, 4))
    large = rng.key_bits(rng.per_slot(k, 8))
    assert np.array_equal(np.asarray(small), np.asarray(large)[:4])


def test_draws_reproducible():
    k = rng.derive(rng.root_key(5), 3, 2)
    x1 = jax.random.uniform(k, (100,))
    x2 = jax.random.uniform(k, (100,))
    assert jnp.array_equal(x1, x2)


def test_split_reproducible():
    k = rng.root_key(9)
    a = rng.split(k, 4)
    b = rng.split(k, 4)
    assert rng.keys_equal(a, b)
    # the 4 split keys differ from each other
    bits = np.asarray(rng.key_bits(a)).reshape(4, -1)
    assert len({tuple(r) for r in bits}) == 4


def test_name_to_int_stable():
    # Stable across processes/runs (CRC32-based), and deterministic here.
    assert rng.name_to_int("movement") == rng.name_to_int("movement")
    assert rng.name_to_int("movement") != rng.name_to_int("reproduction")
    # exact known CRC32 value (locks the algorithm)
    import zlib
    assert rng.name_to_int("movement") == int(zlib.crc32(b"movement") & 0x7FFFFFFF)


def test_stream_named_determinism():
    k = rng.root_key(2)
    assert rng.keys_equal(rng.stream(k, "a"), rng.stream(k, "a"))
    assert not rng.keys_equal(rng.stream(k, "a"), rng.stream(k, "b"))
