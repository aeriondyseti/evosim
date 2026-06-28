"""Counter-based deterministic RNG (per SPEC: same-device bit-exact determinism).

EvoSim never threads a mutable RNG state through the simulation. Instead, all randomness
is *derived* from a single root key plus integer coordinates using JAX's counter-based
Threefry PRNG. Given the same root seed and the same coordinates you always get the same
key — independent of evaluation order, parallelism, or how many other draws happened.

Coordinate convention used by the scheduler / systems:

    root = root_key(seed)
    system_key = derive(root, tick, system_id[, stream])      # one key per system per tick
    slot_keys  = per_slot(system_key, capacity)               # one stable key per entity slot

``per_slot`` is keyed by *slot index* via ``fold_in``, which makes the per-entity stream
stable as capacity grows (``per_slot(k, 8)[:4] == per_slot(k, 4)``) — important because the
population buffers grow by doubling.

Keys are JAX *typed* PRNG keys (``jax.random.key``). Use :func:`key_bits` to obtain the raw
uint32 representation for hashing, checkpointing, or equality comparison.
"""

from __future__ import annotations

import zlib

import jax
import jax.numpy as jnp
from jax import random

__all__ = [
    "Key",
    "root_key",
    "fold_in",
    "derive",
    "split",
    "per_slot",
    "name_to_int",
    "stream",
    "key_bits",
    "keys_equal",
]

#: A JAX typed PRNG key (or a batch thereof).
Key = jax.Array


def root_key(seed: int) -> Key:
    """Create the root key for a run from an integer seed."""
    return random.key(seed)


def fold_in(key: Key, data: int) -> Key:
    """Deterministically mix an integer into ``key`` (counter increment)."""
    return random.fold_in(key, data)


def derive(key: Key, *coords: int) -> Key:
    """Derive a sub-key by folding in a sequence of integer coordinates in order.

    Order matters: ``derive(k, 1, 2) != derive(k, 2, 1)``.
    """
    for c in coords:
        key = random.fold_in(key, c)
    return key


def split(key: Key, num: int = 2) -> Key:
    """Split ``key`` into ``num`` independent keys (shape ``(num,)``)."""
    return random.split(key, num)


def per_slot(key: Key, capacity: int) -> Key:
    """Return one stable key per slot ``[0, capacity)`` by folding the slot index in.

    The prefix-stability property holds: the first ``n`` keys are identical regardless of
    ``capacity >= n``, so growing the population buffers does not perturb existing streams.
    """
    idx = jnp.arange(capacity, dtype=jnp.uint32)
    return jax.vmap(lambda i: random.fold_in(key, i))(idx)


def name_to_int(name: str) -> int:
    """Stable (process-independent) integer hash of a string, for named streams.

    Uses CRC32 so the value is reproducible across processes (unlike ``hash()``).
    """
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


def stream(key: Key, name: str) -> Key:
    """Derive a named sub-stream key from a stable hash of ``name``."""
    return random.fold_in(key, name_to_int(name))


def key_bits(key: Key) -> jax.Array:
    """Raw uint32 representation of a key (or batch of keys) for hashing/equality."""
    return random.key_data(key)


def keys_equal(a: Key, b: Key) -> bool:
    """True if two keys (or key batches) have identical underlying bits."""
    return bool(jnp.array_equal(key_bits(a), key_bits(b)))
