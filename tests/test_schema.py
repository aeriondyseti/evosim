"""Unit tests for evosim.schema (Field / Schema / SoA allocation)."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from evosim.schema import Field, Schema, RESERVED_FIELDS


# --- Field ------------------------------------------------------------------

def test_field_shape_int_promoted_to_tuple():
    f = Field(name="v", shape=3)
    assert f.shape == (3,)


def test_field_scalar_shape_default():
    f = Field(name="e")
    assert f.shape == ()


def test_field_dtype_canonicalized():
    assert Field(name="x", dtype="float32").dtype == "float32"
    assert Field(name="x", dtype=np.int16).dtype == "int16"
    assert Field(name="x", dtype="bool").dtype == "bool"


def test_field_invalid_name_raises():
    with pytest.raises(ValueError):
        Field(name="not an identifier")


def test_field_invalid_dtype_raises():
    with pytest.raises(ValueError):
        Field(name="x", dtype="not-a-dtype")


def test_field_buffer_shape():
    assert Field(name="p", shape=(2,)).buffer_shape(10) == (10, 2)
    assert Field(name="e").buffer_shape(10) == (10,)


def test_field_allocate_shape_dtype_default():
    f = Field(name="e", dtype="float32", default=1.5)
    buf = f.allocate(4)
    assert buf.shape == (4,)
    assert buf.dtype == np.float32
    assert np.allclose(np.asarray(buf), 1.5)


def test_field_allocate_vector_default():
    f = Field(name="p", dtype="int16", shape=(2,), default=7)
    buf = f.allocate(3)
    assert buf.shape == (3, 2)
    assert buf.dtype == np.int16
    assert np.all(np.asarray(buf) == 7)


def test_field_allocate_bool_default_false():
    buf = Field(name="alive", dtype="bool", default=False).allocate(5)
    assert buf.dtype == np.bool_
    assert not np.any(np.asarray(buf))


def test_field_negative_capacity_raises():
    with pytest.raises(ValueError):
        Field(name="e").allocate(-1)


def test_field_64bit_without_x64_raises():
    # Tests run with x64 disabled by default; allocating a 64-bit field must error.
    assert not jax.config.read("jax_enable_x64")
    with pytest.raises(ValueError, match="x64"):
        Field(name="big", dtype="float64").allocate(2)


def test_field_rename():
    f = Field(dtype="int8")
    assert f.rename("trait").name == "trait"


# --- Schema -----------------------------------------------------------------

def test_schema_kwargs_assign_names():
    s = Schema(position=Field(dtype="int16", shape=(2,)), energy=Field(dtype="float32"))
    assert s["position"].name == "position"
    assert s["energy"].name == "energy"


def test_schema_auto_adds_reserved():
    s = Schema(energy=Field(dtype="float32"))
    assert "alive" in s and "id" in s
    assert s["alive"].dtype == "bool"
    assert s["id"].dtype == "int32"
    assert RESERVED_FIELDS == {"alive", "id"}


def test_schema_user_names_excludes_reserved():
    s = Schema(energy=Field(dtype="float32"), genome=Field(dtype="float32", shape=(4,)))
    assert set(s.user_names) == {"energy", "genome"}
    assert "alive" not in s.user_names


def test_schema_order_preserved():
    s = Schema(a=Field(dtype="int8"), b=Field(dtype="int8"), c=Field(dtype="int8"))
    # user fields precede the auto-added reserved fields
    assert s.names[:3] == ("a", "b", "c")


def test_schema_duplicate_name_raises():
    with pytest.raises(ValueError):
        Schema(Field(name="x", dtype="int8"), Field(name="x", dtype="int8"))


def test_schema_positional_requires_name():
    with pytest.raises(ValueError):
        Schema(Field(dtype="int8"))


def test_schema_explicit_reserved_not_duplicated():
    s = Schema(alive=Field(dtype="bool", default=True), energy=Field(dtype="float32"))
    # explicit alive kept (default True), not overwritten; still exactly one 'alive'
    assert s.names.count("alive") == 1
    assert bool(np.asarray(s["alive"].allocate(2))[0]) is True


def test_schema_allocate_returns_all_buffers():
    s = Schema(position=Field(dtype="int16", shape=(2,)), energy=Field(dtype="float32"))
    bufs = s.allocate(6)
    assert set(bufs) == {"position", "energy", "alive", "id"}
    assert bufs["position"].shape == (6, 2)
    assert bufs["energy"].shape == (6,)
    assert bufs["alive"].shape == (6,)
    assert bufs["id"].shape == (6,)
    # id default is -1
    assert np.all(np.asarray(bufs["id"]) == -1)


def test_schema_len_and_contains():
    s = Schema(energy=Field(dtype="float32"))
    assert len(s) == 3  # energy + alive + id
    assert "energy" in s
    assert "nope" not in s
