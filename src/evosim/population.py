"""Dynamic population management over fixed-capacity SoA buffers.

Per SPEC: population lives in capacity-sized buffers tracked by the ``alive`` mask, grows by
doubling when full, and is periodically compacted. All operations are deterministic.

Two execution layers:

- **Hot-loop (jit-able, static shape):** :func:`kill`, :func:`spawn`, :func:`compact`.
  These never change ``capacity``.
- **Host-level (changes shape, triggers recompile — rare):** :func:`grow`, :func:`grow_to_fit`.
  Call these between ticks when :func:`spawn` reports overflow.

Spawn semantics (deterministic free-slot claim): candidate births are ranked by their order
in the input; the lowest-ranked accepted births claim the available free slots (in ascending
slot order). If there are more births than free slots, the surplus is *dropped* and reported
as overflow so the caller can :func:`grow_to_fit` and retry. Newly placed agents receive
contiguous unique ids starting at ``state.next_id``.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

from .schema import RESERVED_FIELDS
from .state import State

__all__ = ["SpawnResult", "n_free", "kill", "spawn", "compact", "grow", "grow_to_fit"]


class SpawnResult(NamedTuple):
    """Result of :func:`spawn`. Counts are int32 scalar arrays (jit-friendly)."""

    state: State
    n_placed: jax.Array
    n_overflow: jax.Array


def n_free(state: State) -> jax.Array:
    """Number of free (dead) slots as an int32 scalar array."""
    return jnp.asarray(state.capacity, dtype=jnp.int32) - state.n_alive


# -- hot-loop ops ------------------------------------------------------------

def kill(state: State, kill_mask: jax.Array) -> State:
    """Mark slots where ``kill_mask`` is True as dead (alive=False, id=-1)."""
    new_alive = jnp.logical_and(state.alive, jnp.logical_not(kill_mask))
    new_id = jnp.where(kill_mask, jnp.asarray(-1, jnp.int32), state.ids)
    return state.set_many({"alive": new_alive, "id": new_id})


def spawn(state: State, child_data: Mapping[str, jax.Array],
          birth_mask: jax.Array | None = None) -> SpawnResult:
    """Place new agents into free slots.

    Parameters
    ----------
    child_data:
        ``{field: array}`` of candidate offspring values, each of shape ``(M, *field.shape)``.
        Reserved fields (``alive``/``id``) are ignored if present — the framework sets them.
    birth_mask:
        Optional bool ``(M,)`` selecting which candidates are real births. Defaults to all.

    Returns
    -------
    SpawnResult(state, n_placed, n_overflow)
    """
    # Determine M from the first user-provided array.
    user_items = [(k, jnp.asarray(v)) for k, v in child_data.items() if k not in RESERVED_FIELDS]
    if not user_items:
        raise ValueError("child_data must contain at least one user field")
    M = user_items[0][1].shape[0]
    for k, v in user_items:
        if v.shape[0] != M:
            raise ValueError(f"child_data[{k!r}] has leading dim {v.shape[0]}, expected {M}")

    accepted = (
        jnp.ones((M,), dtype=bool) if birth_mask is None else jnp.asarray(birth_mask, dtype=bool)
    )

    cap = state.capacity
    free = jnp.logical_not(state.alive)
    nfree = jnp.sum(free.astype(jnp.int32))

    # Ascending free-slot order: free slots (alive==0) come first, stably by index.
    order = jnp.argsort(state.alive.astype(jnp.int32), stable=True)  # (capacity,)

    # Rank of each accepted birth among accepted births (0-based).
    birth_rank = jnp.cumsum(accepted.astype(jnp.int32)) - 1  # (M,)
    fits = birth_rank < nfree
    do_place = jnp.logical_and(accepted, fits)

    safe_rank = jnp.clip(birth_rank, 0, cap - 1)
    target = jnp.where(do_place, order[safe_rank], cap)  # cap == OOB sentinel (dropped)

    new_components = dict(state.components)
    # Write user fields at the claimed slots.
    for k, v in user_items:
        new_components[k] = new_components[k].at[target].set(v, mode="drop")
    # Framework-managed fields.
    new_alive = new_components["alive"].at[target].set(True, mode="drop")
    new_ids = new_components["id"].at[target].set(
        (state.next_id + birth_rank).astype(jnp.int32), mode="drop"
    )
    new_components["alive"] = new_alive
    new_components["id"] = new_ids

    n_placed = jnp.sum(do_place.astype(jnp.int32))
    n_overflow = jnp.sum(accepted.astype(jnp.int32)) - n_placed
    new_state = state.replace(
        components=new_components,
        next_id=state.next_id + n_placed,
    )
    return SpawnResult(new_state, n_placed, n_overflow)


def compact(state: State) -> State:
    """Move alive agents to the front (dense), preserving their relative order."""
    order = jnp.argsort(jnp.logical_not(state.alive).astype(jnp.int32), stable=True)
    new_components = {name: comp[order] for name, comp in state.components.items()}
    return state.replace(components=new_components)


# -- host-level ops (change capacity; trigger recompile) ---------------------

def grow(state: State, new_capacity: int) -> State:
    """Return a state with buffers reallocated to ``new_capacity`` (data preserved).

    New slots are dead (alive=False, id=-1). Changes static shape — do this outside jit.
    """
    if new_capacity < state.capacity:
        raise ValueError(f"new_capacity {new_capacity} < current {state.capacity}")
    if new_capacity == state.capacity:
        return state
    new_components = {}
    for field in state.schema:
        buf = field.allocate(new_capacity)
        buf = buf.at[: state.capacity].set(state.components[field.name])
        new_components[field.name] = buf
    return state.replace(components=new_components, capacity=new_capacity)


def grow_to_fit(state: State, n_needed: int) -> State:
    """Grow (by doubling) until at least ``n_needed`` free slots exist."""
    cap = state.capacity
    alive = int(state.n_alive)  # host sync (host-level op)
    while cap - alive < n_needed:
        cap *= 2
    return grow(state, cap)
