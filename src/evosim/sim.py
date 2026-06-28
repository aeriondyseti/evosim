"""Simulation assembly — the top-level orchestrator.

:class:`Simulation` bundles a scheduler, an optional world, a seed (root RNG key), and the
backend, exposing convenient run methods:

- :meth:`Simulation.run` — fast ``scan``-based run at fixed capacity (optionally recording
  on-device metrics via ``record``).
- :meth:`Simulation.run_recorded` — host-loop run driving :mod:`recorders` (snapshots etc.).
- :meth:`Simulation.run_with_growth` — host-loop run that grows population capacity (by
  doubling) whenever free slots run low, for simulations with births.
- :meth:`Simulation.run_ensemble` — the ``vmap``-over-worlds primitive: run many independent
  worlds in parallel (each with its own derived root key), for replicates / parameter sweeps.

All randomness flows from ``seed`` through counter-based derivation, so runs are deterministic.
"""

from __future__ import annotations

from typing import Callable

import jax

from . import population, rng
from .backend import Backend, get_backend
from .recorders import Recorder, run_recorded
from .scheduler import Scheduler
from .schema import Schema
from .state import State

__all__ = ["Simulation"]


class Simulation:
    """Orchestrates a scheduler + world + RNG over a simulation state."""

    def __init__(self, scheduler: Scheduler, world=None, seed: int = 0,
                 schema: Schema | None = None, backend: Backend | None = None,
                 params: dict | None = None):
        self.scheduler = scheduler
        self.world = world
        self.seed = int(seed)
        self.schema = schema
        self.backend = backend or get_backend()
        self.params = params or {}
        self.root_key = rng.root_key(self.seed)

    # -- helpers -------------------------------------------------------------
    def new_state(self, capacity: int, fields: dict | None = None) -> State:
        """Allocate an empty state from the simulation's schema."""
        if self.schema is None:
            raise ValueError("Simulation has no schema; pass schema=... or build State directly")
        return State.create(self.schema, capacity, fields)

    # -- runs ----------------------------------------------------------------
    def step(self, state: State, jit: bool = True) -> State:
        """Advance a single tick."""
        return self.scheduler.step(state, self.root_key, self.world, self.backend,
                                   self.params, jit=jit)

    def run(self, state: State, n_steps: int, record: Callable[[State], object] | None = None,
            jit: bool = True):
        """Run ``n_steps`` ticks via ``scan`` (fixed capacity)."""
        return self.scheduler.run(state, n_steps, self.root_key, self.world, self.backend,
                                  self.params, record=record, jit=jit)

    def run_recorded(self, state: State, n_steps: int, recorders: list[Recorder] | tuple = (),
                     jit: bool = True, record_initial: bool = False) -> State:
        """Run ``n_steps`` ticks in a host loop, driving ``recorders`` each tick."""
        return run_recorded(self.scheduler, state, n_steps, self.root_key, recorders,
                            self.world, self.backend, self.params, jit=jit,
                            record_initial=record_initial)

    def run_with_growth(self, state: State, n_steps: int, min_free: int = 1,
                        jit: bool = True) -> State:
        """Run in a host loop, growing capacity (doubling) when free slots fall below ``min_free``.

        Use for simulations with net population growth. The jitted tick transparently
        recompiles when capacity changes (rare).
        """
        tick = self.scheduler.make_tick(self.root_key, self.world, self.backend, self.params)
        if jit:
            tick = self.backend.jit(tick)
        s = state
        for _ in range(n_steps):
            if int(population.n_free(s)) < min_free:
                s = population.grow_to_fit(s, min_free)
            s = tick(s)
        return s

    def run_ensemble(self, init_fn: Callable[[jax.Array], State], n_worlds: int, n_steps: int,
                     record: Callable[[State], object] | None = None):
        """Run ``n_worlds`` independent worlds in parallel via ``vmap`` (fixed capacity).

        ``init_fn(key) -> State`` builds one world's initial state (same shapes across worlds);
        each world gets its own root key derived from the simulation seed. Returns a batched
        State (leading world axis), or ``(batched_state, stacked_records)`` if ``record`` given.
        """
        keys = rng.split(self.root_key, n_worlds)
        world = self.world
        scheduler = self.scheduler
        params = self.params

        def one(key):
            s0 = init_fn(key)
            tick = scheduler.make_tick(key, world, self.backend, params)
            if record is None:
                def body(carry, _):
                    return tick(carry), None
            else:
                def body(carry, _):
                    ns = tick(carry)
                    return ns, record(ns)
            final, ys = jax.lax.scan(body, s0, None, length=n_steps)
            return final if record is None else (final, ys)

        return jax.vmap(one)(keys)
