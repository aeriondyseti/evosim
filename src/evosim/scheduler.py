"""Scheduler: phased-stage execution of systems with deterministic per-system RNG.

The scheduler holds an ordered list of stages; systems are registered into a stage and run
in registration order within it. Each tick runs every system in stage order, then increments
``state.tick``.

Determinism: each system gets a key ``rng.derive(root_key, tick, system_index)`` where
``system_index`` is the system's stable position in the global execution order. Because keys
are counter-based, results are identical regardless of evaluation order or parallelism.

Execution:

- :meth:`Scheduler.make_tick` builds a ``tick_fn(state) -> state`` closure.
- :meth:`Scheduler.run` runs ``n_steps`` ticks via ``backend.scan`` (jit-compiled, fixed
  capacity). For simulations that need population *growth* (capacity change), step the tick
  in a host loop and call ``population.grow_to_fit`` between ticks instead.
"""

from __future__ import annotations

from typing import Callable, Sequence

from . import rng
from .backend import Backend, get_backend
from .state import State
from .system import DEFAULT_STAGES, Context, System, SystemFn

__all__ = ["Scheduler"]


class Scheduler:
    """Registers systems into stages and runs ticks."""

    def __init__(self, stages: Sequence[str] = DEFAULT_STAGES):
        self.stages: tuple[str, ...] = tuple(stages)
        self._by_stage: dict[str, list[System]] = {s: [] for s in self.stages}

    # -- registration --------------------------------------------------------
    def add(self, sys_or_fn, stage: str | None = None, name: str | None = None):
        """Register a :class:`System` (or raw ``fn`` + ``stage``). Returns the input."""
        if isinstance(sys_or_fn, System):
            sysobj = sys_or_fn
        else:
            if stage is None:
                raise ValueError("stage is required when registering a raw function")
            sysobj = System(name=name or getattr(sys_or_fn, "__name__", "system"),
                            stage=stage, fn=sys_or_fn)
        if sysobj.stage not in self._by_stage:
            raise ValueError(f"unknown stage {sysobj.stage!r}; stages are {self.stages}")
        self._by_stage[sysobj.stage].append(sysobj)
        return sys_or_fn

    def ordered(self) -> list[System]:
        """All systems in global execution order (stage order, then registration order)."""
        out: list[System] = []
        for s in self.stages:
            out.extend(self._by_stage[s])
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_stage.values())

    # -- execution -----------------------------------------------------------
    def make_tick(self, root_key, world=None, backend: Backend | None = None,
                  params: dict | None = None) -> Callable[[State], State]:
        """Build a ``tick_fn(state) -> state`` running all systems once and bumping tick."""
        systems = self.ordered()
        params = params or {}

        def tick_fn(state: State) -> State:
            t = state.tick
            s = state
            for idx, sysobj in enumerate(systems):
                key = rng.derive(root_key, t, idx)
                ctx = Context(tick=t, key=key, world=world, backend=backend, params=params)
                s = sysobj.fn(s, ctx)
            return s.increment_tick()

        return tick_fn

    def step(self, state: State, root_key, world=None, backend: Backend | None = None,
             params: dict | None = None, jit: bool = True) -> State:
        """Run a single tick (jit-compiled by default)."""
        backend = backend or get_backend()
        tick_fn = self.make_tick(root_key, world, backend, params)
        if jit:
            tick_fn = backend.jit(tick_fn)
        return tick_fn(state)

    def run(self, state: State, n_steps: int, root_key, world=None,
            backend: Backend | None = None, params: dict | None = None,
            record: Callable[[State], object] | None = None, jit: bool = True):
        """Run ``n_steps`` ticks via ``scan``.

        Returns the final state, or ``(final_state, stacked_records)`` if ``record`` is given
        (``record(state)`` is called after each tick and its outputs are stacked along axis 0).
        Capacity is fixed for the whole run (no growth inside ``scan``).
        """
        backend = backend or get_backend()
        tick_fn = self.make_tick(root_key, world, backend, params)

        if record is None:
            def body(carry: State, _):
                return tick_fn(carry), None
        else:
            def body(carry: State, _):
                ns = tick_fn(carry)
                return ns, record(ns)

        def run_scan(s0: State):
            return backend.scan(body, s0, None, length=n_steps)

        if jit:
            run_scan = backend.jit(run_scan)
        final, ys = run_scan(state)
        return final if record is None else (final, ys)
