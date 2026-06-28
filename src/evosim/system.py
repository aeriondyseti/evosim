"""Systems: the free-form, vectorized functions that advance the simulation.

A *system* is a pure function ``(state, ctx) -> state`` that operates on the whole
population (and/or environment fields) at once. Systems are registered into a named
*stage* of the :class:`~evosim.scheduler.Scheduler`, which runs stages in a fixed order
each tick (per SPEC: ``sense -> decide -> act -> spawn -> death -> environment -> cleanup``).

Each system receives a :class:`Context` carrying the current tick, a per-system RNG key
(derived deterministically by the scheduler), the optional world, and the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax

from .state import State

__all__ = ["Context", "System", "SystemFn", "system", "DEFAULT_STAGES"]

#: Canonical stage order (per SPEC). Schedulers may use a custom tuple instead.
DEFAULT_STAGES: tuple[str, ...] = (
    "sense",
    "decide",
    "act",
    "interact",
    "spawn",
    "death",
    "environment",
    "cleanup",
)


@dataclass(frozen=True)
class Context:
    """Per-system, per-tick execution context.

    Attributes
    ----------
    tick:
        Current tick (int32 scalar array).
    key:
        RNG key derived for *this* system at *this* tick (counter-based; deterministic).
    world:
        Optional world object (static topology / query helpers). May be ``None``.
    backend:
        The active compute backend. May be ``None`` (defaults resolved by caller).
    params:
        Optional free-form mapping of static parameters for the system.
    """

    tick: jax.Array
    key: jax.Array
    world: Any = None
    backend: Any = None
    params: dict[str, Any] = field(default_factory=dict)


SystemFn = Callable[[State, Context], State]


@dataclass(frozen=True)
class System:
    """A named system bound to a stage."""

    name: str
    stage: str
    fn: SystemFn

    def __call__(self, state: State, ctx: Context) -> State:
        return self.fn(state, ctx)


def system(stage: str, name: str | None = None) -> Callable[[SystemFn], System]:
    """Decorator marking a function as a :class:`System` in ``stage``.

    >>> @system("act")
    ... def move(state, ctx):
    ...     return state
    """

    def deco(fn: SystemFn) -> System:
        return System(name=name or fn.__name__, stage=stage, fn=fn)

    return deco
