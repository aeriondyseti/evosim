"""Pluggable recorders + a host-loop runner.

Two data-collection paths (per SPEC):

- **On-device metrics (fast path):** pass ``metricset.record_fn()`` to ``Scheduler.run`` and
  the per-tick values are stacked by ``scan`` with no host involvement.
- **Host-side recorders (flexible path):** :func:`run_recorded` steps the simulation in a
  host loop (each tick jit-compiled) and calls recorders after every tick. This supports
  snapshots to disk and arbitrary Python-side collection, at the cost of per-tick host sync.

Recorders implement ``record(state)`` (called each tick) and ``result()``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .backend import Backend, get_backend
from .metrics import MetricSet
from .state import State

__all__ = ["Recorder", "MetricsRecorder", "SnapshotRecorder", "run_recorded"]


class Recorder:
    """Base recorder. Subclasses override :meth:`record` and :meth:`result`."""

    def record(self, state: State) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def result(self):  # pragma: no cover - interface
        raise NotImplementedError


class MetricsRecorder(Recorder):
    """Collect a :class:`MetricSet` every ``every`` ticks into host arrays."""

    def __init__(self, metricset: MetricSet, every: int = 1):
        self.metricset = metricset
        self.every = max(1, int(every))
        self._rows: list[dict[str, np.ndarray]] = []
        self._ticks: list[int] = []
        self._i = 0

    def record(self, state: State) -> None:
        if self._i % self.every == 0:
            vals = self.metricset.compute(state)
            self._rows.append({k: np.asarray(v) for k, v in vals.items()})
            self._ticks.append(int(state.tick))
        self._i += 1

    def result(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {"tick": np.asarray(self._ticks)}
        if self._rows:
            for k in self._rows[0]:
                out[k] = np.stack([r[k] for r in self._rows])
        return out


class SnapshotRecorder(Recorder):
    """Snapshot selected components and/or fields every ``every`` ticks."""

    def __init__(self, components: Iterable[str] = (), fields: Iterable[str] = (),
                 every: int = 1):
        self.components = tuple(components)
        self.fields = tuple(fields)
        self.every = max(1, int(every))
        self._snaps: list[tuple[int, dict[str, np.ndarray]]] = []
        self._i = 0

    def record(self, state: State) -> None:
        if self._i % self.every == 0:
            snap: dict[str, np.ndarray] = {}
            for name in self.components:
                snap[name] = np.asarray(state[name])
            for name in self.fields:
                snap[f"field::{name}"] = np.asarray(state.get_field(name))
            self._snaps.append((int(state.tick), snap))
        self._i += 1

    def result(self) -> list[tuple[int, dict[str, np.ndarray]]]:
        return self._snaps

    def save_npz(self, path: str) -> None:
        """Save snapshots to a .npz file (arrays stacked per key; ``ticks`` array included)."""
        if not self._snaps:
            np.savez(path, ticks=np.asarray([]))
            return
        ticks = np.asarray([t for t, _ in self._snaps])
        keys = self._snaps[0][1].keys()
        data = {k: np.stack([snap[k] for _, snap in self._snaps]) for k in keys}
        data["ticks"] = ticks
        np.savez(path, **data)


def run_recorded(scheduler, state: State, n_steps: int, root_key, recorders: Sequence[Recorder] = (),
                 world=None, backend: Backend | None = None, params: dict | None = None,
                 jit: bool = True, record_initial: bool = False) -> State:
    """Step the simulation ``n_steps`` ticks in a host loop, invoking recorders each tick.

    Returns the final state; collected data lives in each recorder's ``result()``.
    """
    backend = backend or get_backend()
    tick = scheduler.make_tick(root_key, world, backend, params)
    if jit:
        tick = backend.jit(tick)
    s = state
    if record_initial:
        for r in recorders:
            r.record(s)
    for _ in range(n_steps):
        s = tick(s)
        for r in recorders:
            r.record(s)
    return s
