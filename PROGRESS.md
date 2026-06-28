# EvoSim — Build Progress Ledger

> Durable state for autonomous build. **Read this first on every iteration to re-orient**
> (context may have been compacted). Update it after every chunk of work.
> Source of truth for design = `SPEC.md`.

Last updated: 2026-06-28 (iteration 3 — rng + state done)

## Mission
Build the `evosim` library + 3 demos to completion per SPEC.md, bottom-up, with tests +
determinism golden-masters alongside. Stop only when library + 3 demos are done and `uv run pytest` passes.

## Environment notes (important for resuming)
- Repo: `D:\Development\evosim-framework` (standalone git repo). Always use absolute paths;
  the harness may pin cwd elsewhere.
- Tooling: **uv**. Run things with `uv run ...` (e.g. `uv run pytest`). Add deps with `uv add`.
- Python 3.13 (>=3.11 required). JAX = **CPU only** on Windows (no native GPU) — correctness &
  determinism dev here; GPU perf validated on Linux/WSL2 later. This does not change architecture.
- Determinism: use JAX counter-based RNG (threefry). fp32 default; allow per-field dtypes.
  For fp64 fields, enable `jax.config.update("jax_enable_x64", True)`.

## How I work the loop
Each iteration: (1) read this file, (2) pick the next unchecked task, (3) implement + test it,
(4) run `uv run pytest`, (5) check the box & jot notes below, (6) commit, (7) ScheduleWakeup.

## Module layout (target)
```
src/evosim/
  schema.py        Field, Schema, dtype handling, SoA buffer allocation
  state.py         State PyTree (component arrays, alive mask, ids, capacity, tick, rng)
  rng.py           counter-based key derivation per (tick, system, slot)
  backend.py       backend abstraction (JAX first): jit/scan/array ops/rng
  population.py    capacity grow(double), spawn/birth (deterministic free-slot claim), death, compaction
  system.py        System type + execution context
  scheduler.py     phased stages (sense->decide->act->spawn/death->environment->cleanup), tick via scan
  world/
    base.py        World protocol: topology, spatial_index, query, field layers
    grid.py        toric 2D grid: cell binning, cell/Moore-neighborhood queries
    fields.py      named 2D field layers + update systems (diffusion, regrowth, decay)
  operators/
    mutation.py    gaussian, uniform, bitflip, per-gene rate
    crossover.py   uniform, n-point, none
    selection.py   tournament, roulette, truncation, elitism
  interaction.py   spatial pairing + deterministic resolution (scatter-min/segmented arbitration)
  metrics.py       on-device per-tick reducers
  recorders.py     recorder interface; metrics recorder; snapshot recorder (npz; zarr optional)
  checkpoint.py    save/load full state + rng + tick (deterministic resume)
  sim.py           Simulation assembly + run loop + vmap-over-worlds primitive
examples/
  conway.py        Conway's Game of Life (minimal grid/field, no genome)
  foragers.py      evolving foragers (full ALife slice)
  ga_benchmark.py  classic GA (explicit-fitness path)
tests/             unit tests per module + determinism golden-masters + perf smoke
```

## Task checklist

### Phase 0 — Bootstrap
- [x] Move repo to `D:\Development\evosim-framework`, `git init`, `uv init --lib --name evosim`
- [x] Add deps (`jax`, `numpy`; dev `pytest`); verify `import jax` on CPU (jax 0.10.2, cpu)
- [x] Configure pyproject (Apache-2.0, requires-python>=3.11, pytest config), add LICENSE, .gitignore
- [x] Initial commit

### Phase 1 — Core engine
- [x] `schema.py` (Field/Schema/dtypes/SoA alloc) + tests (21 tests pass)
- [x] `rng.py` (counter-based key derivation) + tests (9 tests)
- [x] `state.py` (State PyTree, registration as JAX pytree) + tests (11 tests)
- [ ] `backend.py` (abstraction + JAX backend) + tests
- [ ] `population.py` (grow/spawn/death/compaction, deterministic claim) + tests
- [ ] `system.py` + `scheduler.py` (phased stages, tick via scan) + tests

### Phase 2 — World
- [ ] `world/base.py` (World protocol)
- [ ] `world/grid.py` (toric grid + cell/Moore queries, cell binning) + tests
- [ ] `world/fields.py` (field layers + diffusion/regrowth/decay) + tests

### Phase 3 — Genetics & interactions
- [ ] `operators/mutation.py` + tests
- [ ] `operators/crossover.py` + tests
- [ ] `operators/selection.py` + tests
- [ ] `interaction.py` (pairing + deterministic conflict resolution) + tests

### Phase 4 — Observability & persistence
- [ ] `metrics.py` (reducers) + tests
- [ ] `recorders.py` (interface + metrics + snapshot) + tests
- [ ] `checkpoint.py` (save/load, deterministic resume golden-master) + tests

### Phase 5 — Assembly
- [ ] `sim.py` (Simulation assembly, run loop, vmap-over-worlds) + tests

### Phase 6 — Demos (vertical validation)
- [ ] `examples/conway.py` + test (known-pattern golden-master, e.g. blinker/glider)
- [ ] `examples/foragers.py` + test (runs, population dynamics, evolution signal)
- [ ] `examples/ga_benchmark.py` + test (fitness improves over generations)

### Phase 7 — Hardening
- [ ] Determinism golden-masters across core (same seed -> identical state hash)
- [ ] Perf smoke benchmarks (agent-ticks/s) — CPU baseline numbers recorded
- [ ] README with quickstart + examples
- [ ] Full `uv run pytest` green; final review pass

## Running log (newest first)
- iter 3 (rng + state): `rng.py` counter-based keys (9 tests) committed. `state.py` immutable
  State PyTree (components/tick/next_id + static schema/capacity), jit/vmap-safe, fingerprint
  for golden-masters (11 tests). 41 total tests pass. Decision: RNG root key lives in the
  Simulation, not State (derived from root+tick). Next: backend.py abstraction (JAX first).
- iter 2 (schema): pyproject finalized (Apache-2.0, py>=3.11, numpy>=1.26 floor since numpy 2.5
  dropped 3.11), LICENSE + .gitignore added. `schema.py` (Field/Schema/SoA, x64 guard) +
  21 tests passing. Recorders extra (zarr/pyarrow) resolves. Next: rng.py (counter-based keys).
- iter 1 (bootstrap): repo moved, git+uv initialized, scaffold created. SPEC.md + PROGRESS.md written.

## Open questions / blockers
- (none yet)
