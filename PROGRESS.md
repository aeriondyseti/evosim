# EvoSim — Build Progress Ledger

> Durable state for autonomous build. **Read this first on every iteration to re-orient**
> (context may have been compacted). Update it after every chunk of work.
> Source of truth for design = `SPEC.md`.

Last updated: 2026-06-28 (iteration 12 — PHASE 7 DONE; PROJECT COMPLETE ✅)

## ✅ COMPLETE — library + all 3 demos built. Loop stopped.
## Post-completion: added PyGame visualization (evosim.viz). 177 tests pass.

### Visualization (added after core completion)
- LIBRARY: `src/evosim/viz/render.py` — pure-numpy, GUI-agnostic renderers: GridRenderer
  heatmaps, AgentRenderer rasterization, ScatterRenderer/scatter_image (continuous/non-grid
  point clouds), colormaps, compose. `evosim.viz` has NO GUI dep (needs no extra).
- EXAMPLE: `src/evosim/examples/pygame_viewer.py` — run_live (window loop; legend toggle H,
  pause SPACE, step S/., speed arrows, quit ESC/Q). Supports grid `layers` OR a custom
  `frame_fn`/`overlay_fn` (so non-grid sims work). + PygameViewer Recorder. PyGame is shipped
  as an example of consuming the viz renderers, NOT in the library.
- `demos` extra (pygame) in pyproject — renamed from `viz` so installing the renderers never
  pulls a GUI toolkit. ALL THREE demos take `--view`:
    - conway: life heatmap; foragers: food field as BACKGROUND image (frame_fn) + agents as
      distinct DOTS on top via agent_overlay() helper (pygame circles, color-by-energy, outlined)
      so agents are visually separable from cells.
    - ga_benchmark.run_view: NON-SPATIAL viz — population in genome space (dims 0,1) colored by
      fitness, optimum marker + best-objective convergence curve (frame_fn + ScatterRenderer).
- SCALE PoC: examples/foragers_large.py — reuses foragers sim at ~1e6 agents on a 1024x1024
  grid, rasterized rendering (GridRenderer food + AgentRenderer scatter, O(cells), one blit/
  frame). CPU bench: ~8 ticks/s @ 1M agents (~8M agent-ticks/s). `--headless-bench N` for timing.
- launchers: scripts/run_{conway,foragers,ga_benchmark,foragers_large}_view.{ps1,bat}.
- tests: test_viz.py (renderers incl scatter, no pygame) + test_example_viewer.py (pygame +
  GA viewer, headless via SDL dummy). 184 tests pass. Determinism/fast path untouched.

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
- [x] `backend.py` (abstraction + JAX backend) + tests (8 tests)
- [x] `population.py` (grow/spawn/death/compaction, deterministic claim) + tests (10 tests)
- [x] `system.py` + `scheduler.py` (phased stages, tick via scan) + tests (11 tests)
- [x] `state.py` extended with environment `fields` (grids/layers) + tests (4 tests)

### Phase 2 — World
- [x] `world/base.py` (World ABC: ndim/shape/wrap)
- [x] `world/grid.py` (ToricGrid2D: cell_id/move/cell_counts/scatter+gather_field/
      moore_sum/von_neumann_sum/accumulate_neighborhood/random_positions) + tests (13)
- [x] `world/fields.py` (decay/regrow/diffuse/map_field/life_like) + tests (6)
      NOTE: Conway rule validated here (life_like blinker oscillates, block still) —
      conway demo will just wire ToricGrid2D + life_like("cells").

### Phase 3 — Genetics & interactions
- [x] `operators/mutation.py` (gaussian/uniform/bitflip) + tests
- [x] `operators/crossover.py` (clone/uniform/one_point/n_point/blend) + tests
- [x] `operators/selection.py` (tournament/roulette/truncation/elitism) + tests (19 total)
- [x] `interaction.py` (resolve_claims scatter-min arbitration, resolve_cell_claims,
      mutual_match pairing, lottery_priorities) + tests (12)

### Phase 4 — Observability & persistence
- [x] `metrics.py` (masked_mean/var, population, mean/var/sum_of, genetic_diversity,
      MetricSet w/ compute + record_fn, standard builder) + tests (7)
- [x] `recorders.py` (Recorder, MetricsRecorder, SnapshotRecorder+save_npz, run_recorded
      host loop) + tests (6). NOTE: recorders fire on 1st/(1+every)/... invocation.
- [x] `checkpoint.py` (npz save/load full state+fields+rng+schema, deterministic resume
      golden-master PASSES) + tests (4)

### Phase 5 — Assembly
- [x] `sim.py` (Simulation: run/run_recorded/run_with_growth/run_ensemble vmap-over-worlds)
      + tests (7). Library is now feature-complete. Exported Simulation.

### Phase 6 — Demos (vertical validation)
- [x] `examples/conway.py` + test — glider/blinker/block golden-masters PASS (6 tests).
      NOTE: demos live in src/evosim/examples/ (importable as evosim.examples.*, runnable
      via `python -m evosim.examples.conway`).
- [x] `examples/foragers.py` + test — full ALife: taxis move, food-field eat w/ lottery
      arbitration, reproduction+mutation, metabolism/death. Emergent selection: mean
      efficiency gene rises ~0 -> 0.06 over 400 steps; pop stabilizes at carrying capacity.
      5 tests (survival, births, selection signal, determinism). VALIDATED.
- [x] `examples/ga_benchmark.py` + test — generational GA (sphere/rastrigin), explicit-fitness
      path: evaluate -> elitism + tournament + uniform crossover + gaussian mutation.
      Sphere converges 4.3 -> 0.024; elitism gives monotonic non-increasing best. 6 tests.

### Phase 7 — Hardening
- [x] Determinism golden-masters across core (test_determinism.py: same-seed identical,
      diff-seed differs, scan==host-loop) + per-module determinism tests throughout
- [x] Perf smoke benchmark (test_perf_smoke.py): ~31.5M agent-ticks/s on CPU (50k agents x
      100 steps in ~159ms). GPU targets validated separately on Linux/WSL2 per SPEC.
- [x] README with quickstart (verified runnable), demos, architecture, perf, install
- [x] Full `uv run pytest` green — 168 tests pass

## Running log (newest first)
- iter 12 (PHASE 7 DONE — PROJECT COMPLETE): perf smoke (~31.5M agent-ticks/s CPU),
  consolidated determinism golden-masters (test_determinism.py), README (quickstart verified).
  168 tests pass. Library + all 3 demos complete and tested -> LOOP STOPPED.
- iter 11 (demo 3; PHASE 6 DONE): ga_benchmark.py (generational GA, sphere converges to
  ~0.02, rastrigin improves, elitism monotonic; 6 tests). ALL 3 DEMOS COMPLETE. 164 tests
  pass. Remaining: Phase 7 hardening — perf smoke benchmark, README quickstart, final review.
- iter 10 (demos 1+2): conway.py (glider/blinker/block golden-masters, 6 tests) + foragers.py
  (full ALife, emergent selection signal confirmed: eff gene 0->0.06; 5 tests). 158 tests pass.
  Demos in src/evosim/examples/. Next: ga_benchmark.py (explicit-fitness GA path), then
  Phase 7 hardening (perf smoke, README).
- iter 9 (sim assembly; PHASE 5 DONE — LIBRARY FEATURE-COMPLETE): sim.py Simulation
  (run/run_recorded/run_with_growth/run_ensemble vmap-over-worlds), 7 tests. 147 tests pass.
  Library core is complete; remaining = Phase 6 demos (conway, foragers, ga_benchmark) +
  Phase 7 hardening (golden-masters, perf smoke, README). Next: examples/conway.py.
- iter 8 (observability + persistence; PHASE 4 DONE): metrics.py (on-device reducers +
  MetricSet), recorders.py (MetricsRecorder/SnapshotRecorder + run_recorded host loop),
  checkpoint.py (npz save/load, deterministic resume golden-master passes). Exported all.
  140 tests pass. Next: Phase 5 assembly — sim.py (Simulation wrapper, run loop with growth,
  vmap-over-worlds primitive). Then Phase 6 demos.
- iter 7 (genetics + interaction; PHASE 3 DONE): operators/{mutation,crossover,selection}
  (19 tests) + interaction.py (resolve_claims deterministic scatter-min arbitration with
  index tie-break, resolve_cell_claims, mutual_match, lottery_priorities; 12 tests).
  operators+interaction exported. 124 tests pass. Next: Phase 4 observability/persistence —
  metrics.py, recorders.py, checkpoint.py.
- iter 6 (world; PHASE 2 DONE): `world/base.py` (World ABC), `world/grid.py` (ToricGrid2D
  cell-binning + toric neighborhood ops, 13 tests), `world/fields.py` (decay/regrow/diffuse/
  map_field/life_like, 6 tests). Conway rule already validated via life_like. world exported
  in __init__. 93 tests pass. Next: Phase 3 genetics — operators/{mutation,crossover,selection}.
- iter 5 (system + scheduler; PHASE 1 DONE): added env `fields` dict to State (needed for
  agent-less sims like Conway). `system.py` (System/Context/@system, DEFAULT_STAGES =
  sense/decide/act/interact/spawn/death/environment/cleanup). `scheduler.py` (stage-ordered
  registration, per-system counter-RNG keys, make_tick/step/run via scan, record callback).
  Public API exported in __init__. 74 tests pass. Next: world/base.py + world/grid.py
  (toric grid, cell/Moore queries, cell binning).
- iter 4 (backend + population): `backend.py` Backend/JAXBackend (jit/scan/vmap/xp/devices) +
  registry/use_backend (8 tests). `population.py` deterministic spawn (free-slot claim via
  argsort+cumsum, overflow drop, contiguous ids), kill, compact (stable), grow/grow_to_fit
  (host-level, doubling) (10 tests). 59 total pass. Next: system.py + scheduler.py
  (phased stages: sense->decide->act->spawn/death->environment->cleanup; tick via scan).
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
