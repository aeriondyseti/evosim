# EvoSim Framework — Specification (Draft for Sign-off)

A highly modular library for running large-scale (millions of entities) evolutionary
simulations efficiently. **It is a library, not an application**: other (primarily Python)
apps `import` it and implement their own simulations on top. There is no mandatory CLI/app;
example runners and the viewer are just optional consumers of the core.

---

## 1. Decisions locked in

| Area | Decision |
|---|---|
| Language / API | **Python** authoring API |
| Hot-loop core | **JAX** (XLA) — same code on CPU/GPU/TPU; behind a **backend abstraction** (JAX first, others later) |
| Paradigm | **Agent-based, entities in a world** |
| Update model | **Configurable** scheduler (tick, generational, continuous/event later) |
| Perf architecture | **Hybrid CPU+GPU**, data-oriented **SoA** |
| World model | **Pluggable world shapes**; first module = **discrete toric 2D grid** |
| Selection | **Pluggable**: emergent natural selection AND explicit fitness |
| Genome | **User-defined / pluggable**, but **array-shaped** (typed schema → SoA) |
| Population growth | **Capacity + double-on-grow**, alive-mask + periodic compaction |
| Determinism | **Same-device bit-exact** (counter-based RNG, ordered reductions) |
| Spatial queries | **Pluggable per world**: grid → cell/Moore neighborhood; continuous → radius / kNN |
| Composition API | **ECS** (Components + Systems + Scheduler); **free-form system functions** |
| Scheduler | **Phased stages** (recommended): `sense → decide → act → spawn/death → environment → cleanup` |
| Heterogeneity | **Single uniform agent layout first**, archetypes later |
| Environment | **Both** continuous grid *layers/fields* AND resources-as-entities |
| Lineage | **Opt-in lightweight** (parent id + birth tick default; richer optional) |
| Genetic operators | **Library + user override** (mutation/crossover/selection) |
| Interactions | **Spatial-query + deterministic resolution systems**; message-passing layer later |
| Conflict resolution | **Deterministic arbitration** (scatter-min / segmented reduction; priority + RNG tiebreak) |
| Observability | **Aggregate metrics** (on-device reducers) + **pluggable recorders** |
| Visualization | **Decoupled**: headless engine + optional **native GPU window** viewer (deferred) |
| Checkpointing | **Full deterministic resume** (state + RNG) |
| Precision | **Mixed / per-field dtypes** (fp32 default for floats) |
| Scale target | **Single-GPU first, designed for scale-out**; `vmap`-over-worlds available as a primitive |
| Experiments | **No orchestration layer** — consuming app owns it |
| Audience / distribution | **Open-source research library**, pip-installable |
| Perf benchmark goal | **1M agents @ ≥30–60 ticks/s** (simple logic) and **≥100M agent-ticks/s** batch throughput |
| v1 milestone | **End-to-end vertical slice** |
| Reference demos | Evolving foragers · Classic GA benchmark · **Conway's Game of Life** (minimal viable usage) |
| License | **Apache-2.0** |
| Package name | **`evosim`** |
| Python / core deps | **Python ≥3.11**; hard deps `jax`, `numpy`; optional `zarr`/`pyarrow` (recorders) |
| Tooling | **uv** for env / deps / running (`uv init`, `uv add`, `uv run`) |
| Dev platform | **Windows + CPU JAX** for dev/test; GPU perf validated on Linux/WSL2 later |
| Viewer toolkit | **moderngl** (deferred build) |
| Repo | standalone git repo at `evosim-framework` |

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│ Consuming app (yours) — defines components, systems, config   │
└───────────────┬───────────────────────────────────────────────┘
                │ imports
┌───────────────▼───────────────────────────────────────────────┐
│ EvoSim public API                                              │
│  • Schema/Component registry   • System registration           │
│  • Scheduler (phased stages)   • Operator library              │
│  • World modules               • Recorders / metrics           │
│  • Checkpoint / resume         • vmap-over-worlds primitive     │
├────────────────────────────────────────────────────────────────┤
│ Core engine (state container, SoA buffers, masks, RNG, compaction)
├────────────────────────────────────────────────────────────────┤
│ Backend abstraction  →  [JAX backend]  ( [Numba/Rust] later )   │
└────────────────────────────────────────────────────────────────┘
        Optional, decoupled consumers: native viewer, recorders sinks
```

### Core principles
- **No per-entity Python in the hot loop.** Systems are pure functions over batched SoA arrays, JIT-compiled.
- **State is a single immutable PyTree** threaded through the tick (`state -> state'`), enabling `jit`, `scan`, `vmap`, and deterministic checkpointing.
- **Everything pluggable is a registered function/module** conforming to a typed interface.

---

## 3. Data model

### 3.1 Component schema (typed → SoA)
Users declare components as named, typed, shaped fields. The framework owns the
structure-of-arrays buffers (one array per field, length = capacity) and per-field dtype.

```python
schema = Schema(
    position = Field(dtype="int16", shape=(2,)),     # grid cell coords (toric)
    energy   = Field(dtype="float32"),
    genome   = Field(dtype="float32", shape=(G,)),   # user-defined width G
    alive    = Field(dtype="bool"),                  # framework-managed mask
    parent   = Field(dtype="int32"),                 # opt-in lineage
    birth_t  = Field(dtype="int32"),                 # opt-in lineage
)
```

- `alive` mask + `id` are framework-managed reserved fields.
- Mixed dtypes supported per field (int8 traits, fp16 positions, fp32 genome, …).
- Single uniform agent layout in v1; archetype support (multiple layouts/tables) is a later additive feature.

### 3.2 Population buffers
- Fixed **capacity** N; logical population ≤ N tracked by `alive` mask.
- **Births** write into free slots (claimed deterministically); when no free slots and growth needed → **double capacity** (one-time recompile for the new static shape).
- **Deaths** clear the mask; **compaction** runs periodically to keep arrays dense (improves locality and neighbor-query cost).

### 3.3 RNG
- **Counter-based** (JAX Threefry). A root seed + deterministic key derivation per
  (tick, system, entity-slot) gives reproducible, parallel-safe randomness without
  sequential state. Saved in checkpoints for bit-exact resume.

---

## 4. World modules

Interface (per world module):
- `topology` (shape + boundary; toric grid first)
- `spatial_index` build/refresh over agent positions
- `query` API: grid exposes **cell / Moore-neighborhood**; continuous exposes **radius / kNN**
- **Field layers**: named 2D arrays co-located with the world (resource, pheromone, terrain,
  hazard), each with its own update system (diffusion, regrowth, decay).
- Resources may also be modeled as ordinary entities; both supported.

**First module:** discrete **toric 2D grid**. World shape is swappable (other grids, continuous 2D, 3D, graph) via the same interface.

### Spatial index implementation (recommended defaults)
- Grid: cell-bin via sort/segment (counting sort by cell id) → O(N) neighbor gather.
- Continuous: uniform spatial-hash grid sized to interaction radius.

---

## 5. Scheduler & systems

- **Phased stages** (fixed, extensible): `sense → decide → act → spawn/death → environment → cleanup/compaction`.
- Users **register system functions** into a stage; order within a stage is explicit.
- Each system: `(state, world, rng_key, ctx) -> state'`, pure & JIT-able, vectorized over all agents.
- Tick = run all stages in order. Whole tick is `jit`-compiled; long runs use `lax.scan`.
- Update model is configurable by composing stages (discrete tick first; generational and
  continuous/event-driven schedulers are later additive schedulers over the same systems).
- Future: intra-stage auto-parallelism via declared component read/write sets (DAG).

---

## 6. Interactions & conflict resolution (the hard part)

- **Interactions** (mate, fight, eat, communicate): a system uses spatial queries to find
  candidate pairs/targets, then a **deterministic resolution pass** applies effects.
- **Conflicts** (two agents claim same cell / food / mate): resolved by **scatter-min /
  segmented reduction** — each contested target deterministically keeps one winner ranked by
  (priority, counter-RNG tiebreak). Stays same-device bit-exact; stochastic lottery is just a
  random priority.
- **Message-passing** layer (agents post to cells/targets, gather delivers) is a later optional
  module built on the same gather primitive.

---

## 7. Genetics

- **Operator library** with user override:
  - Mutation: gaussian, uniform, bitflip, per-gene rate, …
  - Crossover: uniform, n-point, none (asexual).
  - Selection: tournament, roulette, truncation, elitism (explicit-fitness path).
- **Emergent selection**: no fitness fn; reproduction gated by energy/resources, death by
  starvation/predation/age. **Explicit fitness**: user `fitness(state)->scores` drives
  selection operators. Both selectable.
- Sexual reproduction pairing is an interaction (Section 6) feeding crossover.

---

## 8. Observability

- **Aggregate metrics**: on-device per-tick reducers (population, trait mean/var, diversity,
  energy totals, births/deaths) → time-series, cheap.
- **Pluggable recorders**: observer interface that taps state at configurable cadence without
  touching the hot loop; sinks to disk (parquet / zarr / npz) — full or **sampled** snapshots.
- **Lineage** (opt-in): parent id + birth tick by default; richer mutation/event tracking with
  sampling/pruning to bound cost.

---

## 9. Checkpoint / resume

- Serialize full state PyTree + RNG state + tick counter + config hash → bit-exact resume on
  the same device/backend. Versioned format; used for crash recovery, branching, and long runs.

---

## 10. Visualization

- Engine is **headless**. A decoupled, read-only viewer consumes state via the host-loop runner
  so rendering never affects determinism or the fast path.
- **IMPLEMENTED:** `evosim.viz` — a **PyGame** viewer (`run_live`, plus a `PygameViewer`
  `Recorder`) with pure-numpy renderers (`GridRenderer` field heatmaps, `AgentRenderer` agent
  rasterization, colormaps). Behind the `viz` extra; Conway/foragers demos take `--view`; runs
  headless under `SDL_VIDEODRIVER=dummy`. (A native moderngl/GPU window remains possible later
  for very large scenes.)

---

## 11. Performance targets (benchmark suite)

- **1M agents @ ≥30–60 ticks/s** on one modern GPU for simple per-agent logic.
- **≥100M agent-ticks/s** sustained for headless batch runs.
- fp32 default. These are regression benchmarks gating releases.

---

## 12. v1 milestone — end-to-end vertical slice

Proves the whole stack on one path:
1. ECS core: schema/SoA, capacity+grow, masks, compaction, counter-RNG.
2. Backend abstraction with the JAX backend.
3. Phased scheduler + free-form systems.
4. Toric 2D grid world + cell/Moore queries + at least one field layer.
5. Operator library (mutation/crossover/selection) + both selection modes.
6. Deterministic interaction + conflict-resolution primitives.
7. Aggregate metrics + one recorder + checkpoint/resume.
8. **Demos**: Conway's Game of Life (minimal grid/field, no genome) · Evolving foragers
   (full ALife slice) · Classic GA benchmark (explicit-fitness path).
9. Determinism golden-master tests + performance benchmarks.

---

## 13. Settled project defaults

- **License**: **Apache-2.0** (permissive + patent grant; standard for ML/research infra).
- **Python / deps**: **Python ≥3.11**; hard deps `jax`, `numpy`; optional `zarr`/`pyarrow`
  (recorders); viewer deps optional.
- **Tooling**: **uv** manages the environment, dependencies, and task running
  (`uv init`, `uv add`, `uv run pytest`, etc.). `pyproject.toml` is the single source of truth.
- **Dev platform**: developed/tested on **Windows with CPU `jaxlib`** (JAX has no native
  Windows GPU build). XLA is the same backend on GPU, so architecture is unaffected; the
  large-GPU performance targets in §11 are validated later on Linux/WSL2. CPU is sufficient
  for correctness, determinism (same-device bit-exact), and the demos.
- **Package name**: **`evosim`**.
- **Viewer toolkit**: **moderngl** (build deferred; minimal matplotlib/notebook view first).
- **Repo location**: standalone git repo initialized at `evosim-framework` (independent of
  the surrounding rpstack working tree).

---

## 14. Explicitly out of scope for v1 (designed-for, added later)

Archetypes/heterogeneous layouts · continuous/3D/graph worlds · message-passing layer ·
multi-GPU & multi-node scale-out · generational/event-driven schedulers · auto-parallel
DAG scheduler · GP/variable-length genomes · alternate backends (Numba/Rust).
