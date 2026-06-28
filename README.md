# evosim

A modular, high-performance library for large-scale **agent-based evolutionary simulations**,
built on a [JAX](https://github.com/jax-ml/jax) core. Designed to run and update very large
populations (toward millions of entities) efficiently by keeping the hot loop fully vectorized
and JIT-compiled, with **no per-entity Python**.

> Status: alpha. CPU-validated on Windows; the same XLA code targets GPU/TPU (see *Performance*).
> See [`SPEC.md`](SPEC.md) for the full design rationale.

## Highlights

- **Data-oriented ECS** — declare typed components; the framework stores them as
  structure-of-arrays (SoA) buffers and runs your *systems* as pure, vectorized array functions.
- **Pluggable everything** — worlds, genomes, genetic operators, recorders, and the backend are
  all swappable. First world: a toric 2D grid with cell/Moore-neighborhood queries and field layers.
- **Deterministic** — counter-based RNG (Threefry) gives same-device bit-exact runs and exact
  checkpoint/resume.
- **Dynamic populations** — capacity buffers that grow by doubling, with deterministic spawn
  (free-slot claim), death, and compaction.
- **Both selection modes** — emergent natural selection *and* explicit-fitness GA.
- **Deterministic interactions** — contention (eat/move/mate) resolved via scatter-min
  arbitration; symmetric pairing for mating.
- **Observability & persistence** — on-device aggregate metrics, pluggable recorders, full
  save/resume checkpoints.
- **Ensembles** — `vmap`-over-worlds primitive to run many independent worlds (replicates /
  sweeps) in parallel.

## Install

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # core (jax, numpy)
uv sync --extra recorders    # + zarr/pyarrow for disk recorders
uv sync --extra viz          # + pygame for the live viewer
```

JAX has no native Windows GPU build, so development/testing here runs on CPU. The architecture
is backend-agnostic (XLA), so the same code runs on GPU/TPU where a CUDA/TPU `jaxlib` is available.

## Quickstart

```python
import jax.numpy as jnp
import evosim
from evosim import Schema, Field, Scheduler, System, Simulation

# 1. Declare components (stored as SoA buffers; `alive`/`id` are added automatically).
schema = Schema(
    energy=Field(dtype="float32", default=1.0),
    genome=Field(dtype="float32", shape=(4,)),
)

# 2. Write systems: pure (state, ctx) -> state functions, vectorized over all agents.
def metabolize(state, ctx):
    return state.set("energy", state["energy"] - 0.01)

sched = Scheduler()
sched.add(System("metabolize", "death", metabolize))

# 3. Assemble and run.
sim = Simulation(sched, seed=0, schema=schema)
state = sim.new_state(capacity=10_000).set("alive", jnp.ones((10_000,), bool))
final, energy = sim.run(state, n_steps=100,
                        record=lambda s: evosim.metrics.masked_mean(s["energy"], s.alive))
print(final, float(energy[-1]))
```

## Reference demos

Run any demo with `python -m`:

```bash
uv run python -m evosim.examples.conway        # Conway's Life (field-only, minimal usage)
uv run python -m evosim.examples.foragers      # evolving foragers (full ALife, emergent selection)
uv run python -m evosim.examples.ga_benchmark  # classic GA (explicit-fitness optimization)
```

Or use the Windows launchers in [`scripts/`](scripts) (PowerShell `.ps1` or `.bat`), which work
from any directory and pass through extra args:

```powershell
scripts\run_conway.ps1            # or scripts\run_conway.bat
scripts\run_foragers.ps1
scripts\run_ga_benchmark.ps1
```

- **conway** — the smallest viable sim: one environment field + the `life_like` cellular-automaton
  system on a toric grid. No agents.
- **foragers** — agents seek food via taxis on a regrowing food field, contend for it (lottery
  arbitration), reproduce with mutation, and starve. The mean *efficiency* gene rises over
  generations: emergent natural selection.
- **ga_benchmark** — a generational GA minimizing sphere/Rastrigin via elitism + tournament
  selection + crossover + mutation.

## Visualization (optional)

A decoupled, read-only PyGame viewer ships in `evosim.viz` (install `evosim[viz]`). The Conway
and foragers demos accept `--view`:

```bash
uv sync --extra viz
uv run python -m evosim.examples.conway --view          # live cellular automaton
uv run python -m evosim.examples.foragers --view        # food heatmap + agents colored by energy
```

Or use the dedicated viewer launchers in [`scripts/`](scripts) (PowerShell `.ps1` or `.bat`;
extra args pass through, e.g. `--steps`, `--seed`):

```powershell
scripts\run_conway_view.ps1            # or scripts\run_conway_view.bat
scripts\run_foragers_view.ps1
```

Controls (also shown in an on-screen legend, toggle with **H**): **SPACE** pause · **S**/**.**
step one tick while paused · **↑/→** faster · **↓/←** slower · **ESC/Q** quit.

It builds on the existing host-loop runner, so the headless fast path and determinism are
untouched. The renderers are pure-numpy and dependency-free; only the window needs pygame:

```python
from evosim.viz import run_live, GridRenderer, AgentRenderer
run_live(sim, state, n_steps=None,
         layers=[GridRenderer("food", cmap="fire"),
                 AgentRenderer("position", color_by="energy", cmap="viridis")])
```

`PygameViewer` is also a `Recorder`, so a window can be driven by `recorders.run_recorded`
alongside other recorders. Everything runs headless under `SDL_VIDEODRIVER=dummy` (CI).

## Architecture (one tick)

```
State (immutable PyTree: SoA components + env fields + tick + next_id)
   │
Scheduler runs systems in phased stages:
   sense -> decide -> act -> interact -> spawn -> death -> environment -> cleanup
   │  (each system gets a deterministic per-system RNG key: derive(root, tick, system_index))
   v
State'   — whole tick is jit/scan-compiled; capacity fixed within a run (grow between runs)
```

Key modules: `schema`, `state`, `rng`, `backend`, `population`, `system`, `scheduler`,
`world/` (grid + fields), `operators/` (mutation/crossover/selection), `interaction`,
`metrics`, `recorders`, `checkpoint`, `sim`.

## Performance

The hot loop forbids per-entity Python: systems operate on batched SoA arrays and the whole tick
is JIT-compiled. The CPU smoke test (`tests/test_perf_smoke.py`) sustains tens of millions of
agent-ticks/second on a laptop; the large-population GPU targets in [`SPEC.md`](SPEC.md) are
validated separately on Linux/WSL2.

## Development

```bash
uv run pytest                                # full test suite
uv run pytest -s tests/test_perf_smoke.py    # see the throughput number
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
