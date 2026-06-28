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
uv sync --extra demos        # + pygame for the interactive demo viewers
```

The `evosim.viz` renderers are pure-numpy and need no extra; only the interactive PyGame demo
viewers require `--extra demos`.

JAX has no native Windows GPU build, so development/testing here runs on CPU. The architecture
is backend-agnostic (XLA), so the same code runs on GPU/TPU where a CUDA/TPU `jaxlib` is
available. On Linux/WSL2, `uv sync --extra gpu` installs `jax[cuda12]` and the same code runs on
the GPU — validated on NVIDIA RTX 5060 Ti / 4060 Ti (see *Performance*).

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

The library ships **framework-agnostic, dependency-free renderers** in `evosim.viz`
(`GridRenderer`, `AgentRenderer`, `ScatterRenderer`, `compose`, colormaps) that turn a `State`
into RGB images — independent of what eventually paints them. A concrete **live PyGame viewer is
shipped as an example** (`evosim.examples.pygame_viewer`), the same way Conway demonstrates the
core engine; it requires the optional `demos` extra. All three demos accept `--view`:

```bash
uv sync --extra demos
uv run python -m evosim.examples.conway --view          # live cellular automaton
uv run python -m evosim.examples.foragers --view        # food heatmap + agents colored by energy
uv run python -m evosim.examples.ga_benchmark --view    # GA population in genome space + curve
```

The foragers viewer draws the **food field as the background** and **agents as distinct dots**
on top (via `run_live(frame_fn=<field image>, overlay_fn=agent_overlay(...))`), so agents are
easy to tell apart from grid cells. The GA viewer shows the *non-spatial* case: the population
scattered in genome space (dims 0,1), colored by fitness, with the optimum marked and a live
best-objective convergence curve — built with `ScatterRenderer` via `run_live(frame_fn=...)`.

Or use the dedicated viewer launchers in [`scripts/`](scripts) (PowerShell `.ps1` or `.bat`;
extra args pass through, e.g. `--steps`, `--seed`):

```powershell
scripts\run_conway_view.ps1            # or scripts\run_conway_view.bat
scripts\run_foragers_view.ps1
scripts\run_ga_benchmark_view.ps1
```

Controls (also shown in an on-screen legend, toggle with **H**): **SPACE** pause · **S**/**.**
step one tick while paused · **↑/→** faster · **↓/←** slower · **ESC/Q** quit.

### Scaling: the million-agent PoC

The foragers/GA viewers above draw agents as per-agent dots/points (great up to ~thousands).
For **millions**, use the *rasterized* path — agents scattered into the grid via
`AgentRenderer` (O(cells), one image blit/frame, independent of population):

```bash
uv run python -m evosim.examples.foragers_large                      # ~1e6 agents, 1024x1024 grid
uv run python -m evosim.examples.foragers_large --headless-bench 10  # throughput, no window
scripts\run_foragers_large_view.ps1 --agents 250000 --grid 512       # launcher (args pass through)
```

It reuses the foragers simulation at scale and composites `GridRenderer` (food) +
`AgentRenderer` (agents). At 1,000,000 agents it sustains **~8 ticks/s on CPU (≈8M
agent-ticks/s)** and **~560 ticks/s on GPU (≈560M agent-ticks/s, RTX 5060 Ti)** — see
*Performance*.

The example viewers build on the existing host-loop runner, so the headless fast path and
determinism are untouched:

```python
from evosim.viz import GridRenderer, AgentRenderer          # library: renderers (no GUI dep)
from evosim.examples.pygame_viewer import run_live          # example: the PyGame driver

run_live(sim, state, n_steps=None,
         layers=[GridRenderer("food", cmap="fire"),
                 AgentRenderer("position", color_by="energy", cmap="viridis")])
```

The example's `PygameViewer` is also a `Recorder`, so a window can be driven by
`recorders.run_recorded` alongside other recorders. Everything runs headless under
`SDL_VIDEODRIVER=dummy` (CI). Because the renderers are GUI-agnostic, you can write your own
matplotlib / web / moderngl driver against the same `GridRenderer`/`AgentRenderer` API.

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
agent-ticks/second on a laptop. On GPU (Linux/WSL2, `uv sync --extra gpu`) the `foragers_large`
PoC runs **~70× faster than CPU**: at 1M agents, 562M agent-ticks/s on an RTX 5060 Ti (16 GB,
Blackwell) and 485M on an RTX 4060 Ti (8 GB, Ada); at 4M agents on a 2048×2048 grid, 334M
agent-ticks/s on the 5060 Ti. The full test suite passes on GPU. Determinism is same-device
bit-exact (counter-based RNG); CPU and GPU agree statistically but not bit-for-bit (different
reduction order), as noted in [`SPEC.md`](SPEC.md).

> Multi-GPU note: CUDA's default device order is `FASTEST_FIRST`, so `CUDA_VISIBLE_DEVICES`
> indices need not match `nvidia-smi`. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to make them line up.

## Development

```bash
uv run pytest                                # full test suite
uv run pytest -s tests/test_perf_smoke.py    # see the throughput number
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
