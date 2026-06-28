"""Example: million-agent foragers — a scale PoC for the *rasterized* viewer path.

Same simulation as :mod:`evosim.examples.foragers`, but with a vastly larger grid and ~1e6
agents. At this scale the per-agent dot overlay (``agent_overlay``) is far too slow, so this
viewer uses the **rasterized** path: the ``AgentRenderer`` scatters agents into the grid with
``scatter_field`` (O(cells), pure array ops) and is composited over the food field — one image
blit per frame, independent of population size.

Notes:
- The grid is sized so ~1e6 cells exist (one-eater-per-cell sets the carrying capacity), so the
  population isn't instantly starved. Tune with ``--agents`` / ``--grid``.
- The big field image is downscaled to ``--window`` pixels for display.
- The first tick triggers JIT compilation (a few seconds); subsequent ticks are fast.
- CPU JAX handles this as a PoC; the SPEC's GPU targets are where this path really shines.

Run it (requires the ``demos`` extra)::

    python -m evosim.examples.foragers_large
    python -m evosim.examples.foragers_large --agents 250000 --grid 512 --window 700
"""

from __future__ import annotations

import jax

from ..viz import AgentRenderer, GridRenderer, compose
from . import foragers

__all__ = ["large_config", "build", "run_view_large", "main"]


def large_config(n_agents: int = 1_000_000, grid: int = 1024) -> foragers.ForagerConfig:
    """A foragers config sized for ~``n_agents`` on a ``grid x grid`` toric world."""
    return foragers.ForagerConfig(
        height=grid,
        width=grid,
        capacity=int(n_agents * 1.2),  # headroom for births
        n_initial=n_agents,
    )


def build(n_agents: int = 1_000_000, grid: int = 1024, seed: int = 0):
    cfg = large_config(n_agents, grid)
    return foragers.build(cfg, seed=seed), cfg


def run_view_large(n_agents: int = 1_000_000, grid: int = 1024, window: int = 800,
                   seed: int = 0, steps: int | None = None):
    """Open the rasterized viewer for a ~``n_agents`` foragers world."""
    from .pygame_viewer import run_live  # requires the `demos` extra

    sim, cfg = build(n_agents, grid, seed)
    state = foragers.initial_state(sim, cfg, jax.random.key(seed))
    print(f"foragers_large: {n_agents:,} agents on {grid}x{grid} "
          f"({grid * grid:,} cells), capacity {cfg.capacity:,}")
    print("compiling the first tick (a few seconds)...")

    # Rasterized layers: food field background + agents scattered into cells (O(cells)).
    layers = [
        GridRenderer("food", cmap="gray", vmin=0.0, vmax=cfg.food_max),
        AgentRenderer("position", color_by="energy", cmap="viridis", vmin=0.0, vmax=2.0),
    ]
    return run_live(
        sim, state, n_steps=steps, layers=layers, window_size=(window, window), fps=30,
        title="evosim · foragers (1M, rasterized)",
        caption_fn=lambda s: f"evosim · foragers  agents={int(s.n_alive):,} tick={int(s.tick)}",
    )


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Million-agent foragers (rasterized viewer PoC)")
    p.add_argument("--agents", type=int, default=1_000_000)
    p.add_argument("--grid", type=int, default=1024)
    p.add_argument("--window", type=int, default=800)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless-bench", type=int, default=0,
                   help="run N ticks headless and print timing instead of opening a window")
    args = p.parse_args(argv)

    if args.headless_bench > 0:
        _bench(args.agents, args.grid, args.headless_bench, args.seed)
        return

    run_view_large(args.agents, args.grid, args.window, args.seed, args.steps)


def _bench(n_agents: int, grid: int, n_steps: int, seed: int) -> None:
    """Headless throughput check (no window) for the large config."""
    import time

    sim, cfg = build(n_agents, grid, seed)
    state = foragers.initial_state(sim, cfg, jax.random.key(seed))
    tick = sim.backend.jit(sim.scheduler.make_tick(sim.root_key, sim.world, sim.backend,
                                                   sim.params))
    state = tick(state)             # warmup / compile
    jax.block_until_ready(state.components["energy"])
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = tick(state)
    jax.block_until_ready(state.components["energy"])
    dt = time.perf_counter() - t0
    backend = jax.default_backend().upper()
    print(f"foragers_large bench: {n_agents:,} agents on {grid}x{grid}, {n_steps} ticks "
          f"in {dt:.2f}s -> {n_steps / dt:.1f} ticks/s, "
          f"{n_agents * n_steps / dt / 1e6:.1f}M agent-ticks/s ({backend})")


if __name__ == "__main__":
    main()
