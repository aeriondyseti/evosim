"""Classic genetic algorithm — the explicit-fitness demo (function optimization).

A generational GA that minimizes a benchmark function (sphere / Rastrigin) over a real-valued
genome. Each tick is one generation:

1. **evaluate** — compute each individual's fitness (``-objective``, since selection maximizes);
2. **evolve** — elitism keeps the best few; the rest are bred via tournament selection +
   uniform crossover + Gaussian mutation (all from :mod:`evosim.operators`).

This validates the explicit-fitness path and the generational update model. With elitism the
best objective value is monotonically non-increasing across generations.

Run it::

    python -m evosim.examples.ga_benchmark
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from ..operators import crossover, mutation, selection
from ..scheduler import Scheduler
from ..schema import Field, Schema
from ..sim import Simulation
from ..state import State
from ..system import System

__all__ = ["GAConfig", "FITNESS_FUNCTIONS", "sphere", "rastrigin", "build", "initial_state",
           "best_objective", "run_view", "main"]


def sphere(x: jax.Array) -> jax.Array:
    """Sphere function (min at 0): sum of squares. ``x`` is ``(N, D)`` -> ``(N,)``."""
    return jnp.sum(x ** 2, axis=-1)


def rastrigin(x: jax.Array) -> jax.Array:
    """Rastrigin function (min at 0): highly multimodal. ``x`` is ``(N, D)`` -> ``(N,)``."""
    d = x.shape[-1]
    return 10.0 * d + jnp.sum(x ** 2 - 10.0 * jnp.cos(2.0 * jnp.pi * x), axis=-1)


FITNESS_FUNCTIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
}


@dataclass(frozen=True)
class GAConfig:
    dim: int = 10
    pop_size: int = 256
    elite: int = 2
    tournament_size: int = 3
    mut_sigma: float = 0.1
    crossover_rate: float = 0.5
    init_range: float = 5.0
    objective: str = "sphere"


def _schema(cfg: GAConfig) -> Schema:
    return Schema(
        genome=Field(dtype="float32", shape=(cfg.dim,), default=0.0),
        fitness=Field(dtype="float32", default=0.0),
    )


def _evaluate_system(objective_fn: Callable) -> System:
    def fn(state: State, ctx) -> State:
        return state.set("fitness", -objective_fn(state["genome"]))  # maximize -objective

    return System("evaluate", "decide", fn)


def _evolve_system(cfg: GAConfig) -> System:
    def fn(state: State, ctx) -> State:
        g = state["genome"]
        fit = state["fitness"]
        n = state.capacity
        n_children = n - cfg.elite
        k_t1, k_t2, k_x, k_m = random.split(ctx.key, 4)

        elite = g[selection.elitism(fit, cfg.elite)]
        p1 = g[selection.tournament(k_t1, fit, n_children, cfg.tournament_size)]
        p2 = g[selection.tournament(k_t2, fit, n_children, cfg.tournament_size)]
        children = crossover.uniform(k_x, p1, p2, rate=cfg.crossover_rate)
        children = mutation.gaussian(k_m, children, sigma=cfg.mut_sigma)

        new_g = jnp.concatenate([elite, children], axis=0)
        return state.set("genome", new_g)

    return System("evolve", "act", fn)


def build(cfg: GAConfig = GAConfig(), seed: int = 0) -> Simulation:
    """Build the GA simulation (one tick == one generation)."""
    objective_fn = FITNESS_FUNCTIONS[cfg.objective]
    sched = Scheduler()
    sched.add(_evaluate_system(objective_fn))
    sched.add(_evolve_system(cfg))
    return Simulation(sched, seed=seed, schema=_schema(cfg), params={"cfg": cfg})


def initial_state(sim: Simulation, cfg: GAConfig, key: jax.Array) -> State:
    """Random initial population uniform in ``[-init_range, init_range]^D``."""
    g = random.uniform(key, (cfg.pop_size, cfg.dim),
                       minval=-cfg.init_range, maxval=cfg.init_range).astype(jnp.float32)
    idx = jnp.arange(cfg.pop_size)
    s = State.create(sim.schema, cfg.pop_size)
    s = s.set_many({"genome": g, "alive": jnp.ones((cfg.pop_size,), dtype=bool),
                    "id": idx.astype(jnp.int32)})
    return s.replace(next_id=jnp.asarray(cfg.pop_size, dtype=jnp.int32))


def best_objective(state: State, cfg: GAConfig) -> jax.Array:
    """Best (lowest) objective value in the current population."""
    return jnp.min(FITNESS_FUNCTIONS[cfg.objective](state["genome"]))


def _draw_curve(screen, vals, size) -> None:
    """Draw a best-objective convergence curve in a translucent bottom-left panel."""
    import pygame
    if len(vals) < 2:
        return
    w, h = size
    pw, ph = min(240, w - 16), 90
    x0, y0 = 8, h - ph - 8
    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    screen.blit(panel, (x0, y0))
    v = np.asarray(vals, dtype=float)
    vmin, vmax = float(v.min()), float(v.max())
    denom = (vmax - vmin) or 1.0
    n = len(v)
    pts = []
    for i, val in enumerate(v):
        px = x0 + int(i / (n - 1) * (pw - 1))
        # best (min) at the bottom, start (max) at the top -> curve descends as it converges
        py = y0 + (ph - 1) - int((val - vmin) / denom * (ph - 1))
        pts.append((px, py))
    pygame.draw.lines(screen, (90, 220, 255), False, pts, 2)


def run_view(cfg: GAConfig = GAConfig(), seed: int = 0, steps: int | None = None):
    """Live visualization of the GA: population in genome space (dims 0,1) colored by fitness,
    with the optimum marked and a best-objective convergence curve. One frame = one generation.

    Demonstrates visualizing a *non-spatial* sim: it uses the library's world-free
    :class:`~evosim.viz.ScatterRenderer` via ``run_live(frame_fn=...)``.
    """
    if cfg.dim < 2:
        raise ValueError("run_view needs dim >= 2 to scatter genome dims 0 and 1")
    from ..viz import ScatterRenderer
    from .pygame_viewer import run_live

    sim = build(cfg, seed=seed)
    state = initial_state(sim, cfg, jax.random.key(seed))
    objective = FITNESS_FUNCTIONS[cfg.objective]
    b = cfg.init_range
    scatter = ScatterRenderer(x=("genome", 0), y=("genome", 1), color_by="fitness",
                              bounds=((-b, b), (-b, b)), resolution=(320, 320), cmap="viridis")
    hist = {"last_tick": -1, "best": []}

    def frame_fn(s):
        return scatter.render_image(s)

    def overlay_fn(screen, s, size):
        import pygame
        w, h = size
        ox = int((0 - (-b)) / (2 * b) * (w - 1))
        oy = int((1 - (0 - (-b)) / (2 * b)) * (h - 1))  # y flipped
        pygame.draw.line(screen, (0, 255, 0), (ox - 7, oy), (ox + 7, oy), 2)
        pygame.draw.line(screen, (0, 255, 0), (ox, oy - 7), (ox, oy + 7), 2)
        t = int(s.tick)
        if t != hist["last_tick"]:
            hist["last_tick"] = t
            hist["best"].append(float(jnp.min(objective(s["genome"]))))
        _draw_curve(screen, hist["best"], size)

    return run_live(sim, state, n_steps=steps, frame_fn=frame_fn, overlay_fn=overlay_fn,
                    px_per_cell=2, fps=30,
                    title=f"evosim · GA ({cfg.objective}) genome space",
                    caption_fn=lambda s: f"evosim · GA  gen={int(s.tick)} "
                                         f"best={float(jnp.min(objective(s['genome']))):.4f}")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Classic GA benchmark (evosim demo)")
    p.add_argument("--view", action="store_true",
                   help="live PyGame visualization in genome space")
    p.add_argument("--objective", default="sphere", choices=sorted(FITNESS_FUNCTIONS))
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--pop", type=int, default=256)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = GAConfig(dim=args.dim, pop_size=args.pop, objective=args.objective)

    if args.view:
        run_view(cfg, seed=args.seed, steps=args.steps)
        return

    sim = build(cfg, seed=args.seed)
    state = initial_state(sim, cfg, jax.random.key(args.seed))
    gens = args.steps or 120
    objective_fn = FITNESS_FUNCTIONS[cfg.objective]
    final, recs = sim.run(state, gens, record=lambda s: jnp.min(objective_fn(s["genome"])))
    best = np.asarray(recs)
    print(f"GA benchmark: {cfg.objective}, dim={cfg.dim}, pop={cfg.pop_size}, {gens} gens")
    print(f"best objective: start={best[0]:.4f}  end={best[-1]:.6f}")
    print(f"best series (every 20): {[round(float(x), 4) for x in best[::20]]}")


if __name__ == "__main__":
    main()
