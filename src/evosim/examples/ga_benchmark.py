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
           "best_objective", "main"]


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


def main() -> None:
    cfg = GAConfig(dim=10, pop_size=256, objective="sphere")
    sim = build(cfg, seed=0)
    state = initial_state(sim, cfg, jax.random.key(0))
    gens = 120
    objective_fn = FITNESS_FUNCTIONS[cfg.objective]
    final, recs = sim.run(state, gens, record=lambda s: jnp.min(objective_fn(s["genome"])))
    best = np.asarray(recs)
    print(f"GA benchmark: {cfg.objective}, dim={cfg.dim}, pop={cfg.pop_size}, {gens} gens")
    print(f"best objective: start={best[0]:.4f}  end={best[-1]:.6f}")
    print(f"best series (every 20): {[round(float(x), 4) for x in best[::20]]}")


if __name__ == "__main__":
    main()
