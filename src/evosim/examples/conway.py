"""Conway's Game of Life — the minimal evosim demo (field-only, no agents).

Demonstrates the smallest viable usage: a single environment field (the cell grid) on a toric
2D world, advanced by the library's :func:`evosim.world.fields.life_like` cellular-automaton
system. No genome, no population — just the scheduler + world + a field layer.

Run it::

    python -m evosim.examples.conway
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..scheduler import Scheduler
from ..schema import Schema
from ..sim import Simulation
from ..state import State
from ..world import ToricGrid2D, life_like

__all__ = [
    "GLIDER", "BLINKER", "BLOCK",
    "empty_grid", "stamp", "random_grid",
    "build", "initial_state", "run_history", "population_series", "render", "main",
]

# Pattern coordinates as (row, col) offsets.
GLIDER = ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2))
BLINKER = ((0, 0), (0, 1), (0, 2))
BLOCK = ((0, 0), (0, 1), (1, 0), (1, 1))


def empty_grid(height: int, width: int) -> jax.Array:
    return jnp.zeros((height, width), dtype=jnp.int32)


def stamp(grid: jax.Array, coords, top: int = 0, left: int = 0) -> jax.Array:
    """Set cells given by ``coords`` (row,col offsets) live, with toric wrap."""
    h, w = grid.shape
    for (r, c) in coords:
        grid = grid.at[(top + r) % h, (left + c) % w].set(1)
    return grid


def random_grid(key: jax.Array, height: int, width: int, density: float = 0.3) -> jax.Array:
    return (jax.random.uniform(key, (height, width)) < density).astype(jnp.int32)


def build(height: int, width: int, seed: int = 0,
          born=(3,), survive=(2, 3)) -> Simulation:
    """Build a Conway simulation on a ``height x width`` toric grid."""
    schema = Schema()  # no agents — reserved alive/id only, capacity 0
    sched = Scheduler(stages=("environment",))
    sched.add(life_like("cells", born=born, survive=survive))
    world = ToricGrid2D(height, width)
    return Simulation(sched, world=world, seed=seed, schema=schema)


def initial_state(sim: Simulation, grid: jax.Array) -> State:
    """Build the initial state from a cell grid."""
    return State.create(sim.schema, 0, fields={"cells": grid})


def run_history(sim: Simulation, state: State, n_steps: int):
    """Run and return ``(final_state, history)`` where history is ``(n_steps, H, W)``."""
    final, hist = sim.run(state, n_steps, record=lambda s: s.get_field("cells"))
    return final, hist


def population_series(history: jax.Array) -> np.ndarray:
    """Number of live cells per recorded step."""
    h = np.asarray(history)
    return h.reshape(h.shape[0], -1).sum(axis=1)


def render(grid: jax.Array, alive: str = "#", dead: str = ".") -> str:
    g = np.asarray(grid)
    return "\n".join("".join(alive if c else dead for c in row) for row in g)


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Conway's Game of Life (evosim demo)")
    p.add_argument("--view", action="store_true", help="live PyGame visualization")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--density", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.view:
        h = args.height or 80
        w = args.width or 120
        sim = build(h, w, seed=args.seed)
        state = initial_state(sim, random_grid(jax.random.key(args.seed), h, w, args.density))
        from ..viz import GridRenderer, run_live
        run_live(sim, state, n_steps=args.steps,
                 layers=[GridRenderer("cells", cmap="green", vmin=0, vmax=1)],
                 px_per_cell=8, fps=15, title="evosim · Conway's Life")
        return

    h = args.height or 20
    w = args.width or 40
    steps = args.steps or 60
    sim = build(h, w, seed=args.seed)
    grid0 = random_grid(jax.random.key(args.seed), h, w, density=args.density)
    state = initial_state(sim, grid0)
    final, hist = run_history(sim, state, steps)
    pops = population_series(hist)
    print(f"Conway's Life {h}x{w}, {steps} steps")
    print(f"live cells: start={int(grid0.sum())}  end={int(final.get_field('cells').sum())}")
    print(f"population series (every 10): {pops[::10].tolist()}")
    print("final grid:")
    print(render(final.get_field("cells")))


if __name__ == "__main__":
    main()
