"""Evolving foragers — the full agent-based ALife slice (emergent natural selection).

Agents live on a toric grid with a regrowing ``food`` field. Each tick they:

1. **move** — biased random walk: step toward the richest von-Neumann neighbour (taxis),
   with some exploration / random steps when no food is sensed;
2. **eat** — agents contend for the food in their cell; a deterministic lottery
   (:func:`evosim.interaction.resolve_cell_claims`) grants the cell's food to one winner;
3. **reproduce** — agents above an energy threshold split: a child takes half the energy and a
   mutated copy of the genome (free slots claimed via :func:`evosim.population.spawn`);
4. **metabolism / death** — every agent pays an energy cost that *decreases* with its heritable
   ``efficiency`` gene; agents at zero energy die.

There is no explicit fitness function — selection is emergent. Because efficient agents pay
less to live, the mean ``efficiency`` gene drifts upward over generations: the evolution signal.

Run it::

    python -m evosim.examples.foragers
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .. import interaction, population
from ..operators import mutation
from ..scheduler import Scheduler
from ..schema import Field, Schema
from ..sim import Simulation
from ..state import State
from ..system import System
from ..world import ToricGrid2D

__all__ = ["ForagerConfig", "build", "initial_state", "schema_for", "main"]

# von Neumann move offsets (row, col): N, S, W, E.
_OFFSETS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


@dataclass(frozen=True)
class ForagerConfig:
    height: int = 32
    width: int = 32
    capacity: int = 4096
    n_initial: int = 400
    food_init: float = 0.3       # initial food per cell
    food_max: float = 1.0        # cap per cell
    regrow: float = 0.03         # food added per cell per tick
    init_energy: float = 1.0
    base_cost: float = 0.05      # metabolic cost at efficiency gene = 0
    eff_scale: float = 0.5       # how strongly the efficiency gene reduces cost
    gene_clip: float = 4.0       # clip efficiency gene to [-clip, clip]
    mut_sigma: float = 0.05      # mutation std on reproduction
    repro_threshold: float = 1.5  # energy needed to reproduce
    explore_rate: float = 0.25   # prob of a random (non-taxis) move


def schema_for(cfg: ForagerConfig) -> Schema:
    return Schema(
        position=Field(dtype="int16", shape=(2,)),
        energy=Field(dtype="float32", default=0.0),
        genome=Field(dtype="float32", shape=(1,), default=0.0),  # genome[0] = efficiency
    )


def _move_system(cfg: ForagerConfig) -> System:
    def fn(state: State, ctx) -> State:
        w = ctx.world
        pos = state["position"]
        alive = state.alive
        food = state.get_field("food")
        n = pos.shape[0]
        # food in each of the 4 neighbours
        neigh = w.wrap(pos[:, None, :] + _OFFSETS[None, :, :])     # (N, 4, 2)
        fvals = w.gather_field(neigh.reshape(-1, 2), food).reshape(n, 4)
        best = jnp.argmax(fvals, axis=1)
        greedy = _OFFSETS[best]
        k1, k2 = random.split(ctx.key)
        rand_off = _OFFSETS[random.randint(k1, (n,), 0, 4)]
        explore = random.uniform(k2, (n,)) < cfg.explore_rate
        no_food = jnp.max(fvals, axis=1) <= 0.0
        use_rand = jnp.logical_or(explore, no_food)
        chosen = jnp.where(use_rand[:, None], rand_off, greedy)
        new_pos = w.move(pos, chosen)
        new_pos = jnp.where(alive[:, None], new_pos, pos)
        return state.set("position", new_pos.astype(state["position"].dtype))

    return System("move", "act", fn)


def _eat_system(cfg: ForagerConfig) -> System:
    def fn(state: State, ctx) -> State:
        w = ctx.world
        pos = state["position"]
        alive = state.alive
        food = state.get_field("food")
        pri = interaction.lottery_priorities(ctx.key, pos.shape[0])
        res = interaction.resolve_cell_claims(w, pos, pri, valid=alive)
        food_at = w.gather_field(pos, food)
        gain = jnp.where(res.won, food_at, 0.0)
        energy = state["energy"] + gain
        consumed = w.scatter_field(pos, res.won.astype(jnp.float32), alive) > 0.0
        new_food = jnp.where(consumed, 0.0, food)
        return state.set("energy", energy).set_field("food", new_food)

    return System("eat", "interact", fn)


def _reproduce_system(cfg: ForagerConfig) -> System:
    def fn(state: State, ctx) -> State:
        alive = state.alive
        energy = state["energy"]
        parents = jnp.logical_and(alive, energy >= cfg.repro_threshold)
        child_energy = energy * 0.5
        child_genome = mutation.gaussian(ctx.key, state["genome"], sigma=cfg.mut_sigma,
                                         clip=(-cfg.gene_clip, cfg.gene_clip))
        child_pos = state["position"]
        # parents keep half their energy
        state = state.set("energy", jnp.where(parents, energy * 0.5, energy))
        res = population.spawn(
            state,
            {"position": child_pos, "energy": child_energy, "genome": child_genome},
            birth_mask=parents,
        )
        return res.state

    return System("reproduce", "spawn", fn)


def _metabolism_system(cfg: ForagerConfig) -> System:
    def fn(state: State, ctx) -> State:
        g0 = state["genome"][:, 0]
        cost = cfg.base_cost * jnp.exp(-cfg.eff_scale * g0)
        energy = state["energy"] - jnp.where(state.alive, cost, 0.0)
        state = state.set("energy", energy)
        dead = jnp.logical_and(state.alive, energy <= 0.0)
        return population.kill(state, dead)

    return System("metabolism", "death", fn)


def _regrow_system(cfg: ForagerConfig) -> System:
    def fn(state: State, ctx) -> State:
        food = jnp.minimum(state.get_field("food") + cfg.regrow, cfg.food_max)
        return state.set_field("food", food)

    return System("regrow", "environment", fn)


def build(cfg: ForagerConfig = ForagerConfig(), seed: int = 0) -> Simulation:
    """Build the forager simulation."""
    schema = schema_for(cfg)
    world = ToricGrid2D(cfg.height, cfg.width)
    sched = Scheduler()
    sched.add(_move_system(cfg))
    sched.add(_eat_system(cfg))
    sched.add(_reproduce_system(cfg))
    sched.add(_metabolism_system(cfg))
    sched.add(_regrow_system(cfg))
    return Simulation(sched, world=world, seed=seed, schema=schema, params={"cfg": cfg})


def initial_state(sim: Simulation, cfg: ForagerConfig, key: jax.Array) -> State:
    """Seed ``n_initial`` agents at random positions with small genome noise."""
    k1, k2 = random.split(key)
    s = State.create(sim.schema, cfg.capacity,
                     fields={"food": jnp.full((cfg.height, cfg.width), cfg.food_init,
                                              dtype=jnp.float32)})
    pos = sim.world.random_positions(k1, cfg.capacity).astype(jnp.int16)
    genome = (0.01 * random.normal(k2, (cfg.capacity, 1))).astype(jnp.float32)
    idx = jnp.arange(cfg.capacity)
    alive = idx < cfg.n_initial
    energy = jnp.where(alive, cfg.init_energy, 0.0).astype(jnp.float32)
    ids = jnp.where(alive, idx, -1).astype(jnp.int32)
    s = s.set_many({"position": pos, "genome": genome, "energy": energy,
                    "alive": alive, "id": ids})
    return s.replace(next_id=jnp.asarray(cfg.n_initial, dtype=jnp.int32))


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evolving foragers (evosim demo)")
    p.add_argument("--view", action="store_true", help="live PyGame visualization")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = ForagerConfig()
    sim = build(cfg, seed=args.seed)
    state = initial_state(sim, cfg, jax.random.key(args.seed))

    if args.view:
        from ..viz import GridRenderer, compose
        from .pygame_viewer import agent_overlay, run_live
        # Food field as the background image; agents drawn as distinct dots on top.
        field = GridRenderer("food", cmap="fire", vmin=0.0, vmax=cfg.food_max)
        run_live(
            sim, state, n_steps=args.steps,
            frame_fn=lambda s: compose([field], s, sim.world),
            overlay_fn=agent_overlay(sim.world, color_by="energy", cmap="viridis",
                                     vmin=0.0, vmax=2.0, radius_frac=0.42),
            px_per_cell=16, fps=30, title="evosim · foragers",
            caption_fn=lambda s: f"evosim · foragers  pop={int(s.n_alive)} tick={int(s.tick)}")
        return

    def record(s):
        from .. import metrics
        return {"pop": s.n_alive, "eff": metrics.masked_mean(s["genome"][:, 0], s.alive)}

    steps = args.steps or 400
    final, recs = sim.run(state, steps, record=record)
    pop = np.asarray(recs["pop"])
    eff = np.asarray(recs["eff"])
    print(f"Foragers {cfg.height}x{cfg.width}, cap={cfg.capacity}, {steps} steps")
    print(f"population: start={cfg.n_initial} end={int(pop[-1])} (min={pop.min()}, max={pop.max()})")
    print(f"mean efficiency gene: start={eff[0]:.4f} end={eff[-1]:.4f}")
    print(f"pop series (every 50): {pop[::50].tolist()}")
    print(f"eff series (every 50): {[round(float(x), 3) for x in eff[::50]]}")


if __name__ == "__main__":
    main()
