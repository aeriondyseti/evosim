"""PyGame live viewer (optional; requires ``evosim[viz]``).

A decoupled, read-only consumer of the engine (per SPEC): it steps the simulation on the host
loop and draws each frame, so determinism and the headless fast path are untouched.

- :func:`run_live` — open a window and run the sim, compositing renderer layers each frame.
  An on-screen legend (toggle ``H``) documents the controls: SPACE pause, S/. step one tick
  while paused, UP/RIGHT faster, DOWN/LEFT slower, ESC/Q quit.
- :class:`PygameViewer` — a :class:`~evosim.recorders.Recorder` so a window can be driven by
  ``recorders.run_recorded`` alongside other recorders.

Both work headless (e.g. CI) when ``SDL_VIDEODRIVER=dummy`` is set.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .render import AgentRenderer, GridRenderer, compose

try:  # optional dependency
    import pygame
except ImportError:  # pragma: no cover - exercised only when pygame is absent
    pygame = None

__all__ = ["run_live", "PygameViewer"]


def _require_pygame() -> None:
    if pygame is None:
        raise ImportError(
            "PyGame is required for the viewer. Install it with `uv sync --extra viz` "
            "(or `pip install evosim[viz]`)."
        )


def _default_layers(state, world) -> list:
    """Pick sensible default layers: every field as a heatmap, then agents if present."""
    layers: list = []
    for name in state.fields:
        layers.append(GridRenderer(name, cmap="viridis"))
    if "position" in state.components:
        layers.append(AgentRenderer("position", color_by=None))
    if not layers:
        raise ValueError("no fields or 'position' component to render; pass layers=...")
    return layers


def _make_surface(img):
    # img is (H, W, 3); pygame surfarray expects (W, H, 3) (x, y).
    return pygame.surfarray.make_surface(img.swapaxes(0, 1))


# Control legend shown in the viewer (also documents the keybindings).
_LEGEND_LINES = (
    "SPACE   pause / resume",
    "S / .   step (when paused)",
    "Arrows  speed  +/-",
    "H       toggle this help",
    "ESC/Q   quit",
)


def _draw_panel(screen, font, lines, pos=(8, 8), pad=6,
                fg=(240, 240, 240), bg=(0, 0, 0, 160)):
    """Draw a translucent text panel at ``pos`` (top-left)."""
    surfs = [font.render(t, True, fg) for t in lines]
    width = max((s.get_width() for s in surfs), default=0) + pad * 2
    height = sum(s.get_height() for s in surfs) + pad * 2
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill(bg)
    y = pad
    for s in surfs:
        panel.blit(s, (pad, y))
        y += s.get_height()
    screen.blit(panel, pos)


def run_live(sim, state, n_steps: int | None = None, layers: Sequence | None = None,
             px_per_cell: int = 8, fps: int = 30, steps_per_frame: int = 1,
             title: str = "evosim", caption_fn: Callable | None = None,
             show_legend: bool = True):
    """Open a window and run the simulation live, returning the final state.

    ``n_steps=None`` runs until the window is closed. ``layers`` defaults to a heatmap per field
    plus an agent layer. ``caption_fn(state) -> str`` optionally updates the window title.

    Controls (also shown in the on-screen legend, toggle with ``H``):
    ``SPACE`` pause/resume · ``S``/``.`` step one tick while paused · ``Up``/``Right`` faster ·
    ``Down``/``Left`` slower · ``ESC``/``Q`` quit.
    """
    _require_pygame()
    world = sim.world
    if world is None:
        raise ValueError("run_live requires sim.world to define the grid shape")
    h, w = world.shape
    layers = list(layers) if layers is not None else _default_layers(state, world)

    pygame.init()
    pygame.font.init()
    try:
        size = (w * px_per_cell, h * px_per_cell)
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 18)

        tick = sim.backend.jit(sim.scheduler.make_tick(sim.root_key, world, sim.backend,
                                                       sim.params))
        s = state
        steps_done = 0
        spf = max(1, int(steps_per_frame))
        paused = False
        legend = show_legend
        running = True

        def _do_step():
            nonlocal s, steps_done
            s = tick(s)
            steps_done += 1

        while running:
            step_once = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in (pygame.K_s, pygame.K_PERIOD):
                        step_once = True       # single-step (takes effect while paused)
                    elif event.key in (pygame.K_UP, pygame.K_RIGHT):
                        spf = min(spf * 2, 4096)
                    elif event.key in (pygame.K_DOWN, pygame.K_LEFT):
                        spf = max(spf // 2, 1)
                    elif event.key == pygame.K_h:
                        legend = not legend

            # draw current state
            img = compose(layers, s, world)
            surf = pygame.transform.scale(_make_surface(img), size)
            screen.blit(surf, (0, 0))
            if legend:
                status = f"{'PAUSED' if paused else 'PLAYING'}   x{spf}   tick {int(s.tick)}"
                _draw_panel(screen, font, [*_LEGEND_LINES, "", status])
            else:
                _draw_panel(screen, font, ["H  help"])
            if caption_fn is not None:
                pygame.display.set_caption(caption_fn(s))
            pygame.display.flip()
            clock.tick(fps)

            if n_steps is not None and steps_done >= n_steps:
                break  # final state already drawn
            if paused:
                if step_once:
                    _do_step()
                continue
            for _ in range(spf):
                _do_step()
                if n_steps is not None and steps_done >= n_steps:
                    break
        return s
    finally:
        pygame.quit()


class PygameViewer:
    """A read-only :class:`~evosim.recorders.Recorder` that draws to a window each tick.

    Use with ``recorders.run_recorded``. Sets :attr:`closed` if the window is closed.
    """

    def __init__(self, world, layers: Sequence | None = None, px_per_cell: int = 8,
                 fps: int = 30, title: str = "evosim", caption_fn: Callable | None = None):
        _require_pygame()
        self.world = world
        self.layers = list(layers) if layers is not None else None
        self.px = px_per_cell
        self.fps = fps
        self.caption_fn = caption_fn
        self.closed = False
        h, w = world.shape
        self._size = (w * px_per_cell, h * px_per_cell)
        pygame.init()
        self._screen = pygame.display.set_mode(self._size)
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()

    def record(self, state) -> None:
        if self.closed:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        layers = self.layers if self.layers is not None else _default_layers(state, self.world)
        img = compose(layers, state, self.world)
        surf = pygame.transform.scale(_make_surface(img), self._size)
        self._screen.blit(surf, (0, 0))
        if self.caption_fn is not None:
            pygame.display.set_caption(self.caption_fn(state))
        pygame.display.flip()
        self._clock.tick(self.fps)

    def result(self):
        return None

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            pygame.quit()
