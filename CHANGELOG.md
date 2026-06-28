# Changelog

All notable changes to **evosim** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 the public API may change between minor versions.

## [Unreleased]

## [0.1.0] — 2026-06-28

First tagged release. The library and its three reference demos are complete, with
tests and determinism golden-masters throughout.

### Added
- **Core library** (`import evosim`): SoA component `Schema`/`Field` and immutable
  `State`; counter-based deterministic RNG (`rng`); pluggable compute `Backend`
  (JAX/XLA); phased `Scheduler` + `system` decorator; `population` management with
  capacity growth; `world/` (toric grid + environment fields, incl. `life_like` CA);
  `operators/` (mutation, crossover, selection); `interaction`, `metrics`,
  `recorders`, and `checkpoint`.
- **`Simulation`** orchestrator with `run` (scan), `run_recorded` (host loop),
  `run_with_growth`, and `run_ensemble` (`vmap` over worlds).
- **Reference demos** under `evosim.examples`: `conway`, `foragers`, `ga_benchmark`
  (`python -m evosim.examples.<name>`).
- **Visualization** (`evosim.viz`): GUI-agnostic, pure-numpy `GridRenderer` /
  `AgentRenderer` / `ScatterRenderer`; optional interactive PyGame viewer
  (`demos` extra).
- **Million-agent scale PoC** (`foragers_large`): rasterized rendering, O(cells) per
  frame, with a `--headless-bench` throughput mode.
- **GPU support** on Linux/WSL2 via the `gpu` extra (`jax[cuda12]`), validated on
  NVIDIA RTX 5060 Ti / 4060 Ti — ~560M agent-ticks/s at 1M agents (~70× the CPU
  baseline); full test suite green on GPU.
- **CI**: GitHub Actions running `pytest` on Python 3.11–3.13.

### Notes
- Determinism is same-device bit-exact (counter-based RNG); CPU and GPU agree
  statistically but not bit-for-bit (different reduction order).

[Unreleased]: https://github.com/aeriondyseti/evosim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aeriondyseti/evosim/releases/tag/v0.1.0
