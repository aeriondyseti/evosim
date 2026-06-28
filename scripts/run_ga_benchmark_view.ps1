#!/usr/bin/env pwsh
# Launch the live PyGame visualizer for the GA benchmark
# (population in genome space, colored by fitness, with convergence curve).
# Controls: SPACE pause | S/. step (paused) | Up/Right faster | Down/Left slower | H help | Esc/Q quit.
# Requires the demos extra:  uv sync --extra demos
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uv run python -m evosim.examples.ga_benchmark --view @args
}
finally {
    Pop-Location
}
