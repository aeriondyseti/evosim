#!/usr/bin/env pwsh
# Launch the live PyGame visualizer for the evolving-foragers demo
# (food heatmap + agents colored by energy).
# Controls: SPACE pause | S/. step (paused) | Up/Right faster | Down/Left slower | H help | Esc/Q quit.
# Requires the demos extra:  uv sync --extra demos
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uv run python -m evosim.examples.foragers --view @args
}
finally {
    Pop-Location
}
