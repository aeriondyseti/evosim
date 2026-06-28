#!/usr/bin/env pwsh
# Launch the live PyGame visualizer for Conway's Game of Life.
# Controls: SPACE pause | Up/Right faster | Down/Left slower | Esc/Q quit.
# Requires the viz extra:  uv sync --extra viz
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uv run python -m evosim.examples.conway --view @args
}
finally {
    Pop-Location
}
