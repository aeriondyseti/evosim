#!/usr/bin/env pwsh
# Launch the million-agent foragers viewer (rasterized AgentRenderer path).
# Defaults: 1,000,000 agents on a 1024x1024 grid, shown in an 800px window.
# Args pass through, e.g.:  scripts\run_foragers_large_view.ps1 --agents 250000 --grid 512
# Controls: SPACE pause | S/. step (paused) | Up/Right faster | Down/Left slower | H help | Esc/Q quit.
# Requires the demos extra:  uv sync --extra demos
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uv run python -m evosim.examples.foragers_large @args
}
finally {
    Pop-Location
}
