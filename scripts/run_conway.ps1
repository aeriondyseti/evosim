#!/usr/bin/env pwsh
# Run the Conway's Game of Life demo (field-only, minimal usage).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uv run python -m evosim.examples.conway @args
}
finally {
    Pop-Location
}
