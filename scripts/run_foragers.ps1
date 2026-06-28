#!/usr/bin/env pwsh
# Run the evolving-foragers demo (full agent-based ALife; emergent natural selection).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uv run python -m evosim.examples.foragers @args
}
finally {
    Pop-Location
}
