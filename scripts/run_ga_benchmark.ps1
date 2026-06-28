#!/usr/bin/env pwsh
# Run the classic GA benchmark demo (explicit-fitness function optimization).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    uv run python -m evosim.examples.ga_benchmark @args
}
finally {
    Pop-Location
}
