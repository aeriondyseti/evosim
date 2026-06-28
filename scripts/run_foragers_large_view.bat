@echo off
REM Launch the million-agent foragers viewer (rasterized AgentRenderer path).
REM Defaults: 1,000,000 agents on a 1024x1024 grid, shown in an 800px window.
REM Args pass through, e.g.:  scripts\run_foragers_large_view.bat --agents 250000 --grid 512
REM Controls: SPACE pause | S/. step (paused) | Up/Right faster | Down/Left slower | H help | Esc/Q quit.
REM Requires the demos extra:  uv sync --extra demos
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.foragers_large %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
