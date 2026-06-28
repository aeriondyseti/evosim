@echo off
REM Launch the live PyGame visualizer for the GA benchmark
REM (population in genome space, colored by fitness, with convergence curve).
REM Controls: SPACE pause | S/. step (paused) | Up/Right faster | Down/Left slower | H help | Esc/Q quit.
REM Requires the demos extra:  uv sync --extra demos
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.ga_benchmark --view %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
