@echo off
REM Launch the live PyGame visualizer for Conway's Game of Life.
REM Controls: SPACE pause | S/. step (paused) | Up/Right faster | Down/Left slower | H help | Esc/Q quit.
REM Requires the viz extra:  uv sync --extra viz
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.conway --view %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
