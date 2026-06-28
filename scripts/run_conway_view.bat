@echo off
REM Launch the live PyGame visualizer for Conway's Game of Life.
REM Controls: SPACE pause | Up/Right faster | Down/Left slower | Esc/Q quit.
REM Requires the viz extra:  uv sync --extra viz
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.conway --view %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
