@echo off
REM Launch the live PyGame visualizer for the evolving-foragers demo
REM (food heatmap + agents colored by energy).
REM Controls: SPACE pause | Up/Right faster | Down/Left slower | Esc/Q quit.
REM Requires the viz extra:  uv sync --extra viz
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.foragers --view %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
