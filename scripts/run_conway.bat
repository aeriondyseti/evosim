@echo off
REM Run the Conway's Game of Life demo (field-only, minimal usage).
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.conway %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
