@echo off
REM Run the evolving-foragers demo (full agent-based ALife; emergent natural selection).
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.foragers %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
