@echo off
REM Run the classic GA benchmark demo (explicit-fitness function optimization).
setlocal
pushd "%~dp0.."
uv run python -m evosim.examples.ga_benchmark %*
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
