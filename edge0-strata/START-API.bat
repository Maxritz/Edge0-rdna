@echo off
rem Launch Edge0-120B as a remote OpenAI/Anthropic-compatible API endpoint on Windows.
rem
rem   Double-click to start. The console shows the ready banner (with the LAN URL and
rem   the Windows Firewall hint); press Ctrl+C or close the window to stop the server.
rem
rem Overrides (set them in this shell before running, or edit the defaults below):
rem   STRATA_API_HOST   listen address          (default 0.0.0.0 = all IPv4 interfaces)
rem   STRATA_API_PORT   listen port             (default 8080)
rem   STRATA_API_KEY    Bearer key for /v1/*     (default edge0-rdna2-2026)
rem   STRATA_CONFIG     engine config JSON      (default serve\config-rdna2.json)
rem
rem To keep a file log as well, run this file from a cmd prompt with:
rem   START-API.bat >> api-server.log 2>&1
setlocal
set "PYTHONUTF8=1"
title Edge0 Strata API
cd /d "%~dp0"

rem ROCm runtime and HIP clang live here on this gfx1031 machine. The engine's DLLs
rem are also staged beside build\strata.exe, but keep the runtime on PATH for tools.
set "PATH=C:\Rocm10\bin;C:\Rocm10\lib\llvm\bin;%PATH%"

if not defined STRATA_API_HOST set "STRATA_API_HOST=0.0.0.0"
if not defined STRATA_API_PORT set "STRATA_API_PORT=8080"
if not defined STRATA_API_KEY  set "STRATA_API_KEY=edge0-rdna2-2026"
if not defined STRATA_CONFIG   set "STRATA_CONFIG=%~dp0serve\config-rdna2.json"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo Starting Edge0 Strata API ...
echo   config : %STRATA_CONFIG%
echo   listen : %STRATA_API_HOST%:%STRATA_API_PORT%
echo   key    : set (read from STRATA_API_KEY; not put on the command line)
echo.

rem serve.server defaults --api-key from STRATA_API_KEY, so passing it here would only
rem expose the secret in the process command line (visible to other local users).
"%PY%" -m serve.server --engine strata --config "%STRATA_CONFIG%" --host "%STRATA_API_HOST%" --port "%STRATA_API_PORT%"

echo.
echo Server stopped.
endlocal
pause
