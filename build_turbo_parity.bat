@echo off
rem Rebuild only kv_turbo_parity in the existing official build dir (ROCm 10.2, gfx1031).
setlocal EnableDelayedExpansion
set "BUILD_DIR=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build"
set "ROCM=C:\Rocm10"
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
cmake --build "%BUILD_DIR%" --target kv_turbo_parity %1 || exit /b 1
echo BUILD_OK
endlocal
