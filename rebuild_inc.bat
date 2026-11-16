@echo off
rem Incremental rebuild of the existing build\ (same env as build_hip_official.bat, no wipe).
setlocal EnableDelayedExpansion
set "ROCM=C:\Rocm10"
set "ROCM_F=%ROCM:\=/%"
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
cmake --build "C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build" --target strata || exit /b 1
echo BUILD_OK
endlocal
