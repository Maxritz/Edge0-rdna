@echo off
REM Reconfigures build-hip-rdna2 with one extra HIP compile flag and rebuilds the engine.  The project's own HIP
REM flags are kept verbatim; the flag is appended, so tools/hip/build_mode.bat -mcumode is CU mode for every kernel
REM and tools/hip/build_mode.bat (no argument) is the toolchain default again (WGP, workgroup_processor_mode 1).
REM
REM   usage: build_mode.bat [-mcumode | -mno-cumode | <anything>]
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
set BASE=--rocm-path=C:/Rocm10 --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode
set EXTRA=%1
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-hip-rdna2 -G Ninja "-DCMAKE_HIP_FLAGS=%BASE% %EXTRA%"
if errorlevel 1 exit /b 1
"C:\Program Files\CMake\bin\cmake.exe" --build build-hip-rdna2 --target strata -- -j 8
if errorlevel 1 exit /b 2
echo built build-hip-rdna2\strata.exe with HIP flags: %BASE% %EXTRA%
