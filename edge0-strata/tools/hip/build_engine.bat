@echo off
REM Rebuilds the HIP engine into the existing build-hip-rdna2 tree (ninja + the ROCm clang CMake configured).
REM Same tree and flags as the binary being served, so an A/B is a source change and nothing else.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
"C:\Program Files\CMake\bin\cmake.exe" --build build-hip-rdna2 --target strata -- -j 8
if errorlevel 1 exit /b 1
echo rebuilt build-hip-rdna2\strata.exe
