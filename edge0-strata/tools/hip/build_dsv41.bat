@echo off
REM Builds the third engine path, strata-dsv41 (DeepSeek-V4.1-Flash, src/dsv41): the option is OFF by default
REM because it changes no existing target, so without this script there is no way to build it from this tree.
REM Same HIP flags and build directory as tools/hip/build_mode.bat; only the target differs.
REM
REM   usage: build_dsv41.bat            build strata-dsv41 (leaves STRATA_BUILD_DSV41=ON in the cache)
REM          build_dsv41.bat <target>   build any other target of this build directory (strata-gguf, a parity test)
REM          build_dsv41.bat configure  configure only
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
set BASE=--rocm-path=C:/Rocm10 --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-hip-rdna2 -G Ninja "-DCMAKE_HIP_FLAGS=%BASE%" -DSTRATA_BUILD_DSV41=ON
if errorlevel 1 exit /b 1
if /I "%1"=="configure" exit /b 0
set TARGET=strata-dsv41
if not "%1"=="" set TARGET=%1
"C:\Program Files\CMake\bin\cmake.exe" --build build-hip-rdna2 --target %TARGET% -- -j 8
if errorlevel 1 exit /b 2
echo built build-hip-rdna2\%TARGET%.exe
