@echo off
rem The same recipe as build_hip_official.bat but with the unit tests on, so the KV parity gates can run.
rem   build_tests.bat            build kv_turbo_parity + kv_q4_parity
setlocal EnableDelayedExpansion

set "SRC=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata"
set "BUILD_DIR=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build-tests"
set "ROCM=C:\Rocm10"
set "ROCM_F=%ROCM:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"

call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"

if exist "%BUILD_DIR%\CMakeCache.txt" goto :build
cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON ^
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DCMAKE_HIP_ARCHITECTURES=gfx1031 ^
  "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
  "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
  "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" || exit /b 1

:build
cmake --build "%BUILD_DIR%" --target kv_turbo_parity kv_q4_parity || exit /b 1
echo TESTS_BUILD_OK
endlocal
