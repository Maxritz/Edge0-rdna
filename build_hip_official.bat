@echo off
rem Exact replica of tools\hip\build_windows.bat's configure+build steps,
rem using the local ROCm 10.2 install at C:\Rocm10 and gfx1031.
setlocal EnableDelayedExpansion

set "SRC=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata"
set "BUILD_DIR=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build"
set "STRATA_HIP_ARCHS=gfx1031"

rem ---- 1. ROCm 10.2 at C:\Rocm10 (forward-slash form for CMake)
set "ROCM=C:\Rocm10"
if not exist "%ROCM%\lib\llvm\bin\clang++.exe" (echo ROCm has no compiler in "%ROCM%" & exit /b 1)
set "ROCM_F=%ROCM:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"

rem ---- 2. MSVC environment (the linker, CRT, Windows SDK that ROCm clang uses)
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"

rem ---- 3. configure (fresh)
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PREFILL_MMQ=ON ^
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%STRATA_HIP_ARCHS%" ^
  "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
  "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
  "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" || exit /b 1

rem ---- 4. build
cmake --build "%BUILD_DIR%" --target strata strata-device || exit /b 1

echo BUILD_OK
endlocal
