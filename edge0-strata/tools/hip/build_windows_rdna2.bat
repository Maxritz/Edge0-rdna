@echo off
rem Build Strata's HIP (AMD) engine for RDNA2 (gfx1030 / gfx1031 / gfx1032) on Windows.
rem
rem   tools\hip\build_windows_rdna2.bat            configure + build build-hip-rdna2\strata.exe and strata-device.exe
rem   tools\hip\build_windows_rdna2.bat tests      also build the HIP unit tests
rem
rem Unlike build_windows.bat, which installs ROCm from AMD's TheRock wheels into .rocm-win, this script uses a ROCm
rem that is ALREADY installed on the PC (default C:\Rocm10) -- the case of a developer machine with a system ROCm.
rem
rem RDNA2 notes (docs\AMD_HIP.md#rdna2-gfx1030):
rem   * gfx1030/gfx1031/gfx1032 are listed by cmake\hip_backend.cmake as unvalidated; the build prints that warning.
rem   * RDNA2 is wave32 with 64 KiB of LDS per workgroup, so the wave32 HIP kernels apply.
rem   * RDNA2 has no v_dot4_i32_iu8: __dp4a takes the signed v_dot4_i32_i8 path (include\strata\hip_compat\intrinsics.hpp).
rem   * RDNA2 has no matrix cores and hipBLASLt ships no gfx103x kernels, so there is no hipBLASLt tuning table;
rem     rocBLAS does carry gfx1031 kernels (shipped as .kpack), and its tuned FP16 GEMMs are reachable through
rem     STRATA_HIP_PROMPT_F16=1 / STRATA_ROCBLAS_TUNING.
rem
rem Settings (environment variables, all optional):
rem   STRATA_HIP_ARCHS     default gfx1031; the RX 6800 / 6900 series is gfx1030, the RX 6600 series gfx1032
rem   ROCM_ROOT            default C:\Rocm10 -- must hold lib\llvm\bin\clang++.exe and lib\llvm\amdgcn\bitcode\ocml.bc
rem   VCVARS               path to vcvars64.bat; auto-detected from the Visual Studio installs when unset
rem   BUILD_DIR            default <repo>\build-hip-rdna2
rem   STRATA_GGML_DIR      a llama.cpp checkout at the pinned commit (default: CMake fetches it)
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1031"
if not defined ROCM_ROOT set "ROCM_ROOT=C:\Rocm10"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-hip-rdna2"
set "TESTS=OFF"
if /i "%~1"=="tests" set "TESTS=ON"

rem ---- 1. ROCm (the installed system toolkit) ----
if not exist "%ROCM_ROOT%\lib\llvm\bin\clang++.exe" (echo No ROCm compiler at "%ROCM_ROOT%\lib\llvm\bin\clang++.exe" - set ROCM_ROOT & exit /b 1)
if not exist "%ROCM_ROOT%\lib\llvm\amdgcn\bitcode\ocml.bc" (echo No ROCm device library in "%ROCM_ROOT%\lib\llvm\amdgcn\bitcode" - set ROCM_ROOT & exit /b 1)
set "ROCM_F=%ROCM_ROOT:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM_ROOT%"
set "ROCM_PATH=%ROCM_ROOT%"
set "PATH=%ROCM_ROOT%\bin;%ROCM_ROOT%\lib\llvm\bin;%PATH%"

rem ---- 2. MSVC: the linker, the C runtime and the Windows SDK (ROCm's clang uses them) ----
if not defined VCVARS (
  for /f "delims=" %%V in ('dir /b /o-n "%ProgramFiles%\Microsoft Visual Studio\*" 2^>nul') do (
    for /f "delims=" %%W in ('dir /b /o-n "%ProgramFiles%\Microsoft Visual Studio\%%V\*" 2^>nul') do (
      if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\%%V\%%W\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\%%V\%%W\VC\Auxiliary\Build\vcvars64.bat"
      )
    )
  )
)
if not defined VCVARS (echo Visual Studio Build Tools with the C++ workload are needed ^(set VCVARS^) & exit /b 1)
call "%VCVARS%" >nul || exit /b 1

rem ---- 3. configure + build ----
set "GGML="
if defined STRATA_GGML_DIR set "GGML=-DSTRATA_GGML_DIR=%STRATA_GGML_DIR:\=/%"
if not exist "%BUILD_DIR%\build.ninja" (
  echo Configuring %BUILD_DIR% for %STRATA_HIP_ARCHS% ...
  cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=%TESTS% ^
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON ^
    "-DCMAKE_HIP_ARCHITECTURES=%STRATA_HIP_ARCHS%" ^
    "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" ^
    "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
    "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" "-DCMAKE_PREFIX_PATH=%ROCM_F%" ^
    "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" %GGML% || exit /b 1
)
if "%TESTS%"=="ON" (
  cmake --build "%BUILD_DIR%" -- -k 0 || exit /b 1
) else (
  cmake --build "%BUILD_DIR%" --target strata strata-device || exit /b 1
)

rem ---- 4. a run folder: the exes beside the ROCm DLLs they load ----
rem Windows resolves amdhip64_7.dll from System32 before it looks at PATH, and some AMD drivers install one there
rem that does not match the toolkit (docs\AMD_HIP.md: "the engine folder holds its own amdhip64_7.dll ... because
rem Windows looks in the program's folder before System32").  Programs in this folder therefore load the toolkit's
rem own DLLs; .kpack is linked in beside them because rocBLAS reads its kernels from <folder>\.kpack.
rem The folder is <repo>\build because that is the path edge0-infer's launch contract runs (edge0/models/*.json).
set "RUN=%SRC%\build"
if exist "%RUN%\strata.exe" del /q "%RUN%\strata.exe"
if not exist "%RUN%" mkdir "%RUN%"
copy /y "%BUILD_DIR%\strata.exe" "%RUN%\" >nul || exit /b 1
if exist "%BUILD_DIR%\strata-device.exe" copy /y "%BUILD_DIR%\strata-device.exe" "%RUN%\" >nul
for %%D in ("%ROCM_ROOT%\bin\*.dll") do copy /y "%%~fD" "%RUN%\" >nul
rem rocBLAS reads its Tensile kernels from "<folder of rocblas.dll>\rocblas\library" (the path it names when it cannot
rem find them), and the ROCm install keeps them in bin\rocblas\library beside bin\rocblas.dll.  Without the copy every
rem prompt/serving GEMM fails at start with "Cannot read ...\rocblas\library\TensileLibrary.dat ... for GPU arch".
if exist "%ROCM_ROOT%\bin\rocblas" if not exist "%RUN%\rocblas" (
  echo Copying rocBLAS kernels for %STRATA_HIP_ARCHS% ...
  xcopy /E /I /Y /Q "%ROCM_ROOT%\bin\rocblas" "%RUN%\rocblas" >nul || exit /b 1
)
rem rocBLAS also looks for a .kpack one level above the DLL folder (docs: "../.kpack/blas_lib_<arch>.kpack").
rem It is only a fallback for an install that ships kernels packed, so link it when there is one.
if exist "%ROCM_ROOT%\bin\..\.kpack" if not exist "%RUN%\..\.kpack" mklink /J "%RUN%\..\.kpack" "%ROCM_ROOT%\.kpack" >nul

rem ---- 5. the PCIe window: Resizable BAR (SAM) ----
rem An RDNA2 engine streams most of a 120B / 145B pack's experts over PCIe (here 26.7 GiB), so the size of the card's
rem BAR matters.  The whole VRAM appears as one BAR only when the firmware enables Above 4G Decoding and Re-Size BAR /
rem Smart Access Memory (with CSM off); no driver setting can turn it on.  hipInfo reports isLargeBar 1 for the wide
rem window and 0 for the 256 MiB one.  Nothing in this build branches on it, so this only records the state a run was
rem made in, to compare before and after the firmware change.
echo.
where hipInfo >nul 2>nul && (echo Resizable BAR ^(isLargeBar 1 = whole VRAM mapped, 0 = 256 MiB window^): & hipInfo | findstr /c:"isLargeBar")
echo.
echo Done. Engine:  %RUN%\strata.exe
echo Device check:  "%RUN%\strata-device.exe" --list-devices
endlocal
