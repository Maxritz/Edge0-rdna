@echo off
REM Builds tools/hip/q3k_nt_bench.hip.cpp for gfx1031 the way the project's HIP build does:
REM the ROCm 7.x clang, from C:\Rocm10\bin so the runtime DLLs beside it are the ones loaded.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Rocm10\bin
set ROOT=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
"C:\Rocm10\lib\llvm\bin\clang++.exe" -x hip --offload-arch=gfx1031 -O3 ^
  -I C:/Rocm10/include --rocm-path=C:/Rocm10 --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode ^
  -L C:/Rocm10/lib -lamdhip64 ^
  -o %ROOT%\tools\hip\q3k_nt_bench.exe %ROOT%\tools\hip\q3k_nt_bench.hip.cpp
if errorlevel 1 exit /b 1
echo built %ROOT%\tools\hip\q3k_nt_bench.exe
