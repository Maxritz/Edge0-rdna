@echo off
REM Builds tools/hip/cache_policy_bench.hip.cpp for gfx1031 and also emits its gfx1031 assembly, so the loads'
REM cache modifiers can be read out of the code rather than assumed.  One binary, three kernels (plain, __ldg,
REM glc+slc); the run prints which policy each line is and checks that all three sums agree.
REM
REM   run: copy the .exe beside ROCm's amdhip64_7.dll (edge0-strata\build) and run it there, otherwise Windows
REM        resolves C:\Windows\System32\amdhip64.dll, which reports no ROCm-capable device.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Rocm10\bin
set ROOT=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
set CL="C:\Rocm10\lib\llvm\bin\clang++.exe"
set FLAGS=-x hip --offload-arch=gfx1031 -O3 -I C:/Rocm10/include --rocm-path=C:/Rocm10 --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode -L C:/Rocm10/lib -lamdhip64
set SRC=%ROOT%\tools\hip\cache_policy_bench.hip.cpp

%CL% %FLAGS% -o %ROOT%\tools\hip\cache_policy_bench.exe "%SRC%"
if errorlevel 1 exit /b 1
%CL% %FLAGS% -S -o %ROOT%\tools\hip\cache_policy_bench.s "%SRC%"
if errorlevel 2 exit /b 2
echo built tools\hip\cache_policy_bench.exe and cache_policy_bench.s
