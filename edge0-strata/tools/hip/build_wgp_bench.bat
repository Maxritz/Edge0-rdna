@echo off
REM Builds tools/hip/wgp_gather_bench.hip.cpp twice for gfx1031: the toolchain default (which this ROCm emits as
REM workgroup_processor_mode 1 = WGP) and -mcumode (workgroup_processor_mode 0 = CU).  Same source, same flags
REM otherwise, so the difference between the two runs is the workgroup placement, nothing else.  The descriptors are
REM printed after the build so "the flag did nothing" cannot pass for a result.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Rocm10\bin
set ROOT=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
set CL="C:\Rocm10\lib\llvm\bin\clang++.exe"
set FLAGS=-x hip --offload-arch=gfx1031 -O3 -I C:/Rocm10/include --rocm-path=C:/Rocm10 --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode -L C:/Rocm10/lib -lamdhip64
set SRC=%ROOT%\tools\hip\wgp_gather_bench.hip.cpp

%CL% %FLAGS% "-DSTRATA_MODE_LABEL=\"default: WGP per the toolchain\"" -o %ROOT%\tools\hip\wgp_gather_default.exe %SRC%
if errorlevel 1 exit /b 1
%CL% %FLAGS% -mcumode "-DSTRATA_MODE_LABEL=\"-mcumode: CU mode\"" -o %ROOT%\tools\hip\wgp_gather_cumode.exe %SRC%
if errorlevel 1 exit /b 2
echo built wgp_gather_default.exe and wgp_gather_cumode.exe
