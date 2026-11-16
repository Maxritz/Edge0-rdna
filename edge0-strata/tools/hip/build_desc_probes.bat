@echo off
REM Two code objects from one source, CU mode and WGP mode, to calibrate the descriptor's WGP bit (kdesc.py).
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Rocm10\bin
set ROOT=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
set CL="C:\Rocm10\lib\llvm\bin\clang++.exe"
set FLAGS=-x hip --offload-arch=gfx1031 -O3 -I C:/Rocm10/include --rocm-path=C:/Rocm10 --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode -c
%CL% %FLAGS% -o %ROOT%\build-hip-rdna2\probe_cu.obj %ROOT%\tools\hip\wgp_gather_bench.hip.cpp
if errorlevel 1 exit /b 1
%CL% %FLAGS% -mno-cumode -o %ROOT%\build-hip-rdna2\probe_wgp.obj %ROOT%\tools\hip\wgp_gather_bench.hip.cpp
if errorlevel 1 exit /b 2
echo built probe_cu.obj and probe_wgp.obj
