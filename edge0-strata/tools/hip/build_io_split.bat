@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >/dev/null
set ROOT=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
"C:\Rocm10\lib\llvm\bin\clang++.exe" -x hip --offload-arch=gfx1031 -O2 -I C:/Rocm10/include --rocm-path=C:/Rocm10 --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode -L C:/Rocm10/lib -lamdhip64 -o %ROOT%\tools\hip\io_split_bench.exe %ROOT%\tools\hip\io_split_bench.cpp
