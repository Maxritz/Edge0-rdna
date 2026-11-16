@echo off
cd /d C:\Users\rr\Desktop\Edge0-rdna
"C:\Rocm10\bin\hipcc.exe" -O2 -x hip --offload-arch=gfx1031 rebar_test.cpp -o rebar_test.exe -I "C:\Rocm10\include" --rocm-path="C:\Rocm10" --rocm-device-lib-path="C:\Rocm10\lib\llvm\amdgcn\bitcode" -L"C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build-hip-rdna2" -lamdhip64 2>&1