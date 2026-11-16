@echo off
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build-hip-rdna2
"C:\Program Files\AMD\ROCm\7.1\bin\amdclang++.exe" -x hip --offload-arch=gfx1031 -O2 -I "C:\Program Files\AMD\ROCm\7.1\include" --rocm-path="C:\Program Files\AMD\ROCm\7.1" -L"C:\Program Files\AMD\ROCm\7.1\lib" -lamdhip64 ..\tools\hip\rebar_probe.cpp -o rebar_probe.exe 2>&1