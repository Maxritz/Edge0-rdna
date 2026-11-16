@echo off
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build-hip-rdna2
"C:\Program Files\AMD\ROCm\7.1\bin\amdclang++.exe" -O2 ..\..\hip_rebar_check.cpp -o hip_rebar_check.exe -I "C:\Program Files\AMD\ROCm\7.1\include" -L"C:\Program Files\AMD\ROCm\7.1\lib" -lamdhip64 2>&1