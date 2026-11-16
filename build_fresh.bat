@echo off
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
if exist build rmdir /s /q build
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1031 -DCMAKE_HIP_COMPILER=C:/Rocm10/lib/llvm/bin/amdclang++.exe -DCMAKE_CXX_COMPILER=C:/Rocm10/lib/llvm/bin/amdclang++.exe -DCMAKE_RC_COMPILER=C:/WinKits/rc.exe -DROCM_PATH=C:/Rocm -DCMAKE_HIP_COMPILER_FORCED=ON -DCMAKE_CXX_COMPILER_FORCED=ON -DCMAKE_RC_COMPILER_FORCED=ON .. 2>&1