@echo off
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
rmdir /s /q build
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1031 -DCMAKE_HIP_COMPILER=C:/Rocm10/lib/llvm/bin/amdclang++.exe -DCMAKE_CXX_COMPILER=C:/Rocm10/lib/llvm/bin/amdclang++.exe -DCMAKE_RC_COMPILER="C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\rc.exe" -DROCM_PATH=C:/Rocm .. 2>&1