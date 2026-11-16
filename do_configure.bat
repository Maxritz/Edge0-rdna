@echo off
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
if exist build rmdir /s /q build
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1031 -DCMAKE_TOOLCHAIN_FILE=../cmake/strata-toolchain.cmake
