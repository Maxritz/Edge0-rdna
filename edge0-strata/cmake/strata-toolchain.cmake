# Strata HIP Toolchain for Windows
#
# Working combination (verified against the prior build-hip-rdna2 that produced a
# running strata.exe):
#   host CXX = MinGW GCC  -> link line uses GNU ld flags it understands
#   HIP      = the ROCm Clang (amdclang++), NOT the hipcc wrapper (CMake rejects hipscc)
#
# The Windows registry stores ROCm as a backslash path (c:\Rocm10) that CMake's
# HIP detection cannot parse; C:/Rocm is a junction to the same install with
# forward slashes.

# Host compiler: MinGW GCC (the linker the successful build used)
set(CMAKE_CXX_COMPILER   "C:/Strawberry/c/bin/c++.exe"     CACHE FILEPATH "CXX compiler")
set(CMAKE_C_COMPILER     "C:/Strawberry/c/bin/gcc.exe"     CACHE FILEPATH "C compiler")
set(CMAKE_RC_COMPILER    "C:/Strawberry/c/bin/windres.exe" CACHE FILEPATH "Windows resource compiler")

# HIP compiler: real ROCm Clang (CMake refuses the hipcc wrapper)
set(CMAKE_HIP_COMPILER   "C:/Rocm10/lib/llvm/bin/amdclang++.exe" CACHE FILEPATH "HIP compiler")

# ROCm locations, forward-slash only
set(CMAKE_HIP_ROCM_PATH  "C:/Rocm" CACHE PATH "ROCm path")
set(CMAKE_HIP_PATH       "C:/Rocm" CACHE PATH "HIP path")
set(CMAKE_HIP_ARCHITECTURES "gfx1031" CACHE STRING "HIP target architecture")

# Device library path (where amdgcn bitcode lives) + host include/lib
set(CMAKE_HIP_FLAGS
    "-DROCM_PATH=C:/Rocm --rocm-path=C:/Rocm --rocm-device-lib-path=C:/Rocm10/lib/llvm/amdgcn/bitcode"
    CACHE STRING "HIP compile flags")
set(CMAKE_HIP_FLAGS_RELEASE "-O3" CACHE STRING "HIP release flags")
set(CMAKE_HIP_FLAGS_DEBUG   "-g"  CACHE STRING "HIP debug flags")

# Environment hints so hipconfig/includes resolve to forward-slash paths
set(ENV{ROCM_PATH}   "C:/Rocm")
set(ENV{HIP_PATH}    "C:/Rocm")
set(ENV{HIP_DEVICE_LIB_PATH} "C:/Rocm10/lib/llvm/amdgcn/bitcode")
set(ENV{HIPCXX}      "C:/Rocm10/lib/llvm/bin/amdclang++.exe")

# Standards
set(CMAKE_CXX_STANDARD 20 CACHE STRING "CXX standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "CXX standard required")
set(CMAKE_C_STANDARD 11 CACHE STRING "C standard")
