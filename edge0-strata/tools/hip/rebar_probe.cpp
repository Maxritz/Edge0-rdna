// tools/hip/rebar_probe.cpp - is Resizable BAR (a large PCIe aperture) actually active on this card?
//
// HIP computes `isLargeBar` from what the firmware exposed, so this one call answers the question that
// decides whether the host can map all of VRAM or only a 256 MB window.  It is a property, not a switch:
// nothing an application sets turns ReBAR on, and the engine has no code path for it either way (its
// `--pcie-frac` tunes the share of expert moves sent over PCIe, not the aperture size).
//
//   build: clang++ -x hip --offload-arch=gfx1031 -O2 -I <rocm>/include --rocm-path=<rocm> \
//            --rocm-device-lib-path=<rocm>/lib/llvm/amdgcn/bitcode -L<rocm>/lib -lamdhip64 -o rebar_probe rebar_probe.cpp
//   run:   rebar_probe        (run it from <rocm>/bin, next to the runtime DLLs, or on PATH)
//
// On the RX 6700 XT this repository is measured on, `isLargeBar` reads 0 while the KMD registry value
// `KMD_RebarControlMode` is 1: the driver is willing, the firmware is not exposing a large aperture.
#include <hip/hip_runtime.h>

#include <cstdio>

int main() {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
        std::printf("no HIP device\n");
        return 1;
    }
    int attr = -1;
    hipDeviceGetAttribute(&attr, hipDeviceAttributeIsLargeBar, 0);
    size_t fr = 0, tot = 0;
    hipMemGetInfo(&fr, &tot);
    char bus[32] = {0};
    hipDeviceGetPCIBusId(bus, sizeof(bus), 0);
    std::printf("device            %s\n", p.name);
    std::printf("gcnArch           %s\n", p.gcnArchName);
    std::printf("isLargeBar (prop) %d\n", p.isLargeBar);
    std::printf("isLargeBar (attr) %d\n", attr);
    std::printf("VRAM              %.2f GiB total, %.2f GiB free\n", tot / 1073741824.0, fr / 1073741824.0);
    std::printf("pci bus           %s (busID %d)\n", bus, p.pciBusID);
    std::printf("canMapHostMemory %d, unifiedAddressing %d\n", p.canMapHostMemory, p.unifiedAddressing);
    return p.isLargeBar ? 0 : 2;   // 2: a 256 MB aperture, not a large one
}
