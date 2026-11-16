// io_split_bench.cpp - the two halves of a prompt's expert staging, measured apart at the pack's own geometry.
//
// The engine moves ~1.16 MB (layer, expert) blobs into VRAM.  For each blob there are two possible costs:
//
//   A. the drive read: the blob is not in RAM, so it comes off E: (what the pinned arena's page faults do)
//   B. the PCIe transfer: the blob is in RAM, so it is a host-to-device copy
//
// This measures both on the same buffer size and blob count, with the same 4-slot ring the engine uses, so the two
// numbers are comparable with the prompt's wait-copy time: bytes / GB/s says which one that time matches.
//
//   usage: io_split_bench.exe <pack experts.bin> [blobs]
//
// A reads unbuffered (FILE_FLAG_NO_BUFFERING) at random 4096-aligned offsets, so the OS cache cannot answer them; B
// copies pinned host memory to VRAM with hipMemcpyAsync and 4 slots in flight.  Nothing here is a model of the
// engine's scheduling, only of its two resources.
#define NOMINMAX
#include <windows.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); return 1; } } while (0)

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <pack experts.bin> [blobs]\n", argv[0]); return 2; }
    const int blobs = argc > 2 ? std::atoi(argv[2]) : 256;
    const int slots = argc > 3 ? std::max(1, std::atoi(argv[3])) : 4;
    // default 1,164,288 B: 4096-aligned, within 100 B of the pack's 1,164,700 B average blob
    // unbuffered reads need a 4096-aligned size, so the blob is rounded down to one
    const size_t blob = ((argc > 4 ? (size_t) (std::atof(argv[4]) * 1048576.0) : (size_t) 1164288) / 4096) * 4096;

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s  pcie %d.%d x%d  blobs %d x %.2f MB = %.2f GB, %d slots\n", prop.name, prop.pciDomainID >= 0 ? 4 : 3,
                0, prop.pciBusID >= 0 ? 16 : 0, blobs, blob / 1048576.0, (double) blobs * blob / 1e9, slots);

    // ---- A: the drive, unbuffered, random offsets inside the pack
    HANDLE f = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (f == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "cannot open %s (%lu)\n", argv[1], GetLastError()); return 1; }
    LARGE_INTEGER sz{};
    GetFileSizeEx(f, &sz);
    const uint64_t nblob = (uint64_t) sz.QuadPart / blob;
    std::vector<uint8_t*> bufs(slots);
    for (auto& b : bufs) {
        b = (uint8_t*) VirtualAlloc(nullptr, blob, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!b) { std::fprintf(stderr, "VirtualAlloc failed\n"); return 1; }
    }
    uint64_t seed = 12345;
    auto next = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return seed >> 33; };
    std::vector<OVERLAPPED> ov(slots);
    std::vector<HANDLE> ev(slots);
    for (int i = 0; i < slots; ++i) { std::memset(&ov[i], 0, sizeof ov[i]); ev[i] = CreateEventA(nullptr, TRUE, FALSE, nullptr); ov[i].hEvent = ev[i]; }
    auto t0 = std::chrono::steady_clock::now();
    int issued = 0, done = 0;
    while (done < blobs) {
        while (issued < blobs && issued - done < slots) {
            const int sl = issued % slots;
            const uint64_t off = (next() % nblob) * blob;
            ov[sl].Offset = (DWORD) (off & 0xffffffffu);
            ov[sl].OffsetHigh = (DWORD) (off >> 32);
            DWORD got = 0;
            if (!ReadFile(f, bufs[sl], (DWORD) blob, &got, &ov[sl]) && GetLastError() != ERROR_IO_PENDING) {
                std::fprintf(stderr, "ReadFile failed (%lu)\n", GetLastError()); return 1; }
            ++issued;
        }
        const int sl = done % slots;
        WaitForSingleObject(ev[sl], INFINITE);
        ResetEvent(ev[sl]);
        ++done;
    }
    const double drive_ms = ms_since(t0);
    CloseHandle(f);
    std::printf("A  drive (unbuffered, %d x %.2f MB): %8.2f ms  %6.2f GB/s\n", blobs, blob / 1048576.0, drive_ms,
                (double) blobs * blob / 1e9 / (drive_ms / 1000.0));
    for (auto& b : bufs) VirtualFree(b, 0, MEM_RELEASE);

    // ---- B: PCIe, pinned host memory -> VRAM, the same ring depth
    std::vector<uint8_t*> host(slots);
    std::vector<void*> dev(slots);
    for (int i = 0; i < slots; ++i) {
        CK(hipHostMalloc((void**) &host[i], blob, hipHostMallocDefault));
        CK(hipMalloc(&dev[i], blob));
        std::memset(host[i], 0x5a, blob);
    }
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::vector<hipEvent_t> evs(slots), evd(slots);
    for (int i = 0; i < slots; ++i) { CK(hipEventCreate(&evs[i])); CK(hipEventCreate(&evd[i])); }
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < blobs; ++i) {
        const int sl = i % slots;
        if (i >= slots) CK(hipEventSynchronize(evd[sl]));   // this slot's previous copy has landed
        CK(hipMemcpyAsync(dev[sl], host[sl], blob, hipMemcpyHostToDevice, s));
        CK(hipEventRecord(evd[sl], s));
    }
    CK(hipStreamSynchronize(s));
    const double pcie_ms = ms_since(t0);
    std::printf("B  PCIe (pinned -> VRAM, %d x %.2f MB): %8.2f ms  %6.2f GB/s\n", blobs, blob / 1048576.0, pcie_ms,
                (double) blobs * blob / 1e9 / (pcie_ms / 1000.0));
    std::printf("=> for the prompt's 16.23 GB (13,941 blobs): drive would take %.0f ms, PCIe %.0f ms\n",
                16230.0 / ((double) blobs * blob / 1e9 / (drive_ms / 1000.0)) * 1000.0,
                16230.0 / ((double) blobs * blob / 1e9 / (pcie_ms / 1000.0)) * 1000.0);
    for (auto& b : host) hipHostFree(b);
    for (auto& d : dev) hipFree(d);
    hipStreamDestroy(s);
    return 0;
}
