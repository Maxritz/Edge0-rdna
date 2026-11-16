// tools/hip/wgp_gather_bench.hip.cpp - does WGP mode (gfx10/11 workgroup spread over a WGP's two CUs, clang
// -mno-cumode) help the kernels that gather streamed expert weights on gfx1031?
//
// WHY.  The prompt path's gather is a pure global->global uint4 copy of the expert bytes (mmq::copy16_kernel and
// copy16_group_kernel in src/prefill/moe_mmq.cu), plus the fused q3k path's copy into dynamic LDS
// (moe_fused_q3k.cu's cp16).  Neither declares any static group segment on this build (llvm-readelf on the
// moe_mmq object: group_segment_fixed_size 0 for all 12 kernels, wavefront_size 32).  With no LDS and 5-6 VGPRs
// these are the kernels whose occupancy is decided by the CU-vs-WGP placement, so the mode is worth measuring
// rather than arguing about.
//
// WHAT IT DOES.  Two kernels at the pack's real geometry - the expert blob is 1,164,800 B (gate/up 704,000 B +
// down 460,800 B), 64 experts = 74.5 MB, far beyond L2:
//   (1) gather_copy  : uint4 copy, block-per-4 KiB, the copy16_kernel shape (no LDS).
//   (2) gather_smem  : uint4 gather into 32 KiB of dynamic LDS, then a reduction from LDS, the cp16 shape.
// The file is built twice - once plain (CU mode, the default) and once with -mno-cumode (WGP mode) - and each
// binary prints which one it is.  Compare the two runs; the outputs must be identical, which the bench checks
// against a host reference so a "faster" result cannot come from doing less work.
//
//   build: see tools/hip/build_wgp_bench.bat (builds wgp_gather_cu.exe and wgp_gather_wgp.exe)
//   run:   wgp_gather_cu.exe [repeats]   /   wgp_gather_wgp.exe [repeats]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                                      \
    do {                                                                                  \
        hipError_t e_ = (x);                                                              \
        if (e_ != hipSuccess) {                                                           \
            std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x,             \
                         hipGetErrorString(e_));                                          \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

#ifndef STRATA_WGP_MODE
#define STRATA_WGP_MODE 0
#endif
// what this build asked for, as the run reports it (the descriptor's workgroup_processor_mode is the truth: it is in
// the metadata note and tools/hip/build_wgp_bench.bat prints it beside each run)
#ifndef STRATA_MODE_LABEL
#define STRATA_MODE_LABEL "unspecified build"
#endif

namespace {

constexpr int kExperts = 64;
constexpr size_t kGuBytes = 704000;        // gate/up of one latent expert (Q3_K, 1280 x 1280)
constexpr size_t kDownBytes = 460800;      // down of the same expert (IQ4_XS, 1280 x 640... rows x cols)
constexpr size_t kBlobBytes = kGuBytes + kDownBytes;   // 1,164,800: the pack's expert stride
constexpr size_t kBlobVec = kBlobBytes / 16;           // 72,800 uint4, exact
constexpr int kThreads = 256;
constexpr int kChunkVec = kThreads;        // 4 KiB a block
constexpr int kLdsBytes = 32 * 1024;       // the fused q3k path's dynamic LDS class

// (1) the copy16 shape: pure global -> global, no LDS, 5-6 VGPRs.
__global__ __launch_bounds__(kThreads) void gather_copy(const uint4* __restrict__ src, uint4* __restrict__ dst) {
    const size_t exp = blockIdx.y;
    const size_t base = exp * kBlobVec;
    const size_t i = (size_t) blockIdx.x * kThreads + threadIdx.x;
    if (i < kBlobVec) dst[base + i] = __ldg(src + base + i);
}

// (2) the cp16 shape: global -> dynamic LDS -> reduce, one block per 32 KiB of the blob.
__global__ __launch_bounds__(kThreads) void gather_smem(const uint4* __restrict__ src, unsigned* __restrict__ out) {
    extern __shared__ uint4 smem[];
    const size_t exp = blockIdx.y;
    const size_t base = exp * kBlobVec;
    unsigned acc = 0;
    for (size_t c = (size_t) blockIdx.x * kLdsBytes / 16; c < kBlobVec; c += (size_t) gridDim.x * kLdsBytes / 16) {
        for (int i = threadIdx.x; i < kLdsBytes / 16; i += kThreads) {
            const size_t g = c + (size_t) i;
            if (g < kBlobVec) smem[i] = __ldg(src + base + g);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kLdsBytes / 16; i += kThreads) {
            const uint4 v = smem[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        __syncthreads();
    }
    __shared__ unsigned red[kThreads];
    red[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned s = 0;
        for (int i = 0; i < kThreads; ++i) s ^= red[i];
        out[exp * gridDim.x + blockIdx.x] = s;
    }
}

template <class F>
float time_kernel(F&& fire, int repeats) {
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) fire();
    HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < repeats; ++i) fire();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return ms / (float) repeats;
}

}  // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 20;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s  built with %s  (clang %s)\n", prop.name, STRATA_MODE_LABEL, __VERSION__);

    const size_t total = (size_t) kExperts * kBlobBytes;
    std::vector<uint8_t> host(total);
    uint32_t s = 0x2545F491u;
    for (auto& v : host) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v = (uint8_t) s;
    }

    uint4* d = nullptr;
    uint4* dback = nullptr;
    unsigned* dout = nullptr;
    const int blocks_copy = (int) ((kBlobVec + kThreads - 1) / kThreads);       // 284 blocks an expert
    const int blocks_smem = (int) std::min<size_t>(kBlobVec / (kLdsBytes / 16), 64);
    HIP_CHECK(hipMalloc(&d, total));
    HIP_CHECK(hipMalloc(&dback, total));
    HIP_CHECK(hipMalloc(&dout, (size_t) kExperts * std::max(blocks_copy, blocks_smem) * sizeof(unsigned)));
    HIP_CHECK(hipMemcpy(d, host.data(), total, hipMemcpyHostToDevice));

    std::printf("blob %.3f MB an expert, %d experts, %.1f MB total (%.0fx l2 %.1f MB)\n\n",
                kBlobBytes / 1048576.0, kExperts, total / 1048576.0,
                total / std::max<double>(1.0, (double) prop.l2CacheSize), prop.l2CacheSize / 1048576.0);

    // (1) copy: the result is checked byte for byte against the source, so the timing is of a real gather.
    const float ms_copy = time_kernel([&] {
        gather_copy<<<dim3(blocks_copy, kExperts), kThreads>>>(d, dback);
    }, repeats);
    std::vector<uint8_t> back(total);
    HIP_CHECK(hipMemcpy(back.data(), dback, total, hipMemcpyDeviceToHost));
    size_t mismatched = 0;
    for (size_t i = 0; i < total; ++i) mismatched += (back[i] != host[i]);

    // (2) smem staging
    HIP_CHECK(hipFuncSetAttribute((const void*) gather_smem, hipFuncAttributeMaxDynamicSharedMemorySize, kLdsBytes));
    const float ms_smem = time_kernel([&] {
        gather_smem<<<dim3((unsigned) blocks_smem, kExperts), kThreads, kLdsBytes>>>(d, dout);
    }, repeats);

    const double gb = total / 1e6;   // bytes per pass
    std::printf("%-24s %8.1f us  %7.1f GB/s  copy check: %s (%zu bytes differ)\n", "gather_copy (no LDS)",
                ms_copy * 1e3, gb / ms_copy, mismatched == 0 ? "exact" : "MISMATCH", mismatched);
    std::printf("%-24s %8.1f us  %7.1f GB/s  lds %d KiB, %d blocks an expert\n", "gather_smem (cp16 shape)",
                ms_smem * 1e3, gb / ms_smem, kLdsBytes / 1024, blocks_smem);

    HIP_CHECK(hipFree(d));
    HIP_CHECK(hipFree(dback));
    HIP_CHECK(hipFree(dout));
    return mismatched == 0 ? 0 : 2;
}
