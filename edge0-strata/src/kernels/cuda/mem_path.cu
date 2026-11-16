// src/kernels/cuda/mem_path.cu - the device's memory path, measured.  Read mem_path.hpp first.
//
// The two numbers that matter, and why they are enough:
//
//   Both kernels are the same code.  Only the buffer size changes.  A 64 MiB buffer is resident in
//   Navi 22's 96 MB Infinity Cache after the first pass; a 512 MiB buffer cannot be, so its pass is
//   served by DRAM.  The ratio is therefore a property of the memory hierarchy, not of the kernel,
//   the grid, the occupancy or the arithmetic - which is exactly what a health check needs.
//
// This file deliberately does not use native_mmvq: a bandwidth check that depends on the quantised
// GEMV would be measuring the thing it is supposed to exonerate.
#include "strata/kernels/mem_path.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr std::size_t kCacheBytes = std::size_t(32) << 20;   // Infinity-Cache resident (96 MB on Navi 22)
constexpr std::size_t kDramBytes = std::size_t(192) << 20;   // past every cache on a consumer card
constexpr int kCachePasses = 8;
constexpr int kDramPasses = 3;
// A health check must never be able to break the run it is checking.  This config can sit at ZERO MiB of
// free VRAM legitimately (512K context + MTP on 12 GiB, ctx_mtp logs), and a 576 MiB probe there made
// "verify: layer 3 never rang" appear where the same build without the probe decoded fine.  So the probe
// needs this much free VRAM to run at all and says so when it does not - and it is 6x smaller than the
// first version (224 MiB total against 576): 32 MiB is still IC-resident and 192 MiB is still past the
// 96 MB Infinity Cache, which is all the ratio test needs (a wedged card reads 0.085 against a healthy
// 0.42 on the 0.15 threshold).
constexpr std::size_t kNeedFreeBytes = std::size_t(1536) << 20;

// 16 bytes a lane, no dependency between iterations: the pattern every card is fastest at, so the
// result is the memory path rather than the access pattern.
__global__ void mem_path_read_kernel(const int4* __restrict__ p, std::size_t n16, int passes, float* sink) {
    const std::size_t tid = (std::size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t) gridDim.x * blockDim.x;
    int acc = 0;
    for (int r = 0; r < passes; ++r)
        for (std::size_t i = tid; i < n16; i += stride) {
            const int4 v = p[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    if (acc == 0x7fffffff) *sink = 1.0f;   // never true in practice; keeps the loads alive
}

// returns GB/s, or 0 on any failure
double timed_read(void* buf, std::size_t bytes, int passes, unsigned grid, int iters) {
    int4* p = static_cast<int4*>(buf);
    float* sink = nullptr;
    if (cudaMalloc(&sink, sizeof(float)) != cudaSuccess) return 0.0;
    cudaEvent_t a = nullptr, b = nullptr;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) {
        if (a != nullptr) cudaEventDestroy(a);
        if (b != nullptr) cudaEventDestroy(b);
        cudaFree(sink);
        return 0.0;
    }
    mem_path_read_kernel<<<grid, 256>>>(p, bytes / 16, passes, sink);   // warm (page tables, clocks)
    cudaDeviceSynchronize();
    double best = 0.0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(a);
        mem_path_read_kernel<<<grid, 256>>>(p, bytes / 16, passes, sink);
        cudaEventRecord(b);
        if (cudaEventSynchronize(b) != cudaSuccess) { best = 0.0; break; }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        const double gbps = (double) bytes * passes / 1048576.0 / ms;
        if (gbps > best) best = gbps;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(sink);
    return best;
}

}  // namespace

bool mem_path_probe(MemPath& out, int device) {
    out = MemPath{};
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
    out.vram_mib = (int) (prop.totalGlobalMem >> 20);
    // never take VRAM from a configuration that has none to spare (see kNeedFreeBytes)
    {
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b < kNeedFreeBytes) {
            out.skipped_for_vram = true;
            out.free_mib = (int) (free_b >> 20);
            return false;
        }
        (void) cudaGetLastError();
    }
    const unsigned grid = (unsigned) (prop.multiProcessorCount > 0 ? prop.multiProcessorCount * 32 : 640);

    void* small = nullptr;
    if (cudaMalloc(&small, kCacheBytes) != cudaSuccess) return false;
    void* large = nullptr;
    const bool have_large = cudaMalloc(&large, kDramBytes) == cudaSuccess;
    if (small != nullptr) cudaMemset(small, 0x5A, kCacheBytes);
    if (have_large) cudaMemset(large, 0x5A, kDramBytes);

    out.cache_gbps = timed_read(small, kCacheBytes, kCachePasses, grid, 2);
    out.dram_gbps = have_large ? timed_read(large, kDramBytes, kDramPasses, grid, 3) : 0.0;
    if (out.cache_gbps > 0.0) out.ratio = out.dram_gbps / out.cache_gbps;
    out.ran = out.cache_gbps > 0.0 && out.dram_gbps > 0.0;

    if (small != nullptr) cudaFree(small);
    if (large != nullptr) cudaFree(large);
    return out.ran;
}

const char* mem_path_verdict(const MemPath& m) {
    // 0.15 sits between the healthy band (0.3-0.6 here) and the degraded one (0.04), so the verdict
    // is a factor of ~4 away from either side rather than a knife edge.
    return m.ratio < 0.15 ? "DEGRADED" : "healthy";
}

bool mem_path_report(const MemPath& m, void* file) {
    std::FILE* f = static_cast<std::FILE*>(file != nullptr ? file : stderr);
    if (m.skipped_for_vram) {
        std::fprintf(f, "strata: memory path: not measured, only %d MiB of VRAM free (the probe needs %d MiB so it "
                        "can never be the reason a tight configuration fails to start)\n",
                     m.free_mib, (int) (kNeedFreeBytes >> 20));
        return true;
    }
    if (!m.ran) {
        std::fprintf(f, "strata: memory path: not measured (allocation refused)\n");
        return true;
    }
    const bool ok = m.ratio >= 0.15;
    std::fprintf(f, "strata: memory path: DRAM %.1f GB/s, cache %.1f GB/s (ratio %.2f) - %s\n",
                 m.dram_gbps, m.cache_gbps, m.ratio, mem_path_verdict(m));
    if (!ok)
        std::fprintf(f,
                     "strata: WARNING: this device is reading its own memory at PCIe speed.  Its caches are fine "
                     "(%.1f GB/s) so the kernels are not the cause: after a TDR the adapter can stay in this state "
                     "until it is reset.  Reboot (or Device Manager -> disable/enable the adapter) before taking any "
                     "measurement; numbers taken now describe the fault, not the engine.\n",
                     m.cache_gbps);
    std::fflush(f);
    return ok;
}

}  // namespace strata::kernels
