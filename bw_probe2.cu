// bw_probe2.cu - separate the four things that could make a 384 GB/s card read at 22 GB/s:
//   the SM clock, the L2/Infinity-Cache path, the DRAM path, or the allocation living in host RAM.
//
//   clang --offload-arch=gfx1031 -O3 -x hip bw_probe2.cu -o bw_probe2.exe
//
// The discriminating test is buffer size: 2 MiB is L2-resident, 64 MiB is Infinity-Cache-resident
// (96 MB on Navi 22) and 1 GiB is DRAM.  If the 64 MiB read is much faster than the 1 GiB one the
// buffer is genuinely in VRAM and only the DRAM path is defective; if every size reads at the same
// ~22 GB/s as a zero-copy host buffer, then "VRAM" here is system RAM behind PCIe and no kernel
// can ever be fast.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ck(x, m) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(e_)); std::exit(2); } } while (0)

__global__ void __launch_bounds__(256, 1) k_read(const int4* __restrict__ p, std::size_t n16, int reps, float* sink) {
    const std::size_t tid = (std::size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t) gridDim.x * blockDim.x;
    int acc = 0;
    for (int r = 0; r < reps; ++r)
        for (std::size_t i = tid; i < n16; i += stride) {
            const int4 v = p[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    if (acc == 0x7fffffff) *sink = 1.0f;
}

__global__ void __launch_bounds__(256, 1) k_write(int4* __restrict__ p, std::size_t n16, float* sink) {
    const std::size_t tid = (std::size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t) gridDim.x * blockDim.x;
    const int4 v = {1, 2, 3, 4};
    for (std::size_t i = tid; i < n16; i += stride) p[i] = v;
    if (n16 == 1) *sink = 1.0f;
}

// fp32 FMA throughput: 4 independent accumulators per lane, no memory traffic at all
__global__ void __launch_bounds__(256, 1) k_fma(float* __restrict__ out, int iters) {
    float a0 = threadIdx.x * 1e-9f, a1 = a0 + 1e-9f, a2 = a0 + 2e-9f, a3 = a0 + 3e-9f;
    const float m = 1.0000001f;
    for (int i = 0; i < iters; ++i) { a0 = fmaf(a0, m, 1e-9f); a1 = fmaf(a1, m, 1e-9f); a2 = fmaf(a2, m, 1e-9f); a3 = fmaf(a3, m, 1e-9f); }
    if (a0 + a1 + a2 + a3 == 1e30f) out[0] = a0;
}

__global__ void __launch_bounds__(256, 1) k_fma16(__half2* __restrict__ out, int iters) {
    __half2 a0 = __float2half2_rn(1e-4f), a1 = a0, a2 = a0, a3 = a0;
    const __half2 m = __float2half2_rn(1.0001f);
    for (int i = 0; i < iters; ++i) { a0 = __hfma2(a0, m, a0); a1 = __hfma2(a1, m, a1); a2 = __hfma2(a2, m, a2); a3 = __hfma2(a3, m, a3); }
    if (__low2float(a0) + __low2float(a1) + __low2float(a2) + __low2float(a3) == 1e30f) out[0] = a0;
}

template<typename F>
static double ms_of(int iters, F f) {
    hipEvent_t a, b;
    ck(hipEventCreate(&a), "ev"); ck(hipEventCreate(&b), "ev");
    f(); ck(hipDeviceSynchronize(), "sync");
    ck(hipEventRecord(a), "rec");
    for (int i = 0; i < iters; ++i) f();
    ck(hipEventRecord(b), "rec");
    ck(hipEventSynchronize(b), "sync");
    float ms = 0.0f;
    ck(hipEventElapsedTime(&ms, a, b), "el");
    ck(hipEventDestroy(a), "ev"); ck(hipEventDestroy(b), "ev");
    return ms / iters;
}

int main() {
    hipDeviceProp_t p{};
    ck(hipGetDeviceProperties(&p, 0), "props");
    const unsigned grid = (unsigned) p.multiProcessorCount * 32;
    std::printf("device %s  CUs=%d  L2=%.1f MiB\n\n", p.name, p.multiProcessorCount, p.l2CacheSize / 1048576.0);
    float* sink = nullptr;
    ck(hipMalloc(&sink, 4), "sink");

    struct Case { const char* what; std::size_t bytes; };
    const Case cases[] = {{"  2 MiB  (L2-resident)", 2u << 20}, {" 32 MiB  (L2+IC-resident)", 32u << 20},
                          {" 64 MiB  (Infinity-Cache-resident, 96 MB)", 64u << 20}, {"  1 GiB  (DRAM)", (std::size_t) 1 << 30}};
    std::printf("1. hipMalloc (device) reads - the same kernel, only the size changes\n");
    for (const Case& c : cases) {
        int4* d = nullptr;
        ck(hipMalloc(&d, c.bytes), "alloc");
        ck(hipMemset(d, 0x5A, c.bytes), "memset");
        const double ms = ms_of(40, [&] { k_read<<<grid, 256>>>(d, c.bytes / 16, 1, sink); });
        std::printf("   %-42s %8.3f ms  %8.1f GB/s\n", c.what, ms, c.bytes / 1048576.0 / ms);
        ck(hipFree(d), "f");
    }

    std::printf("\n2. the same read of a zero-copy HOST buffer (goes over PCIe by definition)\n");
    {
        const std::size_t bytes = (std::size_t) 64 << 20;
        void* h = nullptr;
        int4* dmap = nullptr;
        ck(hipHostMalloc(&h, bytes, hipHostMallocMapped), "host");
        ck(hipHostGetDevicePointer((void**) &dmap, h, 0), "map");
        std::memset(h, 0x33, bytes);
        const double ms = ms_of(40, [&] { k_read<<<grid, 256>>>(dmap, bytes / 16, 1, sink); });
        std::printf("   64 MiB zero-copy host                 %8.3f ms  %8.1f GB/s\n", ms, bytes / 1048576.0 / ms);
        ck(hipHostFree(h), "f");
    }

    std::printf("\n3. write-only 1 GiB (no reads at all)\n");
    {
        const std::size_t bytes = (std::size_t) 1 << 30;
        int4* d = nullptr;
        ck(hipMalloc(&d, bytes), "write");
        const double ms = ms_of(20, [&] { k_write<<<grid, 256>>>(d, bytes / 16, sink); });
        std::printf("   write 1 GiB                           %8.3f ms  %8.1f GB/s\n", ms, bytes / 1048576.0 / ms);
        ck(hipFree(d), "f");
    }

    std::printf("\n4. compute only (no memory traffic): a defect here means clock/CU, not DRAM\n");
    {
        float* o = nullptr;
        ck(hipMalloc(&o, 4), "o");
        const int iters = 20000;
        const double ms = ms_of(20, [&] { k_fma<<<(unsigned) p.multiProcessorCount * 8, 256>>>(o, iters); });
        const double flops = (double) p.multiProcessorCount * 8 * 256 * iters * 4.0 * 2.0 * 2.0;   // 4 FMA, 2 FLOP
        std::printf("   fp32 FMA                              %8.3f ms  %8.1f GFLOP/s  (%.2f TFLOP/s)\n", ms,
                    flops / ms / 1e6, flops / ms / 1e9);
        const double ms16 = ms_of(20, [&] { k_fma16<<<(unsigned) p.multiProcessorCount * 8, 256>>>((__half2*) o, iters); });
        const double flops16 = (double) p.multiProcessorCount * 8 * 256 * iters * 4.0 * 2.0 * 4.0;  // 2 lanes per half2
        std::printf("   fp16x2 FMA                            %8.3f ms  %8.1f GFLOP/s  (%.2f TFLOP/s)\n", ms16,
                    flops16 / ms16 / 1e6, flops16 / ms16 / 1e9);
        ck(hipFree(o), "f");
    }
    ck(hipFree(sink), "f");
    return 0;
}
