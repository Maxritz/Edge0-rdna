// bw_probe3.cu - is the 22 GB/s DRAM path a stuck memory clock waiting to ramp, or a hard limit?
//
//   clang --offload-arch=gfx1031 -O3 -x hip bw_probe3.cu -o bw_probe3.exe
//
// Part 1: 14 consecutive one-second windows of the same 1 GiB read, after two seconds of heavy
//         compute.  A DPM ramp shows as a rising curve; a hard limit is flat.
// Part 2: the same read with a pure-compute kernel saturating the SMs on a second stream, i.e.
//         maximum board power draw - if the table needs a 3D-class load to raise MCLK, this does.
// Part 3: 64 MiB..2 GiB in one pass each, to see where the plateau sits and whether it moves.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
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
__global__ void __launch_bounds__(256, 1) k_burn(float* __restrict__ out, int iters) {
    float a0 = threadIdx.x * 1e-9f, a1 = a0 + 1e-9f, a2 = a0 + 2e-9f, a3 = a0 + 3e-9f;
    const float m = 1.0000001f;
    for (int i = 0; i < iters; ++i) { a0 = fmaf(a0, m, 1e-9f); a1 = fmaf(a1, m, 1e-9f); a2 = fmaf(a2, m, 1e-9f); a3 = fmaf(a3, m, 1e-9f); }
    if (a0 + a1 + a2 + a3 == 1e30f) out[0] = a0;
}

int main() {
    hipDeviceProp_t p{};
    ck(hipGetDeviceProperties(&p, 0), "props");
    const unsigned grid = (unsigned) p.multiProcessorCount * 32;
    const std::size_t bytes = (std::size_t) 1 << 30;
    int4* d = nullptr;
    float* sink = nullptr;
    float* bo = nullptr;
    ck(hipMalloc(&d, bytes), "d");
    ck(hipMalloc(&sink, 4), "sink");
    ck(hipMalloc(&bo, 4), "bo");
    ck(hipMemset(d, 0x5A, bytes), "memset");
    std::printf("device %s  CUs=%d\n\n", p.name, p.multiProcessorCount);

    std::printf("part 0: 2 s of compute first (drive the board into its high power state)\n");
    k_burn<<<(unsigned) p.multiProcessorCount * 8, 256>>>(bo, 400000);
    ck(hipDeviceSynchronize(), "burn");

    std::printf("part 1: 14 x 1 s of the same 1 GiB read\n");
    hipEvent_t e0, e1;
    ck(hipEventCreate(&e0), "ev"); ck(hipEventCreate(&e1), "ev");
    for (int i = 0; i < 14; ++i) {
        ck(hipEventRecord(e0), "rec");
        k_read<<<grid, 256>>>(d, bytes / 16, 40, sink);   // ~1.8 s at 22 GB/s
        ck(hipEventRecord(e1), "rec");
        ck(hipEventSynchronize(e1), "sync");
        float ms = 0;
        ck(hipEventElapsedTime(&ms, e0, e1), "el");
        std::printf("   window %2d: %8.2f ms  %7.1f GB/s\n", i, ms, 40.0 * bytes / 1048576.0 / ms);
    }

    std::printf("\npart 2: the same read while every SM is also doing FMA (max board power)\n");
    {
        ck(hipEventRecord(e0), "rec");
        k_burn<<<(unsigned) p.multiProcessorCount * 16, 256>>>(bo, 60000000);   // long, other stream
        k_read<<<grid, 256>>>(d, bytes / 16, 20, sink);
        ck(hipEventRecord(e1), "rec");
        ck(hipEventSynchronize(e1), "sync");
        float ms = 0;
        ck(hipEventElapsedTime(&ms, e0, e1), "el");
        std::printf("   with compute: %8.2f ms  %7.1f GB/s\n", ms, 20.0 * bytes / 1048576.0 / ms);
        ck(hipDeviceSynchronize(), "sync");
    }

    std::printf("\npart 3: one pass at each size (where is the plateau?)\n");
    for (std::size_t mb : {64u, 128u, 256u, 512u, 1024u, 2048u}) {
        const std::size_t b = mb << 20;
        int4* q = nullptr;
        if (hipMalloc(&q, b) != hipSuccess) { std::printf("   %5zu MiB: alloc failed\n", mb); continue; }
        ck(hipMemset(q, 0x5A, b), "memset");
        const int reps = (int) ((std::size_t) (512u << 20) / b) + 1;
        ck(hipEventRecord(e0), "rec");
        k_read<<<grid, 256>>>(q, b / 16, reps, sink);
        ck(hipEventRecord(e1), "rec");
        ck(hipEventSynchronize(e1), "sync");
        float ms = 0;
        ck(hipEventElapsedTime(&ms, e0, e1), "el");
        std::printf("   %5zu MiB x%-3d %8.2f ms  %7.1f GB/s\n", mb, reps, ms, (double) reps * b / 1048576.0 / ms);
        ck(hipFree(q), "f");
    }
    ck(hipFree(d), "f"); ck(hipFree(sink), "f"); ck(hipFree(bo), "f");
    return 0;
}
