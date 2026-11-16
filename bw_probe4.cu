// bw_probe4.cu - hold a 2 GiB hipMalloc buffer while the OS is asked where it lives.
// Run it, then sample the WDDM GPU memory counters from another shell:
//   Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage'
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

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

int main() {
    hipDeviceProp_t p{};
    ck(hipGetDeviceProperties(&p, 0), "props");
    const std::size_t bytes = (std::size_t) 2 << 30;
    int4* d = nullptr;
    float* sink = nullptr;
    ck(hipMalloc(&d, bytes), "alloc");
    ck(hipMalloc(&sink, 4), "sink");
    ck(hipMemset(d, 0x5A, bytes), "memset");
    std::size_t free_b = 0, total_b = 0;
    ck(hipMemGetInfo(&free_b, &total_b), "info");
    std::printf("holding %.2f GiB; hipMemGetInfo free %.2f GiB\n", bytes / 1073741824.0, free_b / 1073741824.0);
    std::fflush(stdout);
    const unsigned grid = (unsigned) p.multiProcessorCount * 32;
    for (int pass = 0; pass < 8; ++pass) {   // ~16 s of reads, so the allocation must stay resident
        hipEvent_t a, b;
        ck(hipEventCreate(&a), "ev"); ck(hipEventCreate(&b), "ev");
        ck(hipEventRecord(a), "rec");
        k_read<<<grid, 256>>>(d, bytes / 16, 1, sink);
        ck(hipEventRecord(b), "rec");
        ck(hipEventSynchronize(b), "sync");
        float ms = 0;
        ck(hipEventElapsedTime(&ms, a, b), "el");
        std::printf("  pass %d: %7.1f GB/s\n", pass, bytes / 1048576.0 / ms);
        std::fflush(stdout);
    }
    ck(hipFree(d), "f"); ck(hipFree(sink), "f");
    return 0;
}
