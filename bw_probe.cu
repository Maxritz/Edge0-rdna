// bw_probe.cu - what can this machine actually move, measured the only way that counts.
//
//   clang --offload-arch=gfx1031 -O3 -x hip bw_probe.cu -o bw_probe.exe
//
// Every test runs long enough (>= 1.5 s of work after a 0.5 s warm-up) for the card's DPM to be
// at its sustained clocks, because a 12 ms micro-benchmark measures the idle clock, not the card.
//
//   1. VRAM streaming read   1 GiB resident buffer, 16-byte loads, 16 blocks/CU.
//   2. VRAM read+write       copy 1 GiB buffer to another (the classic triad-ish test).
//   3. PCIe host->device     512 MiB pinned and 512 MiB pageable.
//   4. PCIe device->host     512 MiB pinned and 512 MiB pageable.
//   5. The Q8_0 dense GEMV   the engine's kernel at the decode shapes over a 512 MiB working set.
//
// If (1) is far below the card's 384 GB/s then decode is not a kernel problem at all, and the
// same defect is what makes prefill 4.9x slow: both read the same weights the same way.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define ck(x, m) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(e_)); std::exit(2); } } while (0)

// ---------------------------------------------------------------- VRAM streaming read
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

// ---------------------------------------------------------------- VRAM copy
__global__ void __launch_bounds__(256, 1) k_copy(const int4* __restrict__ s, int4* __restrict__ d, std::size_t n16) {
    const std::size_t tid = (std::size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t) gridDim.x * blockDim.x;
    for (std::size_t i = tid; i < n16; i += stride) d[i] = s[i];
}

// ---------------------------------------------------------------- Q8_0 dot (native_mmvq.cu:363/816)
struct Q80Block { __half d; int8_t qs[32]; };
struct Q81Block { __half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q80Block) == 34, "");

__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
    return v;
}
__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = __builtin_amdgcn_sdot4(v, u, sumi, false);
    }
    return __half2float(w->d) * __low2float(x->ds) * float(sumi);
}

// native_small_mmvq_kernel<Q80Block, 8, true> - the kernel the decode dense projections run
__global__ void __launch_bounds__(128, 1) k_engine(const Q80Block* __restrict__ w,
                                                   const Q81Block* __restrict__ x,
                                                   float* __restrict__ y, int n_in, int n_out, int copies) {
    constexpr int WARPS = 4, WARP = 32, Qi = 8, ROWS = 4;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int nb = n_in / 32;
    const std::size_t mat = std::size_t(n_out) * nb;
    float outer = 0.0f;
    for (int c = 0; c < copies; ++c) {
        const Q80Block* wc = w + std::size_t(c) * mat;
        float tmp[ROWS] = {};
        for (int kbx = tid / (Qi / 2); kbx < nb; kbx += BLOCKS_PER_ITER) {
            const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (row0 + i < n_out) tmp[i] += small_q8_dot(wc + std::size_t(row0 + i) * nb + kbx, x + kbx, kqs);
        }
        __shared__ float partial[WARPS - 1][ROWS][WARP];
        if (threadIdx.y > 0)
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
        __syncthreads();
        if (threadIdx.y == 0) {
            for (int i = 0; i < ROWS; ++i) {
                for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
                tmp[i] = warp_sum(tmp[i]);
                if (threadIdx.x == i && row0 + i < n_out) outer += tmp[i];
            }
        }
        __syncthreads();
    }
    if (outer == 1e30f) y[0] = outer;   // keep it live
}

struct Shape { const char* what; int K, N; };

template<typename F>
static double bench_ms(int iters, F f) {
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
    std::printf("device %s\n  CUs=%d  warp=%d  clock(max)=%.0f MHz  mem=%d kHz  bus=%d bit  L2=%d KiB  VRAM=%zu MiB\n",
                p.name, p.multiProcessorCount, p.warpSize, p.clockRate / 1000.0, p.memoryClockRate, p.memoryBusWidth,
                p.l2CacheSize / 1024, p.totalGlobalMem >> 20);
    std::size_t free_b = 0, total_b = 0;
    ck(hipMemGetInfo(&free_b, &total_b), "meminfo");
    std::printf("  hipMemGetInfo: free %.2f GiB of %.2f GiB\n\n", free_b / 1073741824.0, total_b / 1073741824.0);

    // ---------------------------------------------------------------- 1. VRAM streaming read
    {
        const std::size_t bytes = (std::size_t) 1 << 30;   // 1 GiB, ten times the Infinity Cache
        int4* d = nullptr;
        float* sink = nullptr;
        ck(hipMalloc(&d, bytes), "rd");
        ck(hipMalloc(&sink, 4), "sink");
        ck(hipMemset(d, 0x5A, bytes), "memset");
        const unsigned grid = (unsigned) p.multiProcessorCount * 16;
        const double ms = bench_ms(6, [&] { k_read<<<grid, 256>>>(d, bytes / 16, 1, sink); });
        std::printf("1. VRAM streaming read   1 GiB, grid %u x 256: %8.3f ms  %7.1f GB/s\n", grid, ms,
                    bytes / 1048576.0 / ms);
        // and with 64 blocks/CU, in case 16 is not enough occupancy for this access pattern
        const double ms64 = bench_ms(6, [&] { k_read<<<p.multiProcessorCount * 64, 256>>>(d, bytes / 16, 1, sink); });
        std::printf("   same, 64 blocks/CU:                             %8.3f ms  %7.1f GB/s\n", ms64,
                    bytes / 1048576.0 / ms64);
        int4* d2 = nullptr;
        ck(hipMalloc(&d2, bytes), "cp");
        const double msc = bench_ms(6, [&] { k_copy<<<grid, 256>>>(d, d2, bytes / 16); });
        std::printf("2. VRAM read+write       copy 1 GiB -> 1 GiB:     %8.3f ms  %7.1f GB/s (r+w)\n\n", msc,
                    2.0 * bytes / 1048576.0 / msc);
        ck(hipFree(d), "f"); ck(hipFree(d2), "f"); ck(hipFree(sink), "f");
    }

    // ---------------------------------------------------------------- 3/4. PCIe both ways
    {
        const std::size_t bytes = (std::size_t) 512 << 20;
        std::vector<uint8_t> host(bytes, 7);
        void* pin = nullptr;
        ck(hipHostMalloc(&pin, bytes, hipHostMallocDefault), "pin");
        int4* dev = nullptr;
        ck(hipMalloc(&dev, bytes), "pc");
        const double h2d_pin = bench_ms(4, [&] { ck(hipMemcpyAsync(dev, pin, bytes, hipMemcpyHostToDevice, 0), "h2d"); });
        const double d2h_pin = bench_ms(4, [&] { ck(hipMemcpyAsync(pin, dev, bytes, hipMemcpyDeviceToHost, 0), "d2h"); });
        const double h2d_pg = bench_ms(4, [&] { ck(hipMemcpyAsync(dev, host.data(), bytes, hipMemcpyHostToDevice, 0), "h2d"); });
        const double d2h_pg = bench_ms(4, [&] { ck(hipMemcpyAsync(host.data(), dev, bytes, hipMemcpyDeviceToHost, 0), "d2h"); });
        std::printf("3. PCIe host->device    512 MiB pinned %8.3f ms %6.1f GB/s | pageable %8.3f ms %6.1f GB/s\n",
                    h2d_pin, bytes / 1048576.0 / h2d_pin, h2d_pg, bytes / 1048576.0 / h2d_pg);
        std::printf("4. PCIe device->host    512 MiB pinned %8.3f ms %6.1f GB/s | pageable %8.3f ms %6.1f GB/s\n\n",
                    d2h_pin, bytes / 1048576.0 / d2h_pin, d2h_pg, bytes / 1048576.0 / d2h_pg);
        ck(hipFree(dev), "f"); ck(hipHostFree(pin), "f");
    }

    // ---------------------------------------------------------------- 5. the engine's GEMV kernel
    const Shape shapes[] = {
        {"GDN qkv 2560->10240", 2560, 10240},   {"GDN gate 2560->6144", 2560, 6144},
        {"out 6144->2560", 6144, 2560},         {"QSA q 2560->12288", 2560, 12288},
        {"head 2560->106299", 2560, 106299},
    };
    std::printf("5. the engine's Q8_0 GEMV (native_small_mmvq_kernel), working set >= 512 MiB\n");
    for (const Shape& s : shapes) {
        const int nb = s.K / 32;
        const std::size_t bytes = (std::size_t) s.N * nb * 34;
        const int copies = (int) std::max<std::size_t>(1, (std::size_t) (512u << 20) / bytes);
        Q80Block* w = nullptr;
        Q81Block* x = nullptr;
        float* y = nullptr;
        ck(hipMalloc(&w, bytes * copies), "w");
        ck(hipMemset(w, 0x11, bytes * copies), "wm");
        ck(hipMalloc(&x, (std::size_t) nb * 36), "x");
        ck(hipMemset(x, 0x22, (std::size_t) nb * 36), "xm");
        ck(hipMalloc(&y, 4096), "y");
        const unsigned grid = unsigned((s.N + 3) / 4);
        const double ms = bench_ms(std::max(3, (int) (1500.0 / (bytes * copies / 1048576.0 / 20.0))), [&] {
            k_engine<<<grid, dim3(32, 4)>>>(w, x, y, s.K, s.N, copies);
        });
        std::printf("   %-22s %7.2f MB x%-4d %8.3f ms  %6.1f GB/s\n", s.what, bytes / 1048576.0, copies, ms,
                    bytes * copies / 1048576.0 / ms);
        ck(hipFree(w), "f"); ck(hipFree(x), "f"); ck(hipFree(y), "f");
    }
    return 0;
}
