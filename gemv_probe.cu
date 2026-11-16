// gemv_probe.cu - THE decode question, measured: how fast can this card read a Q8_0 weight matrix,
// and how fast does the engine's own kernel read it?
//
//   clang --offload-arch=gfx1031 -O3 -x hip gemv_probe.cu -o gemv_probe.exe
//
// Two kernels over the same bytes, same shapes, same cold buffers (copies cycled so the 96 MiB
// Infinity Cache cannot serve them):
//
//   ceiling : a pure streaming read of the matrix with 16-byte (int4) loads and no dependency
//             between iterations.  This is the achievable bandwidth of the card, not the peak.
//   engine  : a byte-for-byte transcription of native_small_mmvq_kernel<Q80Block, 8, ROWS=4> in
//             edge0-strata/src/kernels/cuda/native_mmvq.cu - the kernel the decode dense
//             projections actually run.  34-byte GGUF blocks leave every int 2-byte aligned, so
//             its inner load is two 16-bit reads (load_int_b2).
//
// A decode token reads every weight exactly once, so the GB/s here IS the tok/s ceiling:
// 4.4 GB/token at 384 GB/s is 11.5 ms/token; at 12 GB/s it is 371 ms/token.
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

// ---- the engine's block types and primitives (native_mmvq.cu:72, :363, :816) ----------------
struct Q80Block { __half d; int8_t qs[32]; };
struct Q81Block { __half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q80Block) == 34, "");
static_assert(sizeof(Q81Block) == 36, "");

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
    const float d0 = __half2float(w->d);
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

// ---- engine: native_small_mmvq_kernel<Q80Block, 8, true> (WARPS=4, WARP=32, ROWS=4) ---------
__global__ void __launch_bounds__(128, 1) k_engine(const Q80Block* __restrict__ w,
                                                   const Q81Block* __restrict__ x,
                                                   float* __restrict__ y, int n_in, int n_out) {
    constexpr int WARPS = 4, WARP = 32, Qi = 8, ROWS = 4;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;   // 32
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
        for (int i = 0; i < ROWS; ++i)
            if (row0 + i < n_out)
                tmp[i] += small_q8_dot(w + std::size_t(row0 + i) * blocks_per_row + kbx, x + kbx, kqs);
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0)
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    __syncthreads();
    if (threadIdx.y > 0) return;
    for (int i = 0; i < ROWS; ++i) {
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// the same kernel with the K loop unrolled by 4: the SAME blocks per thread, the SAME order, so
// every output is bitwise the engine's - only the loads are hoisted to be in flight together.
template<int U>
__global__ void __launch_bounds__(128, 1) k_engine_unroll(const Q80Block* __restrict__ w,
                                                          const Q81Block* __restrict__ x,
                                                          float* __restrict__ y, int n_in, int n_out) {
    constexpr int WARPS = 4, WARP = 32, Qi = 8, ROWS = 4;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;   // 32
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int nb = n_in / 32;
    const int kqs = 2 * (tid % (Qi / 2));
    float tmp[ROWS] = {};
    const Q80Block* wr[ROWS];
    for (int i = 0; i < ROWS; ++i) wr[i] = w + std::size_t(row0 + i) * nb;
    // Same blocks, same order, same expression - only the dependency chain is broken up so that U
    // iterations' weight loads are all in flight before the first dot consumes them.
#pragma unroll U
    for (int kbx = tid / (Qi / 2); kbx < nb; kbx += BLOCKS_PER_ITER) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            if (row0 + i < n_out) tmp[i] += small_q8_dot(wr[i] + kbx, x + kbx, kqs);
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0)
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    __syncthreads();
    if (threadIdx.y > 0) return;
    for (int i = 0; i < ROWS; ++i) {
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ---- ceiling: read the same bytes with 16-byte aligned loads, nothing else --------------------
__global__ void __launch_bounds__(256, 1) k_ceiling(const int4* __restrict__ p, std::size_t n16, float* sink) {
    std::size_t i = (std::size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t) gridDim.x * blockDim.x;
    int acc = 0;
    for (; i < n16; i += stride) {
        const int4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) *sink = 1.0f;
}

struct Shape { const char* what; int K, N; };

template<typename F>
static double bench(int iters, F f) {
    hipEvent_t a, b;
    ck(hipEventCreate(&a), "ev"); ck(hipEventCreate(&b), "ev");
    for (int i = 0; i < 3; ++i) f();
    ck(hipDeviceSynchronize(), "sync");
    ck(hipEventRecord(a), "rec");
    for (int i = 0; i < iters; ++i) f();
    ck(hipEventRecord(b), "rec");
    ck(hipEventSynchronize(b), "sync");
    float ms = 0.0f;
    ck(hipEventElapsedTime(&ms, a, b), "el");
    ck(hipEventDestroy(a), "ev"); ck(hipEventDestroy(b), "ev");
    return ms / iters;
}

static void run(const Shape& s, int iters) {
    const int nb = s.K / 32;
    const std::size_t bytes = (std::size_t) s.N * nb * 34;
    const int copies = (int) std::max<std::size_t>(1, std::min<std::size_t>(8, (std::size_t) (256u << 20) / bytes));
    std::vector<uint8_t> hw(bytes);
    std::mt19937 rng(11);
    for (auto& v : hw) v = (uint8_t) (rng() & 0xFF);
    for (std::size_t i = 0; i < bytes; i += 34) { hw[i] = 0x00; hw[i + 1] = 0x24; }   // d ~ 0.0625
    std::vector<uint8_t> hx((std::size_t) nb * 36);
    for (auto& v : hx) v = (uint8_t) (rng() & 0xFF);
    for (std::size_t i = 0; i < hx.size(); i += 36) { hx[i] = 0x00; hx[i + 1] = 0x24; }

    std::vector<void*> w(copies);
    for (auto& p : w) { ck(hipMalloc(&p, bytes), "w"); ck(hipMemcpy(p, hw.data(), bytes, hipMemcpyHostToDevice), "w"); }
    Q81Block* x = nullptr;
    float* y = nullptr;
    float* sink = nullptr;
    ck(hipMalloc(&x, hx.size()), "x"); ck(hipMemcpy(x, hx.data(), hx.size(), hipMemcpyHostToDevice), "x");
    ck(hipMalloc(&y, (std::size_t) s.N * 4), "y");
    ck(hipMalloc(&sink, 4), "sink");

    const unsigned grid_e = unsigned((s.N + 3) / 4);
    const double wbytes = (double) bytes * copies;
    std::printf("%-24s K=%4d N=%6d  %7.2f MB x%d\n", s.what, s.K, s.N, bytes / 1048576.0, copies);

    const double m_ceil = bench(iters, [&] { k_ceiling<<<640, 256>>>((const int4*) w[0], bytes / 16, sink); });
    const double m_eng = bench(iters, [&] {
        for (int c = 0; c < copies; ++c) k_engine<<<grid_e, dim3(32, 4)>>>((const Q80Block*) w[c], x, y, s.K, s.N);
    });
    const double m_u4 = bench(iters, [&] {
        for (int c = 0; c < copies; ++c)
            k_engine_unroll<4><<<grid_e, dim3(32, 4)>>>((const Q80Block*) w[c], x, y, s.K, s.N);
    });
    std::printf("   ceiling  int4 stream      %8.3f ms %7.1f GB/s\n", m_ceil, wbytes / 1048576.0 / m_ceil);
    std::printf("   engine   warp-share       %8.3f ms %7.1f GB/s   (%.1f%% of ceiling)\n", m_eng,
                wbytes / 1048576.0 / m_eng, 100.0 * m_ceil / m_eng);
    std::printf("   engine   +K unroll 4      %8.3f ms %7.1f GB/s   (%.1f%% of ceiling)\n\n", m_u4,
                wbytes / 1048576.0 / m_u4, 100.0 * m_ceil / m_u4);
    for (auto p : w) ck(hipFree(p), "f");
    ck(hipFree(x), "f"); ck(hipFree(y), "f"); ck(hipFree(sink), "f");
}

int main() {
    hipDeviceProp_t p{};
    ck(hipGetDeviceProperties(&p, 0), "props");
    const double peak = 2.0 * (p.memoryClockRate * 1000.0) * (p.memoryBusWidth / 8.0) / 1e9;
    std::printf("device %s  CUs=%d warp=%d clock=%.0f MHz  mem %d kHz x %d bit = %.0f GB/s peak\n\n", p.name,
                p.multiProcessorCount, p.warpSize, p.clockRate / 1000.0, p.memoryClockRate, p.memoryBusWidth, peak);
    const Shape shapes[] = {
        {"GDN qkv 2560->10240", 2560, 10240},
        {"GDN gate 2560->6144", 2560, 6144},
        {"out 6144->2560", 6144, 2560},
        {"QSA q 2560->12288", 2560, 12288},
        {"HC down 10240->320", 10240, 320},
        {"HC up 320->10240", 320, 10240},
        {"shared down 640->2560", 640, 2560},
        {"head 2560->106299", 2560, 106299},
    };
    for (const Shape& s : shapes) run(s, 40);
    return 0;
}
