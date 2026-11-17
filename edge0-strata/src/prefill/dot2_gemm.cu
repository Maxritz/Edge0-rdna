// src/prefill/dot2_gemm.cu - the RDNA2 packed-fp16 dense GEMM (see wmma_gemm.h).
//
// WHY THIS EXISTS.  The gfx11 dense GEMM in wmma_gemm.cu is WMMA, and RDNA2 has no matrix cores, so on gfx103x
// those kernels compile to nothing and the dense projections fall to rocBLAS.  Measured on this box
// (edge-0-analysis-rdna2.md 0.54): the gdn bucket is 12.4 TFLOP over 3,080 ms = 4.0 TFLOP/s, 16 % of the card's
// 24.6 TFLOP/s fp16 peak.  This kernel computes the same product with V_DOT2_F32_F16, which RDNA2 does have:
// two fp16 multiplies per instruction, f32 accumulate (D = a0*w0 + a1*w1 + acc).
//
// SHAPE.  Block tile BM x BN = 64 x 64, k-slab BK = 32, 256 threads (16 x 16), each thread owning a 4 x 4
// output tile (16 f32 accumulators).  Per k-pair a thread issues 8 LDS + 16 dot2, i.e. two thirds of its
// instructions are the packed multiply - the ratio the peak needs.
//
// NUMERICS.  A different blocking and accumulation tree from rocBLAS: NOT bit-identical, which is the change the
// owner approved.  The harness is tests/hip/gemm_f16_io_parity (rel-L2 against an FP64 reference, the same bound
// the shipped FP16 path already meets).  The dot2 asm follows native_bf16.cu's proven form.
#include "wmma_gemm.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kBM = 64;    // X rows a block owns
constexpr int kBN = 64;    // W rows (= output columns) a block owns
constexpr int kBK = 32;    // k per slab
constexpr int kLDK = kBK + 2;   // padded: without the +2 the 16 thread rows land on four banks

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)

__device__ __forceinline__ float dot2(uint32_t ap, uint32_t wp, float acc) {
    float r;
    __asm__("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(ap), "v"(wp), "v"(acc));
    return r;
}

__global__ void __launch_bounds__(256) dot2_gemm_kernel(const _Float16* __restrict__ X, int ldx,
                                                        const _Float16* __restrict__ W, int ldw,
                                                        float* __restrict__ Y, int M, int N, int K, int ldy,
                                                        int accumulate) {
    __shared__ __align__(16) _Float16 sA[2][kBM][kLDK];
    __shared__ __align__(16) _Float16 sB[2][kBN][kLDK];
    const int tid = threadIdx.x;
    const int m0 = (int) blockIdx.x * kBM;
    const int n0 = (int) blockIdx.y * kBN;
    // stage: kBM rows x kBK halves = kBM x (kBK/8) uint4 per tile, one uint4 a thread (64 x 4 = 256)
    auto stage = [&](int buf, int k0) {
        const int r = tid >> 2, q = tid & 3;      // r < 64, q < 4
        const int m = min(m0 + r, M - 1);         // the tail repeats its last row; its outputs are dropped below
        *reinterpret_cast<uint4*>(&sA[buf][r][8 * q]) =
            *reinterpret_cast<const uint4*>(X + (size_t) m * ldx + k0 + 8 * q);
        const int n = min(n0 + r, N - 1);
        *reinterpret_cast<uint4*>(&sB[buf][r][8 * q]) =
            *reinterpret_cast<const uint4*>(W + (size_t) n * ldw + k0 + 8 * q);
    };
    const int tm = tid >> 4, tn = tid & 15;       // 16 x 16; this thread owns rows tm*4..+4, cols tn*4..+4
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    int buf = 0;
    stage(0, 0);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += kBK) {
        const bool more = k0 + kBK < K;
        if (more) stage(buf ^ 1, k0 + kBK);       // the other buffer, while this one computes
#pragma unroll
        for (int k = 0; k < kBK; k += 2) {
            uint32_t ap[4], bp[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ap[i] = *reinterpret_cast<const uint32_t*>(&sA[buf][tm * 4 + i][k]);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                bp[j] = *reinterpret_cast<const uint32_t*>(&sB[buf][tn * 4 + j][k]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = dot2(ap[i], bp[j], acc[i][j]);
        }
        if (more) {
            __syncthreads();
            buf ^= 1;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + tm * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tn * 4 + j;
            if (n >= N) continue;
            float* y = Y + (size_t) m * ldy + n;
            *y = accumulate ? *y + acc[i][j] : acc[i][j];
        }
    }
}

#endif

}  // namespace

bool strata_pf_gemm_f16_dot2(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                             int64_t ldy, float beta, void* stream) {
    // The ARCH DECISION IS RUNTIME, not __gfx1031__.  This is a host function, and the device macros are only
    // defined in the device pass, so a compile-time guard here compiles to `return false` and the kernel is
    // silently never launched (which is exactly how this first went in: the parity harness then reported the
    // rocBLAS numbers unchanged to six digits, which is what caught it).
    static const bool on_rdna2 = [] {
        const char* v = std::getenv("STRATA_GEMM_DOT2");
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return false;
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, dev) != hipSuccess) return false;
        // major/minor, not gcnArchName: on Windows HIP gcnArchName comes back EMPTY, so a string test silently
        // refuses and the kernel never launches - which is what kept the first two attempts invisible.
        // OPT-IN, default OFF: measured no faster than the rocBLAS call it replaces (gdn 3,074-3,078 ms both ways,
        // alternating, 0.55), so it is not a win as written - it is kept as the working foundation for a better
        // tiling, behind STRATA_GEMM_DOT2=1.
        bool ok = false;
        if (v != nullptr) ok = std::atoi(v) != 0;   // STRATA_GEMM_DOT2=0 is the A/B arm
        std::fprintf(stderr, "strata gemm: the packed-fp16 (V_DOT2_F32_F16) dense GEMM is %s here (arch %d.%d%s%s, %s)\n",
                     ok ? "ON" : "off", (int) p.major, (int) p.minor, p.gcnArchName[0] ? " " : "",
                     p.gcnArchName[0] ? p.gcnArchName : "", ok ? "gfx103x" : "not gfx103x");
        return ok;
    }();
    if (!on_rdna2) return false;
    if (X == nullptr || W == nullptr || Y == nullptr) return false;
    if (T < 32 || N < 8 || K <= 0 || (K % kBK) != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldy < N) return false;
    if (T > (1LL << 30) || N > (1LL << 30) || K > (1LL << 30)) return false;
    const dim3 grid((unsigned) ((T + kBM - 1) / kBM), (unsigned) ((N + kBN - 1) / kBN));
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
    dot2_gemm_kernel<<<grid, 256, 0, (hipStream_t) stream>>>(
        (const _Float16*) X, (int) K, (const _Float16*) W, (int) K, Y, (int) T, (int) N, (int) K, (int) ldy,
        beta == 1.0f ? 1 : 0);
    return hipGetLastError() == hipSuccess;
#else
    (void) grid;
    return false;   // this device pass has no gfx103x target: nothing to launch
#endif
}
