// tools/hip/q3k_nibble_bench.hip.cpp - is a 4-bit item code cheaper than Q3_K itself for the 1280-wide latent
// experts of Edge0-120B-q3k?
//
// WHY.  The GP100 4-bit transcode (include/strata/core/expert_nibble.hpp) is CUDA sm_60 only and hard-wired to a
// 2560-wide expert; this pack's experts are 1280 wide and run on gfx1031.  Before porting that subsystem, measure
// the only question that decides it: at this shape, does decoding 4-bit item codes and dotting them beat decoding
// Q3_K blocks and dotting them, or is Q3_K's 1.33x smaller footprint worth more on this GPU?
//
// WHAT IT DOES.  Both kernels read the same rows, the same int8 activation, and produce the same per-row float;
// only the weight format differs, so the difference is the decode.  Layout A is the pack's real Q3_K blocks
// (110 B / 256 weights: hmask 32, qs 64, scales 12, fp16 d).  Layout B is the transcode's item code: 16 B of
// 4-bit codes per 32 weights + one int8 scale per 16 weights + the block's fp16 d.
//
// The bench is arithmetic, not bandwidth: 1280 rows x 1280 weights is ~0.7-0.9 MB, so it sits in cache and the
// number it reports is "cost per weight of the decode", which is what the transcode claims to reduce.
//
//   build: clang++ -x hip --offload-arch=gfx1031 -O3 -o q3k_nibble_bench q3k_nibble_bench.hip.cpp
//   run:   q3k_nibble_bench [iterations]

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace {

constexpr int kRows = 2 * 640;      // gate rows 0..639, then up rows: the pack's gate/up block
constexpr int kWidth = 1280;        // n_embd of this pack's latent experts
constexpr int kItems = kWidth / 32; // 40 items of 32 weights per row
constexpr int kBlocks = kWidth / 256;
constexpr int kBlockBytes = 110;

// ---------------------------------------------------------------- the two layouts

// A: the pack's bytes, verbatim.
struct RawArena {
    std::vector<uint8_t> w;   // kRows * kWidth/256 * 110
    std::vector<int8_t> x;    // kWidth: the activation, one per weight column
};

// B: one 32-weight item = 16 B of codes (two 4-bit codes a byte); one int8 scale per 16 weights; one fp16 d a
// 256-weight block, shared by that block's 8 items.
struct NibArena {
    std::vector<uint8_t> codes;    // kRows * kItems * 16
    std::vector<int8_t> scale;     // kRows * kItems * 2
    std::vector<uint16_t> d;       // kRows * kBlocks
};

// Q3_K's 16 six-bit scales, group order (tools/canonical_xcheck.py: unpack_q3_k_scales).
__host__ __device__ inline int sc6(const uint8_t* s, int g) {
    const int j = g & 3;
    if (g < 4) return (s[j] & 0x0F) | ((s[8 + j] & 0x03) << 4);
    if (g < 8) return (s[4 + j] & 0x0F) | (((s[8 + j] >> 2) & 0x03) << 4);
    if (g < 12) return ((s[j] >> 4) & 0x0F) | (((s[8 + j] >> 4) & 0x03) << 4);
    return ((s[4 + j] >> 4) & 0x0F) | (((s[8 + j] >> 6) & 0x03) << 4);
}

// the codes of one 32-weight item, weight l in 0..31 -> 0..7 (the value is (code - 4) * d * (sc - 32))
__host__ __device__ inline void item_codes(const uint8_t* block, int half, int j, uint8_t c[32]) {
    const uint8_t* hm = block;             // 32: sub 0 reads [0..15], sub 1 the SAME bytes at [16..31]
    const uint8_t* qs = block + 32 + half * 32;
    const int m = 4 * half + j;            // the hmask bit does NOT reset between halves
    for (int l = 0; l < 16; ++l) {
        const uint8_t h0 = (hm[l] >> m) & 1, h1 = (hm[16 + l] >> m) & 1;
        c[l] = (uint8_t) (((qs[l] >> (2 * j)) & 3) + 4 * h0);
        c[16 + l] = (uint8_t) (((qs[16 + l] >> (2 * j)) & 3) + 4 * h1);
    }
}

// fp16 -> fp32 for the bench's synthetic d (no ldexp: it has to be device code too).
__host__ __device__ inline float f16(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float v;
    if (e == 0) {
        v = (float) m * (1.0f / 16777216.0f);                       // subnormal: m * 2^-24
    } else if (e == 31u) {
        v = m ? (0.0f / 0.0f) : (1.0f / 0.0f);
    } else {
        v = (float) (m | 0x400u) * (1.0f / 16777216.0f) * (float) (1u << (e - 1u));   // * 2^(e-15)
    }
    return s ? -v : v;
}

// ---------------------------------------------------------------- the two kernels
//
// One block a row, 40 threads (one item each), a shared-memory reduction.  The reduction and the activation read
// are identical in both kernels, so the timing difference is the weight decode.

__device__ __forceinline__ void reduce_row(float v, float* smem, int tid, int n, float* out) {
    smem[tid] = v;
    __syncthreads();
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = smem[0];
}

// A: decode the pack's Q3_K block and dot it.
__global__ void row_q3k(const uint8_t* __restrict__ w, const int8_t* __restrict__ x, float* __restrict__ out) {
    __shared__ float smem[64];
    const int row = blockIdx.x, tid = threadIdx.x;
    float acc = 0.f;
    if (tid < kItems) {
        const int k = tid, b = k / 8, half = (k % 8) / 4, j = k % 4;
        const uint8_t* block = w + ((size_t) row * kBlocks + b) * kBlockBytes;
        uint8_t c[32];
        item_codes(block, half, j, c);
        const int g0 = 8 * half + 2 * j;                  // two 16-weight groups per item
        const float d = f16((uint16_t) (block[108] | (block[109] << 8)));
        const int8_t* xr = x + k * 32;
        int s0 = 0, s1 = 0;
        for (int l = 0; l < 16; ++l) {
            s0 += ((int) c[l] - 4) * (int) xr[l];
            s1 += ((int) c[16 + l] - 4) * (int) xr[16 + l];
        }
        acc = (d * (float) (sc6(block + 96, g0) - 32)) * (float) s0 +
              (d * (float) (sc6(block + 96, g0 + 1) - 32)) * (float) s1;
    }
    reduce_row(acc, smem, tid, 64, out + row);
}

// B: decode the 4-bit item code and dot it.
__global__ void row_nib(const uint8_t* __restrict__ codes, const int8_t* __restrict__ scale,
                        const uint16_t* __restrict__ darr, const int8_t* __restrict__ x,
                        float* __restrict__ out) {
    __shared__ float smem[64];
    const int row = blockIdx.x, tid = threadIdx.x;
    float acc = 0.f;
    if (tid < kItems) {
        const int k = tid, b = k / 8;
        const uint8_t* cp = codes + ((size_t) row * kItems + k) * 16;
        const int8_t* sp = scale + ((size_t) row * kItems + k) * 2;
        const float d = f16(darr[(size_t) row * kBlocks + b]);
        const int8_t* xr = x + k * 32;
        int s0 = 0, s1 = 0;
        for (int l = 0; l < 16; ++l) {
            const int c0 = (cp[l >> 1] >> (4 * (l & 1))) & 15;          // two codes a byte
            const int c1 = (cp[8 + (l >> 1)] >> (4 * (l & 1))) & 15;
            s0 += (c0 - 4) * (int) xr[l];
            s1 += (c1 - 4) * (int) xr[16 + l];
        }
        acc = (d * (float) sp[0]) * (float) s0 + (d * (float) sp[1]) * (float) s1;
    }
    reduce_row(acc, smem, tid, 64, out + row);
}

// ---------------------------------------------------------------- host: build the pack's bytes, then transcode

uint32_t rng_state = 0x9E3779B9u;
inline uint32_t rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// A synthetic but realistic arena: d in a plausible range, the 16 scales six-bit signed, qs and hmask free bits
// (all 8 codes appear, as they do in the pack: measured 0..7, all present).
void build_raw(RawArena& a) {
    a.w.assign((size_t) kRows * kBlocks * kBlockBytes, 0);
    for (int r = 0; r < kRows; ++r)
        for (int b = 0; b < kBlocks; ++b) {
            uint8_t* blk = a.w.data() + ((size_t) r * kBlocks + b) * kBlockBytes;
            for (int i = 0; i < 32; ++i) blk[i] = (uint8_t) rng();          // hmask
            for (int i = 0; i < 64; ++i) blk[32 + i] = (uint8_t) rng();     // qs
            for (int i = 0; i < 12; ++i) blk[96 + i] = (uint8_t) rng();     // scales
            const uint32_t d = 0x3000u + (rng() & 0x03FFu);                 // ~2^-13..2^-12
            blk[108] = (uint8_t) (d & 0xFF);
            blk[109] = (uint8_t) (d >> 8);
        }
    a.x.resize(kWidth);
    for (int i = 0; i < kWidth; ++i) a.x[i] = (int8_t) ((int) (rng() % 255) - 127);
}

// The transcode the GP100 path does, at this shape: codes 0..7 (already the Q3_K code), an int8 scale per 16
// weights (sc - 32 fits an int8 exactly), and the block's fp16 d kept as it is.
void transcode(const RawArena& a, NibArena& n) {
    n.codes.assign((size_t) kRows * kItems * 16, 0);
    n.scale.assign((size_t) kRows * kItems * 2, 0);
    n.d.assign((size_t) kRows * kBlocks, 0);
    for (int r = 0; r < kRows; ++r)
        for (int b = 0; b < kBlocks; ++b) {
            const uint8_t* blk = a.w.data() + ((size_t) r * kBlocks + b) * kBlockBytes;
            n.d[(size_t) r * kBlocks + b] = (uint16_t) (blk[108] | (blk[109] << 8));
        }
    for (int r = 0; r < kRows; ++r)
        for (int k = 0; k < kItems; ++k) {
            const int b = k / 8, half = (k % 8) / 4, j = k % 4;
            const uint8_t* blk = a.w.data() + ((size_t) r * kBlocks + b) * kBlockBytes;
            uint8_t c[32];
            item_codes(blk, half, j, c);
            uint8_t* cp = n.codes.data() + ((size_t) r * kItems + k) * 16;
            for (int l = 0; l < 16; ++l) {                          // sub 0 codes [0..15], sub 1 [16..31]
                cp[l >> 1] |= (uint8_t) ((c[l] & 15) << (4 * (l & 1)));
                cp[8 + (l >> 1)] |= (uint8_t) ((c[16 + l] & 15) << (4 * (l & 1)));
            }
            int8_t* sp = n.scale.data() + ((size_t) r * kItems + k) * 2;
            const int g0 = 8 * half + 2 * j;
            sp[0] = (int8_t) (sc6(blk + 96, g0) - 32);
            sp[1] = (int8_t) (sc6(blk + 96, g0 + 1) - 32);
        }
}

int items_ok(const NibArena& n, const RawArena& a) {
    int bad = 0;
    for (int r = 0; r < kRows; ++r)
        for (int k = 0; k < kItems; ++k) {
            const int b = k / 8, half = (k % 8) / 4, j = k % 4;
            const uint8_t* blk = a.w.data() + ((size_t) r * kBlocks + b) * kBlockBytes;
            uint8_t c[32];
            item_codes(blk, half, j, c);
            const uint8_t* cp = n.codes.data() + ((size_t) r * kItems + k) * 16;
            for (int l = 0; l < 16; ++l) {
                if (((cp[l >> 1] >> (4 * (l & 1))) & 15) != c[l]) ++bad;
                if (((cp[8 + (l >> 1)] >> (4 * (l & 1))) & 15) != c[16 + l]) ++bad;
            }
            const int8_t* sp = n.scale.data() + ((size_t) r * kItems + k) * 2;
            const int g0 = 8 * half + 2 * j;
            if (sp[0] != (int8_t) (sc6(blk + 96, g0) - 32) || sp[1] != (int8_t) (sc6(blk + 96, g0 + 1) - 32)) ++bad;
        }
    return bad;
}

float time_kernel(int which, const void* p0, const void* p1, const void* p2, const void* px, float* dout,
                  int iters) {
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) {   // warm-up
        if (which == 0)
            hipLaunchKernelGGL(row_q3k, dim3(kRows), dim3(64), 0, 0, (const uint8_t*) p0, (const int8_t*) px, dout);
        else
            hipLaunchKernelGGL(row_nib, dim3(kRows), dim3(64), 0, 0, (const uint8_t*) p0, (const int8_t*) p1,
                               (const uint16_t*) p2, (const int8_t*) px, dout);
    }
    HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) {
        if (which == 0)
            hipLaunchKernelGGL(row_q3k, dim3(kRows), dim3(64), 0, 0, (const uint8_t*) p0, (const int8_t*) px, dout);
        else
            hipLaunchKernelGGL(row_nib, dim3(kRows), dim3(64), 0, 0, (const uint8_t*) p0, (const int8_t*) p1,
                               (const uint16_t*) p2, (const int8_t*) px, dout);
    }
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    RawArena a;
    build_raw(a);
    NibArena n;
    transcode(a, n);
    const int bad = items_ok(n, a);
    const double weights = (double) kRows * kWidth;
    std::printf("rows %d x width %d = %.2f M weights\n", kRows, kWidth, weights / 1e6);
    std::printf("Q3_K bytes  %zu (%.3f B/weight)\n", a.w.size(), (double) a.w.size() / weights);
    const double nb = (double) n.codes.size() + n.scale.size() + 2.0 * n.d.size();
    std::printf("4-bit bytes %zu + %zu + %zu = %.0f (%.3f B/weight, %.2fx)\n", n.codes.size(), n.scale.size(),
                2 * n.d.size(), nb, nb / weights, nb / (double) a.w.size());
    std::printf("transcode check: %d mismatched weight(s)/scale(s)\n", bad);

    uint8_t* dw = nullptr;
    int8_t* dx = nullptr;
    uint8_t* dc = nullptr;
    int8_t* ds = nullptr;
    uint16_t* dd = nullptr;
    float* dout = nullptr;
    HIP_CHECK(hipMalloc(&dw, a.w.size()));
    HIP_CHECK(hipMalloc(&dx, a.x.size()));
    HIP_CHECK(hipMalloc(&dc, n.codes.size()));
    HIP_CHECK(hipMalloc(&ds, n.scale.size()));
    HIP_CHECK(hipMalloc(&dd, n.d.size() * 2));
    HIP_CHECK(hipMalloc(&dout, kRows * sizeof(float)));
    HIP_CHECK(hipMemcpy(dw, a.w.data(), a.w.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, a.x.data(), a.x.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dc, n.codes.data(), n.codes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds, n.scale.data(), n.scale.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dd, n.d.data(), n.d.size() * 2, hipMemcpyHostToDevice));

    const float ms_q = time_kernel(0, dw, nullptr, nullptr, dx, dout, iters);
    std::vector<float> out_q(kRows);
    HIP_CHECK(hipMemcpy(out_q.data(), dout, kRows * sizeof(float), hipMemcpyDeviceToHost));

    const float ms_n = time_kernel(1, dc, ds, dd, dx, dout, iters);
    std::vector<float> out_n(kRows);
    HIP_CHECK(hipMemcpy(out_n.data(), dout, kRows * sizeof(float), hipMemcpyDeviceToHost));

    int diff = 0;
    for (int i = 0; i < kRows; ++i)
        if (out_q[i] != out_n[i]) ++diff;

    std::printf("\n%-22s %10.3f us  %8.3f ns/weight  %7.1f Mweights/s\n", "Q3_K decode+dot", ms_q * 1e3,
                ms_q * 1e6 / weights, weights / (ms_q * 1e3));
    std::printf("%-22s %10.3f us  %8.3f ns/weight  %7.1f Mweights/s\n", "4-bit code decode+dot", ms_n * 1e3,
                ms_n * 1e6 / weights, weights / (ms_n * 1e3));
    std::printf("\n4-bit vs Q3_K: %.2fx decode time, %.2fx throughput\n", ms_n / ms_q, ms_q / ms_n);
    std::printf("row results identical: %s (%d of %d differ)\n", diff == 0 ? "yes" : "NO", diff, kRows);

    HIP_CHECK(hipFree(dw));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dc));
    HIP_CHECK(hipFree(ds));
    HIP_CHECK(hipFree(dd));
    HIP_CHECK(hipFree(dout));
    return bad == 0 && diff == 0 ? 0 : 2;
}
