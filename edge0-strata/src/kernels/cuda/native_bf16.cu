#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/window_limits.hpp"
#include "s26_tsum.cuh"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace strata::kernels {
namespace {

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
__device__ __forceinline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

// --- RDNA2 fp8 dot2 path (STRATA_DOT2) --------------------------------------------------------
// Weights are held as fp8 (e4m3, 1 B/weight - half the bf16 scalar path's traffic) and unpacked to
// fp16 on the fly; activations are converted to fp16 by V_CVT_PKRTZ_F16_F32 (RTZ). V_DOT2_F32_F16
// multiplies two packed fp16 pairs and accumulates in f32: D = a0*w0 + a1*w1 + acc. The products of
// two fp16s fit exactly in f32. Error vs the scalar path: the weights (e4m3, ~2^-4 relative) and the
// activations (fp16 RTZ, ~2^-11 relative); the sums of the random-sign products stay within a few %.
//
// NUMERICAL NOTE: the accumulate does V_DOT2 first and then adds the accumulator, i.e.
// fl(p0 + p1) then + acc, whereas the scalar kernel does fl(p0 + acc) then + p1. Both round to f32
// at each step; they differ by a few ulps, NOT bit-exact. The entries below keep the scalar kernel
// as reference used whenever STRATA_DOT2 is off or a weight cannot be represented in fp8 (|w| >
// 448, or non-finite): that escape hatch is decided by a maxabs check at first conversion.
__device__ __forceinline__ uint32_t dot2_cvt_pair(float x0, float x1) {
#if defined(__gfx1031__) || defined(__gfx1030__) || defined(__gfx1032__) || defined(__gfx1100__) || \
    defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    uint32_t r;
    __asm__("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(r) : "v"(x0), "v"(x1));
    return r;
#else
    return (uint32_t) (f16_from_f32(x1) << 16) | (uint32_t) f16_from_f32(x0);   // gfx-agnostic fallback: RNE instead of RTZ
#endif
}

__device__ __forceinline__ float dot2_fma(uint32_t ap, uint32_t wp, float acc) {
#if defined(__gfx1031__) || defined(__gfx1030__) || defined(__gfx1032__) || defined(__gfx1100__) || \
    defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    float r;
    __asm__("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(ap), "v"(wp), "v"(acc));
    return r;
#else
    // bit-exact-enough fallback (no ISA dot2 on the guest target): fp16 value exact in f32
    const float a0 = f32_from_f16((uint16_t) ap), a1 = f32_from_f16((uint16_t) (ap >> 16));
    const float w0 = f32_from_f16((uint16_t) wp), w1 = f32_from_f16((uint16_t) (wp >> 16));
    return __fmaf_rn(a0, w0, __fmaf_rn(a1, w1, acc));
#endif
}

// e4m3 (OCP) fp8 -> fp16 bits, exact for normals; the tiny subnormals are handled exactly too,
// so the fp8 path has no unpacking bias beyond the fp8 quantization itself.
__device__ __forceinline__ uint16_t fp8_to_f16(uint32_t b) {
    const uint32_t s = (b >> 7) << 15;
    const uint32_t e4 = (b >> 3) & 0xfu;
    const uint32_t m = b & 0x7u;
    uint32_t bits;
    if (e4 != 0) bits = s | ((e4 + 8) << 10) | (m << 7);
    else if (m == 0) bits = s;
    else if (m >= 4) bits = s | (8 << 10) | ((m - 4) << 8);
    else if (m == 1) bits = s | (6 << 10);
    else bits = s | (7 << 10) | ((m - 2) << 9);
    return (uint16_t) bits;
}

// Single-row kernel mirroring bf16_f32_mmvf_kernel's reductions (same warp/block partial structure).
// inner loop: ld w8 pair (2 bytes, unpacked to fp16), ld input pair (f32), cvt activation pair
// (1 instr), dot2 (1 instr). Weight traffic is 1 B/weight (half the bf16 scalar path).
template <int BLOCK_SIZE>
__global__ void bf16_f32_dot2_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w8,
                                     float* __restrict__ y, int n_in) {
    const int t = threadIdx.x;
    const uint16_t* row2 = reinterpret_cast<const uint16_t*>(w8 + (size_t) blockIdx.x * n_in);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if (t < 32) partials[t] = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint16_t wp8 = row2[pair];
        const uint32_t wp = ((uint32_t) fp8_to_f16(wp8 >> 8) << 16) | fp8_to_f16(wp8 & 0xff);
        const float2 input = inputs2[pair];
        const uint32_t ap = dot2_cvt_pair(input.x, input.y);
        acc = dot2_fma(ap, wp, acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

// Multi-row variant mirroring bf16_f32_mmvf_multi_kernel's EXACT_T=true shape (same pair walk, same
// warp and block reductions; every output identical to a BF16_F32_DOT2_kernel of its own row).
template <int BLOCK_SIZE, int NT>
__global__ void bf16_f32_dot2_multi_kernel(const float* x_, int64_t ldx, const uint8_t* __restrict__ w8,
                                           float* y_, int64_t ldy, int n_in, int n_tok) {
    const float* STRATA_PDL_RESTRICT x = x_;
    float* STRATA_PDL_RESTRICT y = y_;
    const int t = threadIdx.x;
    const uint16_t* row2 = reinterpret_cast<const uint16_t*>(w8 + (size_t) blockIdx.x * n_in);
    __shared__ float partials[NT][32];
    if (t < 32) {
#pragma unroll
        for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
    }
    __syncthreads();
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint16_t wp8 = row2[pair];
        const uint32_t wp = ((uint32_t) fp8_to_f16(wp8 >> 8) << 16) | fp8_to_f16(wp8 & 0xff);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float2 input = __ldg(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + pair);
                const uint32_t ap = dot2_cvt_pair(input.x, input.y);
                acc[k] = dot2_fma(ap, wp, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k)
        if (k < n_tok) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (k < n_tok) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (k < n_tok) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
}

// (dot2_w_cast_kernel, the fp16 weight cache and dot2_on moved to namespace strata::kernels scope
// below the anonymous namespace: CUDA/HIP drops the __global__ attribute on a new-listed
// non-template kernel declared inside an anonymous namespace, which makes the <<<>>> launch
// parse as a binary shift.)

template <int BLOCK_SIZE>
__global__ void bf16_f32_mmvf_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int n_in) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if (t < 32) partials[t] = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT, bool EXACT_T = true>
__global__ void bf16_f32_mmvf_multi_kernel(const float* x_, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* y_, int64_t ldy, int n_in, int n_tok) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[NT][32];
    if (t < 32) {
#pragma unroll
        for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
    }
    __syncthreads();
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    // PDL (pdl.hpp): this thread's first PRE weight pairs are loaded before waiting for the activations
    constexpr int PRE = kPdlPrefetch ? 8 : 0;
    uint32_t wpre[PRE > 0 ? PRE : 1];
#pragma unroll
    for (int q = 0; q < PRE; ++q) {
        const int pair = t + q * BLOCK_SIZE;
        wpre[q] = pair < n_in / 2 ? __ldg(weights2 + pair) : 0u;
    }
    pdl_wait();
    int q = 0;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE, ++q) {
        const uint32_t weight = q < PRE ? wpre[q] : __ldg(weights2 + pair);
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (EXACT_T || k < n_tok) {
                const float2 input = __ldg(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + pair);
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k)
        if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (EXACT_T || k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
}

// S25 (STRATA_MMVF_ROWS=1): RPB output rows per block. Each block read its row's weights once but all T activation
// rows again (n_out blocks x T x n_in floats through L2 - the part that grew with T); here a block reads the
// activations once for RPB rows. Per output, thread t still walks pairs t, t + BLOCK_SIZE, ... with the same two
// ordered FMAs and the same warp and block reductions: bit-identical to bf16_f32_mmvf_multi_kernel.
// S26 STRATA_LFUSE=1 (AUX): one more block computes row 0 of w_aux into y_aux[k * ldy_aux] - the same per-output
// code, so each aux output is bitwise what its own bf16_f32_mmvf_multi_kernel launch gave (the shared expert's gate)
template <int BLOCK_SIZE, int NT, int RPB, bool TS = false, bool AUX = false>
__global__ void bf16_f32_mmvf_rows_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ y, int64_t ldy, int n_in, int n_out, int n_tok,
                                          const uint16_t* __restrict__ w_aux = nullptr, float* __restrict__ y_aux = nullptr,
                                          int64_t ldy_aux = 0) {
    const int t = threadIdx.x;
    int o0 = blockIdx.x * RPB;
    if constexpr (AUX) {
        if (blockIdx.x == gridDim.x - 1) { w = w_aux; y = y_aux; ldy = ldy_aux; n_out = 1; o0 = 0; }
    }
    __shared__ float partials[RPB][NT][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t] = 0.0f;
        __syncthreads();
    }
    float acc[RPB][NT];
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        float2 in[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) in[k] = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            if (o0 + r >= n_out) break;
            const uint32_t weight = reinterpret_cast<const uint32_t*>(w + (size_t) (o0 + r) * n_in)[pair];
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k < n_tok) {
                    acc[r][k] = __fmaf_rn(w0, in[k].x, acc[r][k]);
                    acc[r][k] = __fmaf_rn(w1, in[k].y, acc[r][k]);
                }
            }
        }
    }
    if constexpr (TS) {   // S26 STRATA_TSUM=1: the RPB x NT warp sums (both stages) as transposed butterflies, bitwise the same
        constexpr int V = RPB * NT, P = s26ts::pow2_ceil(V);
        const int lane = t & 31, j = s26ts::tsum_token<P>(lane);
        const bool own = lane == s26ts::tsum_lane<P>(j) && j < V;
        float v[P];
#pragma unroll
        for (int q = 0; q < P; ++q) v[q] = q < V ? acc[q / NT][q % NT] : 0.0f;
        float sum = s26ts::tsum<P>(v, lane);
        if constexpr (BLOCK_SIZE > 32) {
            if (own) partials[j / NT][j % NT][t / 32] = sum;
            __syncthreads();
            if (t >= 32) return;
#pragma unroll
            for (int q = 0; q < P; ++q) v[q] = q < V ? partials[q / NT][q % NT][t] : 0.0f;
            sum = s26ts::tsum<P>(v, lane);
        }
        const int r = j / NT, k = j % NT;
        if (own && o0 + r < n_out && k < n_tok) y[(size_t) k * ldy + o0 + r] = sum;
        return;
    }
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(acc[r][k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t / 32] = acc[r][k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(partials[r][k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int r = 0; r < RPB; ++r)
            if (o0 + r < n_out)
#pragma unroll
                for (int k = 0; k < NT; ++k)
                    if (k < n_tok) y[(size_t) k * ldy + o0 + r] = acc[r][k];
}

template <int B>
void launch_rows(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int n_in, int n_out, int n_tok,
                 cudaStream_t st, const uint16_t* w_aux = nullptr, float* y_aux = nullptr, int64_t ldy_aux = 0) {
    constexpr int RPB = 4;
    const unsigned nb = (unsigned) ((n_out + RPB - 1) / RPB);
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (w_aux != nullptr) {   // S26 STRATA_LFUSE: + one aux row block
#define S26_AUX(TSV) \
        if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB, TSV, true><<<nb + 1, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux, y_aux, ldy_aux); \
        else bf16_f32_mmvf_rows_kernel<B, 8, RPB, TSV, true><<<nb + 1, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux, y_aux, ldy_aux);
        if (ts) { S26_AUX(true) } else { S26_AUX(false) }
#undef S26_AUX
        return;
    }
    if (ts) {
        if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB, true><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
        else bf16_f32_mmvf_rows_kernel<B, 8, RPB, true><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
        return;
    }
    if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
    else bf16_f32_mmvf_rows_kernel<B, 8, RPB><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

// f32 -> e4m3 (OCP) with round-to-nearest from the bit pattern; tiny magnitudes flush to zero
// (an fp8 subnormal contributes <= 2^-6 absolute, far below the fp8 quantization error).
__device__ __forceinline__ uint8_t f32_to_e4m3_rn(float f) {
    const uint32_t b = __float_as_uint(f);
    const uint32_t sign = (b >> 24) & 0x80u;
    const uint32_t exp8 = (b >> 23) & 0xffu;
    const uint32_t m = b & 0x7fffffu;
    int e = (int) exp8 - 127;
    if (e >= 9) return (uint8_t) (sign | 0x7eu);      // saturate to 448 (max normal)
    if (e <= -7) return (uint8_t) sign;              // flush to +-zero
    uint32_t m3 = (m + 0x80000u) >> 20;               // round-to-nearest from the truncated bits
    if (m3 >= 8u) { m3 = 0; ++e; }
    if (e >= 9) return (uint8_t) (sign | 0x7eu);      // rounded up off the top normal
    return (uint8_t) (sign | ((uint32_t) (e + 7) << 3) | (m3 & 0x7u));
}

// fp8 weight cache: dense bf16 weights are process-lifetime constants. On the first use of a pointer
// they are converted (bf16 -> f32 -> e4m3, 1 B/weight) and cached; a tensor that holds a value that
// does not fit e4m3 (|w| > 448 / inf / NaN) is marked unusable forever, and the caller falls back to
// the scalar kernel. That keeps the dot2 path a pure win, and the images SHRINK the VRAM footprint
// (1 B vs 2 B per weight) instead of duplicating it.
template <int R>
__global__ void dot2_e4m3_cast_kernel(const uint16_t* __restrict__ b, uint8_t* __restrict__ h, int64_t count,
                                      unsigned int* __restrict__ bad) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float f = f32_from_bf16(b[i]);
    if (!(abs(f) <= 448.0f)) if (bad) atomicExch(bad, 1u);   // also catches inf/NaN
    h[i] = f32_to_e4m3_rn(f);
}

struct Fp8wCache {
    std::mutex m;
    std::unordered_map<uintptr_t, uint8_t*> ptr;   // bf16 ptr -> fp8 ptr
    std::unordered_map<uintptr_t, bool> no;         // bf16 ptr -> unusable (range check failed)
};

// Returns the fp8 image of `w[count]`, or nullptr when it cannot be represented.
// Conversion always runs to completion on a dedicated side stream, never on the caller's: the first
// GEMV of a window runs inside the verifier's stream CAPTURE, where CUDA calls of any kind are illegal.
// Therefore this function must only be called outside a capture (warm_fp16_cache); the capture-time
// dispatch uses fp8_cached() instead, a plain host-side lookup that never touches CUDA.
uint8_t* fp8_of(const uint16_t* w, int64_t count) {
    static Fp8wCache c;
    static cudaStream_t side = [&] {
        cudaStream_t ss = nullptr;
        if (cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking) != cudaSuccess) { cudaGetLastError(); ss = nullptr; }
        return ss;
    }();
    const uintptr_t key = reinterpret_cast<uintptr_t>(w);
    {
        std::lock_guard<std::mutex> lk(c.m);
        auto f = c.ptr.find(key);
        if (f != c.ptr.end()) return f->second;
        if (c.no.find(key) != c.no.end()) return nullptr;
    }
    if (side == nullptr) return nullptr;
    uint8_t* d = nullptr;
    unsigned int* bad = nullptr;
    cudaError_t e = hipMallocAsync(&d, (size_t) count * sizeof(uint8_t), side);
    if (e != cudaSuccess) { hipGetLastError(); return nullptr; }
    e = hipMallocAsync(&bad, sizeof(unsigned int), side);
    if (e != cudaSuccess) { hipFreeAsync(d, side); hipGetLastError(); return nullptr; }
    e = hipMemsetAsync(bad, 0, sizeof(unsigned int), side);
    if (e != cudaSuccess) { hipFreeAsync(d, side); hipFreeAsync(bad, side); hipGetLastError(); return nullptr; }
    const int64_t grid = (count + 255) / 256;
    dot2_e4m3_cast_kernel<0><<<(unsigned) grid, 256, 0, side>>>(w, d, count, bad);
    e = hipStreamSynchronize(side);
    if (e != cudaSuccess) { hipFreeAsync(d, side); hipFreeAsync(bad, side); hipGetLastError(); return nullptr; }
    unsigned int badv = 0;
    e = hipMemcpyAsync(&badv, bad, sizeof(unsigned int), hipMemcpyDeviceToHost, side);
    if (e != cudaSuccess) { hipFreeAsync(d, side); hipFreeAsync(bad, side); hipGetLastError(); return nullptr; }
    e = hipStreamSynchronize(side);
    hipFreeAsync(bad, side);
    if (e != cudaSuccess) { hipFreeAsync(d, side); hipGetLastError(); return nullptr; }
    {
        std::lock_guard<std::mutex> lk(c.m);
        if (badv != 0 || c.no.find(key) != c.no.end()) { hipFreeAsync(d, side); c.no[key] = true; return nullptr; }
        auto f = c.ptr.find(key);
        if (f != c.ptr.end()) { hipFreeAsync(d, side); return f->second; }   // a racing caller won the conversion
        c.ptr[key] = d;
        return d;
    }
}

bool dot2_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DOT2");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

// Capture-safe lookup: nullptr when the fp8 image is not ready. No CUDA calls, no sync, no
// allocations - just a map lookup - so it is legal from inside the verifier's stream capture.
uint8_t* fp8_cached(const uint16_t* w) {
    static Fp8wCache c;
    const uintptr_t key = reinterpret_cast<uintptr_t>(w);
    std::lock_guard<std::mutex> lk(c.m);
    auto f = c.ptr.find(key);
    return f != c.ptr.end() ? f->second : nullptr;
}

bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream) {
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    // exactly the conditions under which both calls would take the rows / multi kernels at the same block size
    if (!rows || n_out < 64 || n_tok < 2 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || w_aux == nullptr || y_aux == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        return false;
    const cudaStream_t st = (cudaStream_t) stream;
    const int ni = (int) n_in, no = (int) n_out;
    switch (mmvf_block_size(n_in)) {
        case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
    }
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi_aux launch: ") + cudaGetErrorString(result));
    return true;
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > STRATA_VERIFY_MAX_T || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 ||
        x == nullptr || w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows (16 in a wide build), even n_in/ldx, aligned pointers");
    const cudaStream_t st = (cudaStream_t) stream;
    if (dot2_on() && n_tok <= 8) {
        if (uint8_t* w8 = fp8_cached(w)) {
            const int ni = (int) n_in;
#define STRATA_DOT2_M(N) case N: \
    switch (n_tok) { \
        case 1: bf16_f32_dot2_multi_kernel<N, 1><<<dim3((unsigned) n_out), dim3(N), 0, st>>>(x, ldx, w8, y, ldy, ni, n_tok); break; \
        case 2: bf16_f32_dot2_multi_kernel<N, 2><<<dim3((unsigned) n_out), dim3(N), 0, st>>>(x, ldx, w8, y, ldy, ni, n_tok); break; \
        case 3: bf16_f32_dot2_multi_kernel<N, 3><<<dim3((unsigned) n_out), dim3(N), 0, st>>>(x, ldx, w8, y, ldy, ni, n_tok); break; \
        case 4: bf16_f32_dot2_multi_kernel<N, 4><<<dim3((unsigned) n_out), dim3(N), 0, st>>>(x, ldx, w8, y, ldy, ni, n_tok); break; \
        case 5: bf16_f32_dot2_multi_kernel<N, 5><<<dim3((unsigned) n_out), dim3(N), 0, st>>>(x, ldx, w8, y, ldy, ni, n_tok); break; \
        case 6: bf16_f32_dot2_multi_kernel<N, 6><<<dim3((unsigned) n_out), dim3(N), 0, st>>>(x, ldx, w8, y, ldy, ni, n_tok); break; \
        default: bf16_f32_dot2_multi_kernel<N, 8><<<dim3((unsigned) n_out), dim3(N), 0, st>>>(x, ldx, w8, y, ldy, ni, n_tok); break; \
    } break
            switch (mmvf_block_size(n_in)) {
                STRATA_DOT2_M(32); STRATA_DOT2_M(64); STRATA_DOT2_M(96); STRATA_DOT2_M(128);
                STRATA_DOT2_M(160); STRATA_DOT2_M(192); STRATA_DOT2_M(224); STRATA_DOT2_M(256);
            }
#undef STRATA_DOT2_M
            const cudaError_t result = cudaGetLastError();
            if (result != cudaSuccess)
                throw std::runtime_error(std::string("bf16_f32_dot2_multi launch: ") + cudaGetErrorString(result));
            return;
        }
    }
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    if (rows && n_out >= 64 && n_tok <= 8) {   // (the opt-in rows kernels hold 8 columns: a wider window takes the switch below)
        const int ni = (int) n_in, no = (int) n_out;
        switch (mmvf_block_size(n_in)) {
            case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
        }
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
        return;
    }
    // 9..16 rows (wide builds only): one exact instantiation per row count, each row's dot independent of the others.  7 and 8
    // rows stay with the default case below, the instantiation they always had.
#if STRATA_VERIFY_MAX_T > 8
#define STRATA_MMVF_W(N, T) case T: launch_pdl(bf16_f32_mmvf_multi_kernel<N, T, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break;
#define STRATA_MMVF_WIDE(N) \
        STRATA_MMVF_W(N, 9) STRATA_MMVF_W(N, 10) STRATA_MMVF_W(N, 11) STRATA_MMVF_W(N, 12) \
        STRATA_MMVF_W(N, 13) STRATA_MMVF_W(N, 14) STRATA_MMVF_W(N, 15) STRATA_MMVF_W(N, 16)
#else
#define STRATA_MMVF_WIDE(N)
#endif
#define STRATA_MMVF_M(N) case N: \
    switch (n_tok) { \
        STRATA_MMVF_WIDE(N) \
        case 1: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 1, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 2: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 2, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 3: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 3, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 4: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 4, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 5: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 5, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 6: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 6, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        default: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 8, false>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
    } break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
#undef STRATA_MMVF_WIDE
#ifdef STRATA_MMVF_W
#undef STRATA_MMVF_W
#endif
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const cudaStream_t st = (cudaStream_t) stream;
    if (dot2_on()) {
        // RDNA2 dot2: the fp8 (e4m3) weight image, converted once per tensor at Verifier::init
        // (warm_fp16_cache); here we only do a host-side lookup, so this is capture-legal.
        if (uint8_t* w8 = fp8_cached(w)) {
            switch (mmvf_block_size(n_in)) {
                case 32:  bf16_f32_dot2_kernel<32><<<(unsigned) n_out, 32, 0, st>>>(x, w8, y, (int) n_in); break;
                case 64:  bf16_f32_dot2_kernel<64><<<(unsigned) n_out, 64, 0, st>>>(x, w8, y, (int) n_in); break;
                case 96:  bf16_f32_dot2_kernel<96><<<(unsigned) n_out, 96, 0, st>>>(x, w8, y, (int) n_in); break;
                case 128: bf16_f32_dot2_kernel<128><<<(unsigned) n_out, 128, 0, st>>>(x, w8, y, (int) n_in); break;
                case 160: bf16_f32_dot2_kernel<160><<<(unsigned) n_out, 160, 0, st>>>(x, w8, y, (int) n_in); break;
                case 192: bf16_f32_dot2_kernel<192><<<(unsigned) n_out, 192, 0, st>>>(x, w8, y, (int) n_in); break;
                case 224: bf16_f32_dot2_kernel<224><<<(unsigned) n_out, 224, 0, st>>>(x, w8, y, (int) n_in); break;
                default:  bf16_f32_dot2_kernel<256><<<(unsigned) n_out, 256, 0, st>>>(x, w8, y, (int) n_in); break;
            }
            const cudaError_t result = cudaGetLastError();
            if (result != cudaSuccess)
                throw std::runtime_error(std::string("bf16_f32_dot2 launch: ") + cudaGetErrorString(result));
            return;
        }
    }
#define STRATA_MMVF_CASE(N) case N: \
    bf16_f32_mmvf_kernel<N><<<(unsigned) n_out, N, 0, st>>>(x, w, y, (int) n_in); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
}



void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream) {
    // columns contiguous: the existing multi-row kernel (weight read once, each output bitwise its one-row call),
    // up to 8 rows per launch
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        if (nc == 1) bf16_gemv_fp32_mmvf(x + (size_t) c0 * n_in, w, y + (size_t) c0 * n_out, n_in, n_out, stream);
        else bf16_gemv_fp32_mmvf_multi(x + (size_t) c0 * n_in, n_in, w, y + (size_t) c0 * n_out, n_out, n_in, n_out, nc,
                                       stream);
    }
}

namespace {
bool dot2_name_ok(const std::string& name) {
    // Which tensors does bf16_gemv_fp32_mmvf(_multi) actually run on? (verify's routers, indexer
    // projections, the PLE key/value tables, the shared-expert/MTP gates). Converting EVERY bf16
    // tensor (484 of them, including the 1.17 GiB lm-head and the big hc/down/um matrices) doubles
    // the dense VRAM footprint and measured 0 MiB free + paging + a 27% slowdown. So: whitelist by
    // name, with a VRAM budget, and NEVER the head. The hc_attn/hc_ffn matrices (fused_gr reads are
    // mmvf under the hood) are opt-in with STRATA_DOT2_HC=1 when there is room.
    if (name.find("output.weight") != std::string::npos) return false;
    static const char* kNames[] = {
        "ffn_gate_inp.weight",      // MoE router
        "indexer.q_proj.weight",    // QSA indexer q
        "indexer.k_proj.weight",    // QSA indexer k
        "ple_key.weight",           // positional-library embedding
        "ple_value.weight",
        "mlp.gate.weight",          // MTP router / shared gate
        "shared",                   // shared-expert gate (if bf16)
        "attn_q.weight",            // GDN/gated projections the verifier drives through mmvf
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
    };
    for (const char* n : kNames)
        if (name.find(n) != std::string::npos) return true;
    if (std::getenv("STRATA_DOT2_HC") != nullptr) {
        if (name.find("hc_attn_") != std::string::npos || name.find("hc_ffn_") != std::string::npos) return true;
    }
    return false;
}
}  // namespace

void warm_fp16_cache(const core::WeightTable& wt) {
    // STRATA_DOT2: one-time e4m3 conversion of the whitelisted BF16 tensors, outside any stream capture.
    // Verifier::init calls this before the first window is captured; the MMVF entry points then do
    // pure host-side cache lookups and launch the RDNA2 v_dot2 kernels instead of the scalar FMA ones.
    // The images are 1 B/weight (vs 2 B bf16), so this SHRINKS the VRAM footprint.
    if (!dot2_on()) return;
    // a hard byte budget: stop converting once the fp8 images reach it (default 512 MiB)
    uint64_t budget = (uint64_t) 512 * 1024 * 1024;
    if (const char* b = std::getenv("STRATA_DOT2_MAX_MB"))
        budget = (uint64_t) std::atol(b) * 1024 * 1024;
    uint64_t used = 0;
    int64_t n_done = 0;
    for (const auto& entry : wt.all()) {
        const core::WeightRef& ref = entry.second;
        if (ref.kind != core::WeightKind::Bf16InF32 || ref.data == nullptr || ref.elements <= 0) continue;
        if (!dot2_name_ok(entry.first)) continue;
        if (used + (uint64_t) ref.elements > budget) continue;
        if (fp8_of((const uint16_t*) ref.data, ref.elements) != nullptr) {
            used += (uint64_t) ref.elements;
            ++n_done;
        }
    }
    std::fprintf(stderr, "strata dot2: warmed the fp8 (e4m3) weight cache on %lld tensors (%.1f MiB, STRATA_DOT2=1, RDNA2 v_dot2)\n",
                 (long long) n_done, (double) used / 1048576.0);
}

}  // namespace strata::kernels
