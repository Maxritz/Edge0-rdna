// src/prefill/wmma_gemm.h - RDNA3 WMMA FP16 & BF16 GEMM for Strata prefill.
#pragma once

#include <cstdint>

/// Compute Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA instructions.
/// X: T x K row-major (leading dim K), fp16 (uint16_t)
/// W: N x K row-major (leading dim K), fp16 (uint16_t)
/// Y: T x N row-major with leading dimension ldy >= N, fp32 (float)
/// Returns true if executed on WMMA, false if unsupported shape/parameters (fall back to hipblas).
bool strata_wmma_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y,
                          int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                          void* stream = nullptr);

/// Compute Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA instructions.
/// X: T x K row-major (leading dim K), bf16 (uint16_t)
/// W: N x K row-major (leading dim K), bf16 (uint16_t)
/// Y: T x N row-major with leading dimension ldy >= N, fp32 (float)
/// Returns true if executed on WMMA, false if unsupported shape/parameters (fall back to hipblas).
/// S23 (opt-in STRATA_PF_GEMM=1): Y = beta * Y + X . W^T, FP16 in, FP32 out, beta 0 or 1, K a multiple of 32,
/// T >= 64, N >= 512; the 128 x 256 / 128 x 128 grouped-order WMMA kernel. False (nothing launched) otherwise / off gfx11.
bool strata_pf_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        float beta, void* stream = nullptr);
/// The same with row strides ldx >= K and ldw >= K (multiples of 8): a 4 KB-multiple row stride (K 2048 / 4096 /
/// 6144) camps on the memory channels, K + 64 does not (S23 gemm_probe7/8: K 6144 19 -> 35 TFLOPS).
/// RDNA2 (gfx103x) has no matrix cores, so the WMMA kernels above refuse there and the dense GEMM fell to
/// rocBLAS - measured 4.0 TFLOP/s on the gdn projections (edge-0-analysis-rdna2.md 0.54), 16 % of this card's
/// 24.6 TFLOP/s fp16 peak.  This is the same product on V_DOT2_F32_F16, the packed-fp16 form RDNA2 does have:
/// two fp16 multiplies per instruction with an f32 accumulate.  Y = beta * Y + X . W^T, FP16 in, FP32 out,
/// beta 0 or 1, K a multiple of 32, T >= 32, N a multiple of 8.  NOT bit-identical to rocBLAS - a different
/// blocking and accumulation tree, which is the change the owner approved; the harness is
/// tests/hip/gemm_f16_io_parity (rel-L2 against an FP64 reference, the bound the shipped FP16 path already meets).
/// False (nothing launched) otherwise, and off gfx103x.
bool strata_pf_gemm_f16_dot2(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                             int64_t ldy, float beta, void* stream = nullptr);
/// S23 (STRATA_PF_HCDOWN=1): lo[t, n] = sum_k X[t, k] Wd[n, k] (n < nd) and inj[t, n] = sum_k X[t, k] Wi[n, k]
/// (n < ni) in one BF16 WMMA GEMM over X (T x K, token stride ldx >= K, a multiple of 8; Wd / Wi row-major, stride
/// K); lo T x nd and inj T x ni, FP32.  K a multiple of 64, T >= 64.  False (nothing launched) otherwise / off gfx11.
#if defined(STRATA_USE_HIP)
bool strata_pf_hcdown_bf16(const uint16_t* X, int64_t ldx, const uint16_t* Wd, const uint16_t* Wi, int64_t nd,
                           int64_t ni, float* lo, float* inj, int64_t T, int64_t K, void* stream = nullptr);
#else
// wmma_gemm.cu is part of the AMD builds only: another build has no such kernel (prefill.cpp then stops with its error)
inline bool strata_pf_hcdown_bf16(const uint16_t*, int64_t, const uint16_t*, const uint16_t*, int64_t, int64_t, float*, float*,
                                   int64_t, int64_t, void* = nullptr) { return false; }
#endif
/// S (STRATA_HCD_EXACT=1): Y[t, n] = sum_k X[t, k] W[n, k] for N 320, K 10240 (BF16, FP32 out, X token stride ldx >= K, a
/// multiple of 8), bitwise equal to hipBLASLt solution 1176 / 1177 for that shape (its k order, StaggerU included).
/// False (nothing launched) for any other shape / off gfx11.
bool strata_pf_hcdown_exact_bf16(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N,
                                 int64_t K, void* stream = nullptr);
bool strata_pf_gemm_f16_ld(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T,
                           int64_t N, int64_t K, int64_t ldy, float beta, void* stream = nullptr);
/// True when the products above can run on the current device (the gfx11 matrix cores).  Host code that pads an
/// activation's row stride for them (STRATA_PF_PAD, prefill.cpp) must ask this first: on another card no kernel takes
/// a padded X, and Gemm::native would have to copy the rows unpadded.
#if defined(STRATA_USE_HIP)
bool strata_pf_padded_gemm_available();
#else
inline bool strata_pf_padded_gemm_available() { return false; }
#endif

bool strata_wmma_gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream = nullptr);
