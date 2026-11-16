// include/strata/kernels/latent.hpp - the latent MoE's engine-side kernels (new file).
//
// The A part reuses native_expert_grouped (40,40,ELL,I); these two kernels cover the
// only latent-specific data movement left in the verify graph:
//   * latent_plan_rows: rewrite the GPU plan entries' activation rows so A reads the
//     q8_1 rows of xl (row = token*G + label[expert]) instead of the plain token rows.
//   * latent_u_grouped: U for every routed entry: y[r] = yl[r] @ U[labels[ids[r]]].T.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// tok[j] = (base_token + dst[j]/K) * G + labels[ids[dst[j]]] for j < counts[1] (all GPU entries).
/// `ids` points at this window group's ids block; labels = the layer's 512 group labels (device).
void latent_plan_rows(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                      const int32_t* labels_dev, int64_t E, int base_token, int G, int K, void* stream);

/// One block per routed entry (cap_entries of them): quantized yl rows (q8_1) times the route's
/// U[group] slice (Q8_0 or Q4_0 rows [G*H, ELL], fmt 8 or 4); the H-wide result lands at out[r*H .. r*H+H].
/// `n_groups` (= lat_G, the distinct U matrices a layer has) switches to the weight-reuse kernel, which
/// keeps an output column's weights in registers and reuses them over every routed row of that group;
/// 0 keeps the per-row kernel (verify.cpp's and the parity harness's behaviour is then unchanged).
void latent_u_grouped(const uint8_t* u_q8, const uint8_t* yl_q8, const int32_t* ids, const int32_t* labels_dev,
                      int64_t E, int64_t cap_entries, int ELL, int H, float* out, void* stream, int fmt = 8,
                      int n_groups = 0);

/// The weight-reuse U projection (latent_u_grouped's documented `n_groups` variant, made explicit and
/// tunable): the same outputs, bit for bit, from blocks that hold one output column's weights in
/// registers and reuse them over a RUN of R routed rows sharing that column's label.  The per-row kernel
/// re-reads a group's whole U matrix once per routed row (~20 GB a layer at the engine's shape); R = 8
/// cuts that ~4x while keeping ~200k independent blocks.  `scratch` is a device buffer of at least
/// 1024 + 4*(G) + 4*n bytes (the label-bucketed row list, built here so no host upload is needed).
void latent_u_runs(const uint8_t* u_q8, const uint8_t* yl_q8, const int32_t* ids, const int32_t* labels_dev,
                   int64_t E, int64_t n, int ELL, int H, float* out, void* scratch, void* stream, int fmt,
                   int R, int G);

/// The staged U tile: 256 output columns x 64 routed rows a block, both operands staged into 4-byte
/// aligned shared rows (64 codes + two half scales) and the scale applied once per 16-value sub-block,
/// so the inner loop is four __dp4a per sub-block on aligned smem words instead of the per-row kernel's
/// ~9 instructions a 4-value term.  Same outputs as latent_u_grouped up to fp32 rounding (the int32 sums
/// are exact; the scale is hoisted from 4 values to 16).  23,040 B of smem a block, so 2 blocks a CU.
/// `scratch` is the same buffer latent_u_runs takes.  Requires fmt 8, ELL % 64 == 0.
void latent_u_tiled(const uint8_t* u_q8, const uint8_t* yl_q8, const int32_t* ids, const int32_t* labels_dev,
                    int64_t E, int64_t n, int ELL, int H, float* out, void* scratch, void* stream, int fmt,
                    int G);

/// The DECODE-shaped tile: one block = (a label's <= 16 routed rows) x 256 columns, one lane a column, no
/// cross-lane reduction.  Decode's groups carry ~10 rows, where the 64-row tile pads 64 rows for 1 and
/// stages more than the per-row kernel; this one issues ~150 blocks a layer instead of ~3,200.  Same
/// outputs as latent_u_grouped up to fp32 rounding (scale per 16 values instead of per 4).
void latent_u_tiles(const uint8_t* u_q8, const uint8_t* yl_q8, const int32_t* ids, const int32_t* labels_dev,
                    int64_t E, int64_t n, int ELL, int H, float* out, void* scratch, void* stream, int fmt,
                    int G);

/// The same staged tile for the D projection (no groups: every token against the whole [n_out, K]
/// matrix).  `out` is [M, n_out] fp32, the layout the caller then quantizes to the xl rows' q8_1.
void latent_d_tiled(const uint8_t* w_q8, const uint8_t* xq, int64_t M, int K, int64_t n_out, float* out,
                    void* stream, int fmt);

/// knife-0 (STRATA_LATENT_U_GPU): the CPU share's U computed on the GPU into parts_ directly.
/// One block per routed entry; a GPU row (r in dst[0..count)) copies hit_out, a CPU row computes U[labels[ids[r]]] . hq[r].
void latent_u_matvec(const uint8_t* u_q8, const uint8_t* hq, const int32_t* ids, const int32_t* labels_dev,
                     const int32_t* dst, const int32_t* count, int64_t E, int64_t n_entries, int ELL, int H,
                     int G, float* parts, void* stream, int fmt = 8);

}  // namespace strata::kernels