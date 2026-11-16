// include/strata/kernels/kv_turbo.hpp - Turbo3 and Rot4 KV storage for the QSA layers.
//
// `--kv turbo3` and `--kv rot4`, ported from Kraken-x (third_party/Kraken-x/include/krk/turbo_kv.hpp,
// TurboQuant 3-bit, arxiv 2504.19874). Both are 3-bit codecs over the SAME orthonormal Hadamard rotation the
// q4_0 path already uses (kv_q4.hpp), so the query/output rotation points, `st.kv_rot`, the gather-to-fp16
// scratch and therefore the whole attention path are unchanged: this is a codec swap, not a new pipeline.
//
// WHY 3 BITS WITH A LLOYD-MAX CODEBOOK AND NOT A UNIFORM 4-BIT GRID. After an orthonormal rotation every
// coordinate of a head is ~N(0, sigma^2) with the same sigma, so the coordinate distribution is known and a
// Lloyd-Max quantizer can put its 8 levels where the density is. Kraken-x measured the uniform 4-bit grid at
// LOWER MSE (0.0089 vs 0.033) and still breaking the model at the second token, because a max-derived step
// spreads 15 levels over +-max and leaves the core of the distribution - where the attention energy is - on
// about five coarse levels. The codebook below is the textbook N(0,1) 3-bit Lloyd-Max solution
// (kTurboCentroid), reproduced from Kraken-x rather than re-derived; turbo_kv_selftest there re-solves it and
// checks the reproduction, and our round-trip parity check does the same for this port.
//
// DIFFERENCE FROM KRAKEN-X, deliberately. Kraken-x seeds a per-(layer, kv head) +-1 sign pattern before the
// FWHT so a fixed basis cannot align with outliers. This port uses Strata's existing unseeded FWHT256 for both
// formats, because that transform is already verified bit-for-bit against the q4_0 path and is shared with the
// query and output rotations (fwht256_inplace_cuda). An unseeded orthonormal Hadamard still satisfies the
// concentration the codebook needs (entries +-1/sqrt(d), sub-Gaussian); a seeded variant can be added later
// without moving the layout, by folding the signs into the same three call sites.
//
// LAYOUT, per cell (one token) per KV head, at Strata's head_dim = 256:
//
//   turbo3  [ f32 sigma ][ 32 x 3 B ]                                   = 100 B  (0.391 B/value)
//           one sigma per HEAD ROW (sigma = ||x|| / sqrt(d), the rotated coordinate's standard deviation), then
//           3-bit codes packed 8 per group, group g in bytes 4 + 3g .. 4 + 3g + 2, little-endian bit order.
//           NOTE: Kraken-x stores this scale at the END of the row; this port stores it FIRST so the f32 is
//           4-aligned. Byte-compatible with Kraken-x's *size*, not with its byte order - if you diff against
//           third_party/Kraken-x, compare after accounting for the 4-byte header.
//
//   rot4    [ 8 x ( f16 scale ][ 4 x 3 B ] ) ]                          = 112 B  (0.438 B/value)
//           one f16 scale per 32-value block (the block's RMS, not its max), then four groups of 8 values
//           packed in 3 bytes each.
//
// Against f16's 512 B/head that is 5.1x and 4.6x; against the already-shipped int8 (264 B) 2.64x and 2.36x,
// and against q4_0 (144 B) 1.44x and 1.29x. turbo3 is the smaller of the two and is Kraken-x's default.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// Lloyd-Max centroids for N(0,1) at 3 bits (8 levels), and their decision boundaries. Textbook values;
/// Kraken-x's turbo_kv_selftest re-solves the fixed point and checks these rather than trusting the copy.
inline constexpr int kTurboBits = 3;
inline constexpr int kTurboLevels = 1 << kTurboBits;
inline constexpr float kTurboCentroid[kTurboLevels] = {
    -2.15194566f, -1.34390926f, -0.75600525f, -0.24509420f,
     0.24509420f,  0.75600525f,  1.34390926f,  2.15194566f};
inline constexpr float kTurboBoundary[kTurboLevels - 1] = {
    -1.74793313f, -1.05003882f, -0.50062380f, 0.0f, 0.50062380f, 1.05003882f, 1.74793313f};

/// Bytes of one head's turbo3 row: ceil(d/8)*3 code bytes + one f32 scale. d = 256 -> 100.
inline uint64_t kv_turbo_bytes_per_head(int head_dim) {
    return (uint64_t) ((head_dim + 7) / 8) * 3u + 4u;
}
/// Bytes of one head's rot4 row: (d/32) blocks of (f16 scale + 12 code bytes). d = 256 -> 112.
inline uint64_t kv_rot4_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / 32) * 14u; }
/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_turbo_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_turbo_bytes_per_head((int) s.head_dim) * 2;
}
inline uint64_t kv_rot4_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_rot4_bytes_per_head((int) s.head_dim) * 2;
}

/// Append the already-rotated cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident. Same contract as kv_append_q4_step.
void kv_append_turbo_step(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                          const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                          const KvHostPools* host = nullptr);
void kv_append_turbo_steps(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                           int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                           void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, head_dim]; also into `stage`
/// (the identity layout of a one-layer staging pool) when given.
void kv_append_turbo(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, int64_t pos0, int64_t T,
                     const float* K, const float* V, const QsaShapes& s, void* stream,
                     const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
/// The attention kernels are unchanged because they read this scratch, exactly as for q4_0.
void kv_gather_turbo_step(const uint8_t* k_turbo, const uint8_t* v_turbo, const int32_t* page_table,
                          const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                          uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

/// The rot4 codec, same API and the same rotation, a different block quantizer.
void kv_append_rot4_step(uint8_t* k_rot4, uint8_t* v_rot4, const int32_t* page_table, const int32_t* step,
                         const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                         const KvHostPools* host = nullptr);
void kv_append_rot4_steps(uint8_t* k_rot4, uint8_t* v_rot4, const int32_t* page_table, const int32_t* step,
                          int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                          void* stream, const KvHostPools* host = nullptr);
void kv_append_rot4(uint8_t* k_rot4, uint8_t* v_rot4, const int32_t* page_table, int64_t pos0, int64_t T,
                    const float* K, const float* V, const QsaShapes& s, void* stream,
                    const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);
void kv_gather_rot4_step(const uint8_t* k_rot4, const uint8_t* v_rot4, const int32_t* page_table,
                         const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                         uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
