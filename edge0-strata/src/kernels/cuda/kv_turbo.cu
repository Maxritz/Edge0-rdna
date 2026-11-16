// src/kernels/cuda/kv_turbo.cu - see include/strata/kernels/kv_turbo.hpp. Turbo3 and Rot4 KV codecs over the
// same orthonormal Hadamard rotation the q4_0 path uses, so the attention path (which reads the gathered FP16
// scratch) is untouched.
#include "strata/kernels/kv_turbo.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_turbo: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The row byte counts as DEVICE constants: the header's helpers are host-side (as kv_q4.hpp's are), so a kernel
// cannot call them. Kept next to each other so the two definitions cannot drift.
__device__ __forceinline__ long long turbo_row_bytes(int d) { return (long long) ((d + 7) / 8) * 3 + 4; }
__device__ __forceinline__ long long rot4_row_bytes(int d) { return (long long) (d / 32) * 14; }

// The nearest Lloyd-Max level for a standardised coordinate t. Eight levels is a handful of compares; a binary
// search over seven boundaries would not be faster and branches just as unpredictably.
__device__ __forceinline__ int turbo_nearest(float t) {
    int idx = 0;
    while (idx < kTurboLevels - 1 && t > kTurboBoundary[idx]) ++idx;
    return idx;
}

// ---------------------------------------------------------------------------------------------------------
// turbo3: one f32 sigma per HEAD ROW, then ceil(d/8)*3 code bytes. The row is one warp (head_dim = 256, so
// eight values a lane); sigma is ||x|| / sqrt(d), the standard deviation of a rotated coordinate, so the
// codebook stays the unit-variance one.
// ---------------------------------------------------------------------------------------------------------
__global__ void kv_append_turbo_kernel(uint8_t* __restrict__ k_turbo, uint8_t* __restrict__ v_turbo,
                                       const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                       int step_stride, int planes, const float* __restrict__ kcur,
                                       const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size,
                                       KvHostPools host) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, lane = threadIdx.x;   // lane: one of 32, eight values each
    const int per = head_dim / 32;
    const long long base = (long long) step_idx * kv_heads * head_dim + (long long) h * head_dim;
    const float* src = (is_v ? vcur : kcur) + base;

    float x[8];
    float n2 = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x[j] = src[lane * per + j];
        n2 += x[j] * x[j];
    }
    n2 = warp_sum(n2);
    const float sigma = sqrtf(n2) / sqrtf((float) head_dim);
    const float inv = sigma > 0.0f ? 1.0f / sigma : 0.0f;

    uint32_t acc = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc |= (uint32_t) turbo_nearest(x[j] * inv) << (3 * j);

    const long long page = (long long) table[pos / page_size];
    const long long bph = turbo_row_bytes(head_dim);
    if (page >= 0) {
        uint8_t* dst = (is_v ? v_turbo : k_turbo) + ((page * kv_heads + h) * page_size + (pos % page_size)) * bph;
        if (lane == 0) *reinterpret_cast<float*>(dst) = sigma;
        dst[4 + 3 * lane + 0] = (uint8_t) (acc & 0xFFu);
        dst[4 + 3 * lane + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
        dst[4 + 3 * lane + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
    }
    if (host.k_turbo != nullptr) {
        uint8_t* dst = (is_v ? host.v_turbo : host.k_turbo) +
                       ((pos / page_size) * kv_heads + h) * page_size * bph + (pos % page_size) * bph;
        if (lane == 0) *reinterpret_cast<float*>(dst) = sigma;
        dst[4 + 3 * lane + 0] = (uint8_t) (acc & 0xFFu);
        dst[4 + 3 * lane + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
        dst[4 + 3 * lane + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
    }
}

// The prompt path: T consecutive cells from pos0, K then V; also into the staging pool when given.
__global__ void kv_append_turbo_batch_kernel(uint8_t* __restrict__ k_turbo, uint8_t* __restrict__ v_turbo,
                                             const int32_t* __restrict__ table, int64_t pos0,
                                             const float* __restrict__ K, const float* __restrict__ V, int kv_heads,
                                             int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                             KvHostPools stage) {
    const long long t = blockIdx.x, pos = pos0 + t;
    const int h = blockIdx.y, lane = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const int per = head_dim / 32;
    const float* src = (is_v ? V : K) + t * (kv_heads * head_dim) + (long long) h * head_dim;

    float x[8];
    float n2 = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x[j] = src[lane * per + j];
        n2 += x[j] * x[j];
    }
    n2 = warp_sum(n2);
    const float sigma = sqrtf(n2) / sqrtf((float) head_dim);
    const float inv = sigma > 0.0f ? 1.0f / sigma : 0.0f;
    uint32_t acc = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc |= (uint32_t) turbo_nearest(x[j] * inv) << (3 * j);

    const long long page = (long long) table[pos / page_size];
    const long long bph = turbo_row_bytes(head_dim);
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) {
        uint8_t* dst = (is_v ? v_turbo : k_turbo) + ((page * kv_heads + h) * page_size + (pos % page_size)) * bph;
        if (lane == 0) *reinterpret_cast<float*>(dst) = sigma;
        dst[4 + 3 * lane + 0] = (uint8_t) (acc & 0xFFu);
        dst[4 + 3 * lane + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
        dst[4 + 3 * lane + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
    }
    if (host.k_turbo != nullptr) {
        uint8_t* dst = (is_v ? host.v_turbo : host.k_turbo) + row_id * bph;
        if (lane == 0) *reinterpret_cast<float*>(dst) = sigma;
        dst[4 + 3 * lane + 0] = (uint8_t) (acc & 0xFFu);
        dst[4 + 3 * lane + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
        dst[4 + 3 * lane + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
    }
    if (stage.k_turbo != nullptr) {
        uint8_t* dst = (is_v ? stage.v_turbo : stage.k_turbo) + row_id * bph;
        if (lane == 0) *reinterpret_cast<float*>(dst) = sigma;
        dst[4 + 3 * lane + 0] = (uint8_t) (acc & 0xFFu);
        dst[4 + 3 * lane + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
        dst[4 + 3 * lane + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
    }
}

__global__ void kv_gather_turbo_kernel(const uint8_t* __restrict__ k_turbo, const uint8_t* __restrict__ v_turbo,
                                       const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                       const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                       uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const long long wi = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (wi >= n_ids * kv_heads) return;
    const long long id = wi / kv_heads;
    const int h = (int) (wi % kv_heads);
    const int lane = threadIdx.x, per = head_dim / 32;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    if (page < 0) return;   // not resident: the streaming resolver makes it so before this runs
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const long long bph = turbo_row_bytes(head_dim);
    const uint8_t* kr = k_turbo + row * bph;
    const uint8_t* vr = v_turbo + row * bph;
    const float ks = *reinterpret_cast<const float*>(kr);
    const float vs = *reinterpret_cast<const float*>(vr);
    uint32_t kacc = (uint32_t) kr[4 + 3 * lane] | ((uint32_t) kr[5 + 3 * lane] << 8) |
                    ((uint32_t) kr[6 + 3 * lane] << 16);
    uint32_t vacc = (uint32_t) vr[4 + 3 * lane] | ((uint32_t) vr[5 + 3 * lane] << 8) |
                    ((uint32_t) vr[6 + 3 * lane] << 16);
    const long long dst = ((id * kv_heads + h) * head_dim) + lane * per;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        k_scratch[dst + j] = f16_from_f32(ks * kTurboCentroid[(kacc >> (3 * j)) & 7u]);
        v_scratch[dst + j] = f16_from_f32(vs * kTurboCentroid[(vacc >> (3 * j)) & 7u]);
    }
}

// ---------------------------------------------------------------------------------------------------------
// rot4: per 32-value block an f16 scale (the block's RMS, not its max) and four groups of eight 3-bit codes.
// 14 B a block, 112 B a head. The per-32 scale bounds the damage of any one bad estimate, which is why it is
// the more conservative of the two even though turbo3 is smaller.
// ---------------------------------------------------------------------------------------------------------
__global__ void kv_append_rot4_kernel(uint8_t* __restrict__ k_rot4, uint8_t* __restrict__ v_rot4,
                                      const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                      int step_stride, int planes, const float* __restrict__ kcur,
                                      const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size,
                                      KvHostPools host) {
    __shared__ uint8_t s_codes[4][32];
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, lane = threadIdx.x, wy = threadIdx.y;
    const long long base = (long long) step_idx * kv_heads * head_dim + (long long) h * head_dim;
    const float* src = (is_v ? vcur : kcur) + base;
    const int n_blocks = head_dim / 32;
    const long long page = (long long) table[pos / page_size];
    const long long bph = rot4_row_bytes(head_dim);

    uint8_t* dst = nullptr;
    if (page >= 0) dst = (is_v ? v_rot4 : k_rot4) + ((page * kv_heads + h) * page_size + (pos % page_size)) * bph;
    uint8_t* hdst = nullptr;
    if (host.k_rot4 != nullptr)
        hdst = (is_v ? host.v_rot4 : host.k_rot4) + (((pos / page_size) * kv_heads + h) * page_size + (pos % page_size)) * bph;

    for (int b = 0; b < n_blocks; ++b) {
        const float x = src[b * 32 + lane];
        const float s = sqrtf(warp_sum(x * x) / 32.0f);
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        s_codes[wy][lane] = (uint8_t) turbo_nearest(x * inv);
        __syncwarp();
        if (lane < 4) {   // one group of eight codes -> three bytes
            const uint8_t* g = s_codes[wy] + lane * 8;
            uint32_t acc = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc |= (uint32_t) g[j] << (3 * j);
            const size_t off = (size_t) b * 14 + 2 + 3 * lane;
            if (dst != nullptr) {
                dst[off + 0] = (uint8_t) (acc & 0xFFu);
                dst[off + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
                dst[off + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
            }
            if (hdst != nullptr) {
                hdst[off + 0] = (uint8_t) (acc & 0xFFu);
                hdst[off + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
                hdst[off + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
            }
        }
        if (lane == 0) {
            const uint16_t sh = f16_from_f32(s);
            if (dst != nullptr) *reinterpret_cast<uint16_t*>(dst + (size_t) b * 14) = sh;
            if (hdst != nullptr) *reinterpret_cast<uint16_t*>(hdst + (size_t) b * 14) = sh;
        }
        __syncwarp();
    }
}

__global__ void kv_append_rot4_batch_kernel(uint8_t* __restrict__ k_rot4, uint8_t* __restrict__ v_rot4,
                                            const int32_t* __restrict__ table, int64_t pos0,
                                            const float* __restrict__ K, const float* __restrict__ V, int kv_heads,
                                            int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                            KvHostPools stage) {
    __shared__ uint8_t s_codes[4][32];
    const long long t = blockIdx.x, pos = pos0 + t;
    const int h = blockIdx.y, lane = threadIdx.x, wy = threadIdx.y;
    const bool is_v = is_v_grid != 0;
    const float* src = (is_v ? V : K) + t * (kv_heads * head_dim) + (long long) h * head_dim;
    const int n_blocks = head_dim / 32;
    const long long page = (long long) table[pos / page_size];
    const long long bph = rot4_row_bytes(head_dim);
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);

    uint8_t* dst = nullptr;
    if (page >= 0) dst = (is_v ? v_rot4 : k_rot4) + ((page * kv_heads + h) * page_size + (pos % page_size)) * bph;
    uint8_t* hdst = host.k_rot4 != nullptr ? (is_v ? host.v_rot4 : host.k_rot4) + row_id * bph : nullptr;
    uint8_t* sdst = stage.k_rot4 != nullptr ? (is_v ? stage.v_rot4 : stage.k_rot4) + row_id * bph : nullptr;

    for (int b = 0; b < n_blocks; ++b) {
        const float x = src[b * 32 + lane];
        const float s = sqrtf(warp_sum(x * x) / 32.0f);
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        s_codes[wy][lane] = (uint8_t) turbo_nearest(x * inv);
        __syncwarp();
        if (lane < 4) {
            const uint8_t* g = s_codes[wy] + lane * 8;
            uint32_t acc = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc |= (uint32_t) g[j] << (3 * j);
            const size_t off = (size_t) b * 14 + 2 + 3 * lane;
            const uint8_t b0 = (uint8_t) (acc & 0xFFu), b1 = (uint8_t) ((acc >> 8) & 0xFFu), b2 = (uint8_t) ((acc >> 16) & 0xFFu);
            if (dst != nullptr) { dst[off + 0] = b0; dst[off + 1] = b1; dst[off + 2] = b2; }
            if (hdst != nullptr) { hdst[off + 0] = b0; hdst[off + 1] = b1; hdst[off + 2] = b2; }
            if (sdst != nullptr) { sdst[off + 0] = b0; sdst[off + 1] = b1; sdst[off + 2] = b2; }
        }
        if (lane == 0) {
            const uint16_t sh = f16_from_f32(s);
            if (dst != nullptr) *reinterpret_cast<uint16_t*>(dst + (size_t) b * 14) = sh;
            if (hdst != nullptr) *reinterpret_cast<uint16_t*>(hdst + (size_t) b * 14) = sh;
            if (sdst != nullptr) *reinterpret_cast<uint16_t*>(sdst + (size_t) b * 14) = sh;
        }
        __syncwarp();
    }
}

__global__ void kv_gather_rot4_kernel(const uint8_t* __restrict__ k_rot4, const uint8_t* __restrict__ v_rot4,
                                      const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                      const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                      uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const long long wi = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (wi >= n_ids * kv_heads) return;
    const long long id = wi / kv_heads;
    const int h = (int) (wi % kv_heads), lane = threadIdx.x;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    if (page < 0) return;
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const long long bph = rot4_row_bytes(head_dim);
    const uint8_t* kr = k_rot4 + row * bph;
    const uint8_t* vr = v_rot4 + row * bph;
    const int grp = lane / 8, bit = 3 * (lane % 8);
    for (int b = 0; b < head_dim / 32; ++b) {
        const uint16_t ksh = *reinterpret_cast<const uint16_t*>(kr + (size_t) b * 14);
        const uint16_t vsh = *reinterpret_cast<const uint16_t*>(vr + (size_t) b * 14);
        const uint8_t* kg = kr + (size_t) b * 14 + 2 + 3 * grp;
        const uint8_t* vg = vr + (size_t) b * 14 + 2 + 3 * grp;
        const uint32_t ka = (uint32_t) kg[0] | ((uint32_t) kg[1] << 8) | ((uint32_t) kg[2] << 16);
        const uint32_t va = (uint32_t) vg[0] | ((uint32_t) vg[1] << 8) | ((uint32_t) vg[2] << 16);
        const long long dst = ((id * kv_heads + h) * head_dim) + b * 32 + lane;
        k_scratch[dst] = f16_from_f32(f32_from_f16(ksh) * kTurboCentroid[(ka >> bit) & 7u]);
        v_scratch[dst] = f16_from_f32(f32_from_f16(vsh) * kTurboCentroid[(va >> bit) & 7u]);
    }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ turbo3 ----
void kv_append_turbo_steps(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                           int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                           void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_turbo");
    const int planes = (v_turbo == nullptr || (v_turbo == k_turbo && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, 1u, (unsigned) (n_steps * planes));
    kv_append_turbo_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_turbo, v_turbo, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_turbo launch");
}

void kv_append_turbo_step(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                          const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                          const KvHostPools* host) {
    kv_append_turbo_steps(k_turbo, v_turbo, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_turbo(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, int64_t pos0, int64_t T,
                     const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                     const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_turbo");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, 1u);
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_turbo_batch_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
            k_turbo, v_turbo, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, is_v, h,
            st);
    check("kv_append_turbo batch launch");
}

void kv_gather_turbo_step(const uint8_t* k_turbo, const uint8_t* v_turbo, const int32_t* page_table,
                          const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                          uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int64_t total = max_ids * s.n_head_kv;
    const int rows_per_block = 4;
    kv_gather_turbo_kernel<<<dim3((unsigned) ((total + rows_per_block - 1) / rows_per_block)),
                             dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_turbo, v_turbo, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_turbo launch");
}

// -------------------------------------------------------------------------------------------------- rot4 ----
void kv_append_rot4_steps(uint8_t* k_rot4, uint8_t* v_rot4, const int32_t* page_table, const int32_t* step,
                          int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                          void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_rot4");
    const int planes = (v_rot4 == nullptr || (v_rot4 == k_rot4 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, 1u, (unsigned) (n_steps * planes));
    kv_append_rot4_kernel<<<grid, dim3(32, 1), 0, (cudaStream_t) stream>>>(
        k_rot4, v_rot4, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_rot4 launch");
}

void kv_append_rot4_step(uint8_t* k_rot4, uint8_t* v_rot4, const int32_t* page_table, const int32_t* step,
                         const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                         const KvHostPools* host) {
    kv_append_rot4_steps(k_rot4, v_rot4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_rot4(uint8_t* k_rot4, uint8_t* v_rot4, const int32_t* page_table, int64_t pos0, int64_t T,
                    const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                    const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_rot4");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, 1u);
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_rot4_batch_kernel<<<grid, dim3(32, 1), 0, (cudaStream_t) stream>>>(
            k_rot4, v_rot4, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, is_v, h,
            st);
    check("kv_append_rot4 batch launch");
}

void kv_gather_rot4_step(const uint8_t* k_rot4, const uint8_t* v_rot4, const int32_t* page_table,
                         const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                         uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int64_t total = max_ids * s.n_head_kv;
    const int rows_per_block = 4;
    kv_gather_rot4_kernel<<<dim3((unsigned) ((total + rows_per_block - 1) / rows_per_block)),
                            dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_rot4, v_rot4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_rot4 launch");
}

}  // namespace strata::kernels
