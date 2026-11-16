// src/kernels/cuda/latent.cu - the latent MoE's engine-side kernels (new file).
//
// Bit-identical dot to iq_mmvq(8)'s vec_dot_q8_0_q8_1: per block
//   acc += dx * dw * sum_k(xs[k] * ws[k])
// where dx is the q8_1 block's f16 scale (struct {half ds[2]; int8 qs[32]}: the first half,
// bytes 0..1; the values follow at bytes 4..35) and dw the q8_0 block's f16 scale
// ({half d; int8 qs[32]}: bytes 0..1; values at 2..33).  The per-block 32-lane sum is
// computed directly (same integer arithmetic as the dp4a lanes, no alignment assumptions).
#include "strata/kernels/latent.hpp"
#include "strata/kernels/dp4a.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::kernels {

__global__ void latent_plan_rows_k(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                                   const int32_t* labels, int E, int base_token, int G, int K) {
    const int nj = counts[1];   // all GPU entries (the VRAM share + the PCIe share)
    for (int j = threadIdx.x; j < nj; j += blockDim.x) {
        const int d = dst[j];
        const int32_t e = ids[d];
        const int g = (e >= 0 && e < E) ? labels[e] : 0;
        tok[j] = (base_token + d / K) * G + g;
    }
}

void latent_plan_rows(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                      const int32_t* labels_dev, int64_t E, int base_token, int G, int K, void* stream) {
    latent_plan_rows_k<<<1, 64, 0, (cudaStream_t) stream>>>(tok, dst, counts, ids, labels_dev, (int) E, base_token, G, K);
}

// latent_u_k optimized: coalesced dp4a, one warp per output row.
// Old version: one THREAD per row with scalar byte loads and no reuse -> ~25x read
// amplification, ~0.84 ms/layer. New: lanes stride the ELL words of a row as int32
// (built from 2-byte-aligned ushort pairs, q8_0 stores {half d; i8 qs[32]} at +2),
// dp4a per word, warp-shuffle reduction. U[g] rows are read once per (route,row)
// and shared across routes through L2 as before, but now at stream bandwidth.
__device__ __forceinline__ int latent_ld4_i8(const uint8_t* p) {
    const unsigned short a = *(const unsigned short*) p;
    const unsigned short b = *(const unsigned short*) (p + 2);
    return (int) ((unsigned) a | ((unsigned) b << 16));   // a = p0|p1<<8, b<<16 = p2<<16|p3<<24
}


// Q4DU: the weight block's dot with a q8_1 activation block, fmt 8 (Q8_0) or 4 (Q4_0).
// The q8_1 block is always {f16 d; f16 s; int8 qs[32]} (36 B, values at +4); the weight block
// is {f16 d; int8 qs[32]} (34 B) for fmt 8, {f16 d; uint8 nib[16]} (18 B) for fmt 4 with
// element j in byte j's low nibble and element j+16 in its high one, value (code - 8) * d -
// the same centred codes as iq_kernels.cu's vec_dot_q4_0_q8_1 (q4_0_centred).
__device__ __forceinline__ float lat_blk_dot_q4(const uint8_t* wb, const uint8_t* xb, int j) {
    // nibble words are built from byte loads: a Q4_0 block is only 2-byte aligned inside its
    // row (18 B stride), a raw int dereference at +2 is misaligned and faults on sm_120.
    const float dw = __half2float(*(const __half*) wb);
    const float dx = __half2float(*(const __half*) xb);
    // 4 bytes at wb+2+j: each byte's low nibble is element (j+k), its high nibble element (j+k+16).
    int w0 = 0, w16 = 0, xw = 0, xw16 = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int byte_ = (int) wb[2 + j + k];
        w0 |= (((byte_ & 0xF) - 8) & 0xFF) << (8 * k);          // mask: a negative code's sign
        w16 |= ((((byte_ >> 4) & 0xF) - 8) & 0xFF) << (8 * k);  // bits must not bleed into the
        xw |= (int) (uint32_t) xb[4 + j + k] << (8 * k);        // neighbouring byte of the word
        xw16 |= (int) (uint32_t) xb[4 + j + k + 16] << (8 * k); // (dp4a reads s8 itself)
    }
    return dx * dw * (float) (__dp4a(xw, w0, 0) + __dp4a(xw16, w16, 0));
}

__global__ void latent_u_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                           const int32_t* __restrict__ ids, const int32_t* __restrict__ labels, int E, int cap_entries,
                           int ELL, int H, float* __restrict__ out, int fmt) {
    const int r = blockIdx.x;
    if (r >= cap_entries) return;
    const int32_t e = ids[r];
    const int g = (e >= 0 && e < E) ? labels[e] : 0;
    const int nb = ELL / 32;            // value blocks per row
    const int wrow = nb * (fmt == 4 ? 18 : 34);   // Q4DU: the weight row's bytes
    const uint8_t* w = u_q8 + (size_t) g * (size_t) H * (size_t) wrow;
    const uint8_t* x = ylq + (size_t) r * (size_t) (nb * 36);
    float* o = out + (size_t) r * (size_t) H;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = (int) (blockIdx.y * (blockDim.x >> 5)) + warp;
    if (h >= H) return;
    const uint8_t* wb = w + (size_t) h * (size_t) wrow;
    float facc = 0.0f;
    if (fmt == 4) {
        for (int wi = lane; wi < nb * 4; wi += 32) {   // 4 int4 words per block (16 nibble bytes)
            const int b = wi >> 2, j = (wi & 3) * 4;
            facc += lat_blk_dot_q4(wb + (size_t) b * 18, x + (size_t) b * 36, j);
        }
    } else {
    for (int wi = lane; wi < nb * 8; wi += 32) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        const float dx = __half2float(*(const __half*) (x + (size_t) b * 36));
        const float dw = __half2float(*(const __half*) (wb + (size_t) b * 34));
        const int xw = *(const int*) (x + (size_t) b * 36 + 4 + j);
        const int ww = latent_ld4_i8(wb + (size_t) b * 34 + 2 + j);
        facc += dx * dw * (float) __dp4a(xw, ww, 0);
    }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) o[h] = facc;
}

// latent_u_k with this lane's ACTIVATION terms held in registers: one block per routed row, its 8 warps
// walking the H output columns (h, h+8, h+16, ...).  latent_u_k re-reads the row's q8_1 blocks once per
// output column - H times - and the row walk was written into every one of (rows x H/8) blocks; here the
// row is decoded once per (row, warp) and the loop over h is the only thing that repeats.
//
// The per-output arithmetic is latent_u_k's, term for term: the lane's terms in the same
// `wi = lane; wi < nb*8; wi += 32` order, the same fp16 scale loads, the same dp4a and float expression,
// the same 5-step shuffle tree.  So every value is bit-identical; only which block computes it changes.
// Measured on gfx1031 at the engine's shape (ELL 1280, H 2560, rows 5120): 25.25 -> 16.85 ms a layer
// (-33.3 %) with the output memcmp-equal (u_probe.cu).  An earlier attempt kept the WEIGHT column in
// registers and reused it across a group's rows instead; that is 8 % slower in the engine (few long
// blocks that scan every row), so the reuse direction that pays is x, not w.
__global__ void latent_u_walk_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                                const int32_t* __restrict__ ids, const int32_t* __restrict__ labels, int E, int cap,
                                int ELL, int H, float* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= cap) return;
    const int32_t e = ids[r];
    const int g = (e >= 0 && e < E) ? labels[e] : 0;
    const uint8_t* wbase = u_q8 + (size_t) g * (size_t) H * (size_t) (ELL / 32 * 34);
    const uint8_t* x = ylq + (size_t) r * (size_t) (ELL / 32 * 36);
    float* o = out + (size_t) r * (size_t) H;
    const int lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int nb = ELL / 32;
    // this lane's terms, in latent_u_k's order (10 of them for ELL 1280)
    int tbase[16], tword[16], txw[16];
    float tdx[16];
    int nt = 0;
    for (int wi = lane; wi < nb * 8; wi += 32, ++nt) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        tbase[nt] = b;
        tword[nt] = j;
        txw[nt] = *(const int*) (x + (size_t) b * 36 + 4 + j);
        tdx[nt] = __half2float(*(const __half*) (x + (size_t) b * 36));
    }
    for (int h = (threadIdx.x >> 5); h < H; h += nwarp) {
        const uint8_t* wb = wbase + (size_t) h * (size_t) (nb * 34);
        float facc = 0.0f;
        for (int i = 0; i < nt; ++i) {
            const int b = tbase[i];
            const float dw = __half2float(*(const __half*) (wb + (size_t) b * 34));
            const int ww = latent_ld4_i8(wb + (size_t) b * 34 + 2 + tword[i]);
            facc += tdx[i] * dw * (float) __dp4a(txw[i], ww, 0);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
        if (lane == 0) o[h] = facc;
    }
}

// ---- the weight-reuse U kernel ------------------------------------------------------------------
//
// latent_u_k's blocks are (one routed row, 8 output columns): each block reads its 8 weight rows (8 x
// 1360 B = 10.9 KB) and every ROW of a group re-reads that group's whole U matrix, so a layer reads
// ~20 GB through the Infinity Cache to produce 5120 x 2560 outputs (measured: 23.7 ms a layer at an
// ~850 GB/s IC rate, 1.42 TFLOP/s, ~2 % of the card's int8 peak).  The fix is what the header of
// latent_u_grouped has always described: keep an output column's weights in REGISTERS and reuse them
// over the run of routed rows that share its label.
//
// Per (run, h) the lane's terms and their order are latent_u_k's, and the reduction is the same tree,
// so every output is bit-identical to the per-row kernel (the test compares with memcmp).
//
// The run length matters in both directions: long runs cut the weight traffic but starve the device of
// independent blocks, which is why the "scan the whole group" form measured 8 % slower in the engine.
// R is the knob (STRATA_LATENT_U_RUN); the label-bucketed row list is built on the device, in the
// caller's scratch, so no per-layer host upload is needed.
__global__ void lat_u_run_count_k(const int32_t* __restrict__ ids, const int32_t* __restrict__ labels,
                                 int64_t n, int E, int G, int32_t* __restrict__ goff) {
    for (int64_t p = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; p < n;
         p += (int64_t) blockDim.x * gridDim.x) {
        const int32_t e = ids[p];
        const int g = (e >= 0 && e < E) ? labels[e] : 0;
        atomicAdd(&goff[g], 1);
    }
}

// counts -> the group prefix (one thread: G <= 1024 and this runs once a layer)
__global__ void lat_u_run_off_k(int32_t* __restrict__ goff, int G) {
    if (threadIdx.x != 0) return;
    int s = 0;
    for (int g = 0; g < G; ++g) {
        const int c = goff[g];
        goff[g] = s;
        s += c;
    }
    goff[G] = s;
}

__global__ void lat_u_run_place_k(const int32_t* __restrict__ ids, const int32_t* __restrict__ labels,
                                  int64_t n, int E, int32_t* __restrict__ cur, int32_t* __restrict__ rows) {
    for (int64_t p = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; p < n;
         p += (int64_t) blockDim.x * gridDim.x) {
        const int32_t e = ids[p];
        const int g = (e >= 0 && e < E) ? labels[e] : 0;
        rows[atomicAdd(&cur[g], 1)] = (int32_t) p;
    }
}

// One block per (run, 8 output columns).  `goff` is the group prefix; a run is up to R consecutive
// rows of one group's list.  The block resolves its own group from its run index, so no run table
// has to be built: ceil(n / R) + G bounds the launch (each group pads one partial run).
__global__ void latent_u_run_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                               const int32_t* __restrict__ rows, const int32_t* __restrict__ goff,
                               int n_groups, int ELL, int H, float* __restrict__ out, int R) {
    const int lane = threadIdx.x & 31;
    const int h = (int) (blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5));
    if (h >= H) return;
    int run = blockIdx.x, g = 0;
    for (; g < n_groups; ++g) {
        const int cnt = goff[g + 1] - goff[g];
        const int nruns = (cnt + R - 1) / R;
        if (run < nruns) break;
        run -= nruns;
    }
    if (g >= n_groups) return;
    const int cnt = goff[g + 1] - goff[g];
    const int base = goff[g] + run * R;
    const int nr = min(R, cnt - run * R);
    if (nr <= 0) return;

    const int nb = ELL / 32;
    const int wrow = nb * 34;                       // Q8_0: {half d; int8 qs[32]} a block
    const uint8_t* wb = u_q8 + (size_t) g * (size_t) H * (size_t) wrow + (size_t) h * (size_t) wrow;
    // this lane's share of the weight row, in registers: latent_u_k's own term order
    float wdw[16];
    int wword[16], tb[16], tj[16];
    int nt = 0;
    for (int wi = lane; wi < nb * 8; wi += 32, ++nt) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        tb[nt] = b;
        tj[nt] = j;
        wdw[nt] = __half2float(*(const __half*) (wb + (size_t) b * 34));
        wword[nt] = latent_ld4_i8(wb + (size_t) b * 34 + 2 + j);
    }
    for (int i = 0; i < nr; ++i) {
        const int r = rows[base + i];
        const uint8_t* x = ylq + (size_t) r * (size_t) (nb * 36);
        float facc = 0.0f;
        for (int t = 0; t < nt; ++t) {
            const int b = tb[t];
            const float dx = __half2float(*(const __half*) (x + (size_t) b * 36));
            const int xw = *(const int*) (x + (size_t) b * 36 + 4 + tj[t]);
            facc += dx * wdw[t] * (float) __dp4a(xw, wword[t], 0);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
        if (lane == 0) out[(size_t) r * (size_t) H + h] = facc;
    }
}

namespace {
inline void u_run_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "latent_u_runs: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

// The label-bucketed row list the U kernels share: goff[G + 1] (each group's run in rows[]) and
// rows[n] (the routed rows bucketed by their label), laid out in the caller's scratch so a layer
// needs no host upload:
//   +0     goff [G + 1]
//   +256   cur  [G]      (the scatter cursors)
//   +1024  rows [n]
void u_run_tables(const int32_t* ids, const int32_t* labels_dev, int64_t E, int64_t n, void* scratch, int G,
                  cudaStream_t s, int32_t** goff_out, int32_t** rows_out) {
    int32_t* goff = (int32_t*) scratch;
    int32_t* cur = (int32_t*) ((uint8_t*) scratch + 256);
    int32_t* rows = (int32_t*) ((uint8_t*) scratch + 1024);
    u_run_ck(cudaMemsetAsync(goff, 0, (size_t) (G + 1) * 4, s), "u tables memset");
    const int nb_ = (int) std::min<int64_t>((n + 255) / 256, 256);
    lat_u_run_count_k<<<nb_, 256, 0, s>>>(ids, labels_dev, n, (int) E, G, goff);
    lat_u_run_off_k<<<1, 32, 0, s>>>(goff, G);
    u_run_ck(cudaMemcpyAsync(cur, goff, (size_t) G * 4, cudaMemcpyDeviceToDevice, s), "u tables cursors");
    lat_u_run_place_k<<<nb_, 256, 0, s>>>(ids, labels_dev, n, (int) E, cur, rows);
    *goff_out = goff;
    *rows_out = rows;
}
}  // namespace

// ---- the staged U tile -------------------------------------------------------------------------
//
// U is the largest single phase of a prompt (8,954 ms of 28,568) and runs at 1.42 TFLOP/s, ~2 % of this
// card's int8 peak, because every term of latent_u_k's inner loop costs ~9 instructions for 4 MACs: a
// 4-byte activation word, two 2-byte scale loads, a weight word built from two UNALIGNED ushorts, one
// __dp4a and two multiplies.  Three variants that cut BYTES (weight reuse over a run of rows, the
// row-walk) all measured slower, so the limit is instructions per MAC, not traffic - and the fix is to
// stage both operands into 4-byte-aligned shared rows and hoist the scale to one FMA per 32-value block:
//
//   staged weight row (kUPitch 72): 64 codes | {half d(b0), half d(b1)} | pad
//   staged act row    (kUPitch 72): 64 codes | {half d(b0), half d(b1)} | pad
//
// then the inner loop is four __dp4a per 16-value sub-block with 4-byte smem operands, and the scale
// product applies once per sub-block - the fragment mapping, the store pattern and the tile shape are
// lat_q3k_down_kernel's (measured, verified by lat_q3k_down_test), which keeps 2 blocks per CU:
// 23,040 B of smem a block and ~80 VGPRs, so 2 x 512 threads stay resident where the run kernel's
// 77 VGPRs bought 12 waves but a serial per-row dependency chain.
//
// Arithmetic: the per-4-value scaling of latent_u_k becomes one scale per 16 values, so this is NOT
// bit-identical - the int32 sums are exact and the scale is constant across a 32-value block, so the
// difference is only fp32 rounding of the final multiply.  Gate: a double-precision reference /
// the per-row kernel within ~1e-6 relative, then the engine's 32-token output.
constexpr int kUTileThreads = 512;
constexpr int kUTileCols = 256;      // output columns a block owns
constexpr int kUTileRows = 64;       // routed rows a block owns (8 rows a warp, 4 warps a warp-group)
constexpr int kUPitch = 72;          // 64 codes + 2 half scales + 4 pad

inline size_t u_tile_smem_bytes() {
    return (size_t) (kUTileCols + kUTileRows) * kUPitch;
}

__global__ void __launch_bounds__(kUTileThreads, 2)
latent_u_tile_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                const int32_t* __restrict__ rows, const int32_t* __restrict__ goff, int n_groups, int ELL,
                int H, float* __restrict__ out, int64_t n_ident) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wst = smem;                                  // kUTileCols staged weight rows
    uint8_t* ast = smem + (size_t) kUTileCols * kUPitch;  // kUTileRows staged activation rows

    const int nbchunk = ELL >> 6;             // 64-value chunks along K (20 for ELL 1280)
    const size_t wrowb = (size_t) (ELL / 32) * 34;   // a Q8_0 weight row {half d; int8 qs[32]}
    const size_t arowb = (size_t) (ELL / 32) * 36;   // a q8_1 activation row {half d; half s; int8 qs[32]}

    // this tile's group (the rows of one label share their U slice) and its first routed row.
    // rows == nullptr means "one weight matrix for every row, in row order" - the D projection, which
    // has no groups at all: the tile then serves tokens 0..n-1 against the one [G*ELL, N] matrix.
    int tile = blockIdx.x, g = 0, base = 0, nrows = 0, cnt = 0;
    if (rows == nullptr || goff == nullptr) {
        cnt = (int) n_ident;
        base = tile * kUTileRows;
        if (base >= cnt) return;
        nrows = min(kUTileRows, cnt - base);
    } else {
        for (; g < n_groups; ++g) {
            const int c = goff[g + 1] - goff[g];
            const int nt = (c + kUTileRows - 1) / kUTileRows;
            if (tile < nt) break;
            tile -= nt;
        }
        if (g >= n_groups) return;
        cnt = goff[g + 1] - goff[g];
        base = goff[g] + tile * kUTileRows;
        nrows = min(kUTileRows, cnt - tile * kUTileRows);
        if (nrows <= 0) return;
    }

    const int col0 = (int) blockIdx.y * kUTileCols;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mg = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    const int wrow = tid >> 1, whalf = tid & 1;    // 2 threads a weight row, 32 codes each
    const int arow = tid >> 1, ahalf = tid & 1;    // threads < 2*kUTileRows stage the activations

    float acc[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;

    for (int kb = 0; kb < nbchunk; ++kb) {
        __syncthreads();     // one buffer: the previous chunk's compute is done
        {
            const int b = 2 * kb + whalf;
            const int wcol = min(col0 + wrow, H - 1);   // a short last column tile repeats its edge row
            const uint8_t* wp = u_q8 + (size_t) g * (size_t) H * wrowb + (size_t) wcol * wrowb +
                                (size_t) b * 34;
            uint8_t* ws = wst + (size_t) wrow * kUPitch + whalf * 32;
            const uint8_t* qs = wp + 2;
#pragma unroll
            for (int i = 0; i < 8; ++i) *(uint32_t*) (ws + 4 * i) = (uint32_t) latent_ld4_i8(qs + 4 * i);
            *(__half*) (wst + (size_t) wrow * kUPitch + 64 + 2 * whalf) = *(const __half*) wp;
        }
        if (tid < 2 * kUTileRows) {
            const int r = base + min(arow, nrows - 1);     // the tail's spare rows repeat the last
            const int rr = rows != nullptr ? rows[r] : r;
            const int b = 2 * kb + ahalf;
            const uint8_t* xp = ylq + (size_t) rr * arowb + (size_t) b * 36;
            uint8_t* as = ast + (size_t) arow * kUPitch + ahalf * 32;
            *(__half*) (ast + (size_t) arow * kUPitch + 64 + 2 * ahalf) = *(const __half*) xp;
            const uint8_t* qs = xp + 4;                    // q8_1's values are 4-byte aligned
#pragma unroll
            for (int i = 0; i < 8; ++i) *(uint32_t*) (as + 4 * i) = *(const uint32_t*) (qs + 4 * i);
        }
        __syncthreads();
        for (int s = 0; s < 4; ++s) {                       // 16-value sub-blocks of the 64-value chunk
            const int sidx = (s & 2) ? 1 : 0;               // which 32-value block's scale
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int ra = 64 * wf + 16 * i + mg;
                const int rb = ra + 8;
                const uint8_t* wa = wst + (size_t) ra * kUPitch + 16 * s;
                const uint8_t* wb = wst + (size_t) rb * kUPitch + 16 * s;
                const float dwa = __half2float(*(const __half*) (wst + (size_t) ra * kUPitch + 64 + 2 * sidx));
                const float dwb = __half2float(*(const __half*) (wst + (size_t) rb * kUPitch + 64 + 2 * sidx));
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int r0 = nb0 + 8 * n + 2 * tig;
                    const uint8_t* ar0 = ast + (size_t) r0 * kUPitch + 16 * s;
                    const uint8_t* ar1 = ar0 + kUPitch;
                    const float dx0 = __half2float(*(const __half*) (ast + (size_t) r0 * kUPitch + 64 + 2 * sidx));
                    const float dx1 = __half2float(*(const __half*) (ast + (size_t) (r0 + 1) * kUPitch + 64 + 2 * sidx));
                    int d[4] = {0, 0, 0, 0};
#pragma unroll
                    for (int kw = 0; kw < 4; ++kw) {
                        const int b0 = (int) *(const uint32_t*) (ar0 + 4 * kw);
                        const int b1 = (int) *(const uint32_t*) (ar1 + 4 * kw);
                        const int wa4 = (int) *(const uint32_t*) (wa + 4 * kw);
                        const int wb4 = (int) *(const uint32_t*) (wb + 4 * kw);
                        d[0] = __dp4a(wa4, b0, d[0]);
                        d[1] = __dp4a(wa4, b1, d[1]);
                        d[2] = __dp4a(wb4, b0, d[2]);
                        d[3] = __dp4a(wb4, b1, d[3]);
                    }
                    acc[i][n][0] += (dwa * dx0) * (float) d[0];
                    acc[i][n][1] += (dwa * dx1) * (float) d[1];
                    acc[i][n][2] += (dwb * dx0) * (float) d[2];
                    acc[i][n][3] += (dwb * dx1) * (float) d[3];
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int tr = nb0 + 8 * n + 2 * tig + (q & 1);
                if (tr >= nrows) continue;
                const int col = col0 + 64 * wf + 16 * i + mg + (q < 2 ? 0 : 8);
                if (col >= H) continue;
                out[(size_t) (rows != nullptr ? rows[base + tr] : base + tr) * (size_t) H + col] = acc[i][n][q];
            }
}

void latent_u_runs(const uint8_t* u_q8, const uint8_t* ylq, const int32_t* ids, const int32_t* labels_dev,
                   int64_t E, int64_t n, int ELL, int H, float* out, void* scratch, void* stream, int fmt,
                   int R, int G) {
    if (n <= 0 || R <= 0 || fmt != 8 || G <= 0) return;
    if (u_q8 == nullptr || ylq == nullptr || ids == nullptr || labels_dev == nullptr || out == nullptr ||
        scratch == nullptr) {
        std::fprintf(stderr, "latent_u_runs: NULL pointer (scratch is required for the run tables)\n");
        std::abort();
    }
    cudaStream_t s = (cudaStream_t) stream;
    int32_t* goff = nullptr;
    int32_t* rows = nullptr;
    u_run_tables(ids, labels_dev, E, n, scratch, G, s, &goff, &rows);
    const int warps = 8;
    const int64_t nrun = (n + R - 1) / R + G;
    dim3 grid((unsigned) nrun, (unsigned) ((H + warps - 1) / warps));
    latent_u_run_k<<<grid, warps * 32, 0, s>>>(u_q8, ylq, rows, goff, G, ELL, H, out, R);
    u_run_ck(cudaGetLastError(), "latent_u_run");
}

void latent_u_tiled(const uint8_t* u_q8, const uint8_t* ylq, const int32_t* ids, const int32_t* labels_dev,
                    int64_t E, int64_t n, int ELL, int H, float* out, void* scratch, void* stream, int fmt, int G) {
    if (n <= 0 || fmt != 8 || G <= 0 || (ELL & 63) != 0 || scratch == nullptr) return;
    if (u_q8 == nullptr || ylq == nullptr || ids == nullptr || labels_dev == nullptr || out == nullptr) {
        std::fprintf(stderr, "latent_u_tiled: NULL pointer\n");
        std::abort();
    }
    cudaStream_t s = (cudaStream_t) stream;
    int32_t* goff = nullptr;
    int32_t* rows = nullptr;
    u_run_tables(ids, labels_dev, E, n, scratch, G, s, &goff, &rows);
    static bool said = false;   // discoverable, and honest: this path is not bit-identical to the per-row one
    if (!said) {
        said = true;
        std::fprintf(stderr,
                     "latent u: staged tile (256 cols x 64 rows, scale hoisted per 16 values; %d columns, "
                     "ELL %d) - numerically equal to the per-row kernel, not bit-identical "
                     "(latent_u_tile_test: mean_rel 3.7e-07, cos 1.0)\n",
                     (int) H, (int) ELL);
    }
    static bool attr = false;
    if (!attr) {
        u_run_ck(cudaFuncSetAttribute(latent_u_tile_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int) u_tile_smem_bytes()), "u tile smem");
        attr = true;
    }
    const int64_t ntiles = (n + kUTileRows - 1) / kUTileRows + G;
    dim3 grid((unsigned) ntiles, (unsigned) ((H + kUTileCols - 1) / kUTileCols));
    latent_u_tile_k<<<grid, kUTileThreads, u_tile_smem_bytes(), s>>>(u_q8, ylq, rows, goff, G, ELL, H, out, n);
    u_run_ck(cudaGetLastError(), "latent_u_tile");
}

void latent_d_tiled(const uint8_t* w_q8, const uint8_t* xq, int64_t M, int K, int64_t n_out, float* out,
                    void* stream, int fmt) {
    // The D projection: every token against the whole [G*ELL, N] dense matrix, no groups (each token
    // uses G*ELL outputs whatever its route), so the same staged tile serves it with rows == nullptr -
    // identity row order.  T=512 gives 8 full 64-row tiles, i.e. no padding at all, where the U side
    // runs at 39 % fill.
    if (M <= 0 || K <= 0 || n_out <= 0 || fmt != 8 || (K & 63) != 0) return;
    if (w_q8 == nullptr || xq == nullptr || out == nullptr) {
        std::fprintf(stderr, "latent_d_tiled: NULL pointer\n");
        std::abort();
    }
    cudaStream_t s = (cudaStream_t) stream;
    static bool attr = false;
    if (!attr) {
        u_run_ck(cudaFuncSetAttribute(latent_u_tile_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int) u_tile_smem_bytes()), "d tile smem");
        attr = true;
    }
    dim3 grid((unsigned) ((M + kUTileRows - 1) / kUTileRows), (unsigned) ((n_out + kUTileCols - 1) / kUTileCols));
    latent_u_tile_k<<<grid, kUTileThreads, u_tile_smem_bytes(), s>>>(w_q8, xq, nullptr, nullptr, 1, K, (int) n_out,
                                                                    out, M);
    u_run_ck(cudaGetLastError(), "latent_d_tile");
}

void latent_u_grouped(const uint8_t* u_q8, const uint8_t* ylq, const int32_t* ids, const int32_t* labels_dev,
                      int64_t E, int64_t cap_entries, int ELL, int H, float* out, void* stream, int fmt,
                      int n_groups) {
    if (cap_entries <= 0) return;
    if (u_q8 == nullptr || ylq == nullptr || ids == nullptr || labels_dev == nullptr || out == nullptr) {
        std::fprintf(stderr, "latent_u_grouped NULL ptr: u=%p ylq=%p ids=%p labels=%p out=%p E=%lld cap=%lld ELL=%d H=%d\n",
                     (const void*) u_q8, (const void*) ylq, (const void*) ids, (const void*) labels_dev,
                     (const void*) out, (long long) E, (long long) cap_entries, (int) ELL, (int) H);
        std::fflush(stderr);
        std::abort();
    }
    const int warps_per_block = 8;   // 256 threads
    const int rows_per_block = warps_per_block;
    // STRATA_LATENT_U_WALK=1 selects the row-walk kernel.  It is NOT the default: a standalone probe
    // (u_probe.cu, kernels back to back, U hot in the Infinity Cache) measures it 33.3 % faster, but in
    // the engine it measures 37.6 vs 32.7 ms a layer - 15 % SLOWER - because the A stage's expert stream
    // evicts the U matrix between layers and only the per-column kernel's 1.3 M small blocks have enough
    // independent work to hide the cold misses.  Standalone microfessions do not decide; the engine does.
    static const bool u_walk = [] {
        const char* v = std::getenv("STRATA_LATENT_U_WALK");
        return v != nullptr && v[0] == '1';
    }();
    static const bool u_old = [] {
        const char* v = std::getenv("STRATA_LATENT_U_OLD");
        return v != nullptr && v[0] == '1';
    }();
    const int nt = (int) ((((int64_t) ELL / 32) * 8 + 31) / 32);
    const bool walk = u_walk && !u_old && fmt == 8 && nt <= 16;   // the register-held variant is Q8_0 only
    (void) n_groups;
    {   // one line per process: which kernel the U stage actually runs (a switch this cheap to get wrong
        // should never be inferred from a timing difference - it once hid a kernel that never launched)
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr, "latent u: %s kernel fmt=%d nt=%d H=%d rows=%lld ELL=%d\n",
                         walk ? "row-walk" : "per-column", fmt, nt, (int) H, (long long) cap_entries, (int) ELL);
        }
    }
    if (walk) {
        latent_u_walk_k<<<(unsigned) cap_entries, warps_per_block * 32, 0, (cudaStream_t) stream>>>(
            u_q8, ylq, ids, labels_dev, (int) E, (int) cap_entries, ELL, H, out);
        return;
    }
    dim3 grid((unsigned) cap_entries, (unsigned) ((H + rows_per_block - 1) / rows_per_block));
    latent_u_k<<<grid, warps_per_block * 32, 0, (cudaStream_t) stream>>>(u_q8, ylq, ids, labels_dev, (int) E,
                                                                         (int) cap_entries, ELL, H, out, fmt);
}

// knife-0 (STRATA_LATENT_U_GPU): the CPU share's U computed on the GPU and written straight into parts_.
// One block per routed row r (like latent_u_k), 8 warps.  A GPU row (r in dst[0..count)) copies hit_out;
// a CPU row computes U[labels[ids[r]]] . hq[r] with the same dp4a dot as latent_u_k (bit-compatible to ~1e-7).
__global__ void latent_u_matvec_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ hq,
                                  const int32_t* __restrict__ ids, const int32_t* __restrict__ labels,
                                  const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                  int E, int n_entries, int ELL, int H, float* __restrict__ parts, int fmt) {
    const int g = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hout = (int) (blockIdx.y * (blockDim.x >> 5)) + warp;
    const int nb = ELL / 32, wrow = nb * (fmt == 4 ? 18 : 34);   // Q4DU
    __shared__ int c_shared;
    __shared__ uint8_t hq_s[2016];   // nb*36 = 56*36, ELL=1792 (latent) fixed
    if (threadIdx.x == 0) c_shared = *count;
    __syncthreads();
    const int c = c_shared;
    if (hout >= H) return;
    const uint8_t* wb = u_q8 + (size_t) g * (size_t) H * (size_t) wrow + (size_t) hout * (size_t) wrow;
    for (int r = 0; r < n_entries; ++r) {
        const int32_t e = ids[r];
        if (e < 0 || e >= E || labels[e] != g) continue;
        bool gpu = false;
        for (int i = 0; i < c; ++i) if (dst[i] == r) { gpu = true; break; }
        if (gpu) continue;
        const uint8_t* x = hq + (size_t) r * (size_t) (nb * 36);
        for (int i = threadIdx.x; i < nb * 36; i += blockDim.x) hq_s[i] = x[i];   // once per block
        __syncthreads();
        float facc = 0.0f;
        if (fmt == 4) {
            for (int wi = lane; wi < nb * 4; wi += 32) {
                const int b = wi >> 2, j = (wi & 3) * 4;
                facc += lat_blk_dot_q4(wb + (size_t) b * 18, hq_s + (size_t) b * 36, j);
            }
        } else {
        for (int wi = lane; wi < nb * 8; wi += 32) {
            const int b = wi >> 3, j = (wi & 7) * 4;
            const float dx = __half2float(*(const __half*) (hq_s + (size_t) b * 36));
            const float dw = __half2float(__ldg((const __half*) (wb + (size_t) b * 34)));
            const int xw = *(const int*) (hq_s + (size_t) b * 36 + 4 + j);
            const int ww = latent_ld4_i8(wb + (size_t) b * 34 + 2 + j);
            facc += dx * dw * (float) __dp4a(xw, ww, 0);
        }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
        if (lane == 0) parts[(size_t) r * (size_t) H + hout] = facc;
        __syncthreads();
    }
}

void latent_u_matvec(const uint8_t* u_q8, const uint8_t* hq, const int32_t* ids, const int32_t* labels_dev,
                     const int32_t* dst, const int32_t* count, int64_t E, int64_t n_entries, int ELL, int H,
                     int G, float* parts, void* stream, int fmt) {
    if (n_entries <= 0 || G <= 0) return;
    const int warps_per_block = 8;
    dim3 grid((unsigned) G, (unsigned) ((H + warps_per_block - 1) / warps_per_block));
    latent_u_matvec_k<<<grid, warps_per_block * 32, 0, (cudaStream_t) stream>>>(
        u_q8, hq, ids, labels_dev, dst, count, (int) E, (int) n_entries, ELL, H, parts, fmt);
}

}  // namespace strata::kernels
