// u_probe.cu - standalone timing + bit-equality probe for the latent U stage at the shape the engine
// actually runs it (ELL 1280, H 2560, G 8, rows = T*K 5120).  Not part of the build; the point is a
// seconds-long loop for kernel variants, with the per-row kernel as the reference.
//
//   hipcc -O3 --offload-arch=gfx1031 u_probe.cu -o u_probe.exe
//
// Positions are the same as latent_u_k: one warp per output scalar, the lane's 10 terms in the
// `wi = lane + 32i` order, the same 5-step shuffle tree.  Any variant that reproduces those numbers
// byte for byte is a candidate.
#include <hip/hip_fp16.h>

#include "strata/hip_compat/cuda_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define ck(x, m) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static constexpr int ELL = 1280;
static constexpr int H = 2560;
static constexpr int G = 8;
static constexpr int K = 10;
static constexpr int T = 512;
static constexpr int E = 512;
static constexpr int CAP = T * K;
static constexpr int NB = ELL / 32;          // 40 q8_1 blocks a row
static constexpr int WROW = NB * 34;         // Q8_0 row: {f16 d; int8 qs[32]}
static constexpr int XROW = NB * 36;         // q8_1 row: {f16 d; f16 s; int8 qs[32]}

__device__ __forceinline__ int ld4_i8(const uint8_t* p) {
    const unsigned short a = *(const unsigned short*) p;
    const unsigned short b = *(const unsigned short*) (p + 2);
    return (int) ((unsigned) a | ((unsigned) b << 16));
}

// ---- the reference: latent.cu's latent_u_k, verbatim arithmetic -------------------------------------
__global__ void u_row_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                        const int32_t* __restrict__ ids, const int32_t* __restrict__ labels, int nE, int cap,
                        int H_, int out_stride, float* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= cap) return;
    const int32_t e = ids[r];
    const int g = (e >= 0 && e < nE) ? labels[e] : 0;
    const uint8_t* wbase = u_q8 + (size_t) g * (size_t) H_ * (size_t) WROW;
    const uint8_t* x = ylq + (size_t) r * (size_t) XROW;
    float* o = out + (size_t) r * (size_t) out_stride;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = (int) (blockIdx.y * (blockDim.x >> 5)) + warp;
    if (h >= H_) return;
    const uint8_t* wb = wbase + (size_t) h * (size_t) WROW;
    float facc = 0.0f;
    for (int wi = lane; wi < NB * 8; wi += 32) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        const float dx = __half2float(*(const __half*) (x + (size_t) b * 36));
        const float dw = __half2float(*(const __half*) (wb + (size_t) b * 34));
        const int xw = *(const int*) (x + (size_t) b * 36 + 4 + j);
        const int ww = ld4_i8(wb + (size_t) b * 34 + 2 + j);
        facc += dx * dw * (float) __dp4a(xw, ww, 0);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) o[h] = facc;
}

// ---- variant 1: x held in registers, one warp walks many h (x reuse) -------------------------------
__global__ void u_xreuse_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                           const int32_t* __restrict__ ids, const int32_t* __restrict__ labels, int nE, int cap,
                           int H_, int out_stride, float* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= cap) return;
    const int32_t e = ids[r];
    const int g = (e >= 0 && e < nE) ? labels[e] : 0;
    const uint8_t* wbase = u_q8 + (size_t) g * (size_t) H_ * (size_t) WROW;
    const uint8_t* x = ylq + (size_t) r * (size_t) XROW;
    float* o = out + (size_t) r * (size_t) out_stride;
    const int lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    // this lane's terms, in latent_u_k's order: 10 of them for ELL 1280
    int xbase[16], xword[16];
    float dxs[16];
    int nt = 0;
    for (int wi = lane; wi < NB * 8; wi += 32, ++nt) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        xbase[nt] = b * 36;
        xword[nt] = b * 36 + 4 + j;
        dxs[nt] = __half2float(*(const __half*) (x + (size_t) b * 36));
    }
    for (int h = (threadIdx.x >> 5); h < H_; h += nwarp) {
        const uint8_t* wb = wbase + (size_t) h * (size_t) WROW;
        float facc = 0.0f;
        for (int i = 0; i < nt; ++i) {
            const int b = xbase[i] / 36;
            const float dw = __half2float(*(const __half*) (wb + (size_t) b * 34));
            const int j = (xword[i] - 4) - b * 36;
            const int ww = ld4_i8(wb + (size_t) b * 34 + 2 + j);
            facc += dxs[i] * dw * (float) __dp4a(*(const int*) (x + (size_t) xword[i]), ww, 0);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
        if (lane == 0) o[h] = facc;
    }
}

// ---- variant 2: warp holds x AND walks h, weights read once per h (x reuse + coalesced h) ---------
// Same arithmetic; the only difference from variant 1 is that h is a per-warp slab so a block's warps
// write H contiguously.
__global__ void u_xreuse2_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                            const int32_t* __restrict__ ids, const int32_t* __restrict__ labels, int nE, int cap,
                            int H_, int out_stride, float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int rows_per_block = 4;                    // 4 rows a block, 8 warps = 2 warps a row
    const int r0 = blockIdx.x * rows_per_block;
    int xbase[16], xword[16];
    float dxs[16];
    for (int rr = 0; rr < rows_per_block; ++rr) {
        const int r = r0 + rr;
        if (r >= cap) return;
        const int32_t e = ids[r];
        const int g = (e >= 0 && e < nE) ? labels[e] : 0;
        const uint8_t* wbase = u_q8 + (size_t) g * (size_t) H_ * (size_t) WROW;
        const uint8_t* x = ylq + (size_t) r * (size_t) XROW;
        float* o = out + (size_t) r * (size_t) out_stride;
        int nt = 0;
        for (int wi = lane; wi < NB * 8; wi += 32, ++nt) {
            const int b = wi >> 3, j = (wi & 7) * 4;
            xbase[nt] = b;
            xword[nt] = b * 36 + 4 + j;
            dxs[nt] = __half2float(*(const __half*) (x + (size_t) b * 36));
        }
        for (int h = (threadIdx.x >> 5); h < H_; h += nwarp) {
            const uint8_t* wb = wbase + (size_t) h * (size_t) WROW;
            float facc = 0.0f;
            for (int i = 0; i < nt; ++i) {
                const int b = xbase[i];
                const float dw = __half2float(*(const __half*) (wb + (size_t) b * 34));
                const int j = (xword[i] - 4) - b * 36;
                const int ww = ld4_i8(wb + (size_t) b * 34 + 2 + j);
                facc += dxs[i] * dw * (float) __dp4a(*(const int*) (x + (size_t) xword[i]), ww, 0);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
            if (lane == 0) o[h] = facc;
        }
    }
}

// ---- variant 3: two output columns per iteration (two accumulators, one x set) ---------------------
__global__ void u_xreuse2h_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                             const int32_t* __restrict__ ids, const int32_t* __restrict__ labels, int nE, int cap,
                             int H_, int out_stride, float* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= cap) return;
    const int32_t e = ids[r];
    const int g = (e >= 0 && e < nE) ? labels[e] : 0;
    const uint8_t* wbase = u_q8 + (size_t) g * (size_t) H_ * (size_t) WROW;
    const uint8_t* x = ylq + (size_t) r * (size_t) XROW;
    float* o = out + (size_t) r * (size_t) out_stride;
    const int lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    int xbase[16], xword[16];
    float dxs[16];
    int nxw[16];
    int nt = 0;
    for (int wi = lane; wi < NB * 8; wi += 32, ++nt) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        xbase[nt] = b;
        xword[nt] = b * 36 + 4 + j;
        nxw[nt] = *(const int*) (x + (size_t) b * 36 + 4 + j);
        dxs[nt] = __half2float(*(const __half*) (x + (size_t) b * 36));
    }
    for (int h = (threadIdx.x >> 5); h < H_; h += 2 * nwarp) {
        const int h1 = h + nwarp;
        const uint8_t* wb0 = wbase + (size_t) h * (size_t) WROW;
        const uint8_t* wb1 = wbase + (size_t) (h1 < H_ ? h1 : h) * (size_t) WROW;
        float f0 = 0.0f, f1 = 0.0f;
        for (int i = 0; i < nt; ++i) {
            const int b = xbase[i];
            const int j = (xword[i] - 4) - b * 36;
            const float dw0 = __half2float(*(const __half*) (wb0 + (size_t) b * 34));
            const float dw1 = __half2float(*(const __half*) (wb1 + (size_t) b * 34));
            f0 += dxs[i] * dw0 * (float) __dp4a(nxw[i], ld4_i8(wb0 + (size_t) b * 34 + 2 + j), 0);
            f1 += dxs[i] * dw1 * (float) __dp4a(nxw[i], ld4_i8(wb1 + (size_t) b * 34 + 2 + j), 0);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) f0 += __shfl_down_sync(0xffffffffu, f0, off);
        if (lane == 0) o[h] = f0;
        if (h1 < H_) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) f1 += __shfl_down_sync(0xffffffffu, f1, off);
            if (lane == 0) o[h1] = f1;
        }
    }
}

static int occupancy(const void* fn, int block, size_t smem) {
    int n = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, block, smem) != hipSuccess) { (void) hipGetLastError(); return -1; }
    return n;
}

static float timeit(const void* fn, bool v1, bool v2, bool v3, uint8_t* dU, uint8_t* dHq, int32_t* ids, int32_t* labels,
                    float* out, int iters) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev");
    ck(cudaEventCreate(&b), "ev");
    // warm
    for (int it = 0; it < 2; ++it) {
        if (v3) u_xreuse2h_k<<<CAP, 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out); else if (v1) u_xreuse_k<<<CAP, 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out);
        else if (v2) u_xreuse2_k<<<(CAP + 3) / 4, 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out);
        else u_row_k<<<dim3(CAP, H / 8), 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out);
    }
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaEventRecord(a), "rec");
    for (int it = 0; it < iters; ++it) {
        if (v3) u_xreuse2h_k<<<CAP, 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out); else if (v1) u_xreuse_k<<<CAP, 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out);
        else if (v2) u_xreuse2_k<<<(CAP + 3) / 4, 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out);
        else u_row_k<<<dim3(CAP, H / 8), 256>>>(dU, dHq, ids, labels, E, CAP, H, H, out);
    }
    ck(cudaEventRecord(b), "rec");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    ck(cudaEventDestroy(a), "ev");
    ck(cudaEventDestroy(b), "ev");
    return ms / (float) iters;
}

int main() {
    const size_t ubytes = (size_t) G * H * WROW;      // 8 rows-groups
    const size_t hbytes = (size_t) CAP * XROW;
    const size_t obytes = (size_t) CAP * H * 4;
    std::printf("shape: ELL %d H %d G %d rows %d  U %.1f MB  Hq %.1f MB  out %.1f MB  FLOP/layer %.2f G\n", ELL, H, G,
                CAP, ubytes / 1048576.0, hbytes / 1048576.0, obytes / 1048576.0,
                (double) CAP * H * ELL * 2 / 1e9);

    std::vector<uint8_t> hU(ubytes), hHq(hbytes);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& v : hU) v = (uint8_t) byte(rng);
    for (auto& v : hHq) v = (uint8_t) byte(rng);
    // the fp16 scales must be finite: force them to a small positive half
    for (size_t i = 0; i < ubytes; i += WROW)
        for (int b = 0; b < NB; ++b) *(uint16_t*) &hU[i + (size_t) b * 34] = (uint16_t) 0x2400;   // ~0.0625
    for (size_t i = 0; i < hbytes; i += XROW)
        for (int b = 0; b < NB; ++b) *(uint16_t*) &hHq[i + (size_t) b * 36] = (uint16_t) 0x2400;

    std::vector<int32_t> hIds(CAP), hLab(E);
    for (int e = 0; e < E; ++e) hLab[e] = e % G;
    for (int r = 0; r < CAP; ++r) hIds[r] = (int32_t) ((r * 37) % E);   // spread across groups

    uint8_t *dU = nullptr, *dHq = nullptr;
    int32_t *ids = nullptr, *lab = nullptr;
    float *o1 = nullptr, *o2 = nullptr, *o3 = nullptr, *o4 = nullptr;
    ck(cudaMalloc(&dU, ubytes), "u");
    ck(cudaMalloc(&dHq, hbytes), "h");
    ck(cudaMalloc(&ids, (size_t) CAP * 4), "ids");
    ck(cudaMalloc(&lab, (size_t) E * 4), "lab");
    for (auto* p : {&o1, &o2, &o3, &o4}) ck(cudaMalloc(p, obytes), "out");
    ck(cudaMemcpy(dU, hU.data(), ubytes, cudaMemcpyHostToDevice), "u");
    ck(cudaMemcpy(dHq, hHq.data(), hbytes, cudaMemcpyHostToDevice), "h");
    ck(cudaMemcpy(ids, hIds.data(), (size_t) CAP * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemcpy(lab, hLab.data(), (size_t) E * 4, cudaMemcpyHostToDevice), "lab");

    std::printf("occupancy (blocks/cu, 2048 threads/cu): row %d   xreuse %d   xreuse2 %d\n",
                occupancy((const void*) u_row_k, 256, 0), occupancy((const void*) u_xreuse_k, 256, 0),
                occupancy((const void*) u_xreuse2_k, 256, 0));

    ck(cudaMemset(o1, 0, obytes), "m");
    ck(cudaMemset(o2, 0, obytes), "m");
    ck(cudaMemset(o3, 0, obytes), "m");
    u_row_k<<<dim3(CAP, H / 8), 256>>>(dU, dHq, ids, lab, E, CAP, H, H, o1);
    u_xreuse_k<<<CAP, 256>>>(dU, dHq, ids, lab, E, CAP, H, H, o2);
    u_xreuse2_k<<<(CAP + 3) / 4, 256>>>(dU, dHq, ids, lab, E, CAP, H, H, o3);
    ck(cudaMemset(o4, 0, obytes), "m");
    u_xreuse2h_k<<<CAP, 256>>>(dU, dHq, ids, lab, E, CAP, H, H, o4);
    ck(cudaDeviceSynchronize(), "sync");

    std::vector<float> r1(obytes / 4), r2(obytes / 4), r3(obytes / 4), r4(obytes / 4);
    ck(cudaMemcpy(r1.data(), o1, obytes, cudaMemcpyDeviceToHost), "c");
    ck(cudaMemcpy(r2.data(), o2, obytes, cudaMemcpyDeviceToHost), "c");
    ck(cudaMemcpy(r3.data(), o3, obytes, cudaMemcpyDeviceToHost), "c");
    ck(cudaMemcpy(r4.data(), o4, obytes, cudaMemcpyDeviceToHost), "c");
    const int same2 = std::memcmp(r1.data(), r2.data(), obytes) == 0;
    const int same3 = std::memcmp(r1.data(), r3.data(), obytes) == 0;
    const int same4 = std::memcmp(r1.data(), r4.data(), obytes) == 0;
    double s = 0;
    for (float v : r1) s += (double) v * v;
    std::printf("bit-equality vs per-row: xreuse %s   xreuse2 %s   xreuse2h %s   (reference L2 %.6g)\n",
                same2 ? "SAME" : "DIFFER", same3 ? "SAME" : "DIFFER", same4 ? "SAME" : "DIFFER", s);

    const int it = 50;
    const double flop = (double) CAP * H * ELL * 2;
    const double ms0 = timeit(nullptr, false, false, false, dU, dHq, ids, lab, o1, it);
    const double ms1 = timeit(nullptr, true, false, false, dU, dHq, ids, lab, o2, it);
    const double ms2 = timeit(nullptr, false, true, false, dU, dHq, ids, lab, o3, it);
    const double ms3 = timeit(nullptr, false, false, true, dU, dHq, ids, lab, o4, it);
    std::printf("per-row    %8.3f ms  %7.1f GFLOP/s  (engine reports ~32.7 ms/layer)\n", ms0, flop / (ms0 * 1e6));
    std::printf("xreuse     %8.3f ms  %7.1f GFLOP/s  %+.1f%%\n", ms1, flop / (ms1 * 1e6), (ms1 / ms0 - 1.0) * 100.0);
    std::printf("xreuse2    %8.3f ms  %7.1f GFLOP/s  %+.1f%%\n", ms2, flop / (ms2 * 1e6), (ms2 / ms0 - 1.0) * 100.0);
    std::printf("xreuse2h   %8.3f ms  %7.1f GFLOP/s  %+.1f%%\n", ms3, flop / (ms3 * 1e6), (ms3 / ms0 - 1.0) * 100.0);
    return 0;
}

