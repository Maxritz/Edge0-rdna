#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// The Q3_K unpack exactly as lat_q3k_fused.cu does it, timed against the dot it feeds (0.188 ns / 64 values).
// Per 16-value sub-block: 4 code words from qs words at 2-bit field j, 4 hmask words at bit hb, then the staged
// byte order.  Everything here is copied from lat_gu_decode/lat_stage_codes so the measurement is the real one.
static __device__ __forceinline__ unsigned vsub4(unsigned a) {
    return ((a | 0x80808080u) - 0x04040404u) ^ ((a ^ 0xFBFBFBFBu) & 0x80808080u);
}
__device__ __forceinline__ void unpack16(const uint32_t* qg, const uint32_t* hg, int j, int hb, uint32_t (&out)[4]) {
    uint32_t nat[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const uint32_t lo2 = (qg[m] >> (2 * j)) & 0x03030303u;
        const uint32_t hi1 = (hg[m] >> hb) & 0x01010101u;
        nat[m] = vsub4(lo2 | (4u * hi1));
    }
#pragma unroll
    for (int k = 0; k < 4; ++k)
        out[k] = __byte_perm(__byte_perm(nat[0], nat[1], k | ((k + 4) << 4)),
                             __byte_perm(nat[2], nat[3], k | ((k + 4) << 4)), 0x5410u);
}

__global__ void k_unpack(const unsigned* w, const unsigned* h, float* out, int iters, int sub) {
    uint32_t qg[8], hg[8];
    for (int i = 0; i < 8; ++i) { qg[i] = w[i + threadIdx.x]; hg[i] = h[i + threadIdx.x]; }
    uint32_t sink = 0;
    for (int it = 0; it < iters; ++it) {
        uint32_t staged[8][4];
        for (int s = 0; s < sub; ++s)                       // `sub` independent sub-blocks = the ILP knob
            unpack16(qg + 4 * (s & 1), hg + 4 * (s & 1), (it + s) & 3, (it + s) & 7, staged[s]);
        for (int s = 0; s < sub; ++s)
            for (int k = 0; k < 4; ++k) sink += staged[s][k];
    }
    out[threadIdx.x] = (float) sink;
}

int main() {
    const int iters = 100000;
    unsigned *dw, *dh; float* dout;
    unsigned hw[64], hh[64];
    for (int i = 0; i < 64; ++i) { hw[i] = 0x12345678u * (i + 1); hh[i] = 0x0f0f0f0fu * (i + 3); }
    hipMalloc(&dw, 256); hipMalloc(&dh, 256); hipMalloc(&dout, 128);
    hipMemcpy(dw, hw, 256, hipMemcpyHostToDevice); hipMemcpy(dh, hh, 256, hipMemcpyHostToDevice);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    printf("Q3_K unpack (qs 2-bit field + hmask + staged byte order), 40x128, gfx1031\n");
    printf("dot reference (measured earlier): 0.188 ns / 64 values at ILP16\n\n");
    for (int sub : {1, 2, 4, 8}) {
        float ms = 1e9f;
        for (int r = 0; r < 3; ++r) {
            hipEventRecord(a); k_unpack<<<40, 128>>>(dw, dh, dout, iters, sub);
            hipDeviceSynchronize(); hipEventRecord(b); hipEventSynchronize(b);
            float t; hipEventElapsedTime(&t, a, b); if (r == 2) ms = t;
        }
        const double blocks = (double) iters * sub * 40 * 128;   // 16-value sub-blocks
        const double ns64 = 1e6 * ms / (blocks / 4.0);           // ns per 64 values
        printf("unpack ILP%-2d  %7.3f ms   %.3f ns / 64 values   (%.1fx the dot, %.1f Gvalue/s)\n",
               sub, ms, ns64, ns64 / 0.188, blocks * 16.0 / (ms * 1e6));
    }
    return 0;
}