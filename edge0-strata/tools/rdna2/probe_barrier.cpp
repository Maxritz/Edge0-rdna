#include <hip/hip_runtime.h>
#include <cstdio>

// RDNA2-first question for the A stage: what do its 40 block-wide __syncthreads per tile actually cost?
// The kernel runs 512 threads (16 waves) with 2 barriers per 64-value chunk x 20 chunks.  Measured budget:
// 4,631 ms / (62,939 blobs x 5 work items) = 14.7 us per tile-iteration.  If 40 barriers are a large part of
// that, the fix is structural (wave-owned rows, no block sync), not another dot-product tweak.
//
// Each kernel does the same arithmetic between synchronisations so the compiler cannot elide them.

#define BODY                                                                     \
    acc += (float) threadIdx.x * 1.0001f;                                        \
    if (acc > 1e30f) __threadfence_block();                                      // keeps the loop live

__global__ void k_nosync(float* out, int iters) {
    float acc = (float) threadIdx.x;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int b = 0; b < 40; ++b) { BODY }
    }
    out[threadIdx.x] = acc;
}

__global__ void k_blocksync(float* out, int iters) {
    __shared__ float s[1];
    float acc = (float) threadIdx.x;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int b = 0; b < 40; ++b) { BODY __syncthreads(); }
    }
    if (threadIdx.x == 0) s[0] = acc;
    out[threadIdx.x] = acc + s[0];
}

__global__ void k_wavesync(float* out, int iters) {
    float acc = (float) threadIdx.x;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int b = 0; b < 40; ++b) { BODY __syncwarp(); }
    }
    out[threadIdx.x] = acc;
}

int main() {
    const int iters = 20000, threads = 512, blocks = 40;
    float* dout; hipMalloc(&dout, threads * 4);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    float t_no = 0, t_bl = 0, t_wv = 0;
    for (int r = 0; r < 5; ++r) {
        hipEventRecord(a); k_nosync<<<blocks, threads>>>(dout, iters);   hipEventRecord(b); hipEventSynchronize(b);
        float t; hipEventElapsedTime(&t, a, b); t_no = t;
        hipEventRecord(a); k_blocksync<<<blocks, threads>>>(dout, iters); hipEventRecord(b); hipEventSynchronize(b);
        hipEventElapsedTime(&t, a, b); t_bl = t;
        hipEventRecord(a); k_wavesync<<<blocks, threads>>>(dout, iters);  hipEventRecord(b); hipEventSynchronize(b);
        hipEventElapsedTime(&t, a, b); t_wv = t;
    }
    const double nb = (double) iters * 40.0, nslots = nb * blocks;   // barriers per launch, barrier-slots total
    const double per_bl = (t_bl - t_no) * 1e6 / nb;                  // ms -> ns is x1e6, then / barriers
    printf("40-barrier pass, %d threads (16 waves), %d blocks:\n", threads, blocks);
    printf("  no sync      %8.3f ms\n", t_no);
    printf("  __syncthreads %6.3f ms   -> %.1f ns per barrier  (%.1f ns per barrier per block)\n",
           t_bl, per_bl, per_bl);
    printf("  __syncwarp    %6.3f ms   -> %.1f ns per wave sync\n", t_wv, (t_wv - t_no) * 1e6 / nb);
    printf("\nA tile-iteration budget is 14,700 ns (4,631 ms / 314,695 tiles).\n");
    printf("  40 block barriers  = %6.0f ns  = %.0f %% of a tile\n", per_bl * 40.0, 100.0 * per_bl * 40.0 / 14700.0);
    printf("  40 wave  syncs     = %6.0f ns  = %.0f %% of a tile\n", (t_wv - t_no) * 1e6 / nb * 40.0,
           100.0 * ((t_wv - t_no) * 1e6 / nb) * 40.0 / 14700.0);
    return 0;
}
