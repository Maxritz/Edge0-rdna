#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// A/B of the A stage's inner product, same work, on this card:
//   K1 = today: int8 codes x int8 activations (dp4a), then per-16-value float scaling (cvt+fma), which the
//        shipped kernel's ISA shows costs 4 of every 5 instructions.
//   K2 = candidate: Q3_K dequantized to fp16 with the sub-block scale BAKED IN at staging time, activations as
//        fp16, so the loop is only v_dot2_f32_f16 - no cvt, no per-block scaling.
// Both do 4 sub-blocks of 16 values per iteration.  Same 64 values either way.

__global__ void k_int8(const unsigned* w, const unsigned* x, const float* s, float* out, int iters) {
    unsigned wv[16], xv[16];
    for (int i = 0; i < 16; ++i) { wv[i] = w[i + threadIdx.x]; xv[i] = x[i + threadIdx.x]; }
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {                    // one 16-value sub-block
            int d[4] = {0, 0, 0, 0};
#pragma unroll
            for (int kw = 0; kw < 4; ++kw) {
                d[0] = __builtin_amdgcn_sdot4((int) wv[4 * b + kw], (int) xv[4 * b + kw], d[0], false);
                d[1] = __builtin_amdgcn_sdot4((int) wv[4 * b + kw], (int) xv[4 * b + kw + 1], d[1], false);
            }
            const float sw = s[(it + b) & 15];
            // the shipped epilogue: convert then scale, one fma per output
            acc += (sw * (float) d[0]) + (sw * (float) d[1]) + (sw * (float) d[2]) + (sw * (float) d[3]);
        }
    }
    out[threadIdx.x] = acc;
}

__global__ void k_f16(const unsigned* w, const unsigned* x, const float* s, float* out, int iters) {
    unsigned wv[16], xv[16];
    for (int i = 0; i < 16; ++i) { wv[i] = w[i + threadIdx.x]; xv[i] = x[i + threadIdx.x]; }
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {                // 8 halves x 2 operands = 16 values
                asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc)
                             : "v"(wv[(b * 2 + (k >> 2)) & 15]), "v"(xv[(b * 2 + (k >> 2)) & 15]));
            }
        }
        if (s[it & 15] > 1e30f) acc += 1.0f;             // keeps `it` live without touching the loop body
    }
    out[threadIdx.x] = acc;
}

int main() {
    const int iters = 200000;
    unsigned *dw, *dx; float* ds; float* dout;
    unsigned hw[64], hx[64]; float hs[16];
    for (int i = 0; i < 64; ++i) { hw[i] = 0x01020304u + i; hx[i] = 0x05060708u + i; }
    for (int i = 0; i < 16; ++i) hs[i] = 0.001f * (i + 1);
    hipMalloc(&dw, 256); hipMalloc(&dx, 256); hipMalloc(&ds, 64); hipMalloc(&dout, 128);
    hipMemcpy(dw, hw, 256, hipMemcpyHostToDevice); hipMemcpy(dx, hx, 256, hipMemcpyHostToDevice);
    hipMemcpy(ds, hs, 64, hipMemcpyHostToDevice);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    float ms_i = 0, ms_f = 0;
    for (int r = 0; r < 3; ++r) {
        hipEventRecord(a); k_int8<<<40, 128>>>(dw, dx, ds, dout, iters); hipEventRecord(b);
        hipEventSynchronize(b); float t; hipEventElapsedTime(&t, a, b); if (r == 2) ms_i = t;
        hipEventRecord(a); k_f16<<<40, 128>>>(dw, dx, ds, dout, iters); hipEventRecord(b);
        hipEventSynchronize(b); hipEventElapsedTime(&t, a, b); if (r == 2) ms_f = t;
    }
    const double vals = (double) iters * 64.0 * 40 * 128;   // value-groups processed
    printf("int8 dp4a + cvt/scale : %7.3f ms   %5.2f Gvalue/s\n", ms_i, vals / (ms_i * 1e6));
    printf("fp16 v_dot2 (baked sc): %7.3f ms   %5.2f Gvalue/s\n", ms_f, vals / (ms_f * 1e6));
    printf("=> fp16 speedup %.2fx\n", ms_i / ms_f);
    return 0;
}