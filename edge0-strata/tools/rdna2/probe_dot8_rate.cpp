#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Same A-stage inner product, three ways, same 64 values per iteration, per-thread register operands:
//   k_int8 : dp4a (int8 codes x int8 activations) + per-16-value float scaling   -- shipped
//   k_d8   : v_dot8_i32_i4 (int4 codes x int4 activations) + the same scaling    -- the only denser dot
// dot8 is 8 MACs/instruction against dp4a's 4, so its BEST case here is 1.33x (the scaling does not shrink).

__global__ void k_int8(const unsigned* w, const unsigned* x, const float* s, float* out, int iters) {
    unsigned wv[16], xv[16];
    for (int i = 0; i < 16; ++i) { wv[i] = w[i + threadIdx.x]; xv[i] = x[i + threadIdx.x]; }
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            int d[4] = {0, 0, 0, 0};
#pragma unroll
            for (int kw = 0; kw < 4; ++kw) {
                d[0] = __builtin_amdgcn_sdot4((int) wv[4 * b + kw], (int) xv[4 * b + kw], d[0], false);
                d[1] = __builtin_amdgcn_sdot4((int) wv[4 * b + kw], (int) xv[4 * b + kw + 1], d[1], false);
            }
            const float sw = s[(it + b) & 15];
            acc += (sw * (float) d[0]) + (sw * (float) d[1]) + (sw * (float) d[2]) + (sw * (float) d[3]);
        }
    }
    out[threadIdx.x] = acc;
}

__global__ void k_d8(const unsigned* w, const unsigned* x, const float* s, float* out, int iters) {
    unsigned wv[16], xv[16];
    for (int i = 0; i < 16; ++i) { wv[i] = w[i + threadIdx.x]; xv[i] = x[i + threadIdx.x]; }
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {                 // one 16-value sub-block = 2 dot8 instructions
            int d[2] = {0, 0};
            asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(d[0])
                         : "v"(wv[2 * b]), "v"(xv[2 * b]), "v"(0));
            asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(d[1])
                         : "v"(wv[2 * b + 1]), "v"(xv[2 * b + 1]), "v"(0));
            const float sw = s[(it + b) & 15];
            acc += (sw * (float) d[0]) + (sw * (float) d[1]);
        }
    }
    out[threadIdx.x] = acc;
}

int main() {
    const int iters = 200000;
    unsigned *dw, *dx; float* ds; float* dout;
    unsigned hw[64], hx[64]; float hs[16];
    for (int i = 0; i < 64; ++i) { hw[i] = 0x12345678u * (i + 1); hx[i] = 0x23456789u * (i + 1); }
    for (int i = 0; i < 16; ++i) hs[i] = 0.001f * (i + 1);
    hipMalloc(&dw, 256); hipMalloc(&dx, 256); hipMalloc(&ds, 64); hipMalloc(&dout, 128);
    hipMemcpy(dw, hw, 256, hipMemcpyHostToDevice); hipMemcpy(dx, hx, 256, hipMemcpyHostToDevice);
    hipMemcpy(ds, hs, 64, hipMemcpyHostToDevice);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    float ms_i = 0, ms_8 = 0;
    for (int r = 0; r < 3; ++r) {
        hipEventRecord(a); k_int8<<<40, 128>>>(dw, dx, ds, dout, iters); hipEventRecord(b);
        hipEventSynchronize(b); float t; hipEventElapsedTime(&t, a, b); if (r == 2) ms_i = t;
        hipEventRecord(a); k_d8<<<40, 128>>>(dw, dx, ds, dout, iters); hipEventRecord(b);
        hipEventSynchronize(b); hipEventElapsedTime(&t, a, b); if (r == 2) ms_8 = t;
    }
    const double vals = (double) iters * 64.0 * 40 * 128;
    printf("int8 dp4a  (4 MAC/instr) : %7.3f ms   %5.2f Gvalue/s\n", ms_i, vals / (ms_i * 1e6));
    printf("int4 dot8  (8 MAC/instr) : %7.3f ms   %5.2f Gvalue/s\n", ms_8, vals / (ms_8 * 1e6));
    printf("=> dot8 speedup %.2fx  (ceiling from the scaling alone: 1.33x)\n", ms_i / ms_8);
    return 0;
}