#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// The dot forms that ACTUALLY assemble for gfx1031 (validated with -c, not -S), executed and timed.
// Per 16-value sub-block: NINS instructions + one float scale, identical for every variant.
#define VMAC(NAME, ASM, NINS)                                                                             \
    __global__ void NAME(const unsigned* w, const unsigned* x, const float* s, float* out, int iters) {   \
        unsigned wv[16], xv[16];                                                                          \
        for (int i = 0; i < 16; ++i) { wv[i] = w[i + threadIdx.x]; xv[i] = x[i + threadIdx.x]; }           \
        float acc = 0.0f;                                                                                  \
        for (int it = 0; it < iters; ++it) {                                                               \
            for (int b = 0; b < 4; ++b) {                                                                  \
                int sum = 0;                                                                               \
                for (int k = 0; k < NINS; ++k) {                                                           \
                    int d = 0;                                                                             \
                    asm volatile(ASM : "=v"(d)                                                             \
                                 : "v"(wv[(b * NINS + k) & 15]), "v"(xv[(b * NINS + k) & 15]), "v"(0));    \
                    sum += d;                                                                              \
                }                                                                                          \
                acc += s[(it + b) & 15] * (float) sum;                                                     \
            }                                                                                              \
        }                                                                                                  \
        out[threadIdx.x] = acc;                                                                            \
    }

VMAC(k_i8_signed,   "v_dot4_i32_i8 %0, %1, %2, %3", 4)    // 4 MACs/instr, signed (dp4a form)
VMAC(k_u8_unsigned, "v_dot4_u32_u8 %0, %1, %2, %3", 4)    // 4 MACs/instr, unsigned (codes biased +8)
VMAC(k_i4_dot8,     "v_dot8_i32_i4 %0, %1, %2, %3", 2)    // 8 MACs/instr, signed int4
VMAC(k_u4_dot8,     "v_dot8_u32_u4 %0, %1, %2, %3", 2)    // 8 MACs/instr, unsigned int4
VMAC(k_i16_dot2,    "v_dot2_i32_i16 %0, %1, %2, %3", 8)   // 2 MACs/instr, int16
VMAC(k_f16_dot2,    "v_dot2_f32_f16 %0, %1, %2, %3", 8)   // 2 MACs/instr, fp16

int main() {
    const int iters = 100000;
    unsigned *dw, *dx; float* ds; float* dout;
    unsigned hw[64], hx[64]; float hs[16];
    for (int i = 0; i < 64; ++i) { hw[i] = 0x12345678u * (i + 1); hx[i] = 0x23456789u * (i + 1); }
    for (int i = 0; i < 16; ++i) hs[i] = 0.001f * (i + 1);
    hipMalloc(&dw, 256); hipMalloc(&dx, 256); hipMalloc(&ds, 64); hipMalloc(&dout, 128);
    hipMemcpy(dw, hw, 256, hipMemcpyHostToDevice); hipMemcpy(dx, hx, 256, hipMemcpyHostToDevice);
    hipMemcpy(ds, hs, 64, hipMemcpyHostToDevice);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
#define RUN(NAME)                                                                     \
    { float ms = 1e9f; hipError_t e = hipSuccess;                                      \
      for (int r = 0; r < 3; ++r) {                                                   \
          hipEventRecord(a); NAME<<<40, 128>>>(dw, dx, ds, dout, iters);              \
          e = hipDeviceSynchronize(); hipEventRecord(b); hipEventSynchronize(b);      \
          float t; hipEventElapsedTime(&t, a, b); if (r == 2) ms = t; }               \
      const double vals = (double) iters * 64.0 * 40 * 128;                           \
      if (e != hipSuccess) printf("%-14s FAILED: %s\n", #NAME, hipGetErrorString(e)); \
      else printf("%-14s %7.3f ms   %6.2f Gvalue/s\n", #NAME, ms, vals / (ms * 1e6)); }
    RUN(k_i8_signed) RUN(k_u8_unsigned) RUN(k_i4_dot8) RUN(k_u4_dot8) RUN(k_i16_dot2) RUN(k_f16_dot2)
    return 0;
}