#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Does ILP ("collect into several VGPR accumulators, push many dots, combine at the end") change the rate -
// and does it change the dp4a-vs-dot8 ranking?  ILP = independent accumulators in flight.
#define VMAC_ILP(NAME, ASM, NINS, ILP)                                                                    \
    __global__ void NAME(const unsigned* w, const unsigned* x, const float* s, float* out, int iters) {   \
        unsigned wv[16], xv[16];                                                                          \
        for (int i = 0; i < 16; ++i) { wv[i] = w[i + threadIdx.x]; xv[i] = x[i + threadIdx.x]; }           \
        float acc = 0.0f;                                                                                  \
        for (int it = 0; it < iters; ++it) {                                                               \
            int sum[ILP];                                                                                  \
            for (int j = 0; j < ILP; ++j) sum[j] = 0;                                                      \
            for (int k = 0; k < NINS; ++k) {                                                               \
                for (int j = 0; j < ILP; ++j) {                                                            \
                    int d = 0;                                                                             \
                    asm volatile(ASM : "=v"(d)                                                             \
                                 : "v"(wv[(k + j) & 15]), "v"(xv[(k + j) & 15]), "v"(0));                  \
                    sum[j] += d;                                                                           \
                }                                                                                          \
            }                                                                                              \
            for (int j = 0; j < ILP; ++j) acc += s[(it + j) & 15] * (float) sum[j];                        \
        }                                                                                                  \
        out[threadIdx.x] = acc;                                                                            \
    }

VMAC_ILP(k_i8_ilp1, "v_dot4_i32_i8 %0, %1, %2, %3", 4, 1)
VMAC_ILP(k_i8_ilp4, "v_dot4_i32_i8 %0, %1, %2, %3", 4, 4)
VMAC_ILP(k_i8_ilp8, "v_dot4_i32_i8 %0, %1, %2, %3", 4, 8)
VMAC_ILP(k_d8_ilp1, "v_dot8_i32_i4 %0, %1, %2, %3", 2, 1)
VMAC_ILP(k_d8_ilp4, "v_dot8_i32_i4 %0, %1, %2, %3", 2, 4)
VMAC_ILP(k_d8_ilp8, "v_dot8_i32_i4 %0, %1, %2, %3", 2, 8)

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
      if (e != hipSuccess) printf("%-12s FAILED\n", #NAME);                           \
      else printf("%-12s %7.3f ms   %6.2f Gvalue/s\n", #NAME, ms, vals / (ms * 1e6)); }
    RUN(k_i8_ilp1) RUN(k_i8_ilp4) RUN(k_i8_ilp8) RUN(k_d8_ilp1) RUN(k_d8_ilp4) RUN(k_d8_ilp8)
    return 0;
}