#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Dot rate with CORRECT value accounting: an accumulator receives NINS dots x 4 values = 16 values per
// iteration (dp4a, NINS=4) or 2 x 8 = 16 (dot8).  So values = iters * 16 * ILP * threads, not 64 * ILP.
//
//   dp4a : v_dot4_i32_i8,  4 MACs/instr
//   dot8 : v_dot8_i32_i4,  8 MACs/instr

template <int NINS, int ILP, bool DOT8>
__global__ void kdot(const unsigned* w, const unsigned* x, float* out, int iters) {
    unsigned wv[16], xv[16];
    for (int i = 0; i < 16; ++i) { wv[i] = w[i + threadIdx.x]; xv[i] = x[i + threadIdx.x]; }
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
        int sum[ILP];
        for (int j = 0; j < ILP; ++j) sum[j] = 0;
        for (int k = 0; k < NINS; ++k) {
            for (int j = 0; j < ILP; ++j) {
                int d = 0;
                if (DOT8)
                    asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(d)
                                 : "v"(wv[(k + j) & 15]), "v"(xv[(k + j) & 15]), "v"(0));
                else
                    asm volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(d)
                                 : "v"(wv[(k + j) & 15]), "v"(xv[(k + j) & 15]), "v"(0));
                sum[j] += d;
            }
        }
        for (int j = 0; j < ILP; ++j) acc += (float) sum[j];
    }
    out[threadIdx.x] = acc;
}

static const int kIters = 100000, kBlocks = 40, kThreads = 128;

template <int NINS, int ILP, bool DOT8>
static void run(const char* name, const unsigned* w, const unsigned* x, float* out) {
    float ms = 1e9f;
    for (int r = 0; r < 3; ++r) {
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        hipEventRecord(a);
        kdot<NINS, ILP, DOT8><<<kBlocks, kThreads>>>(w, x, out, kIters);
        hipDeviceSynchronize();
        hipEventRecord(b); hipEventSynchronize(b);
        float t; hipEventElapsedTime(&t, a, b);
        if (r == 2) ms = t;
        hipEventDestroy(a); hipEventDestroy(b);
    }
    const double values = (double) kIters * 16.0 * ILP * kBlocks * kThreads;
    printf("%-14s ILP%-3d %7.3f ms  %8.1f Gvalue/s  %.3f ns / 64 values\n",
           name, ILP, ms, values / (ms * 1e6), 1e9 * ms * 64.0 / values);
}

int main() {
    unsigned *dw, *dx; float* dout;
    unsigned hw[64], hx[64];
    for (int i = 0; i < 64; ++i) { hw[i] = 0x12345678u * (i + 1); hx[i] = 0x23456789u * (i + 1); }
    hipMalloc(&dw, 256); hipMalloc(&dx, 256); hipMalloc(&dout, 128);
    hipMemcpy(dw, hw, 256, hipMemcpyHostToDevice); hipMemcpy(dx, hx, 256, hipMemcpyHostToDevice);
    run<4, 1, false>("dp4a", dw, dx, dout);
    run<4, 4, false>("dp4a", dw, dx, dout);
    run<4, 8, false>("dp4a", dw, dx, dout);
    run<4, 16, false>("dp4a", dw, dx, dout);
    run<2, 8, true>("dot8", dw, dx, dout);
    run<2, 16, true>("dot8", dw, dx, dout);
    return 0;
}
