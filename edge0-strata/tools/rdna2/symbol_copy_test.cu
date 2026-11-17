// Minimal check: does cudaMemcpyToSymbol on an anonymous-namespace __device__
// int actually take effect under this HIP toolchain?  The raw3k arm relies on
// exactly this mechanism and its return value is ignored in lat_q3k_fused.cu.
#include <cstdio>
#include <cuda_runtime.h>

namespace {
__device__ int g_anon = 7;
}
__device__ int g_named = 7;

__global__ void read_kernel(int* out) {
    out[0] = g_anon;
    out[1] = g_named;
}

int main() {
    int v = 1, got[2] = {-1, -1};
    cudaError_t e1 = cudaMemcpyToSymbol(g_anon, &v, sizeof(int));
    cudaError_t e2 = cudaMemcpyToSymbol(g_named, &v, sizeof(int));
    int* d = nullptr; cudaMalloc(&d, 8);
    read_kernel<<<1, 1>>>(d);
    cudaDeviceSynchronize();
    cudaMemcpy(got, d, 8, cudaMemcpyDeviceToHost);
    std::printf("anon: err=%d (%s) -> %d (want 1)\n", (int) e1, cudaGetErrorString(e1), got[0]);
    std::printf("named: err=%d (%s) -> %d (want 1)\n", (int) e2, cudaGetErrorString(e2), got[1]);
    return (got[0] == 1 && got[1] == 1) ? 0 : 1;
}
