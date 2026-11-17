// Correctness probe for the buffer-resource formulation, independent of clock state (it compares VALUES, not time).
//   k_a: read 16 bytes with the engine's lat_ld4 (flat dword loads + funnel shifts)
//   k_b: read the same 16 bytes through a buffer descriptor whose base is the blob aligned DOWN to 4096 and whose
//        range is 1 MiB - i.e. the formulation the standalone timing probe used, which the first port did NOT.
// Both write the four aligned words; the host compares them word for word.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

using i32x4 = int __attribute__((ext_vector_type(4)));
struct buf_res { unsigned long long ptr; unsigned range; unsigned config; };

__device__ inline i32x4 make_srsrc(const void* p, unsigned range) {
    buf_res r{(unsigned long long) (uintptr_t) p, range, 0x110000u};
    return *reinterpret_cast<const i32x4*>(&r);
}
__device__ unsigned buf_load_b32(i32x4 s, unsigned voff, unsigned soff, unsigned coh)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ __forceinline__ void lat_ld4(const uint8_t* p, int sh, uint32_t (&w)[4]) {
    const uint32_t* q = (const uint32_t*) (p - (sh >> 3));
    w[0] = __funnelshift_r(q[0], q[1], sh);
    w[1] = __funnelshift_r(q[1], q[2], sh);
    w[2] = __funnelshift_r(q[2], q[3], sh);
    w[3] = __funnelshift_r(q[3], q[4], sh);
}

#define ROW 550
#define NP  8                     /* how many 256-aligned pages of junk precede the data */

__global__ void k_a(const uint8_t* rows, unsigned* out, int nrows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * ROW / 16) return;
    const uint8_t* p = rows + (size_t) i * 16;
    const int sh = 8 * ((int) ((uintptr_t) p & 3));
    uint32_t w[4];
    lat_ld4(p, sh, w);
    for (int k = 0; k < 4; ++k) out[i * 4 + k] = w[k];
}

__global__ void k_b(const uint8_t* rows, unsigned* out, int nrows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * ROW / 16) return;
    const uint8_t* p = rows + (size_t) i * 16;
    const int sh = 8 * ((int) ((uintptr_t) p & 3));
    // THE FORMULATION: 4096-aligned base + a generous range, offset by a flat byte count
    const uint8_t* base = (const uint8_t*) ((uintptr_t) rows & ~(uintptr_t) 4095u);
    const i32x4 s = make_srsrc(base, 1u << 20);
    const uint8_t* a = p - (sh >> 3);
    const unsigned o = (unsigned) (a - base);
    uint32_t q[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) q[k] = buf_load_b32(s, o + 4u * (unsigned) k, 0u, 0u);
    uint32_t w[4];
    w[0] = __funnelshift_r(q[0], q[1], sh);
    w[1] = __funnelshift_r(q[1], q[2], sh);
    w[2] = __funnelshift_r(q[2], q[3], sh);
    w[3] = __funnelshift_r(q[3], q[4], sh);
    for (int k = 0; k < 4; ++k) out[i * 4 + k] = w[k];
}

int main() {
    const int nrows = 64;
    const size_t bytes = (size_t) NP * 4096 + nrows * ROW + 8192;
    uint8_t* d = nullptr;
    if (hipMalloc(&d, bytes) != hipSuccess) { printf("hipMalloc failed\n"); return 1; }
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < bytes; ++i) h[i] = (uint8_t) ((i * 37 + (i >> 8) * 11) & 0xFF);   // cheap non-constant junk
    hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice);
    const uint8_t* rows = d + (size_t) NP * 4096 + 3;      // force a 3-byte misalignment, like a 550 B row

    const size_t nvals = (size_t) (nrows * ROW / 16) * 4;
    unsigned *oa = nullptr, *ob = nullptr;
    hipMalloc(&oa, nvals * 4);
    hipMalloc(&ob, nvals * 4);
    const int threads = 128, blocks = (int) ((nrows * ROW / 16 + threads - 1) / threads);
    k_a<<<blocks, threads>>>(rows, oa, nrows);
    k_b<<<blocks, threads>>>(rows, ob, nrows);
    hipDeviceSynchronize();
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) { printf("launch error: %s\n", hipGetErrorString(e)); return 1; }
    std::vector<unsigned> va(nvals), vb(nvals);
    hipMemcpy(va.data(), oa, nvals * 4, hipMemcpyDeviceToHost);
    hipMemcpy(vb.data(), ob, nvals * 4, hipMemcpyDeviceToHost);
    size_t diff = 0, first = (size_t) -1;
    for (size_t i = 0; i < nvals; ++i) if (va[i] != vb[i]) { if (first == (size_t) -1) first = i; ++diff; }
    printf("=== probe_bufcheck: flat dword loads vs 4096-aligned srsrc (values, not time) ===\n");
    printf("  words compared %zu   differing %zu\n", nvals, diff);
    if (diff) {
        printf("  FIRST DIFF at word %zu (row %zu, word-in-row %zu): flat=%08x  srsrc=%08x\n",
               first, first / 4, first % 4, va[first], vb[first]);
        printf("  VERDICT: FORMULATION WRONG\n");
        return 1;
    }
    printf("  VERDICT: IDENTICAL - the formulation is correct\n");
    return 0;
}
