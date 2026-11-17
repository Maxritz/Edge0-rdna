// Probe: does the buffer-resource addressing mode beat lat_ld4's flat/generic loads?
//
// The A stage's weight rows are 550 bytes, so a row is 2-byte misaligned to any 16-byte boundary and the kernel
// realigns with lat_ld4: five overlapping 32-bit loads + four __funnelshift_r per 16 values.
//   k_ld4  - the shipped path (flat/generic dword loads)
//   k_bufd - the SAME five dword loads and the SAME four funnel shifts, but through a buffer resource
//            (make_srsrc + llvm.amdgcn.raw.buffer.load.i32): identical work, different addressing mode only
//   k_buf  - a 128-bit buffer load + manual extraction (kept to show why that form is not a drop-in)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
struct buf_res { uint64_t ptr; uint32_t range; uint32_t config; };

__device__ inline i32x4 make_srsrc(const void* p, uint32_t range) {
    buf_res r{(uint64_t)p, range, 0x110000u};
    return *reinterpret_cast<const i32x4*>(&r);
}
__device__ uint32_t buf_load_b32(i32x4 s, uint32_t voff, uint32_t soff, uint32_t coh)
    __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ __uint128_t buf_load_b128(i32x4 s, uint32_t voff, uint32_t soff, uint32_t coh)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

// the shipped realignment, verbatim
__device__ __forceinline__ void lat_ld4(const uint8_t* p, int sh, uint32_t (&w)[4]) {
    const uint32_t* q = (const uint32_t*) (p - (sh >> 3));
    w[0] = __funnelshift_r(q[0], q[1], sh);
    w[1] = __funnelshift_r(q[1], q[2], sh);
    w[2] = __funnelshift_r(q[2], q[3], sh);
    w[3] = __funnelshift_r(q[3], q[4], sh);
}

#define ROW 550
#define CHUNKS 20
#define SHIFT 2

__global__ void k_ld4(const uint8_t* rows, int nrows, unsigned* out, int iters) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) % nrows;
    const uint8_t* row = rows + (size_t) r * ROW;
    unsigned acc = 0;
    for (int it = 0; it < iters; ++it)
        for (int c = 0; c < CHUNKS; ++c) {
            uint32_t w[4];
            const int off = ((it + c) * 16) % 480;
            lat_ld4(row + off + SHIFT, SHIFT * 8, w);
            acc += w[0] ^ w[1] ^ w[2] ^ w[3];
        }
    out[r] = acc;
}

__global__ void k_bufd(const uint8_t* rows, int nrows, unsigned* out, int iters) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) % nrows;
    // CORRECT usage: base is the 256-byte-aligned allocation, the row is a flat BYTE OFFSET (voffset)
    i32x4 s = make_srsrc(rows, (uint32_t) (nrows * ROW));
    unsigned acc = 0;
    for (int it = 0; it < iters; ++it)
        for (int c = 0; c < CHUNKS; ++c) {
            const uint32_t base = (uint32_t) (((it + c) * 16) % 480);
            uint32_t q[5];
#pragma unroll
            for (int i = 0; i < 5; ++i) q[i] = buf_load_b32(s, (uint32_t) (r * ROW + base + 4u * i), 0u, 0u);
            uint32_t w0 = __funnelshift_r(q[0], q[1], SHIFT * 8);
            uint32_t w1 = __funnelshift_r(q[1], q[2], SHIFT * 8);
            uint32_t w2 = __funnelshift_r(q[2], q[3], SHIFT * 8);
            uint32_t w3 = __funnelshift_r(q[3], q[4], SHIFT * 8);
            acc += w0 ^ w1 ^ w2 ^ w3;
        }
    out[r] = acc;
}

__global__ void k_buf(const uint8_t* rows, int nrows, unsigned* out, int iters) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) % nrows;
    const uint8_t* row = rows + (size_t) r * ROW;
    i32x4 s = make_srsrc(row, ROW);
    unsigned acc = 0;
    for (int it = 0; it < iters; ++it)
        for (int c = 0; c < CHUNKS; ++c) {
            const uint32_t base = (uint32_t) (((it + c) * 16) % 480);
            __uint128_t v = buf_load_b128(s, base, 0u, 0u);
            uint32_t q4 = buf_load_b32(s, base + 16u, 0u, 0u);
            uint64_t lo = (uint64_t) v, hi = (uint64_t) (v >> 64);
            uint32_t q0 = (uint32_t) lo, q1 = (uint32_t) (lo >> 32);
            uint32_t q2 = (uint32_t) hi, q3 = (uint32_t) (hi >> 32);
            uint32_t w0 = __funnelshift_r(q0, q1, SHIFT * 8);
            uint32_t w1 = __funnelshift_r(q1, q2, SHIFT * 8);
            uint32_t w2 = __funnelshift_r(q2, q3, SHIFT * 8);
            uint32_t w3 = __funnelshift_r(q3, q4, SHIFT * 8);
            acc += w0 ^ w1 ^ w2 ^ w3;
        }
    out[r] = acc;
}

int main(int argc, char** argv) {
    int nrows = argc > 1 ? atoi(argv[1]) : 64;
    const int iters = 20000;
    const int bytes = nrows * ROW + 64;
    uint8_t* d = nullptr;
    if (hipMalloc(&d, bytes) != hipSuccess) { printf("hipMalloc failed\n"); return 1; }
    hipMemset(d, 0x5A, bytes);
    unsigned* out = nullptr;
    hipMalloc(&out, nrows * 4);
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    if (nrows < 1) nrows = 1;
    auto run = [&](void (*k)(const uint8_t*, int, unsigned*, int), const char* nm) {
        k<<<20, 64>>>(d, nrows, out, 10);
        hipDeviceSynchronize();
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {
            hipEventRecord(a);
            k<<<20, 64>>>(d, nrows, out, iters);
            hipEventRecord(b);
            hipEventSynchronize(b);
            float ms = 0; hipEventElapsedTime(&ms, a, b);
            if (ms < best) best = ms;
        }
        double vals = 20.0 * 64.0 * CHUNKS * (double) iters;
        printf("  %-5s rows=%-3d  %8.3f ms   %7.2f Gvalues/s\n", nm, nrows, best, vals / (best * 1e6));
    };
    printf("=== probe_addr: flat dword loads vs buffer-resource dword loads (550 B rows, %d-byte misalign) ===\n", SHIFT);
    run(k_ld4, "ld4");
    run(k_bufd, "bufd");
    run(k_buf, "buf128");
    return 0;
}


