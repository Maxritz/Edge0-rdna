// Direct GPU check: stage the SAME super-block through lat_gu_decode (shipped
// Q3_K layout) and lat_gu_decode_raw3k (converted natural 3-bit layout) and
// compare every staged weight byte and sub-scale, byte-exactly.
//
// Including the .cu makes its anonymous-namespace helpers visible in this TU.
#include "../../src/prefill/lat_q3k_fused.cu"

#include <cstdio>
#include <cstdlib>
#include <vector>

// The included .cu's host wrapper calls lat_q3k_down; this test never calls the
// wrapper, so a stub keeps the standalone link simple.
namespace strata::prefill {
void lat_q3k_down(const uint8_t* const*, const int32_t*, int, const void*, float*, int64_t,
                  void*, const int32_t*, const int32_t*) {}
}

__global__ void cmp_decode_kernel(const uint8_t* qblk, const uint8_t* rblk, unsigned* bad) {
    const int kb = threadIdx.x >> 1;      // 0..3 chunks of the super-block
    const int whalf = threadIdx.x & 1;
    const auto a = strata::prefill::lat_gu_decode(qblk, kb, whalf, true);
    const auto b = strata::prefill::lat_gu_decode_raw3k(rblk, kb, whalf, true);
    unsigned n = 0;
    for (int g = 0; g < 2; ++g) {
        for (int k = 0; k < 4; ++k) n += (a.w[g][k] != b.w[g][k]);
        n += (__float_as_uint(a.sc[g]) != __float_as_uint(b.sc[g]));
    }
    atomicAdd(bad, n);
}

static std::vector<uint8_t> read110(const char* path, long long blob, long long block) {
    const long long off = blob * 1164800LL + block * 110LL;
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("cannot open %s\n", path); std::exit(2); }
    _fseeki64(f, off, SEEK_SET);
    std::vector<uint8_t> v(110);
    if (std::fread(v.data(), 1, 110, f) != 110) { std::printf("short read\n"); std::exit(2); }
    std::fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const char* qp = argc > 1 ? argv[1] : "E:/Edge0-120B-q3k/experts.bin";
    const char* rp = argc > 2 ? argv[2] : "E:/Edge0-120B-raw3k/experts.bin";
    // a handful of super-blocks at different row/block alignments (row pitch 550)
    const long long spots[][2] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},
                                  {1, 0}, {1, 1}, {7, 2}, {12345, 3}, {20000, 4}};
    uint8_t* dq = nullptr; uint8_t* dr = nullptr; unsigned* dbad = nullptr;
    cudaMalloc(&dq, 110); cudaMalloc(&dr, 110); cudaMalloc(&dbad, 4);
    unsigned total = 0;
    for (auto& s : spots) {
        auto q = read110(qp, s[0], s[1]);
        auto r = read110(rp, s[0], s[1]);
        cudaMemcpy(dq, q.data(), 110, cudaMemcpyHostToDevice);
        cudaMemcpy(dr, r.data(), 110, cudaMemcpyHostToDevice);
        cudaMemset(dbad, 0, 4);
        cmp_decode_kernel<<<1, 8>>>(dq, dr, dbad);
        cudaDeviceSynchronize();
        unsigned bad = 0; cudaMemcpy(&bad, dbad, 4, cudaMemcpyDeviceToHost);
        std::printf("blob %5lld block %lld: staged mismatches = %u\n", s[0], s[1], bad);
        total += bad;
    }
    std::printf("TOTAL staged mismatches = %u (expected 0)\n", total);
    return total == 0 ? 0 : 1;
}
