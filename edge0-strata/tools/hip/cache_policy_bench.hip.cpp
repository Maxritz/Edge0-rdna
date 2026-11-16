// tools/hip/cache_policy_bench.hip.cpp - which cache policy should the expert loads use on gfx1031?
//
// WHY.  The RDNA2 ISA (section 2.4, "Device Memory") describes L2 -> read-only L1 -> L0 per WGP, and says that
// "specific cache-less load instructions can force data to be retrieved from device memory during an execution of
// a load clause".  The lever it names is the load's cache policy: GLC (bypass L1) and SLC (bypass L2).  An earlier
// experiment rejected __ldg, which is the read-only L1 path, NOT the cache-less path, so this bench separates the
// three arms: plain, __ldg, and an explicit `global_load_dwordx4 ... glc slc`.
//
// THE WORK.  A streaming read of 64 expert blobs (1,111,808 B each, 67.9 MB, ~23x the 3 MB L2), XOR-folded, with the
// result checked against a host reference: XOR is order-independent, so a scheduling difference cannot fake a pass.
// Each timed region is FIVE SEPARATE LAUNCHES.  Repeating the read inside one kernel lets the compiler hoist the
// loads out of the repetition - it did, and this file first reported 1.4 TB/s on a 384 GB/s card: the reason the
// checksum and the raw milliseconds are printed next to every rate.
//
//   build: tools/hip/build_cache_bench.bat   (also emits cp_dev.s, so the load modifiers are read, not assumed)
//   run:   cache_policy_bench.exe [repeats]  from a directory holding ROCm's amdhip64_7.dll (edge0-strata\build);
//          elsewhere Windows resolves C:\Windows\System32\amdhip64.dll and reports no ROCm-capable device.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                                      \
    do {                                                                                  \
        hipError_t e_ = (x);                                                              \
        if (e_ != hipSuccess) {                                                           \
            std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x,             \
                         hipGetErrorString(e_));                                          \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

enum { kPlain = 0, kLdg = 1, kNoCache = 2, kPolicies = 3 };

static const char* kPolicyName[kPolicies] = {"plain (default policy)", "__ldg (read-only L1)",
                                             "glc+slc (cache-less)"};

__device__ __forceinline__ uint4 load_plain(const uint4* p) { return *p; }
__device__ __forceinline__ uint4 load_ldg(const uint4* p) { return __ldg(p); }

// The ISA's cache-less load.  global_load_dwordx4's destination is four CONSECUTIVE VGPRs, so the asm result is a
// 128-bit integer (clang aligns those to a quad); a uint4 result does not come with that guarantee.
__device__ __forceinline__ uint4 load_nocache(const uint4* p) {
    unsigned __int128 raw;
    asm volatile("global_load_dwordx4 %0, %1, off glc slc" : "=v"(raw) : "v"(p) : "memory");
    uint4 v;
    __builtin_memcpy(&v, &raw, sizeof v);
    return v;
}

__global__ void stream_xor(const uint4* __restrict__ src, unsigned long long n4, unsigned long long* out, int pol) {
    const unsigned long long stride = (unsigned long long) gridDim.x * blockDim.x;
    unsigned long long acc = 0;
    for (unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const uint4 v =
            (pol == kPlain) ? load_plain(src + i) : (pol == kLdg ? load_ldg(src + i) : load_nocache(src + i));
        acc ^= ((unsigned long long) v.x << 32) ^ (unsigned long long) v.y ^
               ((unsigned long long) v.z << 32) ^ (unsigned long long) v.w;
    }
    atomicXor(out, acc);
}

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 3;
    const int launches = 5;
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s  %d WGPs  L2 %.1f MB  %d repeats, %d launches per timed region\n", prop.name,
                prop.multiProcessorCount, prop.l2CacheSize / 1048576.0, repeats, launches);

    const size_t blob = 1111808, experts = 640;  // 712 MB: beyond the 96 MB Infinity Cache, so reads reach DRAM
    const size_t bytes = blob * experts;         // 67.9 MB, ~23x L2: no reuse for a cache to exploit
    const size_t n4 = bytes / sizeof(uint4);

    uint4* d_src = nullptr;
    unsigned long long* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_out, sizeof(*d_out)));
    unsigned long long ref = 0;
    {
        std::vector<uint4> h(n4);
        for (size_t i = 0; i < n4; ++i)
            h[i] = make_uint4((unsigned) i, (unsigned) (i * 3 + 1), (unsigned) (i * 5 + 2), (unsigned) (i * 7 + 3));
        HIP_CHECK(hipMemcpy(d_src, h.data(), bytes, hipMemcpyHostToDevice));
        for (size_t i = 0; i < n4; ++i)
            ref ^= ((unsigned long long) h[i].x << 32) ^ (unsigned long long) h[i].y ^
                   ((unsigned long long) h[i].z << 32) ^ (unsigned long long) h[i].w;
    }

    const int threads = 256;
    const int blocks = std::max(1, prop.multiProcessorCount * 32);
    unsigned long long sums[kPolicies] = {0, 0, 0};
    double best[kPolicies] = {0, 0, 0}, best_ms[kPolicies] = {0, 0, 0};

    for (int rep = 0; rep <= repeats; ++rep) {
        for (int pol = 0; pol < kPolicies; ++pol) {
            HIP_CHECK(hipMemset(d_out, 0, sizeof(*d_out)));
            hipEvent_t a, b;
            HIP_CHECK(hipEventCreate(&a));
            HIP_CHECK(hipEventCreate(&b));
            HIP_CHECK(hipEventRecord(a));
            for (int k = 0; k < launches; ++k)
                stream_xor<<<blocks, threads>>>(d_src, n4, d_out, pol);
            HIP_CHECK(hipEventRecord(b));
            HIP_CHECK(hipEventSynchronize(b));
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, a, b));
            HIP_CHECK(hipEventDestroy(a));
            HIP_CHECK(hipEventDestroy(b));
            if (rep == 0) continue;   // warmup
            const double gbs = (double) bytes * launches / (ms * 1e6);
            if (gbs > best[pol]) {
                best[pol] = gbs;
                best_ms[pol] = ms;
            }
            HIP_CHECK(hipMemcpy(&sums[pol], d_out, sizeof(*d_out), hipMemcpyDeviceToHost));
        }
    }

    std::printf("host reference XOR %llu over %.1f MB\n", ref, bytes / 1048576.0);
    bool same = true;
    for (int pol = 0; pol < kPolicies; ++pol) {
        const bool ok = sums[pol] == ref;
        same = same && ok;
        std::printf("%-24s %8.1f GB/s  %7.3f ms  xor %llu  %s\n", kPolicyName[pol], best[pol], best_ms[pol],
                    sums[pol], ok ? "matches host" : "MISMATCH");
    }
    std::printf("every arm read exactly the same bytes as the host: %s\n",
                same ? "yes" : "NO - the rates above are not trustworthy");
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return same ? 0 : 2;
}
