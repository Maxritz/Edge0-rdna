// bw_probe5.cu - the cliff is between 256 and 512 MiB.  Is it total footprint, per-allocation size,
// or the allocation API?  Every case reads the same number of bytes with the same kernel.
//
//   clang --offload-arch=gfx1031 -O3 -x hip bw_probe5.cu -o bw_probe5.exe
//
//   A 1 x 256 MiB   hipMalloc        (below the cliff)
//   B 1 x 512 MiB   hipMalloc
//   C 1 x 1 GiB     hipMalloc
//   D 8 x 256 MiB   hipMalloc        (2 GiB IN TOTAL, but no single allocation over 256 MiB)
//   E 1 x 1 GiB     hipMallocManaged
//   F 1 x 1 GiB     hipMemCreate/hipMemMap virtual memory (a device-pinned physical allocation)
//   G 1 x 1 GiB     hipHostMalloc mapped (zero copy, PCIe by definition)
//
// D vs C is the decisive pair: same bytes, same kernel, only the allocation granularity differs.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ck(x, m) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(e_)); std::exit(2); } } while (0)

__global__ void __launch_bounds__(256, 1) k_read(const int4* __restrict__ p, std::size_t n16_total,
                                                 const int4* const* parts, const int nparts, float* sink) {
    const std::size_t tid = (std::size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t) gridDim.x * blockDim.x;
    int acc = 0;
    if (nparts == 1) {
        for (std::size_t i = tid; i < n16_total; i += stride) {
            const int4 v = p[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    } else {   // walk the parts as one contiguous stream
        for (int q = 0; q < nparts; ++q) {
            const int4* pp = parts[q];
            for (std::size_t i = tid; i < n16_total / nparts; i += stride) {
                const int4 v = pp[i];
                acc ^= v.x ^ v.y ^ v.z ^ v.w;
            }
        }
    }
    if (acc == 0x7fffffff) *sink = 1.0f;
}

static float g_sink = 0;
static float* d_sink = nullptr;
static unsigned g_grid = 0;

template<typename F>
static double ms_of(int iters, F f) {
    hipEvent_t a, b;
    ck(hipEventCreate(&a), "ev"); ck(hipEventCreate(&b), "ev");
    f(); ck(hipDeviceSynchronize(), "sync");
    ck(hipEventRecord(a), "rec");
    for (int i = 0; i < iters; ++i) f();
    ck(hipEventRecord(b), "rec");
    ck(hipEventSynchronize(b), "sync");
    float ms = 0.0f;
    ck(hipEventElapsedTime(&ms, a, b), "el");
    ck(hipEventDestroy(a), "ev"); ck(hipEventDestroy(b), "ev");
    return ms / iters;
}

static void report(const char* what, std::size_t bytes, double ms) {
    std::printf("  %-46s %8.2f MB  %8.3f ms  %7.1f GB/s\n", what, bytes / 1048576.0, ms, bytes / 1048576.0 / ms);
}

static void one(const char* what, std::size_t bytes, int iters) {
    int4* d = nullptr;
    ck(hipMalloc(&d, bytes), "alloc");
    ck(hipMemset(d, 0x5A, bytes), "memset");
    const double ms = ms_of(iters, [&] { k_read<<<g_grid, 256>>>(d, bytes / 16, nullptr, 1, d_sink); });
    report(what, bytes, ms);
    ck(hipFree(d), "f");
}

int main() {
    hipDeviceProp_t p{};
    ck(hipGetDeviceProperties(&p, 0), "props");
    g_grid = (unsigned) p.multiProcessorCount * 32;
    ck(hipMalloc(&d_sink, 4), "sink");
    std::printf("device %s  CUs=%d  largeBar=%d\n", p.name, p.multiProcessorCount, p.isLargeBar);

    std::printf("\nA/B/C  one hipMalloc, size sweep\n");
    one("hipMalloc 256 MiB", 256u << 20, 8);
    one("hipMalloc 512 MiB", 512u << 20, 6);
    one("hipMalloc 1 GiB", (std::size_t) 1 << 30, 4);

    std::printf("\nD  eight separate 256 MiB hipMallocs, 2 GiB in total\n");
    {
        std::vector<int4*> v(8);
        for (auto& q : v) { ck(hipMalloc(&q, 256u << 20), "a"); ck(hipMemset(q, 0x5A, 256u << 20), "m"); }
        // the parts list must live on the device: a host array of device pointers is not readable there
        const int4** dparts = nullptr;
        ck(hipMalloc((void**) &dparts, 8 * sizeof(const int4*)), "pa");
        ck(hipMemcpy(dparts, v.data(), 8 * sizeof(const int4*), hipMemcpyHostToDevice), "pa");
        const std::size_t total = (std::size_t) 8 << 28;
        const double ms = ms_of(3, [&] { k_read<<<g_grid, 256>>>(nullptr, 256u << 20, dparts, 8, d_sink); });
        report("8 x 256 MiB (2 GiB total)", total, ms);
        ck(hipFree(dparts), "f");
        for (auto& q : v) ck(hipFree(q), "f");
    }

    std::printf("\nE  hipMallocManaged 1 GiB\n");
    {
        const std::size_t bytes = (std::size_t) 1 << 30;
        int4* d = nullptr;
        if (hipMallocManaged((void**) &d, bytes, hipMemAttachGlobal) == hipSuccess) {
            ck(hipMemset(d, 0x5A, bytes), "memset");
            const double ms = ms_of(3, [&] { k_read<<<g_grid, 256>>>(d, bytes / 16, nullptr, 1, d_sink); });
            report("hipMallocManaged 1 GiB", bytes, ms);
            ck(hipFree(d), "f");
        } else {
            std::printf("  hipMallocManaged unavailable\n");
        }
    }

    std::printf("\nF  virtual memory: hipMemAddressReserve + hipMemCreate(device) + hipMemMap, 1 GiB\n");
    {
        const std::size_t bytes = (std::size_t) 1 << 30;
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = 0;
        std::size_t gran = 0;
        if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum) != hipSuccess) {
            std::printf("  granularity query failed\n");
        } else {
            const std::size_t sz = ((bytes + gran - 1) / gran) * gran;
            void* va = nullptr;
            hipMemGenericAllocationHandle_t h{};
            hipError_t e = hipMemAddressReserve(&va, sz, gran, nullptr, 0);
            if (e != hipSuccess) std::printf("  hipMemAddressReserve: %s\n", hipGetErrorString(e));
            if (e == hipSuccess) e = hipMemCreate(&h, sz, &prop, 0);
            if (e != hipSuccess) std::printf("  hipMemCreate: %s\n", hipGetErrorString(e));
            if (e == hipSuccess) e = hipMemMap(va, sz, 0, h, 0);
            if (e != hipSuccess) std::printf("  hipMemMap: %s\n", hipGetErrorString(e));
            if (e == hipSuccess) {
                hipMemAccessDesc ad{};
                ad.location.type = hipMemLocationTypeDevice;
                ad.location.id = 0;
                ad.flags = hipMemAccessFlagsProtReadWrite;
                e = hipMemSetAccess(va, sz, &ad, 1);
                if (e != hipSuccess) std::printf("  hipMemSetAccess: %s\n", hipGetErrorString(e));
            }
            if (e == hipSuccess) {
                ck(hipMemset(va, 0x5A, sz), "memset");
                const double ms = ms_of(3, [&] { k_read<<<g_grid, 256>>>((const int4*) va, sz / 16, nullptr, 1, d_sink); });
                report("VMM device-pinned 1 GiB", sz, ms);
                ck(hipMemUnmap(va, sz), "unmap");
                ck(hipMemRelease(h), "rel");
                ck(hipMemAddressFree(va, sz), "free");
            }
        }
    }

    std::printf("\nG  hipHostMalloc mapped (zero copy) 1 GiB\n");
    {
        const std::size_t bytes = (std::size_t) 1 << 30;
        void* h = nullptr;
        int4* dm = nullptr;
        ck(hipHostMalloc(&h, bytes, hipHostMallocMapped), "host");
        ck(hipHostGetDevicePointer((void**) &dm, h, 0), "map");
        std::memset(h, 0x5A, bytes);
        const double ms = ms_of(3, [&] { k_read<<<g_grid, 256>>>(dm, bytes / 16, nullptr, 1, d_sink); });
        report("zero-copy host 1 GiB", bytes, ms);
        ck(hipHostFree(h), "f");
    }
    ck(hipFree(d_sink), "f");
    return 0;
}
