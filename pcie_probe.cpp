// pcie_probe.cpp - measure the H2D / D2H rates that govern this engine's two hot paths:
//   * prefill streams ~74 GB of 1.16 MB expert blobs host -> device;
//   * decode copies ~52 KB of D-stage rows device -> host per layer, inside the layer graph.
// It separates STREAMING bandwidth (one big transfer) from PER-BLOB PIPELINED rate
// (many small transfers in flight) from SERIALISED latency (one sync per blob),
// because those three differ by more than 10x and only the last one looks like 1.6 GB/s.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

static double now_ms() {
    using C = std::chrono::steady_clock;
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(C::now().time_since_epoch()).count() / 1e6;
}

int main() {
    hipDeviceProp_t p{};
    CK(hipGetDeviceProperties(&p, 0));
    std::printf("GPU: %s (%s)  VRAM %.2f GiB\n", p.name, p.gcnArchName,
                (double) p.totalGlobalMem / 1073741824.0);
    std::printf("canMapHostMemory=%d  unifiedAddressing=%d  pageableAccess=%d\n",
                (int) p.canMapHostMemory, (int) p.unifiedAddressing, (int) p.pageableMemoryAccess);

    const size_t kBlob = 1163 * 1024;          // one expert blob as the pack measures it
    const int kBlobs = 900;                    // ~1.0 GiB of blobs

    void* d = nullptr;
    CK(hipMalloc(&d, kBlob * kBlobs));
    void* h = nullptr;
    CK(hipHostMalloc(&h, kBlob * kBlobs, hipHostMallocDefault));
    std::memset(h, 0x5a, kBlob * kBlobs);
    hipStream_t s{};
    CK(hipStreamCreate(&s));

    // ---- 1. streaming: one 1 GiB H2D and one 1 GiB D2H, timed once each after a warmup
    for (int r = 0; r < 2; ++r) {
        const double t0 = now_ms();
        CK(hipMemcpyAsync(d, h, kBlob * kBlobs, hipMemcpyHostToDevice, s));
        CK(hipStreamSynchronize(s));
        const double t1 = now_ms();
        CK(hipMemcpyAsync(h, d, kBlob * kBlobs, hipMemcpyDeviceToHost, s));
        CK(hipStreamSynchronize(s));
        const double t2 = now_ms();
        const double gb = (double) kBlob * kBlobs / 1e9;
        if (r > 0)
            std::printf("streaming  1 GiB: H2D %8.2f ms (%6.2f GB/s)   D2H %8.2f ms (%6.2f GB/s)\n",
                        t1 - t0, gb / ((t1 - t0) / 1e3), t2 - t1, gb / ((t2 - t1) / 1e3));
    }

    // ---- 2. pipelined: kBlobs separate 1.16 MB H2D copies, no host sync until the end
    for (int r = 0; r < 2; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < kBlobs; ++i)
            CK(hipMemcpyAsync((char*) d + (size_t) i * kBlob, (char*) h + (size_t) i * kBlob, kBlob,
                              hipMemcpyHostToDevice, s));
        CK(hipStreamSynchronize(s));
        const double t1 = now_ms();
        if (r > 0)
            std::printf("pipelined  %d x %.2f MB H2D: %8.2f ms total, %6.2f us/blob (%6.2f GB/s)\n",
                        kBlobs, (double) kBlob / 1048576.0, t1 - t0, (t1 - t0) * 1e3 / kBlobs,
                        (double) kBlob * kBlobs / 1e9 / ((t1 - t0) / 1e3));
    }

    // ---- 3. serialised: one copy and one sync per blob (the shape a per-blob event ring produces)
    {
        const int n = 200;
        const double t0 = now_ms();
        for (int i = 0; i < n; ++i) {
            CK(hipMemcpyAsync((char*) d + (size_t) i * kBlob, (char*) h + (size_t) i * kBlob, kBlob,
                              hipMemcpyHostToDevice, s));
            CK(hipStreamSynchronize(s));
        }
        const double t1 = now_ms();
        std::printf("serialised %d x %.2f MB H2D: %8.2f ms total, %6.1f us/blob (%5.2f GB/s)\n",
                    n, (double) kBlob / 1048576.0, t1 - t0, (t1 - t0) * 1e3 / n,
                    (double) kBlob * n / 1e9 / ((t1 - t0) / 1e3));
    }

    // ---- 4. the decode handoff: 52 KB D2H, the size of one layer's D-stage rows at T=1
    {
        const size_t row = 52224;   // act0 10,880 B + xl 41,344 B at T=1 (8 groups x 1280 f32)
        for (int r = 0; r < 2; ++r) {
            const double t0 = now_ms();
            for (int l = 0; l < 48; ++l) {
                CK(hipMemcpyAsync((char*) h + (size_t) l * row, d, row, hipMemcpyDeviceToHost, s));
                CK(hipStreamSynchronize(s));
            }
            const double t1 = now_ms();
            if (r > 0)
                std::printf("decode D2H %d x %zu B, one sync each: %7.2f ms/token, %6.1f us/layer\n",
                            48, row, t1 - t0, (t1 - t0) * 1e3 / 48);
        }
        // the same again but pinned-vs-pageable device memory makes no difference; compare with
        // an event-synchronised variant, which is what a graph node boundary costs
        hipEvent_t ev{};
        CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
        const double t0 = now_ms();
        for (int l = 0; l < 48; ++l) {
            CK(hipMemcpyAsync((char*) h + (size_t) l * row, d, row, hipMemcpyDeviceToHost, s));
            CK(hipEventRecord(ev, s));
            CK(hipEventSynchronize(ev));
        }
        const double t1 = now_ms();
        std::printf("decode D2H %d x %zu B, one event each: %7.2f ms/token, %6.1f us/layer\n",
                    48, row, t1 - t0, (t1 - t0) * 1e3 / 48);
        CK(hipEventDestroy(ev));
    }

    CK(hipStreamDestroy(s));
    CK(hipHostFree(h));
    CK(hipFree(d));
    std::printf("done\n");
    return 0;
}
