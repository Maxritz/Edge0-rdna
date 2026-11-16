#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <iostream>
#include <chrono>
#include <immintrin.h>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __LINE__ << "\n"; \
        exit(1); \
    } \
} while (0)

// RDNA 2 optimized kernel: Vectorized float4 loads/stores (128-bit per thread)
__global__ void __launch_bounds__(256)
rdna2_vector_fma(float4* __restrict__ vram_buf, float scale, float bias, size_t vec_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < vec_count; i += stride) {
        float4 v = vram_buf[i];
        v.x = fmaf(v.x, scale, bias);
        v.y = fmaf(v.y, scale, bias);
        v.z = fmaf(v.z, scale, bias);
        v.w = fmaf(v.w, scale, bias);
        vram_buf[i] = v;
    }
}

// Fast CPU streaming write into PCIe BAR (Write-Combining safe)
void write_to_rebar_vram_avx2(float* __restrict__ bar_ptr, size_t count, float val) {
    __m256 vec = _mm256_set1_ps(val);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        _mm256_stream_ps(bar_ptr + i, vec);
    }
    _mm_sfence();
    for (; i < count; ++i) {
        bar_ptr[i] = val;
    }
}

int main() {
    std::cout << "Starting...\n";
    std::cout.flush();

    HIP_CHECK(hipSetDevice(0));
    std::cout << "hipSetDevice OK\n";
    std::cout.flush();

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cout << "VRAM: " << (props.totalGlobalMem / (1024*1024)) << " MB\n";
    std::cout.flush();

    const size_t num_floats = 4ULL * 1024ULL * 1024ULL; // 4M floats = 16 MB
    const size_t bytes = num_floats * sizeof(float);

    float* vram_ptr = nullptr;

    // Try hipMalloc first (standard)
    std::cout << "Trying hipMalloc(" << (bytes/1024/1024) << " MB)...\n";
    std::cout.flush();
    hipError_t err = hipMalloc(&vram_ptr, bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << "\n";
        return 1;
    }
    std::cout << "hipMalloc OK, ptr = " << vram_ptr << "\n";
    std::cout.flush();

    // Test GPU memset first to verify VRAM works
    std::cout << "Testing GPU memset...\n";
    std::cout.flush();
    HIP_CHECK(hipMemset(vram_ptr, 0x42, bytes));
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "GPU memset OK\n";
    std::cout.flush();

    // Try a simple CPU write (not AVX2) to test if BAR is accessible
    std::cout << "Testing simple CPU write (first 100 floats)...\n";
    std::cout.flush();
    for (size_t i = 0; i < 100; ++i) {
        vram_ptr[i] = 3.14f;
    }
    std::cout << "Simple CPU write OK\n";
    std::cout.flush();

    // Now try AVX2 streaming write
    std::cout << "Testing AVX2 streaming write...\n";
    std::cout.flush();
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        write_to_rebar_vram_avx2(vram_ptr, num_floats, 2.0f);
        std::cout << "AVX2 streaming write OK\n";
    } catch (...) {
        std::cerr << "AVX2 streaming write CRASHED\n";
        HIP_CHECK(hipFree(vram_ptr));
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double write_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "CPU direct BAR write (16 MB): " << write_sec << " s ("
              << (16.0 / write_sec) << " MB/s over PCIe)\n";
    std::cout.flush();

    // Launch RDNA 2 Wave32 vectorized kernel
    const size_t vec_count = num_floats / 4;
    const int threads = 256;
    const int blocks = 40 * 8;

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(
        rdna2_vector_fma,
        dim3(blocks), dim3(threads), 0, 0,
        reinterpret_cast<float4*>(vram_ptr), 3.0f, 1.0f, vec_count
    );
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float gpu_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&gpu_ms, start, stop));

    double gbps = 0.032 / (gpu_ms / 1000.0); // 16 MB read + 16 MB write = 32 MB
    std::cout << "RX 6700 XT Kernel Time: " << gpu_ms << " ms (" << gbps << " GB/s)\n";

    float check = 0.0f;
    HIP_CHECK(hipMemcpy(&check, vram_ptr + (num_floats - 1), sizeof(float), hipMemcpyDeviceToHost));
    std::cout << "Verification value: " << check << " (expected 7.0)\n";

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(vram_ptr));
    return 0;
}