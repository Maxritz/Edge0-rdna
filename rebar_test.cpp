#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void scale_kernel(float* data, float factor, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

int main() {
    int device = 0;
    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    std::cout << "Device: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cout << "Total VRAM: " << (props.totalGlobalMem / (1024 * 1024)) << " MB\n";

    // Allocate 2 GB in VRAM (exceeds the legacy 256 MB non-ReBAR aperture)
    const size_t num_elements = 512 * 1024 * 1024; // 512M floats = 2 GB
    const size_t bytes = num_elements * sizeof(float);

    float* h_pinned = nullptr;
    float* d_vram = nullptr;

    // Pinned host memory maximizes PCIe throughput over ReBAR
    HIP_CHECK(hipHostMalloc(&h_pinned, bytes, hipHostMallocDefault));
    HIP_CHECK(hipMalloc(&d_vram, bytes));

    for (size_t i = 0; i < num_elements; ++i) {
        h_pinned[i] = 1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Transfer 2 GB directly to VRAM
    HIP_CHECK(hipMemcpy(d_vram, h_pinned, bytes, hipMemcpyHostToDevice));

    // Launch kernel on RX 6700 XT (Wave32 native on RDNA 2)
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    hipLaunchKernelGGL(scale_kernel, dim3(blocks), dim3(threads), 0, 0, d_vram, 2.5f, num_elements);
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "2 GB H2D + Compute completed in: " << diff.count() << " s\n";

    HIP_CHECK(hipFree(d_vram));
    HIP_CHECK(hipHostFree(h_pinned));
    return 0;
}