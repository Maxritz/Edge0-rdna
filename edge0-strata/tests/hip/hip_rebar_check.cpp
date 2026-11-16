#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK(call) do { \
    hipError_t e = (call); \
    if (e != hipSuccess) { \
        printf("HIP ERROR: %s : %s\n", #call, hipGetErrorString(e)); \
        return 1; \
    } \
} while(0)

static void flag(const char* name, int value)
{
    printf("  %-38s : %s (%d)\n",
           name, value ? "YES" : "NO", value);
}

int main()
{
    int runtimeVersion = 0;
    int driverVersion = 0;
    int count = 0;

    printf("\n========================================\n");
    printf("       AMD HIP ReBAR DIAGNOSTIC\n");
    printf("========================================\n");

    CHECK(hipRuntimeGetVersion(&runtimeVersion));
    CHECK(hipDriverGetVersion(&driverVersion));
    CHECK(hipGetDeviceCount(&count));

    printf("\nHIP runtime version : %d\n", runtimeVersion);
    printf("HIP driver version  : %d\n", driverVersion);
    printf("HIP device count    : %d\n", count);

    for (int i = 0; i < count; i++)
    {
        hipDeviceProp_t p{};

        CHECK(hipGetDeviceProperties(&p, i));
        CHECK(hipSetDevice(i));

        printf("\n----------------------------------------\n");
        printf("GPU %d: %s\n", i, p.name);
        printf("Architecture: %s\n", p.gcnArchName);
        printf("VRAM: %.2f GiB\n",
               (double)p.totalGlobalMem / (1024.0*1024*1024));

        printf("\n[AMD LARGE BAR DETECTION]\n");

        flag("isLargeBar", p.isLargeBar);

        printf("\n[HOST / DEVICE MEMORY ACCESS]\n");

        flag("canMapHostMemory", p.canMapHostMemory);
        flag("canUseHostPointerForRegisteredMem",
             p.canUseHostPointerForRegisteredMem);

        flag("managedMemory", p.managedMemory);
        flag("directManagedMemAccessFromHost",
             p.directManagedMemAccessFromHost);
        flag("concurrentManagedAccess",
             p.concurrentManagedAccess);
        flag("pageableMemoryAccess",
             p.pageableMemoryAccess);
        flag("pageableMemoryAccessUsesHostPageTables",
             p.pageableMemoryAccessUsesHostPageTables);

        size_t freeMem = 0;
        size_t totalMem = 0;

        CHECK(hipMemGetInfo(&freeMem, &totalMem));

        printf("\n[HIP MEMORY MANAGER]\n");
        printf("Free VRAM : %.2f GiB\n",
               (double)freeMem / (1024.0*1024*1024));
        printf("Total VRAM: %.2f GiB\n",
               (double)totalMem / (1024.0*1024*1024));

        printf("\n[ALLOCATION TEST]\n");

        void* ptr = nullptr;
        size_t testSize = 16 * 1024 * 1024;

        hipError_t alloc = hipMalloc(&ptr, testSize);

        if (alloc == hipSuccess)
        {
            hipError_t memsetResult = hipMemset(ptr, 0x5A, testSize);

            if (memsetResult == hipSuccess)
                memsetResult = hipDeviceSynchronize();

            printf("16 MB device allocation: SUCCESS\n");
            printf("GPU memset/synchronize: %s\n",
                   hipGetErrorString(memsetResult));

            CHECK(hipFree(ptr));
        }
        else
        {
            printf("16 MB allocation failed: %s\n",
                   hipGetErrorString(alloc));
        }

        printf("\n[RESULT]\n");

        if (p.isLargeBar)
            printf("HIP REPORTS LARGE BAR ENABLED.\n");
        else
            printf("HIP REPORTS LARGE BAR DISABLED.\n");

        printf("----------------------------------------\n");
    }

    return 0;
}