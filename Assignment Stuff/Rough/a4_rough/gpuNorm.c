#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

int main()
{
    printf("Hello!\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int numMP = prop.multiProcessorCount;
    int numThreadsPerMP = prop.maxThreadsPerMultiProcessor;
    int warpSize = prop.warpSize;

    printf("No. of MPs - %d\n", numMP);
    printf("No. of threads per MP - %d\n", numThreadsPerMP);
    printf("Warp Size - %d\n", warpSize);
}
