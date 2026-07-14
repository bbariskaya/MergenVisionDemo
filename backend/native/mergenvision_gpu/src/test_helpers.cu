#include <cuda_runtime.h>

__global__ void mergenvision_spin_wait_cycles_kernel(unsigned long long cycles) {
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
        // busy spin
    }
}

extern "C" int mergenvision_spin_wait_cycles(
    unsigned long long cycles,
    cudaStream_t stream) {
    mergenvision_spin_wait_cycles_kernel<<<1, 1, 0, stream>>>(cycles);
    return cudaGetLastError();
}
