#include <stdio.h>

__global__
void hello_kernel() {
    printf("hello world from cuda thread %d\n", int(threadIdx.x));
}

int main(void) {
    hello_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}

