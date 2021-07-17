#include <numeric>

#include <cstdio>
#include <cuda.h>

#include "util.hpp"

__global__
void histogram(int* x, int* bins, int n) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<n) {
        const auto c = x[i];
        bins[c]++;
    }
}

int main(void) {
    const int n = 1024;
    const int c = 16;

    int* x = malloc_managed<int>(n);
    for (auto i=0; i<n; ++i) x[i] = rand()%c;

    int* bins = malloc_managed<int>(c);
    std::fill(bins, bins+c, 0);

    histogram<<<1, n>>>(x, bins, n);
    cudaDeviceSynchronize();

    printf("bins: ");
    for (auto i=0; i<c; ++i) printf("%d ", bins[i]); printf("\n");

    auto sum = std::accumulate(bins, bins+c, 0);
    printf("sum %d, expected %d\n", sum, n);

    cudaFree(x);
    cudaFree(bins);
    return 0;
}
