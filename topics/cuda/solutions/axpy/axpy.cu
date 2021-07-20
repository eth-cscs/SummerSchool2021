#include <iostream>

#include <cuda.h>

#include "util.hpp"

__global__
void axpy(int n, double alpha, const double* x, double* y) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<n) {
        y[i] = y[i] + alpha*x[i];
    }
}

// Extra: Dealing with arrays larger than 1024 using a single block.
// Comment previous definition to use this.
/*
__global__
void axpy(int n, double alpha, const double* x, double* y)
{
    auto i = threadIdx.x;
    int k = (n + 1024 - 1)/1024;
    for (int j = 0; j < k; j++)
    {
        auto index = 1024*j + i;
        if (index < n)
        {
            y[index] = y[index] + alpha*x[index];
        }
    }
}
*/

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;
    auto size_in_bytes = n * sizeof(double);

    cuInit(0);

    std::cout << "memcopy and daxpy test of size " << n << "\n";

    double* x_device = malloc_device<double>(n);
    double* y_device = malloc_device<double>(n);

    double* x_host = malloc_host<double>(n, 1.5);
    double* y_host = malloc_host<double>(n, 3.0);
    double* y      = malloc_host<double>(n, 0.0);

    // copy to device
    auto start = get_time();
    copy_to_device<double>(x_host, x_device, n);
    copy_to_device<double>(y_host, y_device, n);
    auto time_H2D = get_time() - start;

    // calculate grid dimensions
    int num_threads = 128;
    int num_blocks = (n-1)/num_threads + 1;

    // synchronize the host and device so that the timings are accurate
    cudaDeviceSynchronize();

    start = get_time();

    axpy<<<num_blocks, num_threads>>>(n, 2, x_device, y_device);

    cudaDeviceSynchronize();
    auto time_axpy = get_time() - start;

    // check for error in last kernel call
    cuda_check_last_kernel("axpy kernel");

    // copy result back to host
    start = get_time();
    copy_to_host<double>(y_device, y, n);
    auto time_D2H = get_time() - start;

    std::cout << "-------\ntimings\n-------\n";
    std::cout << "H2D:  " << time_H2D << " s\n";
    std::cout << "D2H:  " << time_D2H << " s\n";
    std::cout << "axpy: " << time_axpy << " s\n";
    std::cout << std::endl;
    std::cout << "total: " << time_axpy+time_H2D+time_D2H << " s\n";
    std::cout << std::endl;

    std::cout << "-------\nbandwidth\n-------\n";
    auto H2D_BW = size_in_bytes/1e6*2 / time_H2D;
    auto D2H_BW = size_in_bytes/1e6   / time_D2H;
    std::cout << "H2D BW:  " << H2D_BW << " MB/s\n";
    std::cout << "D2H BW:  " << D2H_BW << " MB/s\n";

    // check for errors
    auto errors = 0;
    for(auto i=0; i<n; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            ++errors;
        }
    }

    std::cout << (errors>0 ? "failed" : "passed") << " with " << errors << " errors\n";

    cudaFree(x_device);
    cudaFree(y_device);

    free(x_host);
    free(y_host);
    free(y);

    return 0;
}

