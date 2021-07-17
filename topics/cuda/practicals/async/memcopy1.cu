#include <iostream>

#include <cuda.h>

#include "util.hpp"
#include "cuda_stream.hpp"
#include "cuda_event.hpp"

#define USE_PINNED

// CUDA kernel implementing axpy:
//      y += alpha*x
__global__
void axpy(int n, double alpha, const double *x, double* y) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid<n) {
        y[tid] += alpha*x[tid];
    }
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 20);
    size_t N = 1 << pow;
    auto size_in_bytes = N * sizeof(double);

    std::cout << "memcopy and daxpy test of length N = " << N
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    cuInit(0);

    double* x_device = malloc_device<double>(N);
    double* y_device = malloc_device<double>(N);

    #ifdef USE_PINNED
    double* x_host = malloc_pinned<double>(N, 1.5);
    double* y_host = malloc_pinned<double>(N, 3.0);
    double* y      = malloc_pinned<double>(N, 0.0);
    #else
    double* x_host = malloc_host<double>(N, 1.5);
    double* y_host = malloc_host<double>(N, 3.0);
    double* y      = malloc_host<double>(N, 0.0);
    #endif

    // synchronize GPU
    cudaDeviceSynchronize();

    // copy to device
    cuda_stream stream; // default stream

    // perform a dummy copy before inserting the event to start timing.
    copy_to_device_async<double>(y_host, y_device, N, stream.stream());

    auto start_event = stream.enqueue_event();
    copy_to_device_async<double>(y_host, y_device, N, stream.stream());
    copy_to_device_async<double>(x_host, x_device, N, stream.stream());
    auto H2D_event = stream.enqueue_event();

    // y += 2 * x
    auto block_dim = 128ul;
    auto grid_dim = (N-1)/block_dim + 1;

    axpy<<<grid_dim, block_dim, 0, stream.stream()>>> (N, 2.0, x_device, y_device);
    auto kernel_event = stream.enqueue_event();

    // copy result back to host
    copy_to_host_async<double>(y_device, y, N, stream.stream());
    auto end_event = stream.enqueue_event();
    end_event.wait();

    auto time_total = end_event.time_since(start_event);
    auto time_H2D   = H2D_event.time_since(start_event);
    auto time_D2H   = end_event.time_since(kernel_event);

    std::cout << "-------\ntimings\n-------" << std::endl;
    std::cout << "H2D   : " << time_H2D << std::endl;
    std::cout << "D2H   : " << time_D2H << std::endl;
    std::cout << "axpy  : " << kernel_event.time_since(H2D_event) << std::endl;
    std::cout << "total : " << time_total << std::endl;

    auto H2D_BW = 2 * size_in_bytes / time_H2D / (1024*1024);
    auto D2H_BW =     size_in_bytes / time_D2H / (1024*1024);
    std::cout << "H2D BW : " << H2D_BW << " MB/s" << std::endl;
    std::cout << "D2H BW : " << D2H_BW << " MB/s" << std::endl;

    // check for errors
    auto errors = 0;
    for(auto i=0; i<N; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            errors++;
        }
    }
    if(errors>0) std::cout << "\n============ FAILED with " << errors << " errors" << std::endl;
    else         std::cout << "\n============ PASSED" << std::endl;

    cudaFree(x_device);
    cudaFree(y_device);
    #ifdef USE_PINNED
    cudaFreeHost(x_host);
    cudaFreeHost(y_host);
    cudaFreeHost(y);
    #else
    free(x_host);
    free(y_host);
    free(y);
    #endif

    return 0;
}

