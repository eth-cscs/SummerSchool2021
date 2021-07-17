#include <iostream>

#include <cuda.h>

#include "util.hpp"
#include "cuda_stream.hpp"
#include "cuda_event.hpp"

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
    size_t pow     = read_arg(argc, argv, 1, 20);
    int num_chunks = read_arg(argc, argv, 2, 1);

    size_t N = 1 << pow;
    auto size_in_bytes = N * sizeof(double);

    std::cout << "memcopy and daxpy test of length N = " << N
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    cuInit(0);

    double* xd = malloc_device<double>(N);
    double* yd = malloc_device<double>(N);

    double* xh = malloc_pinned<double>(N, 1.5);
    double* yh = malloc_pinned<double>(N, 3.0);
    double* y  = malloc_pinned<double>(N, 0.0);

    int chunk_size = N/num_chunks; // assume N % num_chunks == 0

    // precompute kernel launch configuration
    auto block_dim = 128;
    auto grid_dim = (chunk_size-1)/block_dim + 1;

    cuda_stream D2H_stream;
    cuda_stream H2D_stream;
    cuda_stream kernel_stream;

    auto start_event = D2H_stream.enqueue_event();
    for(int i=0; i<num_chunks; ++i) {
        auto offset = i*chunk_size;

        // copy chunk to device
        copy_to_device_async<double>(xh+offset, xd+offset,
                                     chunk_size, H2D_stream.stream());
        copy_to_device_async<double>(yh+offset, yd+offset,
                                     chunk_size, H2D_stream.stream());

        // force the kernel stream to wait for the memcpy
        auto H2D_event = H2D_stream.enqueue_event();
        kernel_stream.wait_on_event(H2D_event);

        // y += 2 * x
        axpy<<<grid_dim, block_dim, 0, kernel_stream.stream()>>>
            (chunk_size, 2.0, xd+offset, yd+offset);
        cuda_check_last_kernel("axpy kernel");

        // copy chunk of result back to host
        auto kernel_event = kernel_stream.enqueue_event();
        D2H_stream.wait_on_event(kernel_event);
        copy_to_host_async<double>(yd+offset, y+offset,
                                   chunk_size, D2H_stream.stream());
    }
    auto end_event = D2H_stream.enqueue_event();
    end_event.wait();

    auto time_total = end_event.time_since(start_event);

    std::cout << "-------\ntimings\n-------" << std::endl;
    std::cout << "total : " << time_total << std::endl;

    // check for errors
    auto errors = 0;
    for(auto i=0; i<N; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            errors++;
        }
    }
    if(errors>0) std::cout << "\n============ FAILED with " << errors << " errors" << std::endl;
    else         std::cout << "\n============ PASSED" << std::endl;

    cudaFree(xd);
    cudaFree(yd);
    cudaFreeHost(xh);
    cudaFreeHost(yh);
    cudaFreeHost(y);

    return 0;
}

