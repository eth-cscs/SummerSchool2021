#include <iostream>

#include <cuda.h>

#include "util.hpp"

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;
    auto size_in_bytes = n * sizeof(double);

    std::cout << "memcopy and daxpy test of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    cuInit(0);

    // initialize cublas
    auto cublas_handle = get_cublas_handle();

    double* x_device = malloc_device<double>(n);
    double* y_device = malloc_device<double>(n);

    double* x_host = malloc_host<double>(n, 1.5);
    double* y_host = malloc_host<double>(n, 3.0);
    double* y      = malloc_host<double>(n, 0.0);

    // start the nvprof profiling
    cudaProfilerStart();

    // copy memory to device
    auto start = get_time();
    copy_to_device<double>(x_host, x_device, n);
    copy_to_device<double>(y_host, y_device, n);

    // y = y + 2 * x
    double alpha = 2.0;
    auto cublas_status =
        cublasDaxpy(cublas_handle, n, &alpha, x_device, 1, y_device, 1);

    auto time_taken = get_time() - start;

    std::cout << "time : " << time_taken << "s\n";

    // copy result back to host
    copy_to_host<double>(y_device, y, n);

    // check for errors
    int errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for(auto i=0; i<n; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            errors++;
        }
    }

    // stop the profiling session
    cudaProfilerStop();

    std::cout << (errors>0 ? "failed" : "passed") << " with " << errors << " errors\n";

    cudaFree(x_device);
    cudaFree(y_device);

    free(x_host);
    free(y_host);
    free(y);

    return 0;
}

