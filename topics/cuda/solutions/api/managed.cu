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

    // initialize cublas
    auto cublas_handle = get_cublas_handle();

    double* x = malloc_managed<double>(n, 1.5);
    double* y = malloc_managed<double>(n, 3.0);


    // start the nvprof profiling
    cudaProfilerStart();
    auto start = get_time(); // start our own timer

    // y = y + 2 * x
    double alpha = 2.0;
    // call to cublas axpy and check for success
    auto cublas_status =
        cublasDaxpy(cublas_handle, n, &alpha, x, 1, y, 1);
    cublas_check_status(cublas_status);

    // wait for the daxpy to finish before stopping timer and testing results.
    cudaDeviceSynchronize();

    // stop the timer
    auto time_taken = get_time() - start;

    // validate the solution
    // this will copy the solution in y back to the host
    int errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for(auto i=0; i<n; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            errors++;
        }
    }

    // stop the profiling session
    cudaProfilerStop();

    std::cout << "time : " << time_taken << "s\n";

    std::cout << (errors>0 ? "failed" : "passed") << " with " << errors << " errors\n";

    cudaFree(x);
    cudaFree(y);

    return 0;
}

