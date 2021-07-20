#include <chrono>
#include <iostream>
#include <numeric>

#define NO_CUDA
#include "util.hpp"

void clear_cache();

// OpenMP implementation of axpy kernel
void axpy(int n, double alpha, const double *x, double* y) {
    #pragma omp parallel for
    #pragma ivdep
    for(auto i=0; i<n; ++i) {
        y[i] += alpha*x[i];
    }
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;

    std::cout << "memcopy and daxpy test of size " << n << "\n";

    double* x = malloc_host<double>(n, 1.5);
    double* y = malloc_host<double>(n, 3.0);

    clear_cache();

    auto start = get_time();
    axpy(n, 2.0, x, y);
    auto time_axpy = get_time() - start;

    std::cout << "\ntimings\n-------\n";
    std::cout << "axpy " << time_axpy << " s\n";
    std::cout << std::endl;

    // check for errors
    auto errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for(auto i=0; i<n; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            errors++;
        }
    }

    if(errors>0) std::cout << "\n============ FAILED with " << errors << " errors\n";
    else         std::cout << "\n============ PASSED\n";

    free(x);
    free(y);

    return 0;
}

void clear_cache() {
    // allocate a large-enough memory buffer to flush the cache
    const auto n = 60*1024*1024;
    auto a = malloc_host<double>(n);

    // fill buffer with ones then find the sum
    std::fill(a, a+n, 1.0);
    double sum = 0.;
    #pragma omp parallel for reduction(+:sum)
    for (auto i=0; i<n; ++i) {
        a[i] *= 2.0;
        sum += a[i];
    }

    // the result needs to have a side effect stop the optimizer
    // removing it.
    if (std::fabs(2*n-sum)/sum > 1e-8) {
        std::cout << "error\n";
    }

    free(a);
}

