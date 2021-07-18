#include <iostream>
#include <cassert>

#include <omp.h>

#define NO_CUDA
#include "util.hpp"

void blur_twice_naive(const double* in , double* out , int n) {
    static double* buffer = malloc_host<double>(n);

    auto blur = [] (int pos, const double* u) {
        return 0.25*( u[pos-1] + 2.0*u[pos] + u[pos+1]);
    };

    #pragma omp parallel for
    for(auto i=1; i<n-1; ++i) {
        buffer[i] = blur(i, in);
    }
    #pragma omp parallel for
    for(auto i=2; i<n-2; ++i) {
        out[i] = blur(i, buffer);
    }
}

void blur_twice(const double* in , double* out , int n) {
    auto const block_size = std::min(2048, n-4);
    assert((n-4)%block_size == 0);
    auto const num_blocks = (n-4)/block_size;
    static double* buffer = malloc_host<double>((block_size+4)*omp_get_max_threads());

    auto blur = [] (int pos, const double* u) {
        return 0.25*( u[pos-1] + 2.0*u[pos] + u[pos+1]);
    };

    #pragma omp parallel for
    for(auto b=0; b<num_blocks; ++b) {
        auto tid = omp_get_thread_num();
        auto first = 2 + b*block_size;
        auto last = first + block_size;

        auto buff = buffer + tid*(block_size+4);
        for(auto i=first-1, j=1; i<(last+1); ++i, ++j) {
            buff[j] = blur(i, in);
        }
        for(auto i=first, j=2;   i<last;   ++i, ++j) {
            out[i] = blur(j, buff);
        }
    }
}

int main(int argc, char** argv) {
    size_t pow    = read_arg(argc, argv, 1, 20);
    size_t nsteps = read_arg(argc, argv, 2, 100);
    bool use_blocking = read_arg(argc, argv, 3, false);
    size_t n = (1 << pow) + 4;

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dispersion 1D test of length n = " << n
              << " : " << size_in_bytes*1e-9 << "MB\n";

    std::cout << "==== " << omp_get_max_threads() << " threads\n";

    auto x0 = malloc_host<double>(n+4, 0.);
    auto x1 = malloc_host<double>(n+4, 0.);

    // set boundary conditions to 1
    x0[0]   = x1[0]   = 1.0;
    x0[1]   = x1[1]   = 1.0;
    x0[n-2] = x1[n-2] = 1.0;
    x0[n-1] = x1[n-1] = 1.0;

    auto tstart = get_time();
    for(auto step=0; step<nsteps; ++step) {
        if (use_blocking) blur_twice(x0, x1, n);
        else              blur_twice_naive(x0, x1, n);
        std::swap(x0, x1);
    }
    auto time = get_time() - tstart;

    //for(auto i=0u; i<10u; ++i) std::cout << x0[i] << " "; std::cout << "\n";

    std::cout << "==== " << time << " seconds : " << 1e3*time/nsteps << " ms/step\n";

    return 0;
}


