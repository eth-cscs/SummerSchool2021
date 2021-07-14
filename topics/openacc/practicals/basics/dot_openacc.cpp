#include <iostream>

#define NO_CUDA
#include "util.h"

// host implementation of dot product
double dot_host(const double *x, const double *y, int n) {
    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += x[i]*y[i];
    }

    return sum;
}

double dot_gpu(const double *x, const double *y, int n) {
    double sum = 0;
    int i;

    // TODO: Offload this loop to the GPU
    for (i = 0; i < n; ++i) {
        sum += x[i]*y[i];
    }

    return sum;
}

int main(int argc, char **argv) {
    size_t pow  = read_arg(argc, argv, 1, 2);
    size_t n = 1 << pow;

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product OpenACC of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i = 0; i < n; ++i) {
        y_h[i] = rand() % 10;
    }

    auto time_host = get_time();
    auto expected = dot_host(x_h, y_h, n);
    time_host = get_time() - time_host;

    auto time_gpu = get_time();
    auto result = dot_gpu(x_h, y_h, n);
    time_gpu = get_time() - time_gpu;
    std::cout << "expected " << expected << " got " << result << ": "
              << (std::abs(expected - result) < 1.e-6 ? "success\n" : "failure\n");
    std::cout << "Host kernel took " << time_host << " s\n";
    std::cout << "GPU kernel took " << time_gpu << " s\n";
    return 0;
}
