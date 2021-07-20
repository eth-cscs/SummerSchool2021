#include <iostream>

#include <cuda.h>

#include "util.hpp"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

// Works for nthreads=WIDTH<=1024 and n<=1024
template <int WIDTH>
__global__
void dot_gpu_kernel(const double *x, const double* y, double *result, int n) {
    __shared__ double buf[WIDTH];
    int i = threadIdx.x;

    buf[i] = i<n? x[i]*y[i]: 0;
    int width = WIDTH/2;

    while (width) {
        __syncthreads();
        if (i<width) {
            buf[i] += buf[i+width];
        }
        width /= 2;
    }

    if (!i) {
        *result = *buf;
    }
}

// Works for arbitrary n with a single thread block
template <int WIDTH>
__global__
void dot_gpu_kernel2(const double *x, const double* y, double *result, int n) {
    __shared__ double buf[WIDTH];
    int i = threadIdx.x;
    int gid = i;

    buf[i] = 0;
    while (gid<n) {
        buf[i] += x[gid]*y[gid];
        gid += WIDTH;
    }
    int width = WIDTH/2;

    while (width) {
        __syncthreads();
        if (i<width) {
            buf[i] += buf[i+width];
        }
        width /= 2;
    }

    if (!i) {
        *result = *buf;
    }
}

template <int WIDTH>
__global__
void dot_gpu_kernel_full(const double *x, const double* y, double *result, int n) {
    __shared__ double buf[WIDTH];
    int i = threadIdx.x;
    int gid = i + blockIdx.x*blockDim.x;

    buf[i] = gid<n? x[gid]*y[gid]: 0;
    int width = WIDTH/2;

    while (width) {
        __syncthreads();
        if (i<width) {
            buf[i] += buf[i+width];
        }
        width /= 2;
    }

    if (!i) {
        atomicAdd(result, *buf);
    }
}

#define WIDTH 128

double dot_gpu(const double *x, const double* y, int n) {
    static double* result = malloc_managed<double>(1);
    //dot_gpu_kernel<WIDTH><<<1, WIDTH>>>(x, y, result, n);
    //dot_gpu_kernel2<WIDTH><<<1, WIDTH>>>(x, y, result, n);
    dot_gpu_kernel_full<WIDTH><<<(n-1)/WIDTH+1, WIDTH>>>(x, y, result, n);

    cudaDeviceSynchronize();
    return *result;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);
    n = pow;

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes*1e-9 << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    auto result   = dot_gpu(x_d, y_d, n);
    auto expected = dot_host(x_h, y_h, n);
    printf("expected %f got %f\n", (float)expected, (float)result);

    return 0;
}

