#include <algorithm>
#include <iostream>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "util.hpp"

void benchmark_gpu(thrust::host_vector<double> values_host)
{
    // fill a vector with random values
    size_t n = values_host.size();
    thrust::device_vector<double> values_device(n);

    auto start = get_time();

    // TODO: copy values to device
    auto h2d_time = get_time() - start;

    // TODO: sort values on device
    auto sort_time = get_time() - h2d_time;

    // TODO: copy result back to host
    auto time_taken = get_time() - start;

    std::cout << "gpu performance including transfers: " << n / time_taken / 1e6 << " million keys/s\n";
    std::cout << "gpu performance without transfers: " << n / sort_time / 1e6 << " million keys/s\n";

    // check for errors
    bool pass = std::is_sorted(values_host.begin(), values_host.end());
    std::cout << "gpu sort: " << (pass ? "passed\n\n" : "failed\n\n");
}

void benchmark_host(thrust::host_vector<double> values_host)
{
    size_t n = values_host.size();

    auto start = get_time();

    // sort values on host
    std::sort(values_host.begin(), values_host.end());

    auto time_taken = get_time();

    std::cout << "host performance: " << n / time_taken / 1e6 << " million keys/s\n";

    // check for errors
    bool pass = std::is_sorted(values_host.begin(), values_host.end());
    std::cout << "host sort: " << (pass ? "passed\n\n" : "failed\n\n");
}

int main(int argc, char** argv)
{
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;
    auto size_in_bytes = n * sizeof(double);

    std::cout << "sort test of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl << std::endl;

    // fill a vector with random values
    thrust::host_vector<double> values_host(n);
    std::generate(values_host.begin(), values_host.end(), drand48);

    // start the nvprof profiling
    cudaProfilerStart();

    benchmark_gpu(values_host);
    benchmark_host(values_host);

    // stop the profiling session
    cudaProfilerStop();

    return 0;
}

