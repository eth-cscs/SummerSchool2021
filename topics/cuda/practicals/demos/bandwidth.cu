#include <iostream>

#include <cuda.h>

#include "util.hpp"
#include "cuda_stream.hpp"

double in_mb(size_t bytes) {
    return bytes*1e-6;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 30);
    size_t n = 1 << pow;

    char* x_device = malloc_device<char>(n);
    char* x_host = malloc_pinned<char>(n);
    char* y_device = malloc_device<char>(n);
    char* y_host = malloc_pinned<char>(n);

    // cuda streams for the transfers
    cuda_stream h2d_stream;
    cuda_stream d2h_stream;

    // perform untimed copy before timing.
    copy_to_device_async<char>(x_host, x_device, 16*1024, h2d_stream.stream());
    copy_to_host_async<char>(y_device, y_host,   16*1024, d2h_stream.stream());
    cudaDeviceSynchronize();

    std::printf("-----------|------------|------------|--------------\n");
    std::printf("%10s | %10s | %10s | %10s\n",
                "size (MB)", "H2D (MB/s)", "D2H (MB/s)", "total (MB/s)");
    std::printf("-----------|------------|------------|--------------\n");
    for (size_t m=1000*1000/8; m<=n; m*=2) {
        auto start_h2d = h2d_stream.enqueue_event();
        copy_to_device_async<char>(x_host, x_device, m, h2d_stream.stream());
        auto stop_h2d = h2d_stream.enqueue_event();

        auto start_d2h = d2h_stream.enqueue_event();
        copy_to_host_async<char>(y_device, y_host, m, d2h_stream.stream());
        auto stop_d2h = d2h_stream.enqueue_event();

        stop_d2h.wait();
        stop_h2d.wait();

        auto time_d2h = stop_d2h.time_since(start_d2h);
        auto time_h2d = stop_h2d.time_since(start_h2d);

        auto mb = in_mb(m*sizeof(char));
        float bw_h2d = mb/time_h2d;
        float bw_d2h = mb/time_d2h;

        std::printf( "%10.2f | %10.1f | %10.1f | %10.1f\n", mb, bw_h2d, bw_d2h, bw_h2d+bw_d2h);
    }

    cudaFree(x_device);
    cudaFree(y_device);
    cudaFreeHost(x_host);
    cudaFreeHost(y_host);

    return 0;
}

