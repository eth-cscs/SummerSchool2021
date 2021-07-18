#include <iostream>

#include <cuda.h>

#include "util.hpp"
#include "cuda_stream.hpp"
#include "cuda_event.hpp"

__global__
void blur(const double *in, double* out, int n) {
    auto i = threadIdx.x + blockDim.x * blockIdx.x + 1;

    if(i<n-1) {
        out[i] = 0.25*(in[i-1] + 2.0*in[i] + in[i+1]);
    }
}

template <int THREADS>
__global__
void blur_twice(const double *in, double* out, int n) {
    __shared__ double buffer[THREADS+4];

    auto block_start = blockDim.x * blockIdx.x;
    auto block_end   = block_start + blockDim.x;

    auto lid = threadIdx.x;
    auto gid = lid + block_start;

    auto blur = [] (int pos, double const* field) {
        return 0.25*(field[pos-1] + 2.0*field[pos] + field[pos+1]);
    };

    if(gid<n-4) {
        auto li = lid+2;
        auto gi = gid+2;

        buffer[li] = blur(gi, in);
        if(threadIdx.x==0) {
            buffer[1] = blur(block_start+1, in);
            buffer[blockDim.x+2] = blur(block_end+2, in);
        }

        __syncthreads();

        out[gi] = blur(li, buffer);
    }
}

int main(int argc, char** argv) {
    size_t pow    = read_arg(argc, argv, 1, 20);
    size_t nsteps = read_arg(argc, argv, 2, 100);
    bool fuse_loops = read_arg(argc, argv, 3, false);
    size_t n = (1 << pow) + 4;

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dispersion 1D test of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    auto x_host = malloc_host<double>(n, 0.);

    // set boundary conditions to 1
    x_host[0]   = 1.0;
    x_host[1]   = 1.0;
    x_host[n-2] = 1.0;
    x_host[n-1] = 1.0;

    auto x0 = malloc_device<double>(n);
    auto x1 = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_host, x0, n);
    copy_to_device<double>(x_host, x1, n);

    // find the launch grid configuration
    constexpr auto block_dim = 128;
    auto grid_dim = (n-4)/block_dim + ((n-4)%block_dim ? 1 : 0);
    auto shared_size = sizeof(double)*(block_dim+4);

    cuda_stream stream;
    auto start_event = stream.enqueue_event();
    for(auto step=0; step<nsteps; ++step) {
        if (fuse_loops) {
            blur_twice<block_dim><<<grid_dim, block_dim, shared_size>>>(x0, x1, n);
        }
        else {
            blur<<<grid_dim, block_dim>>>(x0, x1, n);
            blur<<<grid_dim, block_dim>>>(x0+1, x1+1, n-2);
        }

        std::swap(x0, x1);
    }
    auto stop_event = stream.enqueue_event();

    // copy result back to host
    copy_to_host<double>(x0, x_host, n);

    stop_event.wait();
    auto time = stop_event.time_since(start_event);
    std::cout << "==== " << time << " seconds : " << 1e3*time/nsteps << " ms/step\n";

    return 0;
}

