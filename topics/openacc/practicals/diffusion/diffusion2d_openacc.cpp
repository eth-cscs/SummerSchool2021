#include <iostream>

#include "diffusion2d.hpp"
#include "util.h"

// 2D diffusion example
// the grid has a fixed width of nx=128
// the use specifies the height, ny, as a power of two
// note that nx and ny have 2 added to them to account for halos

int main(int argc, char** argv) {
    // set up parameters
    // first argument is the y dimension = 2^arg
    size_t pow    = read_arg(argc, argv, 1, 8);
    // second argument is the number of time steps
    size_t nsteps = read_arg(argc, argv, 2, 100);
    // third argument is nonzero if shared memory version is to be used
    bool use_shared = read_arg(argc, argv, 3, 0);

    // set domain size
    size_t nx = 128+2;
    size_t ny = (1 << pow)+2;
    double dt = 0.1;

    std::cout << "\n## " << nx << "x" << ny
              << " for " << nsteps << " time steps"
              << " (" << nx*ny << " grid points)\n";

    // allocate memory on device and host
    // note : allocate enough memory for the halo around the boundary
    auto buffer_size = nx*ny;

#ifdef OPENACC_DATA
    // x0, x1 managed by OpenACC's runtime
    double *x0 = new double[buffer_size];
    double *x1 = new double[buffer_size];
#else
    // x0, x1 manually managed with CUDA
    double *x_host = malloc_host_pinned<double>(buffer_size);
    double *x0     = malloc_device<double>(buffer_size);
    double *x1     = malloc_device<double>(buffer_size);
#endif

    double start_diffusion, time_diffusion;

#ifdef OPENACC_DATA
    // TODO: Move data to the GPU
#endif
    {
        // set initial conditions of 0 everywhere
        fill_gpu(x0, 0., buffer_size);
        fill_gpu(x1, 0., buffer_size);

        // set boundary conditions of 1 on south border
        fill_gpu(x0, 1., nx);
        fill_gpu(x1, 1., nx);
        fill_gpu(x0+nx*(ny-1), 1., nx);
        fill_gpu(x1+nx*(ny-1), 1., nx);

        // time stepping loop
        #pragma acc wait
        start_diffusion = get_time();
        for(auto step=0; step<nsteps; ++step) {
            diffusion_gpu(x0, x1, nx-2, ny-2, dt);
#ifdef OPENACC_DATA
            copy_gpu(x0, x1, buffer_size);
#else
            std::swap(x0, x1);
#endif
        }

        #pragma acc wait
        time_diffusion = get_time() - start_diffusion;
    } // end of acc data

#ifdef OPENACC_DATA
    auto x_res = x1;
#else
    copy_to_host<double>(x0, x_host, buffer_size);
    auto x_res = x_host;
#endif

    std::cout << "## " << time_diffusion << "s, "
              << nsteps*(nx-2)*(ny-2) / time_diffusion << " points/second\n\n";

    std::cout << "writing to output.bin/bov\n";
    write_to_file(nx, ny, x_res);
    return 0;
}
