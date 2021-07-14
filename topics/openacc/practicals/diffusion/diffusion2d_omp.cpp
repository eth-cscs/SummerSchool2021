#include <iostream>
#include <fstream>
#include <cstdio>
#include <omp.h>

#define NO_CUDA
#include "util.h"

// 2D diffusion example
// the grid has a fixed width of nx=128
// the use specifies the height, ny, as a power of two
// note that nx and ny have 2 added to them to account for halos

void diffusion_omp(const double *x0, double *x1, int nx, int ny, double dt)
{
    int i, j;
    auto width = nx+2;

    #pragma omp parallel for collapse(2), private(i,j)
    for (j = 1; j < ny+1; ++j) {
        for (i = 1; i < nx+1; ++i) {
            auto pos = i + j*width;
            x1[pos] = x0[pos] + dt * (-4.*x0[pos]
                                      + x0[pos-width] + x0[pos+width]
                                      + x0[pos-1]  + x0[pos+1]);
        }
    }
}

void write_to_file(int nx, int ny, double* data);

int main(int argc, char** argv) {
    // set up parameters
    // first argument is the y dimension = 2^arg
    size_t pow    = read_arg(argc, argv, 1, 8);
    // second argument is the number of time steps
    size_t nsteps = read_arg(argc, argv, 2, 100);
    // third argument is nonzero if shared memory version is to be used
    bool use_shared = read_arg(argc, argv, 3, 0);

    // set domain size
    size_t nx = 128 + 2;
    size_t ny = (1 << pow) + 2;
    double dt = 0.1;

    std::cout << "\n## " << nx << "x" << ny
              << " for " << nsteps << " time steps"
              << " (" << nx*ny << " grid points)\n";

    // allocate memory on device and host
    // note : allocate enough memory for the halo around the boundary
    auto buffer_size = nx*ny;
    double *x0 = malloc_host<double>(buffer_size);
    double *x1 = malloc_host<double>(buffer_size);

    double start_diffusion, time_diffusion;
    // set initial conditions of 0 everywhere
    std::fill(x0, x0 + buffer_size, 0.);
    std::fill(x1, x1 + buffer_size, 0.);

    // set boundary conditions of 1 on south border
    std::fill(x0, x0 + nx, 1.);
    std::fill(x1, x1 + nx, 1.);
    std::fill(x0 + nx*(ny-1), x0 + nx*ny, 1.);
    std::fill(x1 + nx*(ny-1), x1 + nx*ny, 1.);

    std::cout << "Running on " << omp_get_max_threads() << " threads\n";

    // time stepping loop
    start_diffusion = get_time();
    for(auto step=0; step<nsteps; ++step) {
        diffusion_omp(x0, x1, nx-2, ny-2, dt);
        std::swap(x0, x1);
    }
    time_diffusion = get_time() - start_diffusion;


    std::cout << "## " << time_diffusion << "s, "
              << nsteps*(nx-2)*(ny-2) / time_diffusion << " points/second\n\n";

    std::cout << "writing to output.bin/bov\n";
    write_to_file(nx, ny, x0);
    return 0;
}

void write_to_file(int nx, int ny, double* data) {
    {
        FILE* output = fopen("output.bin", "w");
        fwrite(data, sizeof(double), nx * ny, output);
        fclose(output);
    }

    std::ofstream fid("output.bov");
    fid << "TIME: 0.0\n";
    fid << "DATA_FILE: output.bin\n";
    fid << "DATA_SIZE: " << nx << " " << ny << " 1\n";
    fid << "DATA_FORMAT: DOUBLE\n";
    fid << "VARIABLE: phi\n";
    fid << "DATA_ENDIAN: LITTLE\n";
    fid << "CENTERING: nodal\n";
    fid << "BRICK_SIZE: 1.0 1.0 1.0\n";
}
