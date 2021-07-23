#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <openacc.h>

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
    size_t nx = 128;
    size_t ny = 1 << pow;
    double dt = 0.1;

    // initialize MPI
    int mpi_rank, mpi_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (ny % mpi_size) {
        std::cout << "error : global domain dimension " << ny
                  << "must be divisible by number of MPI ranks "
                  << mpi_size << "\n";
        exit(1);
    } else if (mpi_rank == 0) {
        std::cout << "\n## " << mpi_size << " MPI ranks" << std::endl;
        std::cout << "## " << nx << "x" << ny
                  << " : " << nx << "x" << ny/mpi_size << " per rank"
                  << " for " << nsteps << " time steps"
                  << " (" << nx*ny << " grid points)\n";
    }

    ny /= mpi_size;

    // adjust dimensions for halo
    nx += 2;
    ny += 2;

    // allocate memory on device and host
    // note : allocate enough memory for the halo around the boundary
    auto buffer_size = nx*ny;

#ifdef OPENACC_DATA
    double *x0     = malloc_host_pinned<double>(buffer_size);
    double *x1     = malloc_host_pinned<double>(buffer_size);
#else
    double *x_host = (double *) malloc(buffer_size*sizeof(double));
    // double *x_host = malloc_host_pinned<double>(buffer_size);
    double *x0     = malloc_device<double>(buffer_size);
    double *x1     = malloc_device<double>(buffer_size);
#endif

    double start_diffusion, time_diffusion;

#ifdef OPENACC_DATA
    #pragma acc data create(x0[0:buffer_size]) copyout(x1[0:buffer_size])
#endif
    {
        // set initial conditions of 0 everywhere
        fill_gpu(x0, 0., buffer_size);
        fill_gpu(x1, 0., buffer_size);

        // set boundary conditions of 1 on south border
        if (mpi_rank == 0) {
            fill_gpu(x0, 1., nx);
            fill_gpu(x1, 1., nx);
        }

        if (mpi_rank == mpi_size-1) {
            fill_gpu(x0+nx*(ny-1), 1., nx);
            fill_gpu(x1+nx*(ny-1), 1., nx);
        }

        auto south = mpi_rank - 1;
        auto north = mpi_rank + 1;

        // time stepping loop
        #pragma acc wait
        start_diffusion = get_time();
        for(auto step=0; step<nsteps; ++step) {
            MPI_Request requests[4];
            MPI_Status  statuses[4];
            auto num_requests = 0;

#ifdef OPENACC_DATA
            #pragma acc host_data use_device(x0, x1)
#endif
            {
                if (south >= 0) {
                    // x0(:, 0) <- south
                    MPI_Irecv(x0,    nx, MPI_DOUBLE, south, 0, MPI_COMM_WORLD,
                              &requests[0]);
                    // x0(:, 1) -> south
                    MPI_Isend(x0+nx, nx, MPI_DOUBLE, south, 0, MPI_COMM_WORLD,
                              &requests[1]);
                    num_requests += 2;
                }

                // exchange with north
                if(north < mpi_size) {
                    // x0(:, ny-1) <- north
                    MPI_Irecv(x0+(ny-1)*nx, nx, MPI_DOUBLE, north, 0,
                              MPI_COMM_WORLD, &requests[num_requests]);
                    // x0(:, ny-2) -> north
                    MPI_Isend(x0+(ny-2)*nx, nx, MPI_DOUBLE, north, 0,
                              MPI_COMM_WORLD, &requests[num_requests+1]);
                    num_requests += 2;
                }
            }

            MPI_Waitall(num_requests, requests, statuses);

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

    if (mpi_rank == 0) {
        std::cout << "## " << time_diffusion << "s, "
                  << nsteps*(nx-2)*(ny-2)*mpi_size / time_diffusion
                  << " points/second\n\n";

        std::cout << "writing to output.bin/bov\n";
        write_to_file(nx, ny, x_res);
    }

    MPI_Finalize();
    return 0;
}
