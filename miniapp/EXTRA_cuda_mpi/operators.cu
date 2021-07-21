//******************************************
// operators
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
//
// implements
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include <mpi.h>

#include "cuda_helpers.h"
#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

// POD type holding information for device
struct DiffusionParams {
    int nx;
    int ny;
    double alpha;
    double dxs;
    double *x_old;
    double *bndN;
    double *bndE;
    double *bndS;
    double *bndW;
};

// copy of global parameters for kernels to use directly
__device__
DiffusionParams params;

// copies the global parameters to the device
// must be called once at setup, before any of the stencil kernels are executed
void setup_params_on_device(
        int nx, int ny,
        double alpha, double dxs)
{
    auto p = DiffusionParams {
        nx,
        ny,
        alpha,
        dxs,
        data::x_old.device_data(),
        data::bndN.device_data(),
        data::bndE.device_data(),
        data::bndS.device_data(),
        data::bndW.device_data()
    };

    cuda_api_call(
        cudaMemcpyToSymbol(params, &p, sizeof(DiffusionParams))
    );
}

namespace kernels {
    __global__
    void stencil_interior(double* S, const double *U) {
        auto nx = params.nx;
        auto ny = params.ny;

        auto i = threadIdx.x + blockDim.x*blockIdx.x;
        auto j = threadIdx.y + blockDim.y*blockIdx.y;
        auto pos = i + j * nx;

        // stencil is applied to interior grid pints, i.e. (i,j) such that
        //      i \in [1, nx-1)
        //      j \in [1, ny-1)
        auto is_interior = i<(nx-1) && j<(ny-1) && (i>0 && j>0);
        if(is_interior) {
            S[pos] = -(4. + params.alpha) * U[pos]          // central point
                                   + U[pos-1] + U[pos+1]    // east and west
                                   + U[pos-nx] + U[pos+nx]  // north and south
                                   + params.alpha * params.x_old[pos]
                                   + params.dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_east_west(double* S, const double *U) {
        auto j = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        auto find_pos = [&nx] (size_t i, size_t j) {
            return i + j * nx;
        };

        if(j>0 && j<ny-1) {
            // EAST : i = nx-1
            auto pos = find_pos(nx-1, j);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos-nx] + U[pos+nx]
                        + alpha*params.x_old[pos] + params.bndE[j]
                        + dxs * U[pos] * (1.0 - U[pos]);

            // WEST : i = 0
            pos = find_pos(0, j);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos+1] + U[pos-ny] + U[pos+nx]
                        + alpha * params.x_old[pos] + params.bndW[j]
                        + dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_north_south(double* S, const double *U) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        if(i>0 && i<nx-1) {
            // NORTH : j = ny -1
            auto pos = i + nx*(ny-1);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos+1] + U[pos-nx]
                        + alpha*params.x_old[pos] + params.bndN[i]
                        + dxs * U[pos] * (1.0 - U[pos]);
            // SOUTH : j = 0
            pos = i;
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos+1] + U[pos+nx]
                        + alpha * params.x_old[pos] + params.bndS[i]
                        + dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_corners(double* S, const double* U) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        auto find_pos = [&nx] (size_t i, size_t j) {
            return i + j * nx;
        };

        // only 1 thread executes this kernel
        if(i==0) {
            // NORTH-EAST
            auto pos = find_pos(nx-1, ny-1);
            S[pos] = -(4. + alpha) * U[pos]                     // central point
                                   + U[pos-1]    + params.bndE[ny-1] // east and west
                                   + U[pos-nx] + params.bndN[nx-1] // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // SOUTH-EAST
            pos = find_pos(nx-1, 0);
            S[pos] = -(4. + alpha) * U[pos]                     // central point
                                   + U[pos-1]    + params.bndE[0]      // east and west
                                   + params.bndS[nx-1]+ U[pos+nx]  // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // SOUTH-WEST
            pos = find_pos(0, 0);
            S[pos] = -(4. + alpha) * U[pos]                // central point
                                   + params.bndW[0] + U[pos+1]    // east and west
                                   + params.bndS[0] + U[pos+nx] // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // NORTH-WEST
            pos = find_pos(0, ny-1);
            S[pos] = -(4. + alpha) * U[pos]                 // central point
                                   + params.bndW[nx-1]+ U[pos+1] // east and west
                                   + U[pos-nx] + params.bndN[0]  // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);
        }
    }
}

// This function will copy a 1D strip from a 2D field to a 1D buffer.
// It is used to copy the values from along the edge of a field to
// a flat buffer for sending to MPI neighbors.
void pack_buffer(data::Field const& from, data::Field &buffer, int startx, int starty, int stride) {
    int nx = from.xdim();
    int ny = from.ydim();
    int pos = startx + starty*nx;
    auto status = cublasDcopy(
        cublas_handle(), buffer.length(),
        from.device_data() + pos, stride,
        buffer.device_data(),    1
    );
    if(status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "error : cublas copy for boundary condition" << std::endl;
        exit(-1);
    }
}

// Exchange that performs MPI send/recv from/to host memory, and copies
// results from and to the GPU.
void exchange_rdma(data::Field const& U) {
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;


    int nx = domain.nx;
    int ny = domain.ny;

    // NOTE TO TEACHERS:
    //
    // Synchronization of the pack, Isend and Irecv operations is very important for
    // RDMA communication.
    // Students will get subtle bugs in the application if they aren't careful.
    //
    // The Problem:
    //     The Cray MPI uses internal CUDA streams and RDMA to copy from the buffX on
    //     the other MPI rank into the bndX field on this GPU.
    //     The miniapp launches all kernels on the default stream, so the bndX field
    //     may be updated by the MPI_Irecv call at the same time the bndX field
    //     is being read by a kernel running a stencil from the previous iteration.
    //
    // The Solution: There are two possible solutions:
    // option 1. A single call to cudaDeviceSynchronize() before the first MPI_Irecv will
    //        ensure that all kernels that depend on the current values in bndX have
    //        finished executing. This is the simplest and most reliable method.
    // option 2. Call the pack_buffer() function before the MPI_Irecv() for each boundary.
    //        The reason that this works is as a result of a subtle interaction.
    //        pack_buffer() uses cublas to perform the copy.
    //        Cublas calls are blocking, i.e. the host waits until the GPU work is finished,
    //        and they are performed in CUDA stream 0. These two side effects mean
    //        that all operations from previous steps will be completed before the call
    //        Irecv can start filling bndX.
    //        If we were using a kernel we wrote ourselves to perform the pack, the problem
    //        would still persist, because the kernel would not block on the host side,
    //        so I don't consider this to be a very robust solution.
    //
    // This issue often doesn't affect 1 MPI rank, and usually isn't triggered with 2 MPI
    // ranks. However, with 4 MPI ranks and large domains (512x512 and greater), the solution
    // won't converge, and it will happen at different points on each run.
    // If students get to this point, get them to set the CUDA_LAUNCH_BLOCKING=1 environment
    // variable, and the problem will go away.
    // Then work with them to isolate the issue by placing cudaDeviceSynchronize() calls in
    // the code. I would suggest that they put a cudaDeviceSynchronize() at the top of
    // the diffusion() function, where it will fix the problem. Then get them to zero in on
    // exactly where it has to be placed.

    cudaDeviceSynchronize();

    int i, j;
    int num_requests = 0;
    MPI_Request requests[8];
    MPI_Status statuses[8];

    if(domain.neighbour_north>=0) {
        i = 0;
        j = ny - 1; // NORTH
        pack_buffer(U, buffN, i, j, 1);
        MPI_Irecv(bndN.device_data(), nx, MPI_DOUBLE, domain.neighbour_north, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
        MPI_Isend(buffN.device_data(), nx, MPI_DOUBLE, domain.neighbour_north, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
    }
    if(domain.neighbour_south>=0) {
        i = 0;
        j = 0; // SOUTH
        pack_buffer(U, buffS, i, j, 1);
        MPI_Irecv(bndS.device_data(), nx, MPI_DOUBLE, domain.neighbour_south, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
        MPI_Isend(buffS.device_data(), nx, MPI_DOUBLE, domain.neighbour_south, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
    }
    if(domain.neighbour_east>=0) {
        i = nx - 1; // EAST
        j = 0;
        pack_buffer(U, buffE, i, j, nx);
        MPI_Irecv(bndE.device_data(), ny, MPI_DOUBLE, domain.neighbour_east, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
        MPI_Isend(buffE.device_data(), ny, MPI_DOUBLE, domain.neighbour_east, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
    }
    if(domain.neighbour_west>=0) {
        i = 0; // WEST
        j = 0;
        pack_buffer(U, buffW, i, j, nx);
        MPI_Irecv(bndW.device_data(), ny, MPI_DOUBLE, domain.neighbour_west, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
        MPI_Isend(buffW.device_data(), ny, MPI_DOUBLE, domain.neighbour_west, 0, MPI_COMM_WORLD, &(requests[num_requests++]));
    }

    MPI_Waitall(num_requests, requests, statuses);
}

// overlap communication by computation by splitting the exchange
void start_exchange_rdma(data::Field const& U, MPI_Request requests[], int& num_requests) {
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;

    int nx = domain.nx;
    int ny = domain.ny;
    num_requests = 0;

    cudaDeviceSynchronize();

    int i, j;

    if(domain.neighbour_north>=0) {
        i = 0;
        j = ny - 1; // NORTH
        pack_buffer(U, buffN, i, j, 1);
        MPI_Irecv(bndN.device_data(), nx, MPI_DOUBLE, domain.neighbour_north, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Isend(buffN.device_data(), nx, MPI_DOUBLE, domain.neighbour_north, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if(domain.neighbour_south>=0) {
        i = 0;
        j = 0; // SOUTH
        pack_buffer(U, buffS, i, j, 1);
        MPI_Irecv(bndS.device_data(), nx, MPI_DOUBLE, domain.neighbour_south, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Isend(buffS.device_data(), nx, MPI_DOUBLE, domain.neighbour_south, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if(domain.neighbour_east>=0) {
        i = nx - 1; // EAST
        j = 0;
        pack_buffer(U, buffE, i, j, nx);
        MPI_Irecv(bndE.device_data(), ny, MPI_DOUBLE, domain.neighbour_east, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Isend(buffE.device_data(), ny, MPI_DOUBLE, domain.neighbour_east, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if(domain.neighbour_west>=0) {
        i = 0; // WEST
        j = 0;
        pack_buffer(U, buffW, i, j, nx);
        MPI_Irecv(bndW.device_data(), ny, MPI_DOUBLE, domain.neighbour_west, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Isend(buffW.device_data(), ny, MPI_DOUBLE, domain.neighbour_west, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
}

void wait_exchange_rdma(MPI_Request requests[], int num_requests) {
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    MPI_Status statuses[8];
    MPI_Waitall(num_requests, requests, statuses);
}

void diffusion(data::Field const& U, data::Field &S)
{
    using data::options;
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = domain.nx;
    int ny = domain.ny;

    static bool is_initialized = false;
    if (!is_initialized) {
        setup_params_on_device(nx, ny, alpha, dxs);
        is_initialized = true;
    }

    //do exchange
    int num_requests = 0;
    MPI_Request requests[8];
    start_exchange_rdma(U, requests, num_requests);
    //exchange_rdma(U);

    // apply stencil to the interior grid points
    auto calculate_grid_dim = [] (size_t n, size_t block_dim) {
        return (n+block_dim-1)/block_dim;
    };
    dim3 block_dim(8, 8); // use 8x8 thread block dimensions
    dim3 grid_dim(
        calculate_grid_dim(nx, block_dim.x),
        calculate_grid_dim(ny, block_dim.y));
    kernels::stencil_interior<<<grid_dim, block_dim>>>(S.device_data(), U.device_data());

    wait_exchange_rdma(requests, num_requests);

    // apply stencil at boundaries
    auto bnd_grid_dim_y = calculate_grid_dim(ny, 64);
    kernels::stencil_east_west<<<bnd_grid_dim_y, 64>>>(S.device_data(), U.device_data());

    auto bnd_grid_dim_x = calculate_grid_dim(nx, 64);
    kernels::stencil_north_south<<<bnd_grid_dim_x, 64>>>(S.device_data(), U.device_data());

    kernels::stencil_corners<<<1, 1>>>(S.device_data(), U.device_data());
}

} // namespace operators
