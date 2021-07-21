#include <iostream>

#include <cmath>

#include <mpi.h>

#include "data.h"

namespace data{

// fields that hold the solution
Field x_new;
Field x_old;

// fields that hold the boundary points
Field bndN;
Field bndE;
Field bndS;
Field bndW;

// buffers used during boundary halo communication
Field buffN;
Field buffE;
Field buffS;
Field buffW;

Discretization options;
SubDomain      domain;

void SubDomain::init(int mpi_rank, int mpi_size, Discretization& discretization)
{
    // determine the number of subdomains in the x and y dimensions
    int dims[2] = { 0, 0 };
    MPI_Dims_create(mpi_size, 2, dims);

    ndomy = dims[0];
    ndomx = dims[1];

    // create a 2D non-periodic cartesian topology
    int periods[2] = { 0, 0 };

    // retrieve coordinates of the rank in the topology
    int coords[2];
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &comm_cart);
    MPI_Cart_coords(comm_cart, mpi_rank, 2, coords);

    domy = coords[0]+1;
    domx = coords[1]+1;

    // set neighbours for all directions
    // i.e. set neighbour_south neighbour_north neighbour_east neighbour_west
    MPI_Cart_shift(comm_cart, 0, 1, &neighbour_south, &neighbour_north);
    MPI_Cart_shift(comm_cart, 1, 1, &neighbour_west, &neighbour_east);

    // get bounding box
    nx = discretization.nx / ndomx;
    ny = discretization.ny / ndomy;
    startx = (domx-1)*nx+1;
    starty = (domy-1)*ny+1;

    // adjust for grid dimensions that do not divided evenly between the
    // sub-domains
    if( domx == ndomx )
        nx = discretization.nx - startx + 1;
    if( domy == ndomy )
        ny = discretization.ny - starty + 1;

    endx = startx + nx -1;
    endy = starty + ny -1;

    // get total number of grid points in this sub-domain
    N = nx*ny;

    rank = mpi_rank;
    size = mpi_size;
}

// print domain decomposition information to stdout
void SubDomain::print() {
    for(int i=0; i<size; i++) {
        if(rank == i) {
            std::cout << "rank " << rank << "/" << size
                      << " : (" << domx << "," << domy << ")"
                      << " neigh N:S " << neighbour_north << ":" << neighbour_south
                      << " neigh E:W " << neighbour_east << ":" << neighbour_west
                      << " local dims " << nx << " x " << ny
                      << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

} // namespace data
