//******************************************
// operators
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
//
// implements
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

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

// const qualifier issues a "cannot determine bounds for array" error in PGI
// void diffusion(const data::Field &U, data::Field &S)
void diffusion(data::Field &U, data::Field &S)
{
    using data::options;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = options.nx;
    int ny = options.ny;
    int iend  = nx - 1;
    int jend  = ny - 1;

    // TODO: Offload the computation of all the grid points to the GPU.
    //       Check the compiler output to verify that you get the parallelism you expect.

    // the interior grid points
    {
    for (int j=1; j < jend; j++) {
        for (int i=1; i < iend; i++) {
            S(i,j) = -(4. + alpha) * U(i,j)               // central point
                                    + U(i-1,j) + U(i+1,j) // east and west
                                    + U(i,j-1) + U(i,j+1) // north and south
                                    + alpha * x_old(i,j)
                                    + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the east boundary
    {
        int i = nx - 1;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1) + U(i,j+1)
                        + alpha*x_old(i,j) + bndE[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the west boundary
    {
        int i = 0;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the north boundary (plus NE and NW corners)
    {
        int j = ny - 1;

        {
            int i = 0; // NW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndW[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // north boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j-1)
                        + alpha*x_old(i,j) + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx-1; // NE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndE[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the south boundary
    {
        int j = 0;

        {
            int i = 0; // SW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // south boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx - 1; // SE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndE[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }
    } // end acc kernels
}

} // namespace operators
