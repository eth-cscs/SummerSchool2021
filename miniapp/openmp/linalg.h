// linear algebra subroutines
// Ben Cumming @ CSCS

#ifndef LINALG_H
#define LINALG_H

#include "data.h"

namespace linalg
{
    using data::Field;

    extern bool cg_initialized;
    extern Field r, Ap, p, Fx, Fxold, v, xold; // 1d

    // initialize temporary storage fields used by the cg solver
    // I do this here so that the fields are persistent between calls
    // to the CG solver. This is useful if we want to avoid malloc/free calls
    // on the device for the OpenACC implementation (feel free to suggest a better
    // method for doing this)
    void cg_init(const int N);

    ////////////////////////////////////////////////////////////////////////////////
    //  blas level 1 reductions
    ////////////////////////////////////////////////////////////////////////////////

    // computes the inner product of x and y
    // x and y are vectors on length N
    double ss_dot(Field const& x, Field const& y, const int N);

    // computes the 2-norm of x
    // x is a vector on length N
    double ss_norm2(Field const& x, const int N);

    // sets entries in a vector to value
    // x is a vector on length N
    // value is th
    void ss_fill(Field& x, const double value, const int N);

    ////////////////////////////////////////////////////////////////////////////////
    //  blas level 1 vector-vector operations
    ////////////////////////////////////////////////////////////////////////////////

    // computes y := alpha*x + y
    // x and y are vectors on length N
    // alpha is a scalar
    void ss_axpy(Field& y, const double alpha, Field const& x, const int N);

    // computes y = x + alpha*(l-r)
    // y, x, l and r are vectors of length N
    // alpha is a scalar
    void ss_add_scaled_diff(Field& y, Field const& x, const double alpha,
        Field const& l, Field const& r, const int N);

    // computes y = alpha*(l-r)
    // y, l and r are vectors of length N
    // alpha is a scalar
    void ss_scaled_diff(Field& y, const double alpha,
        Field const& l, Field const& r, const int N);

    // computes y := alpha*x
    // alpha is scalar
    // y and x are vectors on length n
    void ss_scale(Field& y, const double alpha, Field& x, const int N);

    // computes linear combination of two vectors y := alpha*x + beta*z
    // alpha and beta are scalar
    // y, x and z are vectors on length n
    void ss_lcomb(Field& y, const double alpha, Field& x, const double beta,
        Field const& z, const int N);

    // copy one vector into another y := x
    // x and y are vectors of length N
    void ss_copy(Field& y, Field const& x, const int N);

    // conjugate gradient solver
    // solve the linear system A*x = b for x
    // the matrix A is implicit in the objective function for the diffusion equation
    // the value in x constitute the "first guess" at the solution
    // x(N)
    // ON ENTRY contains the initial guess for the solution
    // ON EXIT  contains the solution
    void ss_cg(Field& x, Field const& b, const int maxiters, const double tol, bool& success);
}

#endif // LINALG_H

