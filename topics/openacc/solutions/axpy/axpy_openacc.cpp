#include <iostream>
#include "util.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void axpy(int n, double alpha, const double *x, double* y) {

#pragma omp parallel for
    for(auto i=0; i<n; ++i) {
        y[i] += alpha*x[i];
    }
}

// OpenACC implementation of axpy kernel
void axpy_gpu(int n, double alpha, const double *x, double* y) {

    int i;
    #pragma acc parallel loop copyin(x[0:n]) copy(y[0:n])
    for(i = 0; i < n; ++i) {
        y[i] += alpha*x[i];
    }
}

// version informations
void print_versions(void) {

#ifdef _OPENMP
    int cscs_omp_v, ompnumthreadid, ompnumthreads1;
    cscs_omp_v = _OPENMP;
    char* ompnumthreads2 = getenv ("OMP_NUM_THREADS");

    #pragma omp parallel private(ompnumthreadid) shared(ompnumthreads1)
    {
        ompnumthreadid = omp_get_thread_num();
        // #pragma omp barrier
        #pragma omp master
        {
            ompnumthreads1 = omp_get_num_threads();
            std::cout << "OPENMP version: " << cscs_omp_v
                      << " num_threads=" << ompnumthreads1
                      << " OMP_NUM_THREADS=" << ompnumthreads2
                      << std::endl; 
        }
    }
#endif

#ifdef _OPENACC
    std::cout << "OPENACC version: " << _OPENACC << std::endl;
#endif

// MPI_Get_version()


}

int main(int argc, char** argv)
{
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;

    std::cout << "memcopy and daxpy test of size " << n << "\n";

    double* x = malloc_host<double>(n, 1.5);
    double* y = malloc_host<double>(n, 3.0);

    // use dummy fields to avoid cache effects, which make results harder to
    // interpret use 1<<24 to ensure that cache is completely purged for all n
    double* x_ = malloc_host<double>(n, 1.5);
    double* y_ = malloc_host<double>(n, 3.0);

    // openmp version:
    auto start = get_time();
    axpy(n, 2.0, x_, y_);
    auto time_axpy_omp = get_time() - start;

    // openacc version:
    start = get_time();
    axpy_gpu(n, 2.0, x, y);
    auto time_axpy_gpu = get_time() - start;

    std::cout << "-------\ntimings\n-------\n";
    std::cout << "axpy (openmp): "  << time_axpy_omp << " s\n";
    std::cout << "axpy (openacc): " << time_axpy_gpu << " s\n";

    // check for errors
    auto errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for (auto i = 0; i < n; ++i) {
        if (std::fabs(6.-y[i]) > 1e-15) {
            ++errors;
        }
    }

    if (errors > 0) {
        std::cout << "\n============ FAILED with " << errors << " errors\n";
    } else {
        std::cout << "\n============ PASSED\n";
    }

    free(x);
    free(y);
    return 0;
}
