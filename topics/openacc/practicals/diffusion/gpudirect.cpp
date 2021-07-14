#include <iostream>
#include <mpi.h>


constexpr size_t BUFF_SIZE = 100


int main() {
    double *A = new double[BUFF_SIZE];
    double *B = new double[BUFF_SIZE];
    double *C = new double[BUFF_SIZE];

    for (auto i = 0; i < BUFF_SIZE; ++i) {
        A[i] = 1.
        B[i] = 1.
    }

    for (auto step = 0; step < 10; ++step) {
        MPI_Request request;
    }


    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
