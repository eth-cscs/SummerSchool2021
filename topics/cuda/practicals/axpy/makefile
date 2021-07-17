NVCC_FLAGS=-arch=sm_60 -std=c++11 -O3 -g -I../include
GCC_FLAGS=-std=c++11 -O3 -fopenmp -I../include

LINK=-lcublas -lcuda

all : axpy axpy_omp newton

axpy : axpy.cu
	nvcc $(NVCC_FLAGS) axpy.cu -o axpy $(LINK)

axpy_omp : axpy_omp.cpp
	g++ ${GCC_FLAGS} axpy_omp.cpp -o axpy_omp -march=core-avx2

newton : newton.cu
	nvcc $(NVCC_FLAGS) newton.cu -o newton $(LINK)

clean :
	rm -f axpy axpy_omp newton *.nvvp
