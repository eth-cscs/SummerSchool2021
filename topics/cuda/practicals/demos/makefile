NVCC_FLAGS=-arch=sm_60 -std=c++11 -O3 -g -G -I../include
GCC_FLAGS=-std=c++11 -O3 -fopenmp

LINK=-lcublas -lcuda

CUDA_BENCHMARKS= bandwidth

BENCHMARKS=$(CUDA_BENCHMARKS)

all : $(BENCHMARKS)

bandwidth : bandwidth.cu
	nvcc $(NVCC_FLAGS) bandwidth.cu -o bandwidth $(LINK)

clean :
	rm -f $(BENCHMARKS) *.nvvp
