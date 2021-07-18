NVCC_FLAGS=-arch=sm_60 -std=c++11 -O3 -I../include
GCC_FLAGS=-std=c++11 -O3 -fopenmp -I../include

LINK=-lcublas -lcuda

CUDA_BENCHMARKS= blur blur_twice dot string_reverse hist
OPENMP_BENCHMARKS= blur_twice_omp

all : $(CUDA_BENCHMARKS) $(OPENMP_BENCHMARKS)

#----------------------------------------
# CUDA benchmarks
#----------------------------------------
blur : blur.cu
	nvcc $(NVCC_FLAGS) blur.cu -o blur $(LINK)

blur_twice : blur_twice.cu
	nvcc $(NVCC_FLAGS) blur_twice.cu -o blur_twice $(LINK)

dot : dot.cu
	nvcc $(NVCC_FLAGS) dot.cu -o dot $(LINK)

string_reverse : string_reverse.cu
	nvcc $(NVCC_FLAGS) string_reverse.cu -o string_reverse $(LINK)

hist : hist.cu
	nvcc $(NVCC_FLAGS) hist.cu -o hist $(LINK)

#----------------------------------------
# openmp benchmarks
#----------------------------------------
blur_twice_omp : blur_twice.cpp
	g++ $(GCC_FLAGS) blur_twice.cpp -o blur_twice_omp

clean :
	rm -f ${CUDA_BENCHMARKS}
	rm -f ${OPENMP_BENCHMARKS}
