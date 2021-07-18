NVCC_FLAGS=-arch=sm_60 -std=c++11 -O3 -g -G
NVCC_LINK=-lcublas -lcuda

GCC_FLAGS=-std=c++11 -O3 -fopenmp ${CRAY_CUDATOOLKIT_INCLUDE_OPTS}
GCC_LINK=-lcublas -lcuda ${CRAY_CUDATOOLKIT_POST_LINK_OPTS}


CUDA_BENCHMARKS=explicit managed

BENCHMARKS=$(CUDA_BENCHMARKS)

all : $(BENCHMARKS)

explicit : explicit.cu util.hpp
	nvcc $(NVCC_FLAGS) explicit.cu -o explicit $(NVCC_LINK)

managed : managed.cu util.hpp
	nvcc $(NVCC_FLAGS) managed.cu -o managed $(NVCC_LINK)

clean :
	rm -f $(BENCHMARKS) *.nvvp
