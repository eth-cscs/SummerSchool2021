NVCC_FLAGS=-arch=sm_60 -std=c++14 -O3 -g -G
NVCC_LINK=-lcublas -lcuda

GCC_FLAGS=-std=c++14 -O3 -fopenmp ${CRAY_CUDATOOLKIT_INCLUDE_OPTS}
GCC_LINK=-lcublas -lcuda ${CRAY_CUDATOOLKIT_POST_LINK_OPTS}


CUDA_BENCHMARKS=explicit managed sort

BENCHMARKS=$(CUDA_BENCHMARKS)

all : $(BENCHMARKS)

explicit : explicit.cu util.hpp
	nvcc $(NVCC_FLAGS) explicit.cu -o explicit $(NVCC_LINK)

managed : managed.cu util.hpp
	nvcc $(NVCC_FLAGS) managed.cu -o managed $(NVCC_LINK)

# this performance test must be compiled without debug flags (-g -G)
# otherwise performance suffers a lot
sort : sort.cu util.hpp
	nvcc -O3 sort.cu -o sort $(NVCC_LINK)

clean :
	rm -f $(BENCHMARKS) *.nvvp
