NVCC_FLAGS=-arch=sm_60 -std=c++11 -O3 -I../include

LINK=-lcublas -lcuda

CUDA_BENCHMARKS=memcopy1 memcopy2 memcopy3

all : $(CUDA_BENCHMARKS)

#----------------------------------------
# CUDA benchmarks
#----------------------------------------
memcopy1 : memcopy1.cu
	nvcc $(NVCC_FLAGS) memcopy1.cu -o memcopy1 $(LINK)

memcopy2 : memcopy2.cu
	nvcc $(NVCC_FLAGS) memcopy2.cu -o memcopy2 $(LINK)

memcopy3 : memcopy3.cu
	nvcc $(NVCC_FLAGS) memcopy3.cu -o memcopy3 $(LINK)

clean :
	rm -f ${CUDA_BENCHMARKS}
