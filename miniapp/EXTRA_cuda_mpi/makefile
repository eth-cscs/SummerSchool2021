CXX=CC
CUDA=nvcc
CXXFLAGS=-O3 -fopenmp
CUDAFLAGS=-O3 -std=c++11 -arch=sm_60

IFLAGS=-I$(MPICH_DIR)/include

SOURCES = stats.cu  data.cu  operators.cu  linalg.cu    main.cu
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o     main.o      cuda_helpers.o

#ONESIDED_LIB=-Wl,--whole-archive,-ldmapp,--no-whole-archive

.SUFFIXES: .cpp

all: main

stats.o: stats.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) $(IFLAGS) -c stats.cu

data.o: data.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) $(IFLAGS) -c data.cu

operators.o: operators.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) $(IFLAGS) -c operators.cu

linalg.o: linalg.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) $(IFLAGS) -c linalg.cu

cuda_helpers.o: cuda_helpers.cu
	$(CUDA) $(CUDAFLAGS) $(IFLAGS) -c cuda_helpers.cu

main.o: main.cu $(HEADERS) unit_tests.h
	$(CUDA) $(CUDAFLAGS) $(IFLAGS) -c main.cu

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o main -lcublas $(ONESIDED_LIB)

clean:
	rm -f main
	rm -f *.o
	rm -f *.i
	rm -f output.*
