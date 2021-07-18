CXX=CC
CUDA=nvcc
CXXFLAGS=-O3 -fopenmp
CUDAFLAGS=-O3 -std=c++11 -arch=sm_60

SOURCES = stats.cu  data.cu  operators.cu  linalg.cu    main.cu
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .cpp

all: main unit_tests

stats.o: stats.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c stats.cu

data.o: data.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c data.cu

operators.o: operators.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c operators.cu

linalg.o: linalg.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c linalg.cu

main.o: main.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c main.cu

unit_tests.o: unit_tests.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c unit_tests.cu

main: $(OBJ) main.o
	$(CXX) $(CXXFLAGS) $(OBJ) main.o -o main -lcublas

unit_tests: data.o linalg.o unit_tests.o
	$(CXX) $(CXXFLAGS) unit_tests.o $(OBJ) -o unit_tests -lcublas
#	srun ./unit_tests                       # run with interactive session
#	srun -Cgpu ./unit_tests                 # run without interactive session

clean:
	rm -f main
	rm -f unit_tests
	rm -f *.o
	rm -f *.i
	rm -f output.*
