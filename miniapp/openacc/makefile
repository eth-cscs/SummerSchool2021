CXX = pgc++
CXXFLAGS = -std=c++11 -O3 -acc -Minfo=accel -ta=tesla:pinned -Mnorpath
LDFLAGS = -L$(CUDATOOLKIT_HOME)/lib64

SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp   main.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .cpp

all: main unit_tests

stats.o: stats.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c stats.cpp

data.o: data.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c data.cpp

operators.o: operators.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c operators.cpp

linalg.o: linalg.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c linalg.cpp

main.o: main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c main.cpp

unit_tests.o: unit_tests.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c unit_tests.cpp

main: $(OBJ) main.o
	$(CXX) $(CXXFLAGS) $(OBJ) main.o -o main $(LDFLAGS) -lcublas

unit_tests: data.o linalg.o unit_tests.o
	$(CXX) $(CXXFLAGS) unit_tests.o $(OBJ) -o unit_tests $(LDFLAGS) -lcublas
#	srun ./unit_tests                       # run with interactive session
#	srun -Cgpu ./unit_tests                 # run without interactive session

clean:
	rm -f main
	rm -f unit_tests
	rm -f *.o
	rm -f *.i
	rm -f output.*
