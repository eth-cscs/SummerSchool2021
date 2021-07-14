.PHONY: all clean

CRAY ?= 1
PGI  ?= 0
GCC  ?= 0
VERBOSE ?= 0
CPPFLAGS ?= -DNO_NAIVE

override CPPFLAGS += -I../common

ifeq ($(PE_ENV), CRAY)
	CXX = CC
	FC = ftn
	CXXFLAGS = -hstd=c++11 -O3 -homp -hacc -dynamic
	FCFLAGS  = -O3 -homp -hacc -dynamic
	LDFLAGS  = -lcublas
	ifeq ($(VERBOSE), 1)
		CXXFLAGS += -hmsgs
		FCFLAGS  += -hmsgs
	endif
endif

ifeq ($(PE_ENV), PGI)
	CXX = pgc++
	FC  = pgf90
	CXXFLAGS = -DPGI -std=c++11 -O3 -mp=allcores -acc -Minfo=accel -ta=tesla:cc60,pinned
	FCFLAGS  = -O3 -mp=allcores -acc -Minfo=accel -ta=tesla:cc60,pinned
	LDFLAGS  += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart -lcublas
endif

ifeq ($(PE_ENV), GNU)
	CXX = g++
	FC  = gfortran
	override CPPFLAGS += -I$(CRAY_CUDATOOLKIT_DIR)/include
	CXXFLAGS = -std=c++11 -O3 -fopenmp -fopenacc
	FCFLAGS  = -O3 -fopenacc -fopenmp
	LDFLAGS  += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart -lcublas
endif

OPENACC_BENCHMARKS= gemm.openacc

BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

util.o: ../common/util.f03
	$(FC) $(CPPFLAGS) -c $(FCFLAGS) $<

gemm.openacc: gemm.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# axpy.openacc.fort: axpy_openacc.f90 util.o
#	$(FC) $(FCFLAGS) $< util.o -o $@ $(LDFLAGS)

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
