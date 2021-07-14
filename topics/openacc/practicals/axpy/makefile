.PHONY: all clean

CRAY ?= 1
PGI  ?= 0
GCC  ?= 0
VERBOSE ?= 0
override CPPFLAGS += -I../common

ifeq ($(PE_ENV), CRAY)
	CXX = CC
	FC = ftn
	CXXFLAGS = -hstd=c++11 -O3 -homp -hacc -dynamic
	FCFLAGS  = -O3 -homp -hacc -dynamic
	ifeq ($(VERBOSE), 1)
		CXXFLAGS += -hmsgs
		FCFLAGS  += -hmsgs
	endif
endif

ifeq ($(PE_ENV), PGI)
	CXX = pgc++
	FC  = pgf90
	CXXFLAGS = -DPGI -std=c++11 -O3 -acc -Minfo=accel -ta=tesla:pinned
	FCFLAGS  = -O3 -acc -Minfo=accel -ta=tesla:pinned
	LDFLAGS += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart
endif

ifeq ($(PE_ENV), GNU)
	# not tested
	CXX = g++
	FC  = gfortran
	CXXFLAGS = -std=c++11 -O3 -fopenmp -fopenacc -I$(CRAY_CUDATOOLKIT_DIR)/include
	FCFLAGS  = -O3 -fopenacc -fopenmp
	LDFLAGS += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart
endif

OPENACC_BENCHMARKS= axpy.openacc axpy.openacc.fort

BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

util.o: ../common/util.f03
	$(FC) $(CPPFLAGS) -c $(FCFLAGS) $<

axpy.openacc: axpy_openacc.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

axpy.openacc.fort: axpy_openacc.f90 util.o
	$(FC) $(CPPFLAGS) $(FCFLAGS) $< util.o -o $@ $(LDFLAGS)

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
