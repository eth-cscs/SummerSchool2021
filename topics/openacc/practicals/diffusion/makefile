.PHONY: all clean

CRAY ?= 1
PGI  ?= 0
GCC  ?= 0
VERBOSE ?= 0
override CPPFLAGS += -I../common

CXX     = CC
MPI_CXX = CC
FC      = ftn
MPI_FC  = ftn

ifeq ($(PE_ENV), CRAY)
	CXX = CC
	FC  = ftn
	CXXFLAGS = -hstd=c++11 -O3 -hacc -homp -dynamic
	FCFLAGS  = -O3 -homp -hacc -dynamic
	ifeq ($(VERBOSE), 1)
		CXXFLAGS += -hmsgs
		FCFLAGS  += -hmsgs
	endif
endif

ifeq ($(PE_ENV), PGI)
	CXX = pgc++
	FC  = pgf90
	override CPPFLAGS += -I$(CRAY_CUDATOOLKIT_DIR)/include
	CXXFLAGS = -g -DPGI -std=c++11 -O3 -acc -Minfo=accel -ta=tesla:cc60,pinned -Mnorpath
	FCFLAGS  = -g -O3 -acc -Minfo=accel -ta=tesla:cc60,pinned -Mnorpath
	LDFLAGS += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart -Mnorpath
endif

ifeq ($(PE_ENV), GNU)
	# not tested
	CXX = g++
	FC  = gfortran
	CXXFLAGS = -std=c++11 -O3 -fopenmp -fopenacc -I$(CRAY_CUDATOOLKIT_DIR)/include
	FCFLAGS  = -O3 -fopenacc -fopenmp
	LDFLAGS += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart
endif

CXX_BENCHMARKS = diffusion2d.openacc diffusion2d.openacc.cuda diffusion2d.omp \
	diffusion2d.openacc.mpi diffusion2d.openacc.cuda.mpi
FTN_BENCHMARKS = diffusion2d.openacc.fort diffusion2d.openacc.fort.mpi

BENCHMARKS=$(CXX_BENCHMARKS) $(FTN_BENCHMARKS)

all: $(BENCHMARKS)

util.o: ../common/util.f03
	$(FC) $(CPPFLAGS) -c $(FCFLAGS) $<

diffusion2d.o: diffusion2d.f03
	$(FC) $(CPPFLAGS) -c $(FCFLAGS) $<

diffusion2d.openacc: diffusion2d_openacc.cpp diffusion2d.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DOPENACC_DATA $< -o $@ $(LDFLAGS)

diffusion2d.openacc.cuda: diffusion2d_openacc.cpp diffusion2d.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

diffusion2d.openacc.mpi: diffusion2d_openacc_mpi.cpp diffusion2d.hpp
	$(MPI_CXX) $(CPPFLAGS) $(CXXFLAGS) -DOPENACC_DATA $< -o $@ $(LDFLAGS)

diffusion2d.openacc.cuda.mpi: diffusion2d_openacc_mpi.cpp diffusion2d.hpp
	$(MPI_CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

diffusion2d.omp: diffusion2d_omp.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

diffusion2d.openacc.fort: diffusion2d_openacc.f90 util.o diffusion2d.o
	$(FC) $(CPPFLAGS) $(FCFLAGS) $< util.o diffusion2d.o -o $@

diffusion2d.openacc.fort.mpi: diffusion2d_openacc_mpi.f90 util.o diffusion2d.o
	$(MPI_FC) $(CPPFLAGS) $(FCFLAGS) $< util.o diffusion2d.o -o $@

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod output.*
