#pragma once

#include <iostream>

#include <cassert>

#include "cuda_helpers.h"

namespace data
{

namespace kernels {
    __global__
    static void fill_device(double* ptr, double val, size_t n) {
        auto i = threadIdx.x + blockIdx.x*blockDim.x;
        auto grid_step = blockDim.x*gridDim.x;

        while(i < n) {
            ptr[i] = val;
            i += grid_step;
        }
    }
}

// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct Discretization
{
    int nx;       // x dimension
    int ny;       // y dimension
    int N;        // grid dimension (nx*ny)
    int nt;       // number of time steps
    double dt;    // time step size
    double dx;    // distance between grid points
    double alpha; // dx^2/(D*dt)
};

// thin wrapper around a pointer that can be accessed as either a 2D or 1D array
// Field has dimension xdim * ydim in 2D, or length=xdim*ydim in 1D
class Field {
    public:
    // default constructor
    Field()
    :   host_ptr_(nullptr),
        device_ptr_(nullptr),
        xdim_(0),
        ydim_(0)
    {};

    // constructor
    Field(int xdim, int ydim)
    :   host_ptr_(nullptr),
        device_ptr_(nullptr),
        xdim_(xdim),
        ydim_(ydim)
    {
        init(xdim, ydim);
    };

    // destructor
    ~Field() {
        free();
    }

    void init(int xdim, int ydim) {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif

        free();
        allocate(xdim, ydim);
        fill(0.);
    }

    double*       host_data()         { return host_ptr_; }
    const double* host_data()   const { return host_ptr_; }

    double*       device_data()       { return device_ptr_; }
    const double* device_data() const { return device_ptr_; }

    // access via (i,j) pair
    inline double&       operator() (int i, int j)        {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return host_ptr_[i+j*xdim_];
    }
    inline double const& operator() (int i, int j) const  {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return host_ptr_[i+j*xdim_];
    }

    // access as a 1D field
    inline double      & operator[] (int i) {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return host_ptr_[i];
    }
    inline double const& operator[] (int i) const {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return host_ptr_[i];
    }

    int xdim()   const { return xdim_; }
    int ydim()   const { return ydim_; }
    int length() const { return xdim_*ydim_; }

    /////////////////////////////////////////////////
    // helpers for coordinating host-device transfers
    /////////////////////////////////////////////////
    // TODO : implement the body of update_host() and update_device()
    void update_host() {
        cudaMemcpy(host_ptr_, device_ptr_, xdim_*ydim_*sizeof(double), cudaMemcpyDeviceToHost);
    }

    void update_device() {
        cudaMemcpy(device_ptr_, host_ptr_, xdim_*ydim_*sizeof(double), cudaMemcpyHostToDevice);
    }

    private:

    void allocate(int xdim, int ydim) {
        xdim_ = xdim;
        ydim_ = ydim;
        host_ptr_ = new double[xdim*ydim];
        cuda_check_status( cudaMalloc(&device_ptr_, xdim*ydim*sizeof(double)) );
    }

    // set to a constant value
    void fill(double val) {
        // launch kernel to fill device buffer
        auto const n = xdim_*ydim_;
        auto const thread_dim = 192;
        auto const block_dim = n/thread_dim + (n%thread_dim ? 1:0);
        kernels::fill_device<<<block_dim, thread_dim>>>(device_ptr_, val, n);

        // parallel fill on host
        #pragma omp parallel for
        for(int i=0; i<xdim_*ydim_; ++i)
            host_ptr_[i] = val;
    }

    void free() {
        if(host_ptr_)   delete[] host_ptr_;
        if(device_ptr_) cudaFree(device_ptr_);
        host_ptr_   = nullptr;
        device_ptr_ = nullptr;
    }

    double* device_ptr_;
    double* host_ptr_;

    int xdim_;
    int ydim_;
};

// fields that hold the solution
extern Field x_new; // 2d
extern Field x_old; // 2d

// fields that hold the boundary values
extern Field bndN; // 1d
extern Field bndE; // 1d
extern Field bndS; // 1d
extern Field bndW; // 1d

extern Discretization options;

} // namespace data

