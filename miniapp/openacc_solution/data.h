#pragma once

#include <iostream>
#include <cassert>
#include <openacc.h>

namespace data
{

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
    :   xdim_(0),
        ydim_(0),
        ptr_(nullptr)
    {};

    // constructor
    Field(int xdim, int ydim)
    :   xdim_(xdim),
        ydim_(ydim),
        ptr_(nullptr)
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

    double*       host_data()         { return ptr_; }
    const double* host_data()   const { return ptr_; }

    double*       device_data()       { return (double *) acc_deviceptr(ptr_); }
    const double* device_data() const { return (double *) acc_deviceptr(ptr_); }

    // access via (i,j) pair
    #pragma acc routine seq
    inline double&       operator() (int i, int j)        {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

    #pragma acc routine seq
    inline double const& operator() (int i, int j) const  {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

    // access as a 1D field
    #pragma acc routine seq
    inline double      & operator[] (int i) {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

    #pragma acc routine seq
    inline double const& operator[] (int i) const {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

    int xdim()   const { return xdim_; }
    int ydim()   const { return ydim_; }
    int length() const { return xdim_*ydim_; }

    /////////////////////////////////////////////////
    // helpers for coordinating host-device transfers
    /////////////////////////////////////////////////
    void update_host() {
        #pragma acc update host(ptr_[0:xdim_*ydim_])
    }

    void update_device() {
        #pragma acc update device(ptr_[0:xdim_*ydim_]) async(0)
    }

    private:

    void allocate(int xdim, int ydim) {
        xdim_ = xdim;
        ydim_ = ydim;
        ptr_ = new double[xdim*ydim];
        #pragma acc enter data copyin(this) async(0)
        #pragma acc enter data create(ptr_[0:xdim*ydim]) async(0)
    }

    // set to a constant value
    void fill(double val) {
        // initialize the host and device copy at the same time
        #pragma acc parallel loop async(0)
        for(int i=0; i<xdim_*ydim_; ++i)
            ptr_[i] = val;

        #pragma omp parallel for
        for(int i=0; i<xdim_*ydim_; ++i)
            ptr_[i] = val;
    }

    void free() {
        if (ptr_) {
            #pragma acc exit data delete(ptr_[0:xdim_*ydim_], this)
            delete[] ptr_;
        }

        ptr_ = nullptr;
    }

    double* ptr_;
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
