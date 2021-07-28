#pragma once

#include <iostream>

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <openacc.h>

static inline void cuda_check_last_kernel(std::string const& errstr) {
    auto status = cudaGetLastError();
    if(status != cudaSuccess) {
        std::cout << "error: CUDA kernel launch :" << errstr << " : "
                  << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
}

static inline void cuda_check_status(cudaError_t error_code) {
    if(error_code != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(error_code) << std::endl;
        exit(-1);
    }
}

static inline cublasHandle_t& cublas_handle() {
    static cublasHandle_t cublas_handle;
    static bool is_intialized = false;
    if(!is_intialized) {
        auto status = cublasCreate(&cublas_handle);

        if(status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "error: unable to initialize cublas" << std::endl;
            exit(-1);
        }

        // Get OpenACC default stream
        cudaStream_t acc_stream= (cudaStream_t) acc_get_cuda_stream(0);

        // Set cuBLAS stream
        cublasSetStream(cublas_handle, acc_stream);
    }

    return cublas_handle;
}
