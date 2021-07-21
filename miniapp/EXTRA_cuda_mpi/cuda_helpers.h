#pragma once

#include <iostream>

#include <cstdlib>
#include <cublas_v2.h>

void cuda_check_last_kernel(std::string const& errstr);
void cuda_api_call(cudaError_t error_code);
cublasHandle_t& cublas_handle();
