#include <cstdio>
#include <iostream>
#include <fstream>


void diffusion_gpu(const double *x0, double *x1, int nx, int ny, double dt)
{
    int i, j;
    auto width = nx+2;

#ifdef OPENACC_DATA
    // TODO: Offload the following loops to the GPU
#else
    // TODO: Offload the following loops to the GPU; data is managed by CUDA here
#endif
    for (j = 1; j < ny+1; ++j) {
        for (i = 1; i < nx+1; ++i) {
            auto pos = i + j*width;
            x1[pos] = x0[pos] + dt * (-4.*x0[pos]
                                      + x0[pos-width] + x0[pos+width]
                                      + x0[pos-1]  + x0[pos+1]);
        }
    }
}

template<typename T>
void copy_gpu(T *dst, const T *src, int n)
{
    int i;

#ifdef OPENACC_DATA
    // TODO: Offload the following loops to the GPU
#else
    // TODO: Offload the following loops to the GPU; data is managed by CUDA here
#endif
    for (i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}


template <typename T>
void fill_gpu(T *v, T value, int n)
{
    int i;

#ifdef OPENACC_DATA
    // TODO: Offload the following loops to the GPU
#else
    // TODO: Offload the following loops to the GPU; data is managed by CUDA here
#endif
    for (i = 0; i < n; ++i)
        v[i] = value;
}

void write_to_file(int nx, int ny, double* data) {
    {
        FILE* output = fopen("output.bin", "w");
        fwrite(data, sizeof(double), nx * ny, output);
        fclose(output);
    }

    std::ofstream fid("output.bov");
    fid << "TIME: 0.0" << std::endl;
    fid << "DATA_FILE: output.bin" << std::endl;
    fid << "DATA_SIZE: " << nx << " " << ny << " 1" << std::endl;;
    fid << "DATA_FORMAT: DOUBLE" << std::endl;
    fid << "VARIABLE: phi" << std::endl;
    fid << "DATA_ENDIAN: LITTLE" << std::endl;
    fid << "CENTERING: nodal" << std::endl;
    fid << "BRICK_SIZE: 1.0 1.0 1.0" << std::endl;
}
