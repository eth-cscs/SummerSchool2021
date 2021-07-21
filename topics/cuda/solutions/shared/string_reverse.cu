#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "util.hpp"

// TODO : implement a kernel that reverses a string of length n in place
__global__
 void reverse_string(char* str, int n) {
    __shared__ char buffer[1024];

    int i = threadIdx.x;
    if (i<n) {
        buffer[i] = str[i];
        __syncthreads();
        str[i] = buffer[n-i-1];
    }
 }

__global__
 void reverse_string2(char* str, int n) {
    int i = threadIdx.x;
    if (i<n/2) {
        const char tmp = str[n-i-1];
        str[n-i-1] = str[i];
        str[i] = tmp;
    }
 }

int main(int argc, char** argv) {
    // check that the user has passed a string to reverse
    if(argc<2) {
        std::cout << "useage : ./string_reverse \"string to reverse\"\n" << std::endl;
        exit(0);
    }

    // determine the length of the string, and copy in to buffer
    auto n = strlen(argv[1]);
    auto string = malloc_managed<char>(n+1);
    std::copy(argv[1], argv[1]+n, string);
    string[n] = 0; // add null terminator

    std::cout << "string to reverse:\n" << string << "\n";

    // call the string reverse kernel
    reverse_string2<<<1,n>>>(string, n);

    // print reversed string
    cudaDeviceSynchronize();
    std::cout << "reversed string:\n" << string << "\n";

    // free memory
    cudaFree(string);

    return 0;
}

