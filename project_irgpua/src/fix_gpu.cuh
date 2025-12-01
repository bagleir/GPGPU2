#pragma once

#include "image.hh"
#include <cuda_runtime.h>

void fix_image_gpu(Image& to_fix);

namespace gpu_kernels {
    __global__ void compute_predicate(const int* input, int* predicate, int size, int garbage_val);
    __global__ void scatter_compact(const int* input, int* output, const int* scan_result, int size, int garbage_val);
    __global__ void fix_pixels_map(int* buffer, int size);
    __global__ void compute_histogram(const int* buffer, int* histogram, int size);
    __global__ void histogram_scan(int* histogram, int size);
    __global__ void histogram_equalization(int* buffer, const int* histogram, int size, int cdf_min);
    __global__ void compute_partial_sums(const int* buffer, uint64_t* partial_sums, int size);
    __global__ void block_scan_kernel(const int* input, int* output, int* block_sums, int size);
    __global__ void add_block_sums(int* data, const int* block_sums, int size);
    __global__ void find_first_nonzero(const int* histogram, int* result, int size);
}

void check_cuda_error(cudaError_t error, const char* file, int line);
#define CUDA_CHECK(error) check_cuda_error(error, __FILE__, __LINE__)
int* allocate_device_memory(size_t size);
void free_device_memory(int* ptr);
void copy_host_to_device(int* d_ptr, const int* h_ptr, size_t size);
void copy_device_to_host(int* h_ptr, const int* d_ptr, size_t size);
