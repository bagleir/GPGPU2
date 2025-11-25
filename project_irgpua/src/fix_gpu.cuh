#pragma once

#include "image.hh"
#include <cuda_runtime.h>

// GPU image fixing function
void fix_image_gpu(Image& to_fix);

// Individual kernels for the pipeline
namespace gpu_kernels {

// Step 1: Stream Compaction - Remove garbage values (-27)
__global__ void compute_predicate(const int* input, int* predicate, int size, int garbage_val);
__global__ void scatter_compact(const int* input, int* output, const int* scan_result, int size, int garbage_val);

// Step 2: Map transformation - Fix pixel values
__global__ void fix_pixels_map(int* buffer, int size);

// Step 3: Histogram
__global__ void compute_histogram(const int* buffer, int* histogram, int size);
__global__ void histogram_scan(int* histogram, int size);
__global__ void histogram_equalization(int* buffer, const int* histogram, int size, int cdf_min);

// Statistics
__global__ void compute_partial_sums(const int* buffer, uint64_t* partial_sums, int size);

// Scan operations (for compaction)
__global__ void block_scan_kernel(const int* input, int* output, int* block_sums, int size);
__global__ void add_block_sums(int* data, const int* block_sums, int size);

// Utility kernels
__global__ void find_first_nonzero(const int* histogram, int* result, int size);

} // namespace gpu_kernels

// Helper functions
void check_cuda_error(cudaError_t error, const char* file, int line);
#define CUDA_CHECK(error) check_cuda_error(error, __FILE__, __LINE__)

// Memory allocation helpers
int* allocate_device_memory(size_t size);
void free_device_memory(int* ptr);
void copy_host_to_device(int* d_ptr, const int* h_ptr, size_t size);
void copy_device_to_host(int* h_ptr, const int* d_ptr, size_t size);