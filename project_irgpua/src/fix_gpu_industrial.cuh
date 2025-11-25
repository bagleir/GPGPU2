#pragma once

#include "image.hh"
#include <cuda_runtime.h>

// GPU image fixing function - Industrial version using CUB/Thrust
void fix_image_gpu_industrial(Image& to_fix);

// Statistics computation using Thrust
uint64_t compute_image_sum_gpu_industrial(const int* buffer, int size);

// Helper functions
void check_cuda_error_industrial(cudaError_t error, const char* file, int line);
#define CUDA_CHECK_INDUSTRIAL(error) check_cuda_error_industrial(error, __FILE__, __LINE__)

// Minimal custom kernels (only what can't be done with CUB/Thrust)
namespace gpu_kernels_industrial {

// Map transformation - simple enough to keep custom
__global__ void fix_pixels_map(int* buffer, int size);

// Histogram equalization - needs custom implementation
__global__ void histogram_equalization(int* buffer, const int* histogram, int size, int cdf_min);

} // namespace gpu_kernels_industrial