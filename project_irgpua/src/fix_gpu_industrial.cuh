#pragma once

#include "image.hh"
#include <cuda_runtime.h>

void fix_image_gpu_industrial(Image& to_fix);

uint64_t compute_image_sum_gpu_industrial(const int* buffer, int size);

void check_cuda_error_industrial(cudaError_t error, const char* file, int line);
#define CUDA_CHECK_INDUSTRIAL(error) check_cuda_error_industrial(error, __FILE__, __LINE__)

namespace gpu_kernels_industrial {

__global__ void fix_pixels_map(int* buffer, int size);

__global__ void histogram_equalization(int* buffer, const int* histogram, int size, int cdf_min);

}