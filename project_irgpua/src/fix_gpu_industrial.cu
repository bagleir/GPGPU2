#include "fix_gpu_industrial.cuh"

// CUB and Thrust includes
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <cmath>

// ============================================================================
// ERROR CHECKING
// ============================================================================

void check_cuda_error_industrial(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " 
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// MINIMAL CUSTOM KERNELS (only what CUB/Thrust can't do easily)
// ============================================================================

namespace gpu_kernels_industrial {

// Map transformation kernel (simple, keep custom)
__global__ void fix_pixels_map(int* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int mod = idx % 4;
        if (mod == 0)
            buffer[idx] += 1;
        else if (mod == 1)
            buffer[idx] -= 5;
        else if (mod == 2)
            buffer[idx] += 3;
        else if (mod == 3)
            buffer[idx] -= 8;
    }
}

// Histogram equalization kernel
__global__ void histogram_equalization(int* buffer, const int* histogram, 
                                       int size, int cdf_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int pixel = buffer[idx];
        float normalized = ((histogram[pixel] - cdf_min) / 
                           static_cast<float>(size - cdf_min)) * 255.0f;
        buffer[idx] = static_cast<int>(roundf(normalized));
    }
}

} // namespace gpu_kernels_industrial

// ============================================================================
// THRUST FUNCTORS
// ============================================================================

// Predicate to identify garbage values
struct is_not_garbage {
    __host__ __device__
    bool operator()(const int& x) const {
        return x != -27;
    }
};

// Functor for pixel transformation with index
struct fix_pixel_functor {
    __host__ __device__
    int operator()(const thrust::tuple<int, int>& t) const {
        int pixel = thrust::get<0>(t);
        int idx = thrust::get<1>(t);
        int mod = idx % 4;
        
        if (mod == 0)
            return pixel + 1;
        else if (mod == 1)
            return pixel - 5;
        else if (mod == 2)
            return pixel + 3;
        else // mod == 3
            return pixel - 8;
    }
};

// Predicate to find first non-zero
struct is_nonzero {
    __host__ __device__
    bool operator()(const int& x) const {
        return x != 0;
    }
};

// ============================================================================
// MAIN INDUSTRIAL PIPELINE FUNCTION
// ============================================================================

void fix_image_gpu_industrial(Image& to_fix) {
    const int image_size = to_fix.width * to_fix.height;
    const int original_size = to_fix.size();
    constexpr int garbage_val = -27;
    
    // ========================================================================
    // STEP 1: STREAM COMPACTION using Thrust
    // ========================================================================
    // Using Thrust device_vector for automatic memory management
    
    thrust::device_vector<int> d_buffer(to_fix.buffer, to_fix.buffer + original_size);
    
    // Remove all -27 values using Thrust's remove_if
    // This replaces: predicate computation + scan + scatter
    auto new_end = thrust::remove_if(d_buffer.begin(), d_buffer.end(), 
                                     [] __device__ (int x) { return x == garbage_val; });
    
    // Resize to actual size after compaction
    d_buffer.erase(new_end, d_buffer.end());
    
    // Verify we have the expected size
    if (d_buffer.size() != static_cast<size_t>(image_size)) {
        std::cerr << "Warning: Compacted size " << d_buffer.size() 
                  << " doesn't match expected " << image_size << std::endl;
    }
    
    // ========================================================================
    // STEP 2: MAP TRANSFORMATION using Thrust transform
    // ========================================================================
    
    // Option 1: Using Thrust transform with counting iterator (more elegant)
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_buffer.begin(), thrust::counting_iterator<int>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(d_buffer.end(), thrust::counting_iterator<int>(image_size))),
        d_buffer.begin(),
        fix_pixel_functor()
    );
    
    // Option 2: Using custom kernel (simpler, commented out)
    /*
    int* d_buffer_ptr = thrust::raw_pointer_cast(d_buffer.data());
    const int block_size = 256;
    const int grid_size = (image_size + block_size - 1) / block_size;
    gpu_kernels_industrial::fix_pixels_map<<<grid_size, block_size>>>(d_buffer_ptr, image_size);
    CUDA_CHECK_INDUSTRIAL(cudaDeviceSynchronize());
    */
    
    // ========================================================================
    // STEP 3: HISTOGRAM using CUB
    // ========================================================================
    
    thrust::device_vector<int> d_histogram(256, 0);
    int* d_hist_ptr = thrust::raw_pointer_cast(d_histogram.data());
    int* d_buffer_ptr = thrust::raw_pointer_cast(d_buffer.data());
    
    // CUB histogram - much more efficient than custom atomics
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    int num_levels = 257;      // 256 bins + 1
    int lower_level = 0;       // Minimum value (inclusive)
    int upper_level = 256;     // Maximum value (exclusive)
    
    // First call to get required temp storage size
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_buffer_ptr, d_hist_ptr,
        num_levels, lower_level, upper_level, 
        image_size
    );
    
    // Allocate temp storage
    CUDA_CHECK_INDUSTRIAL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Second call to actually compute histogram
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_buffer_ptr, d_hist_ptr,
        num_levels, lower_level, upper_level, 
        image_size
    );
    
    CUDA_CHECK_INDUSTRIAL(cudaDeviceSynchronize());
    CUDA_CHECK_INDUSTRIAL(cudaFree(d_temp_storage));
    
    // ========================================================================
    // STEP 4: INCLUSIVE SCAN using Thrust (for CDF)
    // ========================================================================
    
    // Thrust inclusive scan - replaces custom scan kernel
    thrust::inclusive_scan(d_histogram.begin(), d_histogram.end(), 
                          d_histogram.begin());
    
    // ========================================================================
    // STEP 5: FIND FIRST NON-ZERO (cdf_min) using Thrust
    // ========================================================================
    
    // Find first non-zero value using Thrust
    auto it = thrust::find_if(d_histogram.begin(), d_histogram.end(), 
                              is_nonzero());
    
    int cdf_min = 0;
    if (it != d_histogram.end()) {
        cdf_min = *it;
    }
    
    // ========================================================================
    // STEP 6: HISTOGRAM EQUALIZATION
    // ========================================================================
    
    // This needs custom kernel as it requires complex per-pixel computation
    const int block_size = 256;
    const int grid_size = (image_size + block_size - 1) / block_size;
    
    gpu_kernels_industrial::histogram_equalization<<<grid_size, block_size>>>(
        d_buffer_ptr, d_hist_ptr, image_size, cdf_min
    );
    
    CUDA_CHECK_INDUSTRIAL(cudaDeviceSynchronize());
    
    // ========================================================================
    // COPY RESULTS BACK TO HOST
    // ========================================================================
    
    thrust::copy(d_buffer.begin(), d_buffer.end(), to_fix.buffer);
}

// ============================================================================
// STATISTICS: Compute sum using Thrust reduce
// ============================================================================

uint64_t compute_image_sum_gpu_industrial(const int* buffer, int size) {
    // Wrap raw pointer in Thrust device pointer
    thrust::device_ptr<const int> d_ptr(buffer);
    
    // Single line to compute sum using Thrust reduce!
    // This replaces all the custom reduction kernel code
    uint64_t total = thrust::reduce(
        d_ptr, 
        d_ptr + size, 
        0ULL,  // Initial value
        thrust::plus<uint64_t>()
    );
    
    return total;
}

// ============================================================================
// ALTERNATIVE: Sort using Thrust (OPTIONAL for project)
// ============================================================================

// If you want to sort on GPU instead of CPU:
/*
void sort_images_gpu_industrial(std::vector<Image::ToSort>& to_sort) {
    // Copy to device
    thrust::device_vector<Image::ToSort> d_to_sort(to_sort.begin(), to_sort.end());
    
    // Sort using Thrust (single line!)
    thrust::sort(d_to_sort.begin(), d_to_sort.end(),
        [] __device__ (const Image::ToSort& a, const Image::ToSort& b) {
            return a.total < b.total;
        }
    );
    
    // Copy back
    thrust::copy(d_to_sort.begin(), d_to_sort.end(), to_sort.begin());
}
*/