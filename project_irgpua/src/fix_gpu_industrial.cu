#include "fix_gpu_industrial.cuh"

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

void check_cuda_error_industrial(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " 
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}


namespace gpu_kernels_industrial {

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

} 

struct is_not_garbage {
    __host__ __device__
    bool operator()(const int& x) const {
        return x != -27;
    }
};

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
        else
            return pixel - 8;
    }
};

struct is_nonzero {
    __host__ __device__
    bool operator()(const int& x) const {
        return x != 0;
    }
};


void fix_image_gpu_industrial(Image& to_fix) {
    const int image_size = to_fix.width * to_fix.height;
    const int original_size = to_fix.size();
    constexpr int garbage_val = -27;
    
    
    thrust::device_vector<int> d_buffer(to_fix.buffer, to_fix.buffer + original_size);
    
    auto new_end = thrust::remove_if(d_buffer.begin(), d_buffer.end(), 
                                     [] __device__ (int x) { return x == garbage_val; });
    
    d_buffer.erase(new_end, d_buffer.end());
    
    if (d_buffer.size() != static_cast<size_t>(image_size)) {
        std::cerr << "Warning: Compacted size " << d_buffer.size() 
                  << " doesn't match expected " << image_size << std::endl;
    }
    
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_buffer.begin(), thrust::counting_iterator<int>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(d_buffer.end(), thrust::counting_iterator<int>(image_size))),
        d_buffer.begin(),
        fix_pixel_functor()
    );
    
    thrust::device_vector<int> d_histogram(256, 0);
    int* d_hist_ptr = thrust::raw_pointer_cast(d_histogram.data());
    int* d_buffer_ptr = thrust::raw_pointer_cast(d_buffer.data());
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    int num_levels = 257;     
    int lower_level = 0;       
    int upper_level = 256;   
    
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_buffer_ptr, d_hist_ptr,
        num_levels, lower_level, upper_level, 
        image_size
    );
    
    CUDA_CHECK_INDUSTRIAL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_buffer_ptr, d_hist_ptr,
        num_levels, lower_level, upper_level, 
        image_size
    );
    
    CUDA_CHECK_INDUSTRIAL(cudaDeviceSynchronize());
    CUDA_CHECK_INDUSTRIAL(cudaFree(d_temp_storage));
    
    thrust::inclusive_scan(d_histogram.begin(), d_histogram.end(), 
                          d_histogram.begin());
    
    auto it = thrust::find_if(d_histogram.begin(), d_histogram.end(), 
                              is_nonzero());
    
    int cdf_min = 0;
    if (it != d_histogram.end()) {
        cdf_min = *it;
    }
    
    const int block_size = 256;
    const int grid_size = (image_size + block_size - 1) / block_size;
    
    gpu_kernels_industrial::histogram_equalization<<<grid_size, block_size>>>(
        d_buffer_ptr, d_hist_ptr, image_size, cdf_min
    );
    
    CUDA_CHECK_INDUSTRIAL(cudaDeviceSynchronize());
    
    thrust::copy(d_buffer.begin(), d_buffer.end(), to_fix.buffer);
}


uint64_t compute_image_sum_gpu_industrial(const int* buffer, int size) {
    thrust::device_ptr<const int> d_ptr(buffer);
    
    uint64_t total = thrust::reduce(
        d_ptr, 
        d_ptr + size, 
        0ULL,  
        thrust::plus<uint64_t>()
    );
    
    return total;
}