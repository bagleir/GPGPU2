#include "fix_gpu.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
}

int* allocate_device_memory(size_t size) {
    int* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(int)));
    return d_ptr;
}

void free_device_memory(int* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void copy_host_to_device(int* d_ptr, const int* h_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(int), cudaMemcpyHostToDevice));
}

void copy_device_to_host(int* h_ptr, const int* d_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size * sizeof(int), cudaMemcpyDeviceToHost));
}

namespace gpu_kernels {

__global__ void compute_predicate(const int* input, int* predicate, int size, int garbage_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        predicate[idx] = (input[idx] != garbage_val) ? 1 : 0;
    }
}

__global__ void block_scan_kernel(const int* input, int* output, int* block_sums, int size) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    temp[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();
    
    int offset = 1;
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = temp[blockDim.x - 1];
        }
        temp[blockDim.x - 1] = 0;
    }
    
    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    if (idx < size) {
        output[idx] = temp[tid];
    }
}

__global__ void add_block_sums(int* data, const int* block_sums, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && blockIdx.x > 0) {
        data[idx] += block_sums[blockIdx.x];
    }
}

__global__ void scatter_compact(const int* input, int* output, const int* scan_result, 
                                int size, int garbage_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && input[idx] != garbage_val) {
        output[scan_result[idx]] = input[idx];
    }
}

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

__global__ void compute_histogram(const int* buffer, int* histogram, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int local_hist[256];
    
    if (threadIdx.x < 256) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    if (idx < size) {
        atomicAdd(&local_hist[buffer[idx]], 1);
    }
    __syncthreads();
    
    if (threadIdx.x < 256) {
        atomicAdd(&histogram[threadIdx.x], local_hist[threadIdx.x]);
    }
}

__global__ void histogram_scan(int* histogram, int size) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    
    if (tid < size) {
        temp[tid] = histogram[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    for (int stride = 1; stride < size; stride *= 2) {
        int val = 0;
        if (tid >= stride && tid < size) {
            val = temp[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride && tid < size) {
            temp[tid] += val;
        }
        __syncthreads();
    }
    
    if (tid < size) {
        histogram[tid] = temp[tid];
    }
}

__global__ void find_first_nonzero(const int* histogram, int* result, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < size; i++) {
            if (histogram[i] != 0) {
                *result = histogram[i];
                return;
            }
        }
        *result = 0;
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


__global__ void compute_partial_sums(const int* buffer, uint64_t* partial_sums, int size) {
    extern __shared__ uint64_t sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < size) ? static_cast<uint64_t>(buffer[idx]) : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

}

void fix_image_gpu(Image& to_fix) {
    const int image_size = to_fix.width * to_fix.height;
    const int original_size = to_fix.size();
    constexpr int garbage_val = -27;
    
    const int block_size = 256;
    const int grid_size = (original_size + block_size - 1) / block_size;
    const int grid_size_image = (image_size + block_size - 1) / block_size;
    
    int* d_buffer;
    int* d_predicate;
    int* d_scan_result;
    int* d_compacted;
    int* d_histogram;
    int* d_cdf_min;
    
    CUDA_CHECK(cudaMalloc(&d_buffer, original_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_predicate, original_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan_result, original_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_compacted, image_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf_min, sizeof(int)));
    
    copy_host_to_device(d_buffer, to_fix.buffer, original_size);
    CUDA_CHECK(cudaMemset(d_histogram, 0, 256 * sizeof(int)));
   
    gpu_kernels::compute_predicate<<<grid_size, block_size>>>(
        d_buffer, d_predicate, original_size, garbage_val
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int num_blocks = grid_size;
    int* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    
    size_t shared_mem_size = block_size * sizeof(int);
    gpu_kernels::block_scan_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        d_predicate, d_scan_result, d_block_sums, original_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if (num_blocks > 1) {
        int* h_block_sums = new int[num_blocks];
        CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), 
                             cudaMemcpyDeviceToHost));
        
        int sum = 0;
        for (int i = 0; i < num_blocks; i++) {
            int temp = h_block_sums[i];
            h_block_sums[i] = sum;
            sum += temp;
        }
        
        CUDA_CHECK(cudaMemcpy(d_block_sums, h_block_sums, num_blocks * sizeof(int), 
                             cudaMemcpyHostToDevice));
        delete[] h_block_sums;
        
        gpu_kernels::add_block_sums<<<num_blocks, block_size>>>(
            d_scan_result, d_block_sums, original_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    gpu_kernels::scatter_compact<<<grid_size, block_size>>>(
        d_buffer, d_compacted, d_scan_result, original_size, garbage_val
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_block_sums));
    
    
    gpu_kernels::fix_pixels_map<<<grid_size_image, block_size>>>(
        d_compacted, image_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    gpu_kernels::compute_histogram<<<grid_size_image, block_size>>>(
        d_compacted, d_histogram, image_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    shared_mem_size = 256 * sizeof(int);
    gpu_kernels::histogram_scan<<<1, 256, shared_mem_size>>>(
        d_histogram, 256
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    gpu_kernels::find_first_nonzero<<<1, 1>>>(
        d_histogram, d_cdf_min, 256
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_cdf_min;
    CUDA_CHECK(cudaMemcpy(&h_cdf_min, d_cdf_min, sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_kernels::histogram_equalization<<<grid_size_image, block_size>>>(
        d_compacted, d_histogram, image_size, h_cdf_min
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    
    copy_device_to_host(to_fix.buffer, d_compacted, image_size);
    
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_predicate));
    CUDA_CHECK(cudaFree(d_scan_result));
    CUDA_CHECK(cudaFree(d_compacted));
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_cdf_min));
}

uint64_t compute_image_sum_gpu(const int* buffer, int size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    int* d_buffer;
    uint64_t* d_partial_sums;
    
    CUDA_CHECK(cudaMalloc(&d_buffer, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_sums, grid_size * sizeof(uint64_t)));
    
    copy_host_to_device(d_buffer, buffer, size);
    
    size_t shared_mem_size = block_size * sizeof(uint64_t);
    gpu_kernels::compute_partial_sums<<<grid_size, block_size, shared_mem_size>>>(
        d_buffer, d_partial_sums, size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint64_t* h_partial_sums = new uint64_t[grid_size];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, grid_size * sizeof(uint64_t), 
                         cudaMemcpyDeviceToHost));
    
    uint64_t total = 0;
    for (int i = 0; i < grid_size; i++) {
        total += h_partial_sums[i];
    }
    
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_partial_sums));
    
    return total;
}