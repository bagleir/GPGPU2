#include "image.hh"
#include "pipeline.hh"
#include "fix_gpu.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <cuda_runtime.h>

uint64_t compute_image_sum_gpu(const int* buffer, int size);

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{    
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable device found!" << std::endl;
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    
    std::string images_path = "/home/gablacav/Desktop/ING3/GPU/project_irgpua/images";
    if (!std::filesystem::exists(images_path)) {
        std::cout << "AFS path not found, trying local 'images' directory..." << std::endl;
        images_path = "./images";
        if (!std::filesystem::exists(images_path)) {
            std::cerr << "Error: Images directory not found!" << std::endl;
            std::cerr << "Please provide images in './images' directory" << std::endl;
            return EXIT_FAILURE;
        }
    }
    
    for (const auto& dir_entry : recursive_directory_iterator(images_path))
        filepaths.emplace_back(dir_entry.path());
    
    std::cout << "Found " << filepaths.size() << " images to process" << std::endl;
    
    auto start_load = std::chrono::high_resolution_clock::now();
    Pipeline pipeline(filepaths);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count();
    std::cout << "Pipeline loaded in " << load_time << " ms" << std::endl;
    
    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);
    
    std::cout << "\n[2/4] Processing images on GPU..." << std::endl;
    auto start_compute = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        
        fix_image_gpu(images[i]);
        
        if ((i + 1) % 10 == 0 || i == nb_images - 1) {
            #pragma omp critical
            std::cout << "  Processed " << (i + 1) << "/" << nb_images << " images" << std::endl;
        }
    }
    
    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "GPU processing completed in " << compute_time << " ms" << std::endl;
    std::cout << "Average time per image: " << (compute_time / (float)nb_images) << " ms" << std::endl;
    
    std::cout << "\n[3/4] Computing image statistics on GPU..." << std::endl;
    auto start_stats = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        
        image.to_sort.total = compute_image_sum_gpu(image.buffer, image_size);
    }
    
    auto end_stats = std::chrono::high_resolution_clock::now();
    auto stats_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_stats - start_stats).count();
    std::cout << "Statistics computed in " << stats_time << " ms" << std::endl;
    
    std::cout << "\n[4/4] Sorting images by pixel sum..." << std::endl;
    auto start_sort = std::chrono::high_resolution_clock::now();
    
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, &images] () mutable
    {
        return images[n++].to_sort;
    });
    
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });
    
    auto end_sort = std::chrono::high_resolution_clock::now();
    auto sort_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_sort - start_sort).count();
    std::cout << "Sorting completed in " << sort_time << " ms" << std::endl;
    
    std::cout << "\nWriting output images..." << std::endl;
    for (int i = 0; i < nb_images; ++i)
    {
        std::ostringstream oss;
        oss << "GPU_Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }
    
    std::cout << "RESULTS SUMMARY:" << std::endl;
    std::cout << "First 5 images (sorted by total):" << std::endl;
    for (int i = 0; i < std::min(5, nb_images); ++i) {
        std::cout << "  Image #" << to_sort[i].id 
                  << " - Total: " << to_sort[i].total << std::endl;
    }
    
    if (nb_images > 10) {
        std::cout << "..." << std::endl;
        std::cout << "Last 5 images:" << std::endl;
        for (int i = std::max(0, nb_images - 5); i < nb_images; ++i) {
            std::cout << "  Image #" << to_sort[i].id 
                      << " - Total: " << to_sort[i].total << std::endl;
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
    
    std::cout << "PERFORMANCE METRICS:" << std::endl;
    std::cout << "Loading time:     " << load_time << " ms" << std::endl;
    std::cout << "GPU compute time: " << compute_time << " ms" << std::endl;
    std::cout << "Statistics time:  " << stats_time << " ms" << std::endl;
    std::cout << "Sorting time:     " << sort_time << " ms" << std::endl;
    std::cout << "Total time:       " << total_time << " ms" << std::endl;
        
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);
    
    return EXIT_SUCCESS;
}