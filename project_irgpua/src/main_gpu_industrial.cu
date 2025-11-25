#include "image.hh"
#include "pipeline.hh"
#include "fix_gpu_industrial.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <cuda_runtime.h>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    std::cout << "===========================================================" << std::endl;
    std::cout << "    GPU INDUSTRIAL VERSION - IRGPUA PROJECT" << std::endl;
    std::cout << "    Using CUB & Thrust Libraries" << std::endl;
    std::cout << "===========================================================" << std::endl;
    
    // Check for CUDA device
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable device found!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "===========================================================" << std::endl;
    
    // -- Pipeline initialization
    std::cout << "\n[1/4] Loading images from pipeline..." << std::endl;
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // - Get file paths
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    
    // Try AFS path first, fallback to local if not available
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
    
    // - Init pipeline object
    auto start_load = std::chrono::high_resolution_clock::now();
    Pipeline pipeline(filepaths);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count();
    std::cout << "Pipeline loaded in " << load_time << " ms" << std::endl;
    
    // -- Main loop: Process images with GPU using CUB/Thrust
    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);
    
    std::cout << "\n[2/4] Processing images with CUB/Thrust..." << std::endl;
    std::cout << "Libraries used:" << std::endl;
    std::cout << "  • CUB (DeviceHistogram)" << std::endl;
    std::cout << "  • Thrust (remove_if, transform, scan, reduce)" << std::endl;
    std::cout << "  • Minimal custom kernels (only 2!)" << std::endl;
    
    auto start_compute = std::chrono::high_resolution_clock::now();
    
    // Process images one by one (pipeline style)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        
        // Apply GPU industrial pipeline (CUB/Thrust)
        fix_image_gpu_industrial(images[i]);
        
        if ((i + 1) % 10 == 0 || i == nb_images - 1) {
            #pragma omp critical
            std::cout << "  Processed " << (i + 1) << "/" << nb_images << " images" << std::endl;
        }
    }
    
    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "Industrial GPU processing completed in " << compute_time << " ms" << std::endl;
    std::cout << "Average time per image: " << (compute_time / (float)nb_images) << " ms" << std::endl;
    
    // -- Compute statistics using Thrust reduce
    std::cout << "\n[3/4] Computing statistics with Thrust..." << std::endl;
    auto start_stats = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        
        // Use Thrust reduction for sum computation (single line!)
        image.to_sort.total = compute_image_sum_gpu_industrial(image.buffer, image_size);
    }
    
    auto end_stats = std::chrono::high_resolution_clock::now();
    auto stats_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_stats - start_stats).count();
    std::cout << "Statistics computed in " << stats_time << " ms" << std::endl;
    
    // -- Sort images by total
    std::cout << "\n[4/4] Sorting images by pixel sum..." << std::endl;
    auto start_sort = std::chrono::high_resolution_clock::now();
    
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, &images] () mutable
    {
        return images[n++].to_sort;
    });
    
    // CPU sort (could use Thrust sort too, but CPU is fine for small arrays)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });
    
    auto end_sort = std::chrono::high_resolution_clock::now();
    auto sort_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_sort - start_sort).count();
    std::cout << "Sorting completed in " << sort_time << " ms" << std::endl;
    
    // -- Write results
    std::cout << "\nWriting output images..." << std::endl;
    for (int i = 0; i < nb_images; ++i)
    {
        std::ostringstream oss;
        oss << "INDUSTRIAL_Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }
    
    // Print statistics
    std::cout << "\n===========================================================" << std::endl;
    std::cout << "RESULTS SUMMARY:" << std::endl;
    std::cout << "===========================================================" << std::endl;
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
    
    std::cout << "\n===========================================================" << std::endl;
    std::cout << "PERFORMANCE METRICS:" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "Loading time:     " << load_time << " ms" << std::endl;
    std::cout << "GPU compute time: " << compute_time << " ms" << std::endl;
    std::cout << "Statistics time:  " << stats_time << " ms" << std::endl;
    std::cout << "Sorting time:     " << sort_time << " ms" << std::endl;
    std::cout << "Total time:       " << total_time << " ms" << std::endl;
    std::cout << "===========================================================" << std::endl;
    
    std::cout << "\n===========================================================" << std::endl;
    std::cout << "INDUSTRIAL VERSION BENEFITS:" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "✓ Stream compaction:     Thrust::remove_if (1 line)" << std::endl;
    std::cout << "✓ Pixel transformation:  Thrust::transform (elegant)" << std::endl;
    std::cout << "✓ Histogram:            CUB::DeviceHistogram (optimal)" << std::endl;
    std::cout << "✓ Scan (CDF):           Thrust::inclusive_scan (1 line)" << std::endl;
    std::cout << "✓ Statistics:           Thrust::reduce (1 line)" << std::endl;
    std::cout << "✓ Custom kernels:       Only 2 (map & equalization)" << std::endl;
    std::cout << "✓ Code size:            ~200 lines vs ~450 by-hand" << std::endl;
    std::cout << "✓ Performance:          Highly optimized by NVIDIA" << std::endl;
    std::cout << "===========================================================" << std::endl;
    
    std::cout << "\n✓ Done! The internet is safe now :)" << std::endl;
    std::cout << "Output files: INDUSTRIAL_Image#*.pgm" << std::endl;
    std::cout << "\nCompare with by-hand version:" << std::endl;
    std::cout << "  diff GPU_Image#1.pgm INDUSTRIAL_Image#1.pgm" << std::endl;
    std::cout << "  (Files should be identical or nearly identical)" << std::endl;
    
    // Cleaning
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);
    
    return EXIT_SUCCESS;
}