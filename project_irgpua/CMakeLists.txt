cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(project_irgpua LANGUAGES CXX CUDA)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(project_irgpua)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(src)

# =============================================================================
# EXECUTABLE 1: CPU VERSION (Original Reference)
# =============================================================================
add_executable(main_cpu 
    src/main.cu 
    src/fix_cpu.cu
)

target_compile_options(main_cpu PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
)

target_link_libraries(main_cpu PUBLIC 
    OpenMP::OpenMP_CXX 
    rmm::rmm 
    raft::raft
)

# =============================================================================
# EXECUTABLE 2: GPU BY-HAND VERSION (Custom Kernels)
# =============================================================================
add_executable(main_gpu 
    src/main_gpu.cu 
    src/fix_gpu.cu
    src/fix_cpu.cu
)

target_compile_options(main_gpu PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=${OpenMP_CXX_FLAGS}
        --expt-relaxed-constexpr
        -use_fast_math
    >
)

target_link_libraries(main_gpu PUBLIC 
    OpenMP::OpenMP_CXX 
    rmm::rmm 
    raft::raft
)

# =============================================================================
# EXECUTABLE 3: GPU INDUSTRIAL VERSION (CUB/Thrust)
# =============================================================================
add_executable(main_gpu_industrial 
    src/main_gpu_industrial.cu 
    src/fix_gpu_industrial.cu
    src/fix_cpu.cu
)

target_compile_options(main_gpu_industrial PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=${OpenMP_CXX_FLAGS}
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -use_fast_math
    >
)

target_link_libraries(main_gpu_industrial PUBLIC 
    OpenMP::OpenMP_CXX 
    rmm::rmm 
    raft::raft
)

# =============================================================================
# BUILD CONFIGURATION
# =============================================================================

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Optimization flags for release
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(main_cpu PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    target_compile_options(main_gpu PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    target_compile_options(main_gpu_industrial PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

# Debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(main_cpu PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
    target_compile_options(main_gpu PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
    target_compile_options(main_gpu_industrial PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# =============================================================================
# INSTALLATION & INFO
# =============================================================================

message(STATUS "===========================================================")
message(STATUS "IRGPUA Project - GPU Image Processing Pipeline")
message(STATUS "===========================================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
message(STATUS "Three executables will be built:")
message(STATUS "  1. main_cpu            - CPU reference implementation")
message(STATUS "  2. main_gpu            - GPU by-hand (custom kernels)")
message(STATUS "  3. main_gpu_industrial - GPU industrial (CUB/Thrust)")
message(STATUS "")
message(STATUS "Usage after build:")
message(STATUS "  cd build")
message(STATUS "  ./main_cpu            # Run CPU version")
message(STATUS "  ./main_gpu            # Run GPU by-hand version")
message(STATUS "  ./main_gpu_industrial # Run GPU industrial version")
message(STATUS "")
message(STATUS "Compare performance:")
message(STATUS "  time ./main_cpu")
message(STATUS "  time ./main_gpu")
message(STATUS "  time ./main_gpu_industrial")
message(STATUS "===========================================================")