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

set(CMAKE_VERBOSE_MAKEFILE ON)

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