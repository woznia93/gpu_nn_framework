cmake_minimum_required(VERSION 3.20)
project(gpu_nn_framework LANGUAGES CXX)

option(USE_CUDA "Build with CUDA support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive-but-safe optimization flags (no -ffast-math: softmax relies on
# -inf sentinels).
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
endif()

# ── Core library ─────────────────────────────────────────────────────────────
set(NN_SOURCES
    src/tensor.cpp
    src/autograd.cpp
    src/linear.cpp
    src/optim.cpp
)

if(USE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
    endif()
    list(APPEND NN_SOURCES
        cuda/matmul.cu
        cuda/elementwise.cu
    )
endif()

add_library(nn_core STATIC ${NN_SOURCES})
target_include_directories(nn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(USE_CUDA)
    target_compile_definitions(nn_core PUBLIC USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(nn_core PUBLIC CUDA::cudart)
endif()

# OpenMP is optional; the pragmas are no-ops without it.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nn_core PUBLIC OpenMP::OpenMP_CXX)
    # MinGW/GCC: FindOpenMP applies -fopenmp when compiling but not always when
    # linking, leaving GOMP_parallel etc. unresolved. Passing the flag at link
    # time makes the GCC driver add libgomp itself.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(nn_core PUBLIC ${OpenMP_CXX_FLAGS})
    endif()
    message(STATUS "OpenMP found — CPU matmul/linear will be multithreaded")
else()
    message(STATUS "OpenMP not found — building single-threaded")
endif()

# ── Executables ──────────────────────────────────────────────────────────────
add_executable(nn_framework tests/main.cpp)
target_link_libraries(nn_framework PRIVATE nn_core)

add_executable(nn_bench bench/bench.cpp)
target_link_libraries(nn_bench PRIVATE nn_core)
