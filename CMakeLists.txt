cmake_minimum_required(VERSION 3.20)
project(gpu_nn_framework LANGUAGES CXX)

option(USE_CUDA "Build with CUDA support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Aggressive-but-safe optimization flags (no -ffast-math: softmax relies on
# -inf sentinels).
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
endif()

# -march=native must apply in EVERY build type, not just Release: without it
# the AVX2/FMA micro-kernel in src/gemm.cpp compiles out to the scalar
# fallback, costing ~4x. Checked rather than assumed, since not every
# toolchain/CPU accepts it.
include(CheckCXXCompilerFlag)
if(NOT MSVC)
    check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
    if(HAVE_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        check_cxx_compiler_flag("-mavx2 -mfma" HAVE_AVX2_FMA)
        if(HAVE_AVX2_FMA)
            add_compile_options(-mavx2 -mfma)
        else()
            message(WARNING "No AVX2/FMA flags accepted — GEMM will use the "
                            "scalar fallback micro-kernel (expect ~4x slower matmul).")
        endif()
    endif()
else()
    # MSVC: /arch:AVX2 enables the AVX2+FMA intrinsics used by the GEMM
    # micro-kernel. /utf-8 stops the box-drawing characters in comments from
    # being read as the local code page.
    add_compile_options(/arch:AVX2 /utf-8)
    # Release defaults already include /O2, but be explicit so a stray
    # build type can't silently produce an unoptimized benchmark.
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
endif()

if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "CMAKE_BUILD_TYPE is '${CMAKE_BUILD_TYPE}', not Release. "
        "Benchmarks will be misleading — a Debug build measures ~100x slower. "
        "Reconfigure a clean directory with -DCMAKE_BUILD_TYPE=Release.")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Core library ─────────────────────────────────────────────────────────────
set(NN_SOURCES
    src/gemm.cpp
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
