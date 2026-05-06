cmake_minimum_required(VERSION 3.20)
project(gpu_nn_framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── CUDA architecture (adjust for your GPU) ──────────────────────────────────
# sm_86 = RTX 30xx / A-series  |  sm_89 = RTX 40xx  |  sm_75 = Turing
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# SOURCES
set(CPP_SOURCES
	src/tensor/tensor.cpp
	src/autograd/autograd.cpp
	src/layers/linear.cpp
)

set(CUDA_SORUCES
	src/cuda/matmul.cu
	src/cuda/relu.cu
)


# MAIN EXECUTABLE
add_executable(nn_framework
	src/main.cpp
	${CPP_SOURCES}
	${CUDA_SOURCES}
)


target_include_directories(nn_framework PRIVATE 
	src/tensor
	src/autograd
	src/layers 
	src/cuda
)

target_compile_options(nn_framework PRIVATE
	    $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math -Wno-deprecated-gpu-targets>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(nn_framework PRIVATE CUDA::cudart)


