cmake_minimum_required(VERSION 3.20)
project(gpu_nn_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OPTIONAL CUDA
option(USE_CUDA "Build with CUDA support" OFF)

if(USE_CUDA)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
	find_package(CUDAToolkit REQUIRED)
	add_compile_definitions(USE_CUDA)
endif()

# ── CUDA architecture (adjust for your GPU) ──────────────────────────────────
# sm_86 = RTX 30xx / A-series  |  sm_89 = RTX 40xx  |  sm_75 = Turing

# SOURCES
set(CXX_SOURCES
	src/tensor.tensor.cpp
	src/autograd/autograd.cpp
	src/layers/linear.cpp
)

if(USE_CUDA)
	list(APPEND CXX_SOURCES
		src/cuda/matmul.cu
		src/cuda/relu.cu
	)
endif()




# MAIN EXECUTABLE
add_executable(nn_framework
	src/main.cpp
	${CXX_SOURCES}
)


target_include_directories(nn_framework PRIVATE 
	src/tensor
	src/autograd
	src/layers 
	src/cuda
)

target_compile_options(nn_framework PRIVATE
	    $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -Wno-unused-parameter>
		
    

if(USE_CUDA)
	target_compile_options(nn_framework PRIVATE
		        $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
    )
    target_link_libraries(nn_framework PRIVATE CUDA::cudart)
endif()
