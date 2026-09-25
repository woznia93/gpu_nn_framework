#!/usr/bin/env bash
#
# bench/run_cuda_colab.sh
#
# Build and test the CUDA path on a machine with an NVIDIA GPU.
#
# Easiest free option is Google Colab: open colab.research.google.com,
# set Runtime > Change runtime type > T4 GPU, then in a cell run:
#
#     !git clone https://github.com/<you>/gpu_nn_framework
#     !cd gpu_nn_framework && bash bench/run_cuda_colab.sh
#
# (Colab cells run shell commands when prefixed with '!'.)
#
# Works equally well on any Linux box with the CUDA toolkit installed.
#
set -e

echo "=== environment ==========================================="
nvidia-smi || { echo "No NVIDIA driver found — is a GPU runtime selected?"; exit 1; }
nvcc --version | tail -2
cmake --version | head -1

echo
echo "=== configure ============================================="
# CMAKE_CUDA_ARCHITECTURES=native needs CMake 3.24+. If your CMake is older,
# pass the arch explicitly instead: -DCMAKE_CUDA_ARCHITECTURES=75
#   T4 = 75, V100 = 70, A100 = 80, L4/RTX40 = 89, H100 = 90
rm -rf build
cmake -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release

echo
echo "=== build ================================================="
cmake --build build -j"$(nproc)"

echo
echo "=== CPU suite (must still pass with CUDA compiled in) ====="
./build/nn_framework | tail -3

echo
echo "=== CUDA suite ============================================"
./build/nn_cuda_test

echo
echo "=== CPU benchmarks ========================================"
./build/nn_bench
