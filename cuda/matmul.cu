//
// cuda/matmul.cu
//
// Tiled GEMM kernel: C = A @ B
//   A : [M, K]   B : [K, N]   C : [M, N]
//
// Exposed with C++ linkage (not extern "C") because it throws on CUDA
// errors; tensor.cpp still needs no CUDA headers to call it. Compiled by nvcc via CMake when USE_CUDA=ON.
//
// Note: each launch synchronizes for simplicity/correctness. Stream-based
// async execution is on the roadmap.
//

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Fixed: the original macro was missing the line continuations, so the
// translation unit could not compile.
#define CU(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA matmul error: ") +     \
                                     cudaGetErrorString(_e));                 \
    } while (0)

// 16×16 tiles fit comfortably in shared memory and work on all sm_75+ GPUs.
static constexpr int TILE = 16;

__global__ void tiled_matmul_kernel(const float* __restrict__ A,   // [M, K]
                                    const float* __restrict__ B,   // [K, N]
                                    float*       __restrict__ C,   // [M, N]
                                    int M, int K, int N)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // output row in C
    int col = blockIdx.x * TILE + threadIdx.x;   // output col in C
    float acc = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = t * TILE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

void cuda_matmul(const float* A, const float* B, float* C,
                            int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}
