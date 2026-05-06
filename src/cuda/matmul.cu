//
// cuda/matmul.cu
//
//
// Tiled GEMM kernel: C = A @ B
//		A : [M, K]
//		B : [K, N]
//		C : [M, N]
//
// Exposed as a plain-C function so tensor.cpp can call it without
// pulling in CUDA headers everywhere
//
//
// Build Note: compiled by nvcc via CMakeLists; linked into nn_framework
//
//

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Error checker Help (local)
#define CU(call)
	do {
		cudaError_t _e = (call);
		if (_e != cudaSuccess)
			throw std::runtime_error(std::string("CUDA matmul error: ") + cudaGetErrorString(_e));

	} while (0)


// Tile size 
// 16 x 16 tiles fit comfortably in shared memory and work on all sm_75+ GPUs
// Bump to 32 for sm_80+ if needed
static constexpr int TILE = 16;



//
// KERNEL
//

__global__ void tiled_matmul_kernel(
		const float* __restrict__ A,	// [M, K]
		const float* __restrict__ B,	// [K, N]
		float*		 __restrict__ C,	// [M, N]
		int M,
		int K,
		int N)
{
	// Shared memory tiles
	__shared__ float tileA[TILE][TILE];
	__shared__ float tileB[TILE][TILE];

	int row = blockIdx.y * TILE + threadIdx.y; // output row in C
	int col = blockIdx.x * TILE + threadIdx.x; // output col in C

	float acc = 0.0f;

	// Number of tiles along the K dimension
	int num_tiles = (K + TILE - 1) / TILE;

	for (int t = 0; t < num_tiles; ++t) {
		// Load tile of A: row x (t * TILE + tx)
		int a_col = t * TILE + threadIdx.x;
		tileA[threadIdx.y][threadIdx.x] =
			(row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

		// Load tile of B: (t*Tile + ty) x col
		int b_row = t * TILE + threadIdx.y;
		tileB[threadIdx.y][threadIdx.x] =
			(b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

		__syncthreads();

		// Dot product over this tile
		#pragma unroll
		for (int k = 0; k < TILE; ++k) 
			acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
		
		__syncthreads();
	}

	if (row < M && col < N)
		C[row * N + col] = acc;
}

//
// Host-side launcher (called from tensor.cpp)
//

extern "C" void cuda_matmul(
		const float* A, const float* B, float* C,
		int M, int K, int N)
{
	dim3 block(TILE,TILE);
	dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

	tiled_matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
	CU(cudaGetLastError());
	CU(cudaDeviceSynchronize());
}


