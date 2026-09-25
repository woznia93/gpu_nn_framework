//
// cuda/matmul.cu
//
// Tiled GEMM: C = A @ B      A: [M,K]  B: [K,N]  C: [M,N]
//
// Two kernels, dispatched by problem size:
//
//   naive_matmul_kernel — 16x16 tile, one output per thread. Kept for small
//                         problems, where the big kernel's 128x128 block tile
//                         would mostly compute padding.
//
//   tiled_matmul_kernel — 128x128 block tile, 8x8 outputs per thread. Fast path.
//
// Why the second is several times faster, for the same reason the CPU kernel
// got fast: ARITHMETIC INTENSITY.
//
//   One output per thread: each k-step costs 2 shared-memory loads per 1 FMA.
//   The SM stalls on load/store throughput while the FMA units idle.
//
//   8x8 outputs per thread: each k-step loads 8 values of A and 8 of B into
//   registers, then issues 64 FMAs against them — 16 loads per 64 FMAs. Every
//   loaded value is reused 8 times out of registers. That is exactly the
//   register blocking the CPU micro-kernel does with its 6x16 tile.
//
// Layout notes:
//   * As is stored TRANSPOSED (As[k][m]) so the inner loop reads a contiguous
//     run of m for a fixed k — the same motivation as packing on the CPU.
//   * Tile loads use a flat grid-stride loop with bounds checks, so any M/N/K
//     works: non-multiples of the tile, degenerate dims, everything. Out-of-
//     range elements are zero-filled, and zeros contribute nothing to a dot
//     product, so the compute loop needs no special cases.
//

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CU(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA matmul error: ") +     \
                                     cudaGetErrorString(_e));                 \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Small-problem kernel: 16x16 tile, one output per thread
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int TILE = 16;

__global__ void naive_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float*       __restrict__ C,
                                    int M, int K, int N)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    const int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.0f;

        const int b_row = t * TILE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fast kernel: 128x128 block tile, 8x8 register tile per thread
//
//   outputs per block  = BM * BN = 128 * 128 = 16384
//   outputs per thread = TM * TN =   8 *   8 =    64
//   threads per block  = 16384 / 64          =   256
//
//   shared memory = (BK*BM + BK*BN) * 4 B = (8*128 + 8*128) * 4 = 8 KB
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 8;
static constexpr int TM = 8;
static constexpr int TN = 8;
static constexpr int NTHREADS = (BM / TM) * (BN / TN);   // 256

__global__ __launch_bounds__(NTHREADS)
void tiled_matmul_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float*       __restrict__ C,
                         int M, int K, int N)
{
    __shared__ float As[BK][BM];      // transposed
    __shared__ float Bs[BK][BN];

    const int tid       = threadIdx.x;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // This thread's 8x8 sub-tile inside the block tile.
    const int thread_row = (tid / (BN / TN)) * TM;
    const int thread_col = (tid % (BN / TN)) * TN;

    float acc[TM][TN] = {};           // 64 accumulators, held in registers

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A's [BM x BK] block, transposed into As[BK][BM].
        for (int i = tid; i < BM * BK; i += NTHREADS) {
            const int r  = i / BK;
            const int c  = i % BK;
            const int gr = block_row + r;
            const int gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }

        // Load B's [BK x BN] block. Consecutive threads take consecutive
        // columns, so these reads coalesce.
        for (int i = tid; i < BK * BN; i += NTHREADS) {
            const int r  = i / BN;
            const int c  = i % BN;
            const int gr = k0 + r;
            const int gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // 16 register loads feed 64 FMAs per k.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[k][thread_row + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[k][thread_col + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
            }
        }

        __syncthreads();
    }

    // Write back, guarding the edges.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = block_row + thread_row + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = block_col + thread_col + j;
            if (gc < N) C[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher. C++ linkage (not extern "C") because it throws on error:
// throwing across a C boundary is undefined behaviour, and MSVC's /EHc
// assumes extern "C" functions never throw (warning C4297).
// ─────────────────────────────────────────────────────────────────────────────
void cuda_matmul(const float* A, const float* B, float* C, int M, int K, int N)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Choosing between the two kernels is a trade-off between arithmetic
    // intensity and occupancy.
    //
    // The 128x128 tile gives each thread far more work, but it also means a
    // 256x256 problem produces only 2x2 = 4 blocks — on a GPU with 30 SMs,
    // 26 of them sit idle. The 16x16 kernel generates 256 blocks for the same
    // problem and wins despite its worse per-thread efficiency.
    //
    // So require enough blocks to actually occupy the device before using the
    // big tile. (A proper implementation would query multiProcessorCount and
    // add a mid-size 64x64 kernel; this threshold is a reasonable stand-in.)
    const long long fast_blocks =
        (long long)((N + BN - 1) / BN) * ((M + BM - 1) / BM);
    const bool use_fast = (M >= 64 && N >= 64) &&
                          ((long long)M * N * K >= (1 << 18)) &&
                          fast_blocks >= 8;

    if (use_fast) {
        dim3 block(NTHREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        tiled_matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
    } else {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        naive_matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
    }

    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}
