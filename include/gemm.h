#pragma once
//
// gemm.h
//
// A blocked, packed, SIMD single-precision GEMM — the one hot kernel that
// every dense compute path in the framework routes through.
//
//   C += A @ B          A: [M,K]   B: [K,N]   C: [M,N]
//
// A and B are described by *element strides* rather than assumed to be
// row-major-contiguous:
//
//   A[i][k] == A[i*a_rs + k*a_cs]
//   B[k][j] == B[k*b_rs + j*b_cs]
//   C[i][j] == C[i*ldc + j]
//
// That one generalization lets a single kernel serve every product in the
// framework without materializing a transpose:
//
//   matmul       C = X @ W       a_rs=K, a_cs=1   b_rs=N,  b_cs=1
//   Linear fwd   Y = X @ Wᵀ      a_rs=K, a_cs=1   b_rs=1,  b_cs=K   (W is [N,K])
//   Linear dX    dX = G @ W      a_rs=K, a_cs=1   b_rs=N,  b_cs=1
//   Linear dW    dW = Gᵀ @ X     a_rs=1, a_cs=M   b_rs=N,  b_cs=1
//
// C is ACCUMULATED into, never overwritten. Callers pre-fill C — with zeros
// for a plain product, or with the bias row for a Linear forward, which makes
// the bias add free.
//
// Structure follows the standard Goto/BLIS decomposition:
//
//   jc loop  (N / NC)      choose a column panel of B
//     pc loop  (K / KC)      choose a K slice -> pack B panel into L2
//       ic loop  (M / MC)      pack an A block into L2/L1
//         macro-kernel         MR x NR micro-tiles, register-blocked
//
// Packing copies each block into a contiguous, micro-kernel-ordered buffer so
// the inner loop walks memory linearly — this is what turns a bandwidth-bound
// triple loop into a compute-bound one.
//

#include <cstddef>

namespace gemm {

// Micro-tile: 6 rows x 16 columns. With AVX2 that is 12 accumulator registers
// (6 rows x 2 vectors of 8 floats) out of 16 ymm registers, leaving room for
// one broadcast A value and two B vectors. Each FMA pair does 6*16 = 96
// multiply-adds against 6+16 = 22 loads: enough arithmetic intensity to keep
// the FMA units fed.
constexpr int MR = 6;
constexpr int NR = 16;

// Cache blocking. Bp holds KC x NC floats (targets L2/L3), Ap holds MC x KC
// (targets L2). Both are re-streamed from cache many times by the macro-kernel.
constexpr int MC = 192;    // multiple of MR
constexpr int KC = 256;
constexpr int NC = 512;    // multiple of NR

// C[M,N] += A[M,K] @ B[K,N], with the strides described above.
void sgemm(int M, int N, int K,
           const float* A, int a_rs, int a_cs,
           const float* B, int b_rs, int b_cs,
           float* C, int ldc);

// True when the build has an AVX2+FMA micro-kernel compiled in (as opposed to
// the portable scalar fallback). Reported by the benchmark.
bool has_simd_kernel();

}  // namespace gemm
