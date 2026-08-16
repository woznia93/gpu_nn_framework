#include "gemm.h"
#include "simd_pragmas.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

// MSVC never defines __FMA__ (that is a GCC/Clang macro), but /arch:AVX2
// enables the FMA intrinsics too — so on MSVC, __AVX2__ alone is sufficient.
// Requiring both macros silently compiled the fast kernel out under MSVC.
#if defined(_MSC_VER) && !defined(__clang__)
  #if defined(__AVX2__)
    #define GEMM_HAVE_AVX2 1
  #else
    #define GEMM_HAVE_AVX2 0
  #endif
#elif defined(__AVX2__) && defined(__FMA__)
  #define GEMM_HAVE_AVX2 1
#else
  #define GEMM_HAVE_AVX2 0
#endif

#if GEMM_HAVE_AVX2
#include <immintrin.h>
#endif

namespace gemm {

bool has_simd_kernel() { return GEMM_HAVE_AVX2 != 0; }

// ─────────────────────────────────────────────────────────────────────────────
// Aligned scratch buffers for the packed panels
// ─────────────────────────────────────────────────────────────────────────────
namespace {

struct AlignedBuffer {
    float* ptr = nullptr;
    size_t n   = 0;

    void ensure(size_t need)
    {
        if (n >= need) return;
        release();
        // 64-byte alignment: one cache line, also fine for AVX-512 loads.
        size_t bytes = ((need * sizeof(float)) + 63) & ~size_t(63);
#if defined(_WIN32)
        ptr = static_cast<float*>(_aligned_malloc(bytes, 64));
#else
        ptr = static_cast<float*>(std::aligned_alloc(64, bytes));
#endif
        if (!ptr) throw std::bad_alloc();
        n = bytes / sizeof(float);
    }

    void release()
    {
        if (!ptr) return;
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
        ptr = nullptr;
        n = 0;
    }

    ~AlignedBuffer() { release(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Packing
//
// pack_B lays out a KC x nc slice as consecutive NR-wide column strips; within
// a strip, the NR values for k=0 come first, then the NR values for k=1, etc.
// The micro-kernel then reads B with a single linear sweep.
// pack_A does the mirror image with MR-tall row strips.
//
// Both zero-pad partial edge tiles so the micro-kernel never needs bounds
// checks in its inner loop.
// ─────────────────────────────────────────────────────────────────────────────
void pack_B(const float* B, int b_rs, int b_cs, int kc, int nc, float* Bp)
{
    for (int j0 = 0; j0 < nc; j0 += NR) {
        int nr = std::min(NR, nc - j0);
        float* dst = Bp + static_cast<size_t>(j0) * kc;
        for (int p = 0; p < kc; ++p) {
            const float* src = B + static_cast<size_t>(p) * b_rs + static_cast<size_t>(j0) * b_cs;
            int j = 0;
            for (; j < nr; ++j) dst[j] = src[static_cast<size_t>(j) * b_cs];
            for (; j < NR; ++j) dst[j] = 0.0f;          // zero-pad edge tile
            dst += NR;
        }
    }
}

void pack_A(const float* A, int a_rs, int a_cs, int mc, int kc, float* Ap)
{
    for (int i0 = 0; i0 < mc; i0 += MR) {
        int mr = std::min(MR, mc - i0);
        float* dst = Ap + static_cast<size_t>(i0) * kc;
        for (int p = 0; p < kc; ++p) {
            const float* src = A + static_cast<size_t>(i0) * a_rs + static_cast<size_t>(p) * a_cs;
            int i = 0;
            for (; i < mr; ++i) dst[i] = src[static_cast<size_t>(i) * a_rs];
            for (; i < MR; ++i) dst[i] = 0.0f;          // zero-pad edge tile
            dst += MR;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Micro-kernel: C[MR x NR] += Ap[MR x kc] @ Bp[kc x NR]
//
// Accumulators live entirely in registers across the whole kc loop, so each
// packed value loaded from L1 is reused MR (or NR) times. This is where the
// arithmetic intensity comes from.
// ─────────────────────────────────────────────────────────────────────────────
#if GEMM_HAVE_AVX2

inline void micro_kernel(int kc, const float* Ap, const float* Bp,
                         float* C, int ldc, int mr, int nr)
{
    __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
    __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
    __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
    __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();
    __m256 c4a = _mm256_setzero_ps(), c4b = _mm256_setzero_ps();
    __m256 c5a = _mm256_setzero_ps(), c5b = _mm256_setzero_ps();

    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(Bp);
        __m256 b1 = _mm256_loadu_ps(Bp + 8);
        Bp += NR;

        __m256 a;
        a = _mm256_broadcast_ss(Ap + 0);
        c0a = _mm256_fmadd_ps(a, b0, c0a);  c0b = _mm256_fmadd_ps(a, b1, c0b);
        a = _mm256_broadcast_ss(Ap + 1);
        c1a = _mm256_fmadd_ps(a, b0, c1a);  c1b = _mm256_fmadd_ps(a, b1, c1b);
        a = _mm256_broadcast_ss(Ap + 2);
        c2a = _mm256_fmadd_ps(a, b0, c2a);  c2b = _mm256_fmadd_ps(a, b1, c2b);
        a = _mm256_broadcast_ss(Ap + 3);
        c3a = _mm256_fmadd_ps(a, b0, c3a);  c3b = _mm256_fmadd_ps(a, b1, c3b);
        a = _mm256_broadcast_ss(Ap + 4);
        c4a = _mm256_fmadd_ps(a, b0, c4a);  c4b = _mm256_fmadd_ps(a, b1, c4b);
        a = _mm256_broadcast_ss(Ap + 5);
        c5a = _mm256_fmadd_ps(a, b0, c5a);  c5b = _mm256_fmadd_ps(a, b1, c5b);
        Ap += MR;
    }

    const __m256 acc[MR][2] = {
        {c0a, c0b}, {c1a, c1b}, {c2a, c2b}, {c3a, c3b}, {c4a, c4b}, {c5a, c5b}
    };

    if (mr == MR && nr == NR) {
        // Fast path: full tile, accumulate straight into C.
        for (int i = 0; i < MR; ++i) {
            float* c = C + static_cast<size_t>(i) * ldc;
            _mm256_storeu_ps(c,     _mm256_add_ps(_mm256_loadu_ps(c),     acc[i][0]));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(_mm256_loadu_ps(c + 8), acc[i][1]));
        }
    } else {
        // Edge tile: spill to a small buffer, then write only the valid part.
        alignas(32) float tmp[MR][NR];
        for (int i = 0; i < MR; ++i) {
            _mm256_store_ps(tmp[i],     acc[i][0]);
            _mm256_store_ps(tmp[i] + 8, acc[i][1]);
        }
        for (int i = 0; i < mr; ++i) {
            float* c = C + static_cast<size_t>(i) * ldc;
            for (int j = 0; j < nr; ++j) c[j] += tmp[i][j];
        }
    }
}

#else  // portable fallback — still register-blocked, and auto-vectorizable

inline void micro_kernel(int kc, const float* Ap, const float* Bp,
                         float* C, int ldc, int mr, int nr)
{
    float acc[MR][NR] = {};
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < MR; ++i) {
            const float a = Ap[i];
            for (int j = 0; j < NR; ++j) acc[i][j] += a * Bp[j];
        }
        Ap += MR;
        Bp += NR;
    }
    for (int i = 0; i < mr; ++i) {
        float* c = C + static_cast<size_t>(i) * ldc;
        for (int j = 0; j < nr; ++j) c[j] += acc[i][j];
    }
}

#endif

// ─────────────────────────────────────────────────────────────────────────────
// Small-matrix direct path.
// Below a few tens of thousands of MACs, packing costs more than it saves, so
// fall back to a simple i-k-j loop (unit-stride inner, auto-vectorizes).
// ─────────────────────────────────────────────────────────────────────────────
void sgemm_small(int M, int N, int K,
                 const float* A, int a_rs, int a_cs,
                 const float* B, int b_rs, int b_cs,
                 float* C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        float* crow = C + static_cast<size_t>(i) * ldc;
        for (int p = 0; p < K; ++p) {
            const float a = A[static_cast<size_t>(i) * a_rs + static_cast<size_t>(p) * a_cs];
            if (a == 0.0f) continue;
            const float* brow = B + static_cast<size_t>(p) * b_rs;
            if (b_cs == 1) {
                NN_SIMD_LOOP
                for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
            } else {
                for (int j = 0; j < N; ++j) crow[j] += a * brow[static_cast<size_t>(j) * b_cs];
            }
        }
    }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Driver
// ─────────────────────────────────────────────────────────────────────────────
void sgemm(int M, int N, int K,
           const float* A, int a_rs, int a_cs,
           const float* B, int b_rs, int b_cs,
           float* C, int ldc)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    if (static_cast<long long>(M) * N * K < 32768) {
        sgemm_small(M, N, K, A, a_rs, a_cs, B, b_rs, b_cs, C, ldc);
        return;
    }

    // Packed panels are reused across calls; thread_local keeps them warm and
    // avoids a malloc per GEMM (this matters a lot for small training steps).
    //
    // CRITICAL: read the pointers into plain locals BEFORE entering any OpenMP
    // region. Inside a parallel region each worker thread would see its own
    // (never-allocated, null) thread_local instance. The locals are shared by
    // the whole team, so every thread reads the master's packed buffers.
    thread_local AlignedBuffer Apack, Bpack;
    Apack.ensure(static_cast<size_t>(MC) * KC + MR * KC);
    Bpack.ensure(static_cast<size_t>(KC) * NC + KC * NR);
    float* const Ap_base = Apack.ptr;
    float* const Bp_base = Bpack.ptr;

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);

        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);

            pack_B(B + static_cast<size_t>(pc) * b_rs + static_cast<size_t>(jc) * b_cs,
                   b_rs, b_cs, kc, nc, Bp_base);

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);

                pack_A(A + static_cast<size_t>(ic) * a_rs + static_cast<size_t>(pc) * a_cs,
                       a_rs, a_cs, mc, kc, Ap_base);

                // Macro-kernel. Parallelising the column loop keeps every
                // thread working even when M is small (batch 64 training
                // steps), which parallelising over `ic` would not.
                const int n_tiles = (nc + NR - 1) / NR;
                NN_PARALLEL_FOR_PLAIN
                for (int jt = 0; jt < n_tiles; ++jt) {
                    const int j0 = jt * NR;
                    const int nr = std::min(NR, nc - j0);
                    const float* Bp = Bp_base + static_cast<size_t>(j0) * kc;

                    for (int i0 = 0; i0 < mc; i0 += MR) {
                        const int mr = std::min(MR, mc - i0);
                        const float* Ap = Ap_base + static_cast<size_t>(i0) * kc;
                        float* Ctile = C + static_cast<size_t>(ic + i0) * ldc + (jc + j0);
                        micro_kernel(kc, Ap, Bp, Ctile, ldc, mr, nr);
                    }
                }
            }
        }
    }
}

}  // namespace gemm
