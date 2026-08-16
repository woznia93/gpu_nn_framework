#pragma once
//
// simd_pragmas.h
//
// Portable spellings of the vectorization / parallel-loop pragmas.
//
// Why this exists: MSVC's default `/openmp` implements OpenMP 2.0, which
// predates `#pragma omp simd` (introduced in OpenMP 4.0). Using it there fails
// with "error C7660: 'simd': requires '-openmp:experimental'". Rather than
// forcing every user onto an experimental MSVC switch, each compiler gets the
// pragma it actually supports:
//
//   GCC / Clang       -> `omp simd` (OpenMP 4.0+), or the native loop-vectorize
//                        pragma when built without OpenMP
//   MSVC              -> `loop(ivdep)`, its own vectorization hint
//   anything else     -> nothing; the loop still runs, just maybe scalar
//
// NN_SIMD_LOOP           — vectorize this loop (no threading)
// NN_PARALLEL_FOR        — thread this loop across cores, vectorizing too
// NN_PARALLEL_FOR_PLAIN  — thread only (body already hand-vectorized, e.g.
//                          the GEMM macro-kernel; adding `simd` there would be
//                          wrong since the loop body is not a simple stride)
//
// Both are hints. Correctness never depends on them, so a compiler that
// ignores them still produces right answers, only slower.
//

// _Pragma takes a string, which lets us build pragmas inside macros.
#define NN_PRAGMA(x) _Pragma(#x)

#if defined(_MSC_VER) && !defined(__clang__)
    // MSVC: `loop(ivdep)` asserts no loop-carried dependencies, which is the
    // same promise `omp simd` makes. Note MSVC does not expand _Pragma inside
    // macros the way we need, so use __pragma here.
    #define NN_SIMD_LOOP __pragma(loop(ivdep))
    #if defined(_OPENMP)
        #define NN_PARALLEL_FOR       __pragma(omp parallel for schedule(static))
        #define NN_PARALLEL_FOR_PLAIN __pragma(omp parallel for schedule(static))
    #else
        #define NN_PARALLEL_FOR
        #define NN_PARALLEL_FOR_PLAIN
    #endif

#elif defined(_OPENMP)
    // GCC/Clang with OpenMP. _OPENMP is a date: 201307 == OpenMP 4.0, the
    // first version with `simd`.
    #define NN_PARALLEL_FOR_PLAIN NN_PRAGMA(omp parallel for schedule(static))
    #if _OPENMP >= 201307
        #define NN_SIMD_LOOP    NN_PRAGMA(omp simd)
        #define NN_PARALLEL_FOR NN_PRAGMA(omp parallel for simd schedule(static))
    #else
        #define NN_SIMD_LOOP
        #define NN_PARALLEL_FOR NN_PRAGMA(omp parallel for schedule(static))
    #endif

#elif defined(__clang__)
    #define NN_SIMD_LOOP    NN_PRAGMA(clang loop vectorize(enable))
    #define NN_PARALLEL_FOR NN_PRAGMA(clang loop vectorize(enable))
    #define NN_PARALLEL_FOR_PLAIN

#elif defined(__GNUC__)
    #define NN_SIMD_LOOP    NN_PRAGMA(GCC ivdep)
    #define NN_PARALLEL_FOR NN_PRAGMA(GCC ivdep)
    #define NN_PARALLEL_FOR_PLAIN

#else
    #define NN_SIMD_LOOP
    #define NN_PARALLEL_FOR
    #define NN_PARALLEL_FOR_PLAIN
#endif
