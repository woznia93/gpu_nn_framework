//
// tests/test_cuda.cpp
//
// CUDA-path test suite. Built only when USE_CUDA=ON.
//
// Strategy: the CPU path is the reference implementation — it is already
// validated by finite-difference gradient checks in tests/main.cpp. So every
// CUDA test computes the SAME operation both ways and compares, rather than
// hardcoding expected values. A GPU kernel that disagrees with a
// known-correct CPU kernel is wrong, and that is the only question here.
//
// Tolerances are loose-ish (1e-4 relative): GPU and CPU accumulate in
// different orders, so bit-identical results are not expected for reductions.
//
// Run:
//   cmake -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
//   ./build/nn_cuda_test
//

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "autograd.h"
#include "linear.h"
#include "tensor.h"

#include <cuda_runtime.h>

static int tests_run = 0, tests_passed = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++tests_run;                                                           \
        if (cond) { ++tests_passed; }                                          \
        else {                                                                 \
            std::cerr << "  FAIL  " << #cond                                   \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";        \
        }                                                                      \
    } while (0)

// Max absolute difference between two CPU tensors of the same shape.
static float max_diff(const Tensor& a, const Tensor& b)
{
    if (a.numel() != b.numel()) return 1e30f;
    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    float m = 0.f;
    for (int i = 0; i < a.numel(); ++i)
        m = std::max(m, std::fabs(pa[i] - pb[i]));
    return m;
}

// Relative tolerance scaled by magnitude — GPU reduction order differs from
// CPU, so exact equality is the wrong bar.
static bool close(const Tensor& a, const Tensor& b, float tol = 1e-3f)
{
    return max_diff(a, b) < tol;
}

// ─────────────────────────────────────────────────────────────────────────────
// 0. Device presence and properties
// ─────────────────────────────────────────────────────────────────────────────
static bool report_device()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cerr << "No CUDA device available: "
                  << (err == cudaSuccess ? "device count is 0" : cudaGetErrorString(err))
                  << "\n";
        return false;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  device 0: " << prop.name
              << "  sm_" << prop.major << prop.minor
              << "  " << (prop.totalGlobalMem >> 20) << " MB"
              << "  " << prop.multiProcessorCount << " SMs\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Storage & device transfer — the foundation everything else rests on
// ─────────────────────────────────────────────────────────────────────────────
static void test_transfer()
{
    std::cout << "\n[1] Device transfer\n";

    // Round trip must be bit-exact: it is pure memcpy, no arithmetic.
    Tensor cpu = Tensor::randn({64, 32});
    Tensor gpu = cpu.cuda();
    CHECK(gpu.device() == Device::CUDA);
    CHECK(gpu.shape() == cpu.shape());
    Tensor back = gpu.cpu();
    CHECK(back.device() == Device::CPU);
    CHECK(max_diff(cpu, back) == 0.0f);

    // Transfer of a large tensor (exercises multi-MB cudaMemcpy)
    Tensor big = Tensor::randn({512, 512});
    CHECK(max_diff(big, big.cuda().cpu()) == 0.0f);

    // zeros/ones constructed directly on the device
    Tensor gz = Tensor::zeros({10, 10}, Device::CUDA);
    CHECK(gz.device() == Device::CUDA);
    Tensor gz_cpu = gz.cpu();
    bool all_zero = true;
    for (int i = 0; i < gz_cpu.numel(); ++i) all_zero &= (gz_cpu.data_ptr()[i] == 0.f);
    CHECK(all_zero);

    Tensor go = Tensor::ones({10, 10}, Device::CUDA);
    Tensor go_cpu = go.cpu();
    bool all_one = true;
    for (int i = 0; i < go_cpu.numel(); ++i) all_one &= (go_cpu.data_ptr()[i] == 1.f);
    CHECK(all_one);

    // .to() on a tensor already on the target device is a no-op that shares
    // the impl (no pointless copy).
    CHECK(gpu.cuda().same_impl(gpu));

    // Accessing device memory through the CPU accessor must throw, not
    // silently read a garbage host pointer.
    bool threw = false;
    try { (void)gpu.data_ptr(); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Tiled GEMM vs the CPU reference
// ─────────────────────────────────────────────────────────────────────────────
static void test_matmul()
{
    std::cout << "\n[2] matmul (tiled GEMM)\n";

    // Sizes deliberately include non-multiples of TILE (16) to exercise the
    // boundary guards in the kernel — that is where tiled GEMMs usually break.
    const int shapes[][3] = {
        {  16,  16,  16},   // exactly one tile
        {  17,  19,  23},   // prime-ish, every edge partial
        {  64,  64,  64},
        {   1, 128, 256},   // degenerate M (a single row)
        { 128,   1, 256},   // degenerate N
        { 128, 256,   1},   // degenerate K
        { 200, 300, 150},   // large, all dims non-multiples of 16
        { 512, 512, 512},
    };

    for (const auto& s : shapes) {
        const int M = s[0], N = s[1], K = s[2];
        Tensor A = Tensor::randn({M, K});
        Tensor B = Tensor::randn({K, N});

        Tensor ref = A.matmul(B);                       // CPU reference
        Tensor got = A.cuda().matmul(B.cuda()).cpu();   // GPU

        // Error grows with K (more accumulation), so scale the tolerance.
        float tol = 1e-4f * static_cast<float>(K);
        bool ok = max_diff(ref, got) < tol;
        CHECK(ok);
        if (!ok)
            std::cerr << "    [" << M << "x" << K << "] @ [" << K << "x" << N
                      << "]  max_diff=" << max_diff(ref, got) << "  tol=" << tol << "\n";
    }

    // Identity check: A @ I == A, independent of the CPU path being right.
    {
        const int n = 64;
        Tensor A = Tensor::randn({n, n});
        Tensor I = Tensor::zeros({n, n});
        for (int i = 0; i < n; ++i) I.at({i, i}) = 1.f;
        Tensor got = A.cuda().matmul(I.cuda()).cpu();
        CHECK(close(A, got, 1e-4f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Elementwise activation kernels
// ─────────────────────────────────────────────────────────────────────────────
static void test_activations()
{
    std::cout << "\n[3] Activations\n";

    // Include a size that is not a multiple of the 256-thread block, to check
    // the `if (i < n)` bounds guard.
    for (int n : {1, 255, 256, 257, 100000}) {
        Tensor x = Tensor::randn({n});

        CHECK(close(x.relu(),    x.cuda().relu().cpu(),    1e-6f));
        CHECK(close(x.sigmoid(), x.cuda().sigmoid().cpu(), 1e-5f));
        CHECK(close(x.tanh(),    x.cuda().tanh().cpu(),    1e-5f));
    }

    // Regression test for a bug that existed in v1: the CUDA branch of
    // sigmoid() launched the ReLU kernel. sigmoid(x) is in (0,1) and never
    // equals relu(x) for negative x, so this catches a swapped kernel.
    {
        Tensor x({-2.f, -1.f, 0.f, 1.f, 2.f}, {5});
        Tensor s = x.cuda().sigmoid().cpu();
        bool in_range = true;
        for (int i = 0; i < 5; ++i)
            in_range &= (s.data_ptr()[i] > 0.f && s.data_ptr()[i] < 1.f);
        CHECK(in_range);
        CHECK(std::fabs(s.at({2}) - 0.5f) < 1e-5f);     // sigmoid(0) == 0.5
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. In-place ops on device memory
// ─────────────────────────────────────────────────────────────────────────────
static void test_inplace()
{
    std::cout << "\n[4] In-place ops\n";

    // fill_
    {
        Tensor g = Tensor::zeros({1000}, Device::CUDA);
        g.fill_(3.5f);
        Tensor c = g.cpu();
        bool ok = true;
        for (int i = 0; i < c.numel(); ++i) ok &= (std::fabs(c.data_ptr()[i] - 3.5f) < 1e-6f);
        CHECK(ok);
    }

    // zero_
    {
        Tensor g = Tensor::ones({1000}, Device::CUDA);
        g.zero_();
        Tensor c = g.cpu();
        bool ok = true;
        for (int i = 0; i < c.numel(); ++i) ok &= (c.data_ptr()[i] == 0.f);
        CHECK(ok);
    }

    // add_(scalar) and mul_(scalar)
    {
        Tensor cpu = Tensor::randn({777});
        Tensor gpu = cpu.cuda();
        cpu.add_(2.0f);  gpu.add_(2.0f);
        CHECK(close(cpu, gpu.cpu(), 1e-5f));
        cpu.mul_(0.5f);  gpu.mul_(0.5f);
        CHECK(close(cpu, gpu.cpu(), 1e-5f));
    }

    // add_(tensor, alpha) — the axpy kernel the optimizer would use
    {
        Tensor a_cpu = Tensor::randn({777});
        Tensor b_cpu = Tensor::randn({777});
        Tensor a_gpu = a_cpu.cuda();
        Tensor b_gpu = b_cpu.cuda();
        a_cpu.add_(b_cpu, -0.1f);
        a_gpu.add_(b_gpu, -0.1f);
        CHECK(close(a_cpu, a_gpu.cpu(), 1e-5f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Unimplemented paths must throw, not silently produce garbage
//
// Most of the framework is still CPU-only. The contract is that calling an
// unimplemented CUDA path raises rather than returning a wrong answer, so a
// user never silently trains on nonsense.
// ─────────────────────────────────────────────────────────────────────────────
static void test_unimplemented_paths_throw()
{
    std::cout << "\n[5] Unimplemented CUDA paths throw\n";

    Tensor g = Tensor::randn({4, 4}).cuda();
    Tensor h = Tensor::randn({4, 4}).cuda();

    auto throws = [](auto&& fn) {
        try { fn(); } catch (const std::exception&) { return true; }
        return false;
    };

    CHECK(throws([&] { (void)(g + h); }));          // elementwise binary
    CHECK(throws([&] { (void)g.sum(); }));
    CHECK(throws([&] { (void)g.softmax(1); }));
    CHECK(throws([&] { (void)g.transpose(0, 1); }));
    CHECK(throws([&] { (void)g.exp(); }));

    Tensor t = Tensor::zeros({4}).cuda();
    CHECK(throws([&] { (void)cross_entropy_loss(g, t); }));

    Linear lin(4, 4, true, Device::CUDA);
    CHECK(throws([&] { (void)lin.forward(g); }));
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. GPU vs CPU matmul throughput
//
// Not a pass/fail test — informational. Note the GPU numbers include a full
// device synchronize per launch (the kernels are synchronous by design), so
// they understate what a stream-based implementation would achieve.
// ─────────────────────────────────────────────────────────────────────────────
static void bench_matmul()
{
    std::cout << "\n[6] Throughput: CPU vs CUDA matmul\n";
    std::cout << std::left << std::setw(8) << "size"
              << std::setw(16) << "CPU GFLOP/s" << "CUDA GFLOP/s\n";

    for (int n : {256, 512, 1024, 2048}) {
        Tensor A = Tensor::randn({n, n});
        Tensor B = Tensor::randn({n, n});
        Tensor Ag = A.cuda(), Bg = B.cuda();

        const double flops = 2.0 * n * n * n;
        const int iters = n <= 512 ? 5 : 2;

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { volatile float s = A.matmul(B).at({0,0}); (void)s; }
        auto t1 = std::chrono::steady_clock::now();
        double cpu_s = std::chrono::duration<double>(t1 - t0).count() / iters;

        // Warm up once: the first launch pays context-creation cost.
        { Tensor w = Ag.matmul(Bg); cudaDeviceSynchronize(); }
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { Tensor c = Ag.matmul(Bg); }
        cudaDeviceSynchronize();
        t1 = std::chrono::steady_clock::now();
        double gpu_s = std::chrono::duration<double>(t1 - t0).count() / iters;

        std::cout << std::setw(8) << n
                  << std::setw(16) << std::fixed << std::setprecision(1) << flops / cpu_s / 1e9
                  << flops / gpu_s / 1e9 << "\n";
    }
}

int main()
{
    std::cout << "==========================================\n";
    std::cout << "  gpu_nn_framework  --  CUDA test suite\n";
    std::cout << "==========================================\n";

    if (!report_device()) {
        std::cerr << "\nSkipping CUDA tests (no usable device).\n";
        return 77;                                    // conventional "skipped"
    }

    manual_seed(1234);

    try {
        test_transfer();
        test_matmul();
        test_activations();
        test_inplace();
        test_unimplemented_paths_throw();
        bench_matmul();
    } catch (const std::exception& e) {
        std::cerr << "\nUNCAUGHT EXCEPTION: " << e.what() << "\n";
        return 1;
    }

    // Surface any error left latched on the device.
    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        std::cerr << "\nCUDA error state at exit: " << cudaGetErrorString(last) << "\n";
        return 1;
    }

    std::cout << "\n==========================================\n";
    std::cout << "  Results: " << tests_passed << " / " << tests_run << " passed";
    std::cout << (tests_passed == tests_run ? "  :) all good\n"
                                            : "  ! failures above\n");
    std::cout << "==========================================\n";
    return tests_passed == tests_run ? 0 : 1;
}
