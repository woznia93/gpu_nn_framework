//
// bench/bench.cpp
//
// Micro-benchmarks that mirror bench/compare_pytorch.py, so the two can be
// compared side by side:
//
//   1. matmul GFLOP/s at several sizes
//   2. Linear forward (inference, no grad)
//   3. Full training step: MLP 784→256→10, batch 64
//      (forward + cross-entropy + backward + SGD step)
//   4. Elementwise add throughput
//
// Build & run:
//   cmake -B build && cmake --build build -j && ./build/nn_bench
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "autograd.h"
#include "linear.h"
#include "optim.h"
#include "tensor.h"

using clk = std::chrono::steady_clock;

static double seconds(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

template <class F>
static double time_it(F f, int iters, int warmup = 2)
{
    for (int i = 0; i < warmup; ++i) f();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = clk::now();
    return seconds(t0, t1) / iters;
}

static void bench_matmul()
{
    std::cout << "\n-- matmul (C = A @ B, square) --------------------------\n";
    std::cout << std::left << std::setw(8) << "size"
              << std::setw(14) << "ms/iter" << "GFLOP/s\n";
    for (int n : {128, 256, 512, 1024}) {
        Tensor A = Tensor::randn({n, n});
        Tensor B = Tensor::randn({n, n});
        int iters = n <= 256 ? 20 : (n <= 512 ? 8 : 3);
        double t = time_it([&] { volatile float sink = A.matmul(B).at({0, 0}); (void)sink; },
                           iters);
        double gflops = 2.0 * n * n * n / t / 1e9;
        std::cout << std::setw(8) << n
                  << std::setw(14) << std::fixed << std::setprecision(3) << t * 1e3
                  << std::setprecision(2) << gflops << "\n";
    }
}

static void bench_linear_forward()
{
    std::cout << "\n-- Linear forward, inference (batch 64, 784 -> 256) ----\n";
    Linear lin(784, 256);
    Tensor x = Tensor::randn({64, 784});
    NoGradGuard ng;
    double t = time_it([&] { volatile float sink = lin.forward(x).at({0, 0}); (void)sink; }, 50);
    std::cout << "  " << std::fixed << std::setprecision(3) << t * 1e3 << " ms/iter\n";
}

static void bench_train_step()
{
    std::cout << "\n-- full training step (MLP 784->256->10, batch 64) -----\n";
    Sequential net;
    net.add(std::make_shared<Linear>(784, 256));
    net.add(std::make_shared<ReLULayer>());
    net.add(std::make_shared<Linear>(256, 10));
    SGD opt(net.parameters(), 0.01f);

    Tensor X = Tensor::randn({64, 784});
    std::vector<float> yd(64);
    for (int i = 0; i < 64; ++i) yd[i] = static_cast<float>(i % 10);
    Tensor Y(yd, {64});

    double t = time_it([&] {
        opt.zero_grad();
        Tensor loss = cross_entropy_loss(net.forward(X), Y);
        loss.backward();
        opt.step();
    }, 30);
    std::cout << "  " << std::fixed << std::setprecision(3) << t * 1e3
              << " ms/step  (" << std::setprecision(1) << 1.0 / t << " steps/s)\n";
}

static void bench_elementwise()
{
    std::cout << "\n-- elementwise add (10M floats) -------------------------\n";
    Tensor a = Tensor::randn({10'000'000});
    Tensor b = Tensor::randn({10'000'000});
    double t = time_it([&] { volatile float sink = (a + b).at({0}); (void)sink; }, 10);
    std::cout << "  " << std::fixed << std::setprecision(3) << t * 1e3
              << " ms/iter  (" << std::setprecision(2)
              << 10e6 / t / 1e9 << " Gelem/s)\n";
}

int main()
{
    std::cout << "==========================================\n";
    std::cout << "  gpu_nn_framework  —  CPU benchmarks\n";
    std::cout << "  (compare with: python bench/compare_pytorch.py)\n";
    std::cout << "==========================================\n";

    manual_seed(0);
    bench_matmul();
    bench_linear_forward();
    bench_train_step();
    bench_elementwise();

    std::cout << "\ndone.\n";
    return 0;
}
