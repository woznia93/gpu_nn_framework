#!/usr/bin/env python3
"""
bench/compare_pytorch.py

PyTorch mirror of bench/bench.cpp — run both on the same machine and compare:

    ./build/nn_bench
    python bench/compare_pytorch.py

Requires: pip install torch
Runs on CPU by default so it is an apples-to-apples comparison with the
framework's CPU path. Pass --threads N to pin PyTorch's thread count.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_it(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def bench_matmul():
    print("\n-- matmul (C = A @ B, square) --------------------------")
    print(f"{'size':<8}{'ms/iter':<14}GFLOP/s")
    for n in (128, 256, 512, 1024):
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        iters = 20 if n <= 256 else (8 if n <= 512 else 3)
        t = time_it(lambda: (a @ b)[0, 0].item(), iters)
        print(f"{n:<8}{t * 1e3:<14.3f}{2 * n**3 / t / 1e9:.2f}")


def bench_linear_forward():
    print("\n-- Linear forward, inference (batch 64, 784 -> 256) ----")
    lin = nn.Linear(784, 256)
    x = torch.randn(64, 784)
    with torch.no_grad():
        t = time_it(lambda: lin(x)[0, 0].item(), 50)
    print(f"  {t * 1e3:.3f} ms/iter")


def bench_train_step():
    print("\n-- full training step (MLP 784->256->10, batch 64) -----")
    net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    X = torch.randn(64, 784)
    Y = torch.arange(64) % 10

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(net(X), Y)
        loss.backward()
        opt.step()

    t = time_it(step, 30)
    print(f"  {t * 1e3:.3f} ms/step  ({1.0 / t:.1f} steps/s)")


def bench_elementwise():
    print("\n-- elementwise add (10M floats) -------------------------")
    a = torch.randn(10_000_000)
    b = torch.randn(10_000_000)
    t = time_it(lambda: (a + b)[0].item(), 10)
    print(f"  {t * 1e3:.3f} ms/iter  ({10e6 / t / 1e9:.2f} Gelem/s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None,
                        help="torch.set_num_threads(N) for a fair comparison")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    print("==========================================")
    print(f"  PyTorch {torch.__version__}  —  CPU benchmarks")
    print(f"  threads: {torch.get_num_threads()}")
    print("==========================================")

    torch.manual_seed(0)
    bench_matmul()
    bench_linear_forward()
    bench_train_step()
    bench_elementwise()
    print("\ndone.")


if __name__ == "__main__":
    main()
