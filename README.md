# gpu_nn_framework

A neural-network framework built from scratch in C++17 with optional CUDA
acceleration. Implements a full forward/backward pass via a custom
reverse-mode autograd engine, common layers, losses and optimizers — no
external ML libraries.

**v2 is a correctness rewrite.** The original autograd had architectural bugs
that prevented any learning at all (see [What was fixed](#what-was-fixed)).
This version passes a 2,000+ assertion test suite including finite-difference
gradient checks over every parameter of a full MLP.

## Features

- **Tensor** — n-dimensional float array; CPU + CUDA storage; NumPy-style
  broadcasting on `+ - * /`; views (`reshape`), `transpose`, `squeeze`,
  `clone`/`detach`
- **Autograd** — reverse-mode AD over a dynamic graph; single-pass
  topological engine (correct for fan-out/diamond graphs); `NoGradGuard`
  for inference; `manual_seed()` for reproducibility
- **Layers** — `Linear`, `ReLULayer`, `SigmoidLayer`, `TanhLayer`,
  `Dropout`, `Sequential`
- **Losses** — MSE, cross-entropy (fused stable softmax+NLL), binary
  cross-entropy, NLL — all fully differentiable
- **Optimizers** — `SGD` (momentum, weight decay, Nesterov), `Adam`
- **CUDA kernels** — tiled GEMM, elementwise activations, axpy/fill
- **OpenMP** — multithreaded CPU matmul and Linear forward/backward

## Layout

```
include/   tensor.h  autograd.h  linear.h  optim.h
src/       tensor.cpp  autograd.cpp  linear.cpp  optim.cpp
cuda/      matmul.cu  elementwise.cu        (built when USE_CUDA=ON)
tests/     main.cpp                          -> ./build/nn_framework
bench/     bench.cpp  compare_pytorch.py     -> ./build/nn_bench
```

## Requirements

- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- OpenMP (optional — auto-detected; single-threaded without it)
- CUDA Toolkit 11+ (optional)

## Build & run

**CPU only:**
```bash
cmake -B build -DUSE_CUDA=OFF
cmake --build build -j
./build/nn_framework       # test suite
./build/nn_bench           # benchmarks
```

**With CUDA:**
```bash
cmake -B build -DUSE_CUDA=ON
cmake --build build -j
./build/nn_framework
```

Switching between CPU and CUDA builds? Delete the build directory first
(`rm -rf build/`).

## Usage

```cpp
#include "autograd.h"
#include "linear.h"
#include "optim.h"

manual_seed(42);

Sequential net;
net.add(std::make_shared<Linear>(784, 256));
net.add(std::make_shared<ReLULayer>());
net.add(std::make_shared<Dropout>(0.2f));
net.add(std::make_shared<Linear>(256, 10));

SGD opt(net.parameters(), /*lr=*/0.1f, /*momentum=*/0.9f);
// or: Adam opt(net.parameters(), 1e-3f);

for (int epoch = 0; epoch < epochs; ++epoch) {
    opt.zero_grad();
    Tensor loss = cross_entropy_loss(net.forward(X), Y);   // Y: class ids as float
    loss.backward();
    opt.step();
}

net.eval();                       // disables Dropout
NoGradGuard ng;                   // no graph construction during inference
Tensor logits = net.forward(X_test);
```

Tensors are **handles**: copying a `Tensor` copies a pointer, and both handles
refer to the same data and the same autograd node (exactly like PyTorch). Use
`clone()` for a deep copy and `detach()` to drop out of the graph.

## What was fixed

The v1 autograd could not train anything. The root causes:

1. **Graph severed at almost every op.** The copy constructor deep-copied data
   but silently dropped `grad_fn`, and ops "kept" their inputs by deep-copying
   any tensor not created through `make_tensor`. `cross_entropy_loss` therefore
   stored a grad-less copy of the logits, ReLU a grad-less copy of the linear
   output, and so on — backprop stopped one node below the loss and parameters
   never received gradients. Fixed by the handle/`TensorImpl` redesign: every
   consumer holds the *same* graph node, and `GradFn::inputs` owns real handles.
2. **Use-after-free in `LinearBackward`.** It held a fake non-owning
   `shared_ptr` to a stack local inside `Sequential::forward`, which was
   destroyed before `backward()` ran.
3. **Double gradient propagation.** The engine iterated a topological list
   *and* each backward recursively invoked its inputs' backwards, so gradients
   below the root were counted multiple times (up to 2^depth on diamond
   graphs). The engine now does a single topological pass; each node fires
   exactly once with its fully accumulated gradient.
4. **`Tensor::sum(dim)` had undefined behaviour** — `for (int i = dim + i; …)`
   reads `i` in its own initializer (should be `dim + 1`).
5. **`MatMulBackward` computed wrong numbers.** It fed a *strided* transpose
   view into a matmul that reads raw pointers assuming contiguous memory.
   `transpose()` now materializes a contiguous copy.
6. **`sigmoid()` on CUDA launched the ReLU kernel.**
7. **Scalar `+`, `-`, `/` silently detached the graph** (only `*` had a
   backward). All four now have backwards.
8. **`binary_cross_entropy_loss` and `nll_loss` had no backward at all.**
9. **`reshape()` on non-contiguous views silently reordered data** — now
   guarded; views also propagate gradients (they previously dropped them).
10. **The `CU` error macro in `matmul.cu` was missing line continuations**, so
    the CUDA translation unit could not compile; and there was no
    `CMakeLists.txt` despite the README referencing one.
11. **Init wasn't Kaiming.** `tanh(randn)·bound` replaced with PyTorch's
    actual `nn.Linear` default: `U(-1/√fan_in, +1/√fan_in)` for weight & bias.

Improvements beyond fixes: broadcasting on all elementwise ops (with correct
gradient reduction), `softmax`/`log_softmax` with backwards, `Dropout`, `Adam`,
SGD momentum/weight-decay/Nesterov, `NoGradGuard`, `manual_seed`, double
accumulators in reductions/losses, OpenMP + SIMD-vectorized Linear kernels,
fused single-loop optimizer updates (no per-step temporaries), and a
benchmark pair for head-to-head comparison with PyTorch.

## Benchmarks vs PyTorch

```bash
./build/nn_bench
python bench/compare_pytorch.py            # pip install torch
python bench/compare_pytorch.py --threads 1
```

Both run the identical workloads (square matmuls, Linear inference, a full
MLP training step, elementwise add). Honest expectations:

- **Large GEMMs:** PyTorch calls MKL/OpenBLAS (CPU) or cuBLAS (GPU) — decades
  of hand-tuned assembly. A hand-rolled kernel will not beat those; expect
  PyTorch to win by several× on 512²+ matmuls. Closing that gap means
  register blocking + explicit AVX/FMA kernels (on the roadmap).
- **Small tensors / small-batch steps:** this framework has near-zero per-op
  overhead (no dispatcher, no dtype/device dispatch, no Python), so on tiny
  workloads — e.g. small-MLP training steps at batch ≤ 64 — it can match or
  beat PyTorch's per-step latency, especially single-threaded. This is the
  realistic place to "win", and the benchmark pair lets you measure it on
  your machine rather than take anyone's word for it.

## Operation coverage

| Op | CPU | CUDA | Backward |
|---|---|---|---|
| `+ − × ÷` (tensor, broadcast) | ✅ | — | ✅ |
| `+ − × ÷` (scalar) | ✅ | — | ✅ |
| `matmul` | ✅ (OpenMP) | ✅ (tiled) | ✅ |
| `relu / sigmoid / tanh` | ✅ | ✅ | ✅ |
| `softmax / log_softmax` | ✅ | — | ✅ |
| `exp / log / pow / sqrt` | ✅ | — | ✅ |
| `sum / mean` (full or dim) | ✅ | — | ✅ |
| `reshape / transpose / squeeze / unsqueeze` | ✅ | reshape only | ✅ |
| losses (MSE / CE / BCE / NLL) | ✅ | — | ✅ |
| `Linear` fused fwd/bwd | ✅ (OpenMP+SIMD) | — | ✅ |
| in-place `add_ / mul_ / fill_ / zero_` | ✅ | ✅ | n/a |

End-to-end GPU training is **not** possible yet (losses/reductions are
CPU-only); the CUDA kernels accelerate the ops marked above. The CUDA path
compiles against CUDA 11+ but was not run in this environment — treat it as
best-effort until you've run the test suite on a GPU box.

## Roadmap

- [ ] CPU GEMM register blocking + AVX/FMA micro-kernel
- [ ] CUDA paths for sum, softmax, cross-entropy → full GPU training
- [ ] Stream-based async CUDA (currently syncs per launch)
- [ ] Batch normalization, convolutional layers
- [ ] Dataloader / mini-batching utilities
- [ ] Graph freeing after backward (`retain_graph=false` semantics)

## Migration notes (v1 → v2)

- `make_tensor(...)` / `Tensor::self_` are gone — plain `Tensor` values track
  gradients correctly now: `Tensor x({3.f},{1},Device::CPU,true); (x*x).backward();`
- `t.grad` (public member) → `t.grad()` returns a `Tensor` handle; check with
  `.defined()`.
- `Layer::parameters()` returns `std::vector<Tensor>` handles instead of raw
  `Tensor*`.
- `SGD` moved from `linear.h` to `optim.h` (and gained momentum/weight decay);
  `Adam` is new.
- Copying a `Tensor` no longer deep-copies — use `clone()` for that.
- `transpose()` returns a materialized contiguous tensor, not a view.
- `cuda/relu.cu` → `cuda/elementwise.cu` (adds `axpy`, `fill`).

## License

MIT
