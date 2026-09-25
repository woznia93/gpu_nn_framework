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
include/   tensor.h  autograd.h  linear.h  optim.h  gemm.h  simd_pragmas.h
src/       tensor.cpp  autograd.cpp  linear.cpp  optim.cpp  gemm.cpp
cuda/      matmul.cu  elementwise.cu        (built when USE_CUDA=ON)
tests/     main.cpp                          -> ./build/nn_framework
bench/     bench.cpp  compare_pytorch.py     -> ./build/nn_bench
```

## Requirements

- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)

  Vectorization pragmas go through `include/simd_pragmas.h`, which emits the
  spelling each compiler actually supports — MSVC's default `/openmp` is
  OpenMP 2.0 and rejects `#pragma omp simd`, so it gets `loop(ivdep)` instead.
  No `-openmp:experimental` needed.
- OpenMP (optional — auto-detected; single-threaded without it)
- CUDA Toolkit 11+ (optional)

## Build & run

Two rules that cause most build problems:

1. **Always build Release.** A Debug build measures ~100x slower and the
   benchmark numbers are meaningless. CMake warns if you configure otherwise.
2. **Never mix toolchains.** MSVC and MinGW are complete, mutually
   incompatible toolchains. On Windows, CUDA builds *must* be MSVC end to
   end, because nvcc only supports `cl.exe` as its host compiler. Use a
   separate build directory for each toolchain.

After building, `nn_bench` prints which GEMM kernel was compiled in. If it
says `scalar fallback` instead of `AVX2+FMA`, the AVX2 flags did not reach
the compiler and matmul will be ~20x slower — reconfigure a **clean** build
directory (CMake caches flags).

### Linux / macOS — CPU

```bash
cmake -B build -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/nn_framework       # test suite  (expect: all passed)
./build/nn_bench           # benchmarks  (expect: AVX2+FMA)
```

### Linux — CUDA

```bash
nvidia-smi                 # confirm a GPU and driver are present
cmake -B build-cuda -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j
./build-cuda/nn_framework  # CPU suite must still pass
./build-cuda/nn_cuda_test  # GPU kernels vs the CPU reference
```

`CMAKE_CUDA_ARCHITECTURES=native` needs CMake 3.24+. On older CMake, name the
arch: `-DCMAKE_CUDA_ARCHITECTURES=75` (T4/RTX20 = 75, V100 = 70, A100 = 80,
L4/RTX40 = 89, H100 = 90).

### Windows — CPU (MinGW + Ninja)

From an ordinary PowerShell prompt:

```powershell
cmake -B build -G Ninja -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\nn_framework.exe
.\build\nn_bench.exe
```

Ninja is recommended over the Visual Studio generator because it is
single-config: the executables land in `build\` rather than `build\Release\`,
so there is no way to accidentally run a Debug binary.

### Windows — CUDA (MSVC + Ninja)

nvcc requires MSVC on Windows, so this needs the Visual Studio C++ tools
(the free **Build Tools for Visual Studio 2022** with the "Desktop
development with C++" workload is enough) plus the CUDA Toolkit.

**Step 1 — load the MSVC environment** into the current PowerShell session.
This does not persist; repeat it in every new window.

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
        -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation `
                 -DevCmdArguments "-arch=x64 -host_arch=x64"
```

**Step 2 — verify the toolchain.** Both must resolve, and `cl` must point
inside the Visual Studio folder:

```powershell
where.exe cl        # ...\VC\Tools\MSVC\<version>\bin\Hostx64\x64\cl.exe
nvcc --version
```

**Step 3 — configure and build.** `CMAKE_LINKER=link` is required if MinGW is
on your PATH: otherwise CMake pairs MSVC's compiler with MinGW's `ld.exe`,
which fails with `ld.exe: cannot find /nologo`.

```powershell
Remove-Item -Recurse -Force build-cuda -ErrorAction SilentlyContinue
cmake -B build-cuda -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_CXX_COMPILER=cl -DCMAKE_LINKER=link `
      -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda
```

Configure output should say `The CXX compiler identification is MSVC` and
`The CUDA compiler identification is NVIDIA`. If it says GNU, a stale cache
was reused — delete `build-cuda` and retry.

**Step 4 — run.**

```powershell
.\build-cuda\nn_framework.exe
.\build-cuda\nn_cuda_test.exe
```

To avoid repeating step 1, add a helper to your PowerShell profile
(`notepad $PROFILE`) and then just type `vsdev` in any new window:

```powershell
function vsdev {
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
            -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
    Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation `
                     -DevCmdArguments "-arch=x64 -host_arch=x64"
}
```

### No NVIDIA GPU? Use Google Colab

```
!git clone https://github.com/<you>/gpu_nn_framework
!cd gpu_nn_framework && bash bench/run_cuda_colab.sh
```

Set Runtime > Change runtime type > **T4 GPU** first. Colab already has nvcc
and CMake, and being Linux it avoids every Windows toolchain issue above.

### What the CUDA suite checks

`nn_cuda_test` validates each GPU kernel by computing the same operation on
both devices and comparing, using the CPU path as the reference — it is the
one verified by finite-difference gradient checks. Matmul shapes deliberately
include non-multiples of the 16-wide tile and degenerate dimensions, since
tiled GEMMs fail at boundary guards rather than in the interior. The suite
also asserts that *unimplemented* CUDA paths throw rather than silently
returning wrong answers.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `nvcc fatal : Cannot find compiler 'cl.exe'` | MSVC not loaded — do step 1 |
| `ld.exe: cannot find /nologo` | MinGW linker with MSVC compiler — add `-DCMAKE_LINKER=link` |
| `The CXX compiler identification is GNU` on a CUDA build | stale cache — delete the build dir |
| `scalar fallback` in the bench header | AVX2 flags missing — clean reconfigure |
| `ninja: error: loading 'build.ninja'` | configure failed, or wrong directory |
| `CMakeCache.txt directory is different` | build dir was copied/moved — delete it |
| benchmarks ~100x slow | Debug build; with the VS generator add `--config Release` |

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

## CPU performance

The dense compute paths share one blocked, packed, SIMD GEMM
(`src/gemm.cpp`), reached through a stride-generic interface so `matmul`,
`Linear::forward` and both `Linear` backward products use the same kernel
without materializing transposes.

Single core, this container (Xeon @ 2.8 GHz, AVX2+FMA, theoretical peak
~90 GFLOP/s):

| matmul size | before | after |
|---|---|---|
| 128 | 20.9 | ~65 GFLOP/s |
| 256 | 19.5 | ~77 |
| 512 | 18.7 | ~76 |
| 1024 | 14.3 | ~62–77 |

| workload | before | after |
|---|---|---|
| Linear fwd (64x784 -> 256) | 1.43 ms | 0.39 ms |
| MLP training step (batch 64) | 2.70 ms | 0.98 ms (≈1000 steps/s) |
| elementwise add (10M) | 26.6 ms | 23.0 ms |

That is ~70–85% of this core's FMA peak on large GEMMs. The techniques:

1. **Cache blocking** (`MC`/`KC`/`NC`) so the working set stays in L2 rather
   than streaming from DRAM — this is what fixed the flat ~6.5 GFLOP/s
   plateau, which was the signature of a bandwidth-bound kernel.
2. **Packing** — each block is copied into a contiguous, micro-kernel-ordered
   buffer, so the inner loop walks memory linearly with no stride arithmetic.
3. **Register blocking + SIMD** — a 6x16 micro-tile keeps 12 AVX2 accumulators
   live across the whole `k` loop, so each loaded value is reused 6–16 times.
   This is where most of the single-thread gap lived.
4. **Threading over micro-tile columns**, not rows, so batch-64 training steps
   (small M) still parallelize.
5. **No wasted zero-fill** — `Tensor::empty()` skips initialization for buffers
   that are immediately overwritten. On a 10M-element add, the `new float[n]()`
   zero-fill cost as much as the addition itself.

If the benchmark header prints `scalar fallback` instead of `AVX2+FMA`, the
build lost its AVX2 flags and you are leaving ~4x on the table. `-march=native`
is applied in every build type (not just Release) and is probed with
`check_cxx_compiler_flag`, falling back to `-mavx2 -mfma`, then to a warning.

### Build configuration matters enormously

Same machine, same code, three configurations:

| config | matmul 1024 |
|---|---|
| Debug, no OpenMP | 0.68 GFLOP/s |
| Release, naive kernel | 14.3 |
| Release, blocked+SIMD | ~70 |

A Debug build is ~100x off; CMake now emits a warning when you configure one.
Always benchmark with:

```bash
cmake -B build -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
```

## Benchmarks vs PyTorch

```bash
./build/nn_bench
python bench/compare_pytorch.py            # pip install torch
python bench/compare_pytorch.py --threads 1
```

Both run the identical workloads (square matmuls, Linear inference, a full
MLP training step, elementwise add). Honest expectations:

- **Large GEMMs:** with the blocked+packed+AVX2 kernel this is now in the same
  league as MKL/OpenBLAS single-threaded rather than 20x behind. Remaining gap
  comes from AVX-512, software prefetching, and multi-level parallel packing.
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

- [x] CPU GEMM register blocking + AVX/FMA micro-kernel
- [ ] AVX-512 micro-kernel (16x14 tile) + runtime ISA dispatch
- [ ] Software prefetch in the packing loops
- [ ] Fuse activation into the GEMM epilogue (saves a full pass over C)
- [x] CUDA GEMM register tiling (128x128x8 block, 8x8 per thread)
- [ ] CUDA: float4 vectorized loads + double-buffered tiles
- [ ] Caching device allocator (every op currently cudaMallocs its output)
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
