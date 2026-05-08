# gpu_nn_framework

A neural network framework built from scratch in C++ with optional CUDA GPU acceleration.
Implements a full forward/backward pass via a custom autograd engine, common layer types,
and loss functions — no external ML libraries.

## Features

- **Tensor** — n-dimensional array with CPU and CUDA storage, broadcasting, and shape ops
- **Autograd** — reverse-mode automatic differentiation via dynamic computation graph
- **Layers** — Linear, ReLU, Sigmoid, Tanh, Sequential container
- **Loss functions** — MSE, cross-entropy, binary cross-entropy, NLL
- **CUDA kernels** — tiled GEMM, elementwise activations (ReLU, Sigmoid, Tanh)
- **Optimizer** — SGD


## Requirements

- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CUDA Toolkit 11+ (optional)

## Build

**CPU only:**
```bash
cmake -B build -DUSE_CUDA=OFF
cmake --build build -j$(nproc)
./build/nn_framework
```

**With CUDA:**
```bash
cmake -B build -DUSE_CUDA=ON
cmake --build build -j$(nproc)
./build/nn_framework
```

To switch between CPU and CUDA builds, delete the build directory first:
```bash
rm -rf build/
```

## Usage

```cpp
#include "autograd.h"
#include "linear.h"

// Build a network
Sequential net;
net.add(std::make_shared<Linear>(784, 256));
net.add(std::make_shared<ReLULayer>());
net.add(std::make_shared<Linear>(256, 10));

// Forward pass
Tensor logits = net.forward(input);
Tensor loss   = cross_entropy_loss(logits, targets);

// Backward pass
AutogradEngine::backward(loss);

// Optimizer step
SGD optimizer(net.parameters(), /*lr=*/0.01f);
optimizer.step();
optimizer.zero_grad();
```

## Tests

Running the binary executes the built-in test suite:
[1] Tensor basics
[2] Autograd primitives
[3] Loss functions
[4] Linear layer
[5] Training loop (2-layer MLP, synthetic data)
[6] Numerical gradient check

## Roadmap

- [ ] Batch normalization
- [ ] Dropout
- [ ] Adam optimizer
- [ ] Convolutional layers
- [ ] CUDA paths for sum, softmax, cross-entropy
- [ ] Dataloader / batching utilities

## License

MIT
