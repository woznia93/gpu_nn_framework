#pragma once
//
// optim.h
//
// Optimizers. All hold Tensor handles, so step() mutates the real parameters.
//
//   SGD  — lr, momentum, weight decay (L2), optional Nesterov
//   Adam — bias-corrected first/second moments (Kingma & Ba, 2015)
//
// Usage:
//   SGD  opt(net.parameters(), /*lr=*/0.1f, /*momentum=*/0.9f);
//   Adam opt(net.parameters(), /*lr=*/1e-3f);
//
//   opt.zero_grad();
//   loss.backward();
//   opt.step();
//
// CPU-only for now (parameters must live on the CPU).
//

#include "tensor.h"

#include <vector>

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    // Zero all parameter gradients (call before each backward pass).
    void zero_grad();

    const std::vector<Tensor>& parameters() const { return params_; }

protected:
    std::vector<Tensor> params_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SGD:  v ← μ·v + (g + λ·p);   p ← p − lr·v      (or p ← p − lr·g if μ = 0)
// Nesterov uses p ← p − lr·(g + μ·v).
// ─────────────────────────────────────────────────────────────────────────────
class SGD : public Optimizer {
public:
    explicit SGD(std::vector<Tensor> params,
                 float lr           = 1e-3f,
                 float momentum     = 0.0f,
                 float weight_decay = 0.0f,
                 bool  nesterov     = false);

    void step() override;

    float lr() const       { return lr_; }
    void  set_lr(float lr) { lr_ = lr;   }

private:
    float lr_, momentum_, weight_decay_;
    bool  nesterov_;
    std::vector<Tensor> velocity_;        // lazily initialized, one per param
};

// ─────────────────────────────────────────────────────────────────────────────
// Adam:
//   m ← β₁·m + (1−β₁)·g          v ← β₂·v + (1−β₂)·g²
//   m̂ = m/(1−β₁ᵗ)                v̂ = v/(1−β₂ᵗ)
//   p ← p − lr·m̂ / (√v̂ + ε)
// ─────────────────────────────────────────────────────────────────────────────
class Adam : public Optimizer {
public:
    explicit Adam(std::vector<Tensor> params,
                  float lr           = 1e-3f,
                  float beta1        = 0.9f,
                  float beta2        = 0.999f,
                  float eps          = 1e-8f,
                  float weight_decay = 0.0f);

    void step() override;

    float lr() const       { return lr_; }
    void  set_lr(float lr) { lr_ = lr;   }

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    long long t_ = 0;                     // step counter (for bias correction)
    std::vector<Tensor> m_, v_;           // moment buffers, lazily initialized
};
