#pragma once
//
// autograd.h
//
// Reverse-mode automatic differentiation.
//
//   1. AutogradEngine — topological sort + single-pass gradient propagation.
//      Each node's GradFn fires exactly once, after ALL of its consumers have
//      deposited their gradient contributions (correct for fan-out/diamonds,
//      linear time in graph size).
//   2. GradFn subclasses — one per differentiable op. apply() computes the
//      input gradients and accumulates them; it never recurses.
//   3. Loss functions — mse, cross-entropy, binary cross-entropy, nll.
//      All are fused (fast, numerically stable) and fully differentiable.
//
// Usage:
//     Tensor loss = cross_entropy_loss(net.forward(x), y);
//     loss.backward();                  // or AutogradEngine::backward(loss)
//     // every leaf with requires_grad=true now has .grad() filled
//

#include "tensor.h"

#include <memory>
#include <unordered_set>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Engine
// ─────────────────────────────────────────────────────────────────────────────
class AutogradEngine {
public:
    // Backprop from a scalar root (numel == 1, requires_grad == true).
    static void backward(Tensor& root);

    // Backprop from any root with an explicit upstream gradient
    // (shape must match root).
    static void backward(Tensor& root, const Tensor& grad_seed);

    // Zero the .grad of every node reachable from root.
    static void zero_grad(Tensor& root);

private:
    static std::vector<TensorImpl*> build_topo(TensorImpl* root);
    static void dfs(TensorImpl* node,
                    std::unordered_set<TensorImpl*>& visited,
                    std::vector<TensorImpl*>& order);
};

// Zero the .grad of every tensor in params.
void zero_grad(std::vector<Tensor>& params);

// Reduce `grad` (shape of a broadcasted output) back down to `target_shape`
// by summing over broadcast dimensions. Exposed for tests.
Tensor reduce_to_shape(const Tensor& grad, const std::vector<int>& target_shape);

// ─────────────────────────────────────────────────────────────────────────────
// GradFn subclasses
// ─────────────────────────────────────────────────────────────────────────────
struct AddBackward : GradFn {
    void apply(const Tensor& g) override;
    const char* name() const override { return "AddBackward"; }
};
struct SubBackward : GradFn {
    void apply(const Tensor& g) override;
    const char* name() const override { return "SubBackward"; }
};
struct MulBackward : GradFn {
    Tensor saved_a, saved_b;
    void apply(const Tensor& g) override;
    const char* name() const override { return "MulBackward"; }
};
struct DivBackward : GradFn {
    Tensor saved_a, saved_b;
    void apply(const Tensor& g) override;
    const char* name() const override { return "DivBackward"; }
};

struct AddScalarBackward : GradFn {          // c = a + s  (also a - s)
    void apply(const Tensor& g) override;
    const char* name() const override { return "AddScalarBackward"; }
};
struct MulScalarBackward : GradFn {          // c = a * s
    float scalar = 1.f;
    void apply(const Tensor& g) override;
    const char* name() const override { return "MulScalarBackward"; }
};
struct DivScalarBackward : GradFn {          // c = a / s
    float scalar = 1.f;
    void apply(const Tensor& g) override;
    const char* name() const override { return "DivScalarBackward"; }
};

struct MatMulBackward : GradFn {
    Tensor saved_a, saved_b;
    void apply(const Tensor& g) override;
    const char* name() const override { return "MatMulBackward"; }
};

struct ReLUBackward : GradFn {
    Tensor saved_input;
    void apply(const Tensor& g) override;
    const char* name() const override { return "ReLUBackward"; }
};
struct SigmoidBackward : GradFn {
    Tensor saved_output;
    void apply(const Tensor& g) override;
    const char* name() const override { return "SigmoidBackward"; }
};
struct TanhBackward : GradFn {
    Tensor saved_output;
    void apply(const Tensor& g) override;
    const char* name() const override { return "TanhBackward"; }
};
struct ExpBackward : GradFn {
    Tensor saved_output;
    void apply(const Tensor& g) override;
    const char* name() const override { return "ExpBackward"; }
};
struct LogBackward : GradFn {
    Tensor saved_input;
    void apply(const Tensor& g) override;
    const char* name() const override { return "LogBackward"; }
};
struct PowBackward : GradFn {
    Tensor saved_input;
    float  exponent = 1.f;
    void apply(const Tensor& g) override;
    const char* name() const override { return "PowBackward"; }
};
struct SqrtBackward : GradFn {
    Tensor saved_output;
    void apply(const Tensor& g) override;
    const char* name() const override { return "SqrtBackward"; }
};

struct SumBackward : GradFn {
    std::vector<int> input_shape;
    int  dim     = -1;                        // -1 = sum over everything
    void apply(const Tensor& g) override;
    const char* name() const override { return "SumBackward"; }
};

struct SoftmaxBackward : GradFn {             // y = softmax(x, dim)
    Tensor saved_output;
    int dim = -1;
    void apply(const Tensor& g) override;
    const char* name() const override { return "SoftmaxBackward"; }
};
struct LogSoftmaxBackward : GradFn {          // y = log_softmax(x, dim)
    Tensor saved_output;
    int dim = -1;
    void apply(const Tensor& g) override;
    const char* name() const override { return "LogSoftmaxBackward"; }
};

struct ReshapeBackward : GradFn {
    std::vector<int> input_shape;
    void apply(const Tensor& g) override;
    const char* name() const override { return "ReshapeBackward"; }
};
struct TransposeBackward : GradFn {
    int dim0 = 0, dim1 = 1;
    void apply(const Tensor& g) override;
    const char* name() const override { return "TransposeBackward"; }
};

struct MSEBackward : GradFn {
    Tensor saved_pred, saved_target;
    void apply(const Tensor& g) override;
    const char* name() const override { return "MSEBackward"; }
};
struct CrossEntropyBackward : GradFn {
    Tensor saved_softmax, saved_target;       // [N,C] probs, [N] class ids
    void apply(const Tensor& g) override;
    const char* name() const override { return "CrossEntropyBackward"; }
};
struct NLLBackward : GradFn {
    Tensor saved_target;                      // [N] class ids
    int N = 0, C = 0;
    void apply(const Tensor& g) override;
    const char* name() const override { return "NLLBackward"; }
};
struct BCEBackward : GradFn {
    Tensor saved_pred, saved_target;          // pred already clamped to (eps, 1-eps)
    void apply(const Tensor& g) override;
    const char* name() const override { return "BCEBackward"; }
};

struct LinearBackward : GradFn {
    // inputs[0] = x, inputs[1] = weight, inputs[2] = bias (optional)
    Tensor saved_input;                       // [N, in]
    Tensor saved_weight;                      // [out, in]
    int  N = 0, in_f = 0, out_f = 0;
    bool has_bias = false;
    void apply(const Tensor& g) override;
    const char* name() const override { return "LinearBackward"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Loss functions — all return a scalar tensor (shape {1}) wired into the graph
// ─────────────────────────────────────────────────────────────────────────────

// Mean squared error: mean((pred - target)^2). Differentiable w.r.t. both.
Tensor mse_loss(const Tensor& pred, const Tensor& target);

// Binary cross-entropy: -mean( t*log(p) + (1-t)*log(1-p) ).
// pred must be probabilities in (0,1) — apply sigmoid first.
Tensor binary_cross_entropy_loss(const Tensor& pred, const Tensor& target);

// Negative log-likelihood: -mean( log_probs[i, target_i] ).
// log_probs: [N, C]; target: [N] integer class indices stored as float.
Tensor nll_loss(const Tensor& log_probs, const Tensor& target);

// Cross-entropy = numerically-stable softmax + NLL in one fused op.
// logits: [N, C]; target: [N] integer class indices stored as float.
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target);
