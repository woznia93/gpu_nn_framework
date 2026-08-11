#include "autograd.h"
#include "gemm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Accumulate gradient g into tensor t if it participates in autograd.
// This NEVER recurses — the engine drives propagation in topological order.
// (The original recursed here AND iterated a topo list in the engine, so every
// edge below the root fired multiple times and gradients were over-counted.)
static void accumulate_into(Tensor& t, const Tensor& g)
{
    if (t.defined() && t.requires_grad())
        t.accumulate_grad(g);
}

// Reduce an upstream gradient (shape of a broadcast output) back to the shape
// of one input by summing over the broadcast dimensions.
Tensor reduce_to_shape(const Tensor& grad, const std::vector<int>& target_shape)
{
    if (grad.shape() == target_shape) return grad;

    // Scalar-like target: sum everything.
    long long tnum = 1;
    for (int s : target_shape) tnum *= s;
    if (tnum == 1) return grad.sum().reshape(target_shape);

    Tensor cur = grad;
    // Sum away leading dimensions the target doesn't have.
    while (cur.ndim() > static_cast<int>(target_shape.size()))
        cur = cur.sum(0, /*keepdim=*/false);
    // Sum over dimensions where the target has size 1 (broadcast dims).
    for (int i = 0; i < static_cast<int>(target_shape.size()); ++i)
        if (target_shape[i] == 1 && cur.shape()[i] != 1)
            cur = cur.sum(i, /*keepdim=*/true);

    return cur.reshape(target_shape);
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise binary backwards
// ─────────────────────────────────────────────────────────────────────────────

// c = a + b:  dL/da = g, dL/db = g   (reduced over broadcast dims)
void AddBackward::apply(const Tensor& g)
{
    if (inputs.size() > 0) accumulate_into(inputs[0], reduce_to_shape(g, inputs[0].shape()));
    if (inputs.size() > 1) accumulate_into(inputs[1], reduce_to_shape(g, inputs[1].shape()));
}

// c = a - b:  dL/da = g, dL/db = -g
void SubBackward::apply(const Tensor& g)
{
    if (inputs.size() > 0) accumulate_into(inputs[0], reduce_to_shape(g, inputs[0].shape()));
    if (inputs.size() > 1) accumulate_into(inputs[1], reduce_to_shape(g * (-1.0f), inputs[1].shape()));
}

// c = a * b:  dL/da = g ⊙ b, dL/db = g ⊙ a
void MulBackward::apply(const Tensor& g)
{
    if (inputs.size() > 0) accumulate_into(inputs[0], reduce_to_shape(g * saved_b, inputs[0].shape()));
    if (inputs.size() > 1) accumulate_into(inputs[1], reduce_to_shape(g * saved_a, inputs[1].shape()));
}

// c = a / b:  dL/da = g / b, dL/db = -g ⊙ a / b²
void DivBackward::apply(const Tensor& g)
{
    if (inputs.size() > 0)
        accumulate_into(inputs[0], reduce_to_shape(g / saved_b, inputs[0].shape()));
    if (inputs.size() > 1) {
        Tensor gb = (g * saved_a * (-1.0f)) / (saved_b * saved_b);
        accumulate_into(inputs[1], reduce_to_shape(gb, inputs[1].shape()));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar backwards — the original had none for +, - and /, so those ops
// silently cut the graph.
// ─────────────────────────────────────────────────────────────────────────────
void AddScalarBackward::apply(const Tensor& g)
{
    if (!inputs.empty()) accumulate_into(inputs[0], g);
}

void MulScalarBackward::apply(const Tensor& g)
{
    if (!inputs.empty()) accumulate_into(inputs[0], g * scalar);
}

void DivScalarBackward::apply(const Tensor& g)
{
    if (!inputs.empty()) accumulate_into(inputs[0], g * (1.0f / scalar));
}

// ─────────────────────────────────────────────────────────────────────────────
// MatMul: C = A @ B
//   dL/dA = g @ Bᵀ    [M,N] @ [N,K] -> [M,K]
//   dL/dB = Aᵀ @ g    [K,M] @ [M,N] -> [K,N]
// transpose() now materializes a contiguous copy, so the raw-pointer matmul
// below is correct (the original multiplied a strided view as if it were
// contiguous, producing wrong gradient values).
// ─────────────────────────────────────────────────────────────────────────────
void MatMulBackward::apply(const Tensor& g)
{
    if (inputs.size() > 0)
        accumulate_into(inputs[0], g.matmul(saved_b.transpose(0, 1)));
    if (inputs.size() > 1)
        accumulate_into(inputs[1], saved_a.transpose(0, 1).matmul(g));
}

// ─────────────────────────────────────────────────────────────────────────────
// Activations
// ─────────────────────────────────────────────────────────────────────────────

// y = max(x, 0):  dL/dx = g ⊙ 1[x > 0]
void ReLUBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor gx(saved_input.shape(), Device::CPU);
    const float* xp = saved_input.data_ptr();
    const float* gp = g.data_ptr();
    float* qp = gx.data_ptr();
    int n = saved_input.numel();
    for (int i = 0; i < n; ++i) qp[i] = xp[i] > 0.f ? gp[i] : 0.f;
    accumulate_into(inputs[0], gx);
}

// y = σ(x):  dL/dx = g ⊙ y(1 - y)   (uses the saved output — cheaper)
void SigmoidBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor gx(saved_output.shape(), Device::CPU);
    const float* yp = saved_output.data_ptr();
    const float* gp = g.data_ptr();
    float* qp = gx.data_ptr();
    int n = saved_output.numel();
    for (int i = 0; i < n; ++i) qp[i] = gp[i] * yp[i] * (1.f - yp[i]);
    accumulate_into(inputs[0], gx);
}

// y = tanh(x):  dL/dx = g ⊙ (1 - y²)
void TanhBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor gx(saved_output.shape(), Device::CPU);
    const float* yp = saved_output.data_ptr();
    const float* gp = g.data_ptr();
    float* qp = gx.data_ptr();
    int n = saved_output.numel();
    for (int i = 0; i < n; ++i) qp[i] = gp[i] * (1.f - yp[i] * yp[i]);
    accumulate_into(inputs[0], gx);
}

// y = eˣ:  dL/dx = g ⊙ y
void ExpBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    accumulate_into(inputs[0], g * saved_output);
}

// y = ln(x):  dL/dx = g / x   (forward already guarantees x > 0)
void LogBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    accumulate_into(inputs[0], g / saved_input);
}

// y = xⁿ:  dL/dx = g ⊙ n·xⁿ⁻¹
void PowBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    accumulate_into(inputs[0], g * (saved_input.pow(exponent - 1.0f) * exponent));
}

// y = √x:  dL/dx = g / (2y)
void SqrtBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor gx(saved_output.shape(), Device::CPU);
    const float* yp = saved_output.data_ptr();
    const float* gp = g.data_ptr();
    float* qp = gx.data_ptr();
    int n = saved_output.numel();
    for (int i = 0; i < n; ++i) qp[i] = gp[i] / (2.f * yp[i]);
    accumulate_into(inputs[0], gx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sum: dL/dx = broadcast(g) back to the input shape.
// Works for both keepdim=true and keepdim=false — the memory layout of g is
// identical either way (outer × inner).
// ─────────────────────────────────────────────────────────────────────────────
void SumBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor expanded(input_shape, Device::CPU);
    float* ep = expanded.data_ptr();

    if (dim == -1) {
        float gv = g.data_ptr()[0];
        int total = expanded.numel();
        for (int i = 0; i < total; ++i) ep[i] = gv;
    } else {
        int nd = static_cast<int>(input_shape.size());
        int outer = 1, inner = 1, rsize = input_shape[dim];
        for (int i = 0; i < dim; ++i)      outer *= input_shape[i];
        for (int i = dim + 1; i < nd; ++i) inner *= input_shape[i];

        const float* gp = g.data_ptr();
        for (int o = 0; o < outer; ++o)
            for (int r = 0; r < rsize; ++r)
                for (int i = 0; i < inner; ++i)
                    ep[(static_cast<size_t>(o) * rsize + r) * inner + i] =
                        gp[static_cast<size_t>(o) * inner + i];
    }
    accumulate_into(inputs[0], expanded);
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax / LogSoftmax
// ─────────────────────────────────────────────────────────────────────────────

// Shared slice iteration (matches tensor.cpp's layout convention).
template <class RowFn>
static void for_each_slice(const std::vector<int>& shape, int dim, RowFn fn)
{
    int nd = static_cast<int>(shape.size());
    int outer = 1, inner = 1, size = shape[dim];
    for (int i = 0; i < dim; ++i)      outer *= shape[i];
    for (int i = dim + 1; i < nd; ++i) inner *= shape[i];
    for (int o = 0; o < outer; ++o)
        for (int i = 0; i < inner; ++i)
            fn(static_cast<size_t>(o) * size * inner + i, static_cast<size_t>(inner), size);
}

// y = softmax(x):  dL/dx = y ⊙ (g − Σ_dim(g ⊙ y))
void SoftmaxBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor gx(saved_output.shape(), Device::CPU);
    const float* yp = saved_output.data_ptr();
    const float* gp = g.data_ptr();
    float* qp = gx.data_ptr();

    for_each_slice(saved_output.shape(), dim, [&](size_t base, size_t stride, int size) {
        float dot = 0.f;
        for (int s = 0; s < size; ++s) dot += gp[base + s * stride] * yp[base + s * stride];
        for (int s = 0; s < size; ++s)
            qp[base + s * stride] = yp[base + s * stride] * (gp[base + s * stride] - dot);
    });
    accumulate_into(inputs[0], gx);
}

// y = log_softmax(x):  dL/dx = g − softmax(x) ⊙ Σ_dim g,  softmax(x) = exp(y)
void LogSoftmaxBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor gx(saved_output.shape(), Device::CPU);
    const float* yp = saved_output.data_ptr();
    const float* gp = g.data_ptr();
    float* qp = gx.data_ptr();

    for_each_slice(saved_output.shape(), dim, [&](size_t base, size_t stride, int size) {
        float gsum = 0.f;
        for (int s = 0; s < size; ++s) gsum += gp[base + s * stride];
        for (int s = 0; s < size; ++s)
            qp[base + s * stride] = gp[base + s * stride] - std::exp(yp[base + s * stride]) * gsum;
    });
    accumulate_into(inputs[0], gx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape ops
// ─────────────────────────────────────────────────────────────────────────────
void ReshapeBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    accumulate_into(inputs[0], g.contiguous().reshape(input_shape));
}

void TransposeBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    accumulate_into(inputs[0], g.transpose(dim0, dim1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Losses
// ─────────────────────────────────────────────────────────────────────────────

// loss = mean((p − t)²):
//   dL/dp = 2/N (p − t) · g₀        dL/dt = −2/N (p − t) · g₀
void MSEBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    float scale = 2.0f * g.data_ptr()[0] / static_cast<float>(saved_pred.numel());
    Tensor diff = saved_pred - saved_target;
    accumulate_into(inputs[0], diff * scale);
    if (inputs.size() > 1) accumulate_into(inputs[1], diff * (-scale));
}

// loss = −mean(log p[i, tᵢ]),  p = softmax(logits):
//   dL/dlogits[i,j] = g₀/N · (p[i,j] − 1[j == tᵢ])
void CrossEntropyBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;

    int N = saved_softmax.shape()[0];
    int C = saved_softmax.shape()[1];
    float scale = g.data_ptr()[0] / static_cast<float>(N);

    Tensor gx = saved_softmax.clone();
    float* gp = gx.data_ptr();
    const float* tp = saved_target.data_ptr();
    for (int i = 0; i < N; ++i) {
        int cls = static_cast<int>(tp[i]);
        if (cls < 0 || cls >= C)
            throw std::out_of_range("cross_entropy backward: target class out of range");
        gp[static_cast<size_t>(i) * C + cls] -= 1.0f;
    }
    for (int i = 0; i < N * C; ++i) gp[i] *= scale;
    accumulate_into(inputs[0], gx);
}

// loss = −mean(log_probs[i, tᵢ]):   dL/dlog_probs[i, tᵢ] = −g₀/N
void NLLBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    Tensor gx({N, C}, Device::CPU);                  // zero-initialized
    float* gp = gx.data_ptr();
    const float* tp = saved_target.data_ptr();
    float v = -g.data_ptr()[0] / static_cast<float>(N);
    for (int i = 0; i < N; ++i) {
        int cls = static_cast<int>(tp[i]);
        gp[static_cast<size_t>(i) * C + cls] = v;
    }
    accumulate_into(inputs[0], gx);
}

// loss = −mean(t·log p + (1−t)·log(1−p)),  p pre-clamped to (eps, 1−eps):
//   dL/dp = g₀/N · (p − t) / (p(1 − p))
void BCEBackward::apply(const Tensor& g)
{
    if (inputs.empty()) return;
    int n = saved_pred.numel();
    float scale = g.data_ptr()[0] / static_cast<float>(n);
    Tensor gx(saved_pred.shape(), Device::CPU);
    const float* pp = saved_pred.data_ptr();
    const float* tp = saved_target.data_ptr();
    float* qp = gx.data_ptr();
    for (int i = 0; i < n; ++i)
        qp[i] = scale * (pp[i] - tp[i]) / (pp[i] * (1.f - pp[i]));
    accumulate_into(inputs[0], gx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Linear:  y = x Wᵀ + b
//   dL/dX = g @ W          [N,out] @ [out,in] -> [N,in]
//   dL/dW = gᵀ @ X         [out,N] @ [N,in]  -> [out,in]
//   dL/db = Σ_n g          [out]
// inputs hold real owning handles — the original stored a non-owning pointer
// to a local variable inside Sequential::forward, which dangled by the time
// backward ran (use-after-free).
// ─────────────────────────────────────────────────────────────────────────────
void LinearBackward::apply(const Tensor& g)
{
    const float* go = g.data_ptr();

    if (inputs.size() > 0 && inputs[0].defined() && inputs[0].requires_grad()) {
        // dL/dX = G @ W    [N,out] @ [out,in] -> [N,in]
        Tensor gx = Tensor::zeros({N, in_f}, Device::CPU);
        gemm::sgemm(N, in_f, out_f,
                    go,                       /*a_rs=*/out_f, /*a_cs=*/1,
                    saved_weight.data_ptr(),  /*b_rs=*/in_f,  /*b_cs=*/1,
                    gx.data_ptr(), /*ldc=*/in_f);
        accumulate_into(inputs[0], gx);
    }

    if (inputs.size() > 1 && inputs[1].defined() && inputs[1].requires_grad()) {
        // dL/dW = Gᵀ @ X   [out,N] @ [N,in] -> [out,in]
        // Gᵀ[i][k] = G[k][i], expressed with strides: a_rs=1, a_cs=out_f.
        Tensor gw = Tensor::zeros({out_f, in_f}, Device::CPU);
        gemm::sgemm(out_f, in_f, N,
                    go,                      /*a_rs=*/1,    /*a_cs=*/out_f,
                    saved_input.data_ptr(),  /*b_rs=*/in_f, /*b_cs=*/1,
                    gw.data_ptr(), /*ldc=*/in_f);
        accumulate_into(inputs[1], gw);
    }

    if (has_bias && inputs.size() > 2 && inputs[2].defined() && inputs[2].requires_grad()) {
        // dL/db = Σ_n G[n,:]
        Tensor gb = Tensor::zeros({out_f}, Device::CPU);
        float* __restrict gb_p = gb.data_ptr();
        for (int n = 0; n < N; ++n) {
            const float* __restrict grow = go + static_cast<size_t>(n) * out_f;
            #pragma omp simd
            for (int o = 0; o < out_f; ++o) gb_p[o] += grow[o];
        }
        accumulate_into(inputs[2], gb);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Autograd Engine
//
// Correct single-pass reverse-mode:
//   1. Build a topological order over TensorImpl nodes (DFS from the root).
//   2. Visit nodes root-first. When a node is visited, ALL of its consumers
//      have already deposited their gradient contributions, so its GradFn
//      fires exactly once with the fully-accumulated gradient.
// Handle semantics guarantee node identity: every consumer of a tensor holds
// the same TensorImpl, so fan-out (a tensor used twice) accumulates correctly.
// ─────────────────────────────────────────────────────────────────────────────
void AutogradEngine::dfs(TensorImpl* node,
                         std::unordered_set<TensorImpl*>& visited,
                         std::vector<TensorImpl*>& order)
{
    if (!node || visited.count(node)) return;
    visited.insert(node);
    if (node->grad_fn) {
        for (auto& inp : node->grad_fn->inputs) {
            // Prune at tensors that don't require grad — nothing upstream of
            // them can receive gradient anyway.
            if (inp.defined() && inp.requires_grad())
                dfs(inp.impl().get(), visited, order);
        }
    }
    order.push_back(node);                       // post-order: children first
}

std::vector<TensorImpl*> AutogradEngine::build_topo(TensorImpl* root)
{
    std::unordered_set<TensorImpl*> visited;
    std::vector<TensorImpl*> order;
    dfs(root, visited, order);
    std::reverse(order.begin(), order.end());    // root first
    return order;
}

void AutogradEngine::backward(Tensor& root)
{
    if (!root.defined() || root.numel() != 1)
        throw std::runtime_error("backward(): root must be a scalar — reduce with .sum() or "
                                 ".mean(), or pass an explicit gradient seed");
    backward(root, Tensor::ones({1}, root.device()));
}

void AutogradEngine::backward(Tensor& root, const Tensor& grad_seed)
{
    if (!root.defined())
        throw std::runtime_error("backward(): undefined root");
    if (!root.requires_grad())
        throw std::runtime_error("backward(): root has requires_grad=false");
    if (!grad_seed.defined() || grad_seed.numel() != root.numel())
        throw std::runtime_error("backward(): gradient seed shape must match root");

    // No graph construction while computing gradients.
    NoGradGuard no_grad;

    root.accumulate_grad(grad_seed);

    for (TensorImpl* node : build_topo(root.impl().get())) {
        if (!node->grad_fn || !node->grad) continue;
        Tensor g(node->grad);
        node->grad_fn->apply(g);
    }
}

void AutogradEngine::zero_grad(Tensor& root)
{
    if (!root.defined()) return;
    for (TensorImpl* node : build_topo(root.impl().get()))
        if (node->grad) Tensor(node->grad).zero_();
}

void zero_grad(std::vector<Tensor>& params)
{
    for (Tensor& p : params)
        if (p.defined()) p.zero_grad();
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss functions
// ─────────────────────────────────────────────────────────────────────────────

Tensor mse_loss(const Tensor& pred, const Tensor& target)
{
    if (pred.numel() != target.numel())
        throw std::invalid_argument("mse_loss: pred and target must have the same number of elements");
    if (pred.device() != Device::CPU || target.device() != Device::CPU)
        throw std::runtime_error("mse_loss: CUDA path not implemented yet");

    const float* pp = pred.data_ptr();
    const float* tp = target.data_ptr();
    int N = pred.numel();

    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = static_cast<double>(pp[i]) - tp[i];
        sum_sq += d * d;
    }

    Tensor loss({1}, Device::CPU);
    loss.data_ptr()[0] = static_cast<float>(sum_sq / N);

    if (GradMode::is_enabled() && (pred.requires_grad() || target.requires_grad())) {
        loss.set_requires_grad(true);
        auto fn = std::make_shared<MSEBackward>();
        fn->saved_pred   = pred.detach();
        fn->saved_target = target.detach();
        fn->inputs = { pred, target };
        loss.impl()->grad_fn = fn;
    }
    return loss;
}

Tensor binary_cross_entropy_loss(const Tensor& pred, const Tensor& target)
{
    if (pred.numel() != target.numel())
        throw std::invalid_argument("binary_cross_entropy_loss: size mismatch");
    if (pred.device() != Device::CPU || target.device() != Device::CPU)
        throw std::runtime_error("binary_cross_entropy_loss: CUDA path not implemented yet");

    const float eps = 1e-7f;
    int N = pred.numel();

    // Clamp once; save the clamped values so forward and backward agree.
    Tensor clamped(pred.shape(), Device::CPU);
    {
        const float* pp = pred.data_ptr();
        float* cp = clamped.data_ptr();
        for (int i = 0; i < N; ++i)
            cp[i] = std::max(eps, std::min(1.f - eps, pp[i]));
    }

    const float* cp = clamped.data_ptr();
    const float* tp = target.data_ptr();
    double total = 0.0;
    for (int i = 0; i < N; ++i)
        total += tp[i] * std::log(cp[i]) + (1.f - tp[i]) * std::log(1.f - cp[i]);

    Tensor loss({1}, Device::CPU);
    loss.data_ptr()[0] = static_cast<float>(-total / N);

    if (GradMode::is_enabled() && pred.requires_grad()) {
        loss.set_requires_grad(true);
        auto fn = std::make_shared<BCEBackward>();   // fused backward — the original had none
        fn->saved_pred   = clamped;
        fn->saved_target = target.detach();
        fn->inputs = { pred };
        loss.impl()->grad_fn = fn;
    }
    return loss;
}

Tensor nll_loss(const Tensor& log_probs, const Tensor& target)
{
    if (log_probs.ndim() != 2)
        throw std::invalid_argument("nll_loss: log_probs must be 2-D [N, C]");
    if (log_probs.device() != Device::CPU)
        throw std::runtime_error("nll_loss: CUDA path not implemented yet");
    int N = log_probs.shape()[0];
    int C = log_probs.shape()[1];
    if (target.numel() != N)
        throw std::invalid_argument("nll_loss: target size must equal batch size N");

    const float* lp = log_probs.data_ptr();
    const float* tp = target.data_ptr();
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        int cls = static_cast<int>(tp[i]);
        if (cls < 0 || cls >= C)
            throw std::out_of_range("nll_loss: class index out of range");
        total += lp[static_cast<size_t>(i) * C + cls];
    }

    Tensor loss({1}, Device::CPU);
    loss.data_ptr()[0] = static_cast<float>(-total / N);

    if (GradMode::is_enabled() && log_probs.requires_grad()) {
        loss.set_requires_grad(true);
        auto fn = std::make_shared<NLLBackward>();   // backward — the original had none
        fn->saved_target = target.detach();
        fn->N = N; fn->C = C;
        fn->inputs = { log_probs };
        loss.impl()->grad_fn = fn;
    }
    return loss;
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target)
{
    if (logits.ndim() != 2)
        throw std::invalid_argument("cross_entropy_loss: logits must be 2-D [N, C]");
    if (logits.device() != Device::CPU)
        throw std::runtime_error("cross_entropy_loss: CUDA path not implemented yet");

    int N = logits.shape()[0];
    int C = logits.shape()[1];
    if (target.numel() != N)
        throw std::invalid_argument("cross_entropy_loss: target size must equal batch size N");

    const float* lp = logits.data_ptr();
    const float* tp = target.data_ptr();

    // Softmax probabilities are needed by the backward pass, so compute and
    // save them (numerically stable: subtract the row max first).
    Tensor softmax_out({N, C}, Device::CPU);
    float* sp = softmax_out.data_ptr();

    double total_loss = 0.0;
    for (int i = 0; i < N; ++i) {
        const float* row = lp + static_cast<size_t>(i) * C;
        float* srow = sp + static_cast<size_t>(i) * C;

        float mx = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < C; ++j) mx = std::max(mx, row[j]);

        float row_sum = 0.f;
        for (int j = 0; j < C; ++j) {
            srow[j] = std::exp(row[j] - mx);
            row_sum += srow[j];
        }
        for (int j = 0; j < C; ++j) srow[j] /= row_sum;

        int cls = static_cast<int>(tp[i]);
        if (cls < 0 || cls >= C)
            throw std::out_of_range("cross_entropy_loss: class index out of range");
        total_loss += -std::log(std::max(srow[cls], 1e-12f));
    }

    Tensor loss({1}, Device::CPU);
    loss.data_ptr()[0] = static_cast<float>(total_loss / N);

    if (GradMode::is_enabled() && logits.requires_grad()) {
        loss.set_requires_grad(true);
        auto fn = std::make_shared<CrossEntropyBackward>();
        fn->saved_softmax = softmax_out;
        fn->saved_target  = target.detach();
        fn->inputs = { logits };                 // real handle — grads reach the network
        loss.impl()->grad_fn = fn;
    }
    return loss;
}
