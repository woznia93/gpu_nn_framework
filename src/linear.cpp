#include "linear.h"
#include "gemm.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Linear
// ─────────────────────────────────────────────────────────────────────────────
Linear::Linear(int in_features, int out_features, bool use_bias, Device device)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(std::vector<int>{out_features, in_features}, Device::CPU, /*requires_grad=*/true),
      bias_  (std::vector<int>{out_features},              Device::CPU, /*requires_grad=*/true)
{
    if (in_features <= 0 || out_features <= 0)
        throw std::invalid_argument("Linear: feature sizes must be positive");
    init_weights();
    if (device == Device::CUDA) to(device);
}

void Linear::init_weights()
{
    // PyTorch nn.Linear default (kaiming_uniform with a=√5 collapses to this):
    //   weight, bias ~ U(-1/√fan_in, +1/√fan_in)
    float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
    weight_.uniform_(-bound, bound);
    if (use_bias_) bias_.uniform_(-bound, bound);
    else           bias_.zero_();
}

Tensor Linear::forward(const Tensor& input)
{
    // input  : [N, in]      weight : [out, in]
    // output : [N, out] = input @ weightᵀ + bias
    if (input.ndim() != 2)
        throw std::invalid_argument("Linear::forward expects 2-D input [N, in_features]");
    if (input.shape()[1] != in_features_)
        throw std::invalid_argument("Linear::forward: input feature dim mismatch (got " +
                                    input.shape_str() + ", expected [N, " +
                                    std::to_string(in_features_) + "])");
    if (input.device() != weight_.device())
        throw std::runtime_error("Linear::forward: input and parameters on different devices");
    if (input.device() != Device::CPU)
        throw std::runtime_error("Linear::forward: CUDA path not implemented yet "
                                 "(use matmul + add for a GPU forward)");

    const int N   = input.shape()[0];
    const int in  = in_features_;
    const int out = out_features_;

    // y = x @ Wᵀ + b.
    //
    // Pre-fill the output with the bias row, then let sgemm accumulate the
    // product on top — the bias add costs nothing extra. Weight is [out, in],
    // so Wᵀ[k][j] = W[j][k]: b_rs=1, b_cs=in. No transpose is materialized.
    Tensor result = Tensor::empty({N, out}, Device::CPU);
    float* r = result.data_ptr();
    if (use_bias_) {
        const float* b = bias_.data_ptr();
        for (int n = 0; n < N; ++n)
            std::memcpy(r + static_cast<size_t>(n) * out, b, static_cast<size_t>(out) * sizeof(float));
    } else {
        std::memset(r, 0, static_cast<size_t>(N) * out * sizeof(float));
    }

    gemm::sgemm(N, out, in,
                input.data_ptr(),  /*a_rs=*/in, /*a_cs=*/1,
                weight_.data_ptr(), /*b_rs=*/1, /*b_cs=*/in,
                r, /*ldc=*/out);

    bool needs_grad = GradMode::is_enabled() &&
                      (input.requires_grad() || weight_.requires_grad() ||
                       (use_bias_ && bias_.requires_grad()));
    if (needs_grad) {
        result.set_requires_grad(true);
        auto fn = std::make_shared<LinearBackward>();
        fn->saved_input  = input.detach();
        fn->saved_weight = weight_.detach();
        fn->N = N; fn->in_f = in; fn->out_f = out;
        fn->has_bias = use_bias_;
        // Owning handles — the original stored a fake shared_ptr to a stack
        // local that was destroyed before backward() ran.
        fn->inputs = { input, weight_ };
        if (use_bias_) fn->inputs.push_back(bias_);
        result.impl()->grad_fn = fn;
    }
    return result;
}

std::vector<Tensor> Linear::parameters()
{
    if (use_bias_) return { weight_, bias_ };
    return { weight_ };
}

std::string Linear::name() const
{
    return "Linear(" + std::to_string(in_features_) + " -> " +
           std::to_string(out_features_) + (use_bias_ ? "" : ", bias=false") + ")";
}

void Linear::to(Device device)
{
    weight_ = weight_.to(device);
    bias_   = bias_.to(device);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dropout
// ─────────────────────────────────────────────────────────────────────────────
Dropout::Dropout(float p) : p_(p)
{
    if (p < 0.f || p >= 1.f)
        throw std::invalid_argument("Dropout: p must be in [0, 1)");
}

Tensor Dropout::forward(const Tensor& input)
{
    if (!is_training() || p_ == 0.f) return input;
    if (input.device() != Device::CPU)
        throw std::runtime_error("Dropout: CUDA path not implemented yet");

    Tensor mask(input.shape(), Device::CPU);         // requires_grad = false
    std::bernoulli_distribution keep(1.0 - p_);
    const float scale = 1.f / (1.f - p_);
    float* mp = mask.data_ptr();
    auto& rng = global_rng();
    for (int i = 0; i < mask.numel(); ++i)
        mp[i] = keep(rng) ? scale : 0.f;

    return input * mask;                             // MulBackward handles the grad
}

std::string Dropout::name() const
{
    std::ostringstream oss;
    oss << "Dropout(p=" << p_ << ")";
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequential
// ─────────────────────────────────────────────────────────────────────────────
void Sequential::add(std::shared_ptr<Layer> layer)
{
    layers_.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input)
{
    Tensor x = input;
    for (auto& layer : layers_)
        x = layer->forward(x);
    return x;
}

std::vector<Tensor> Sequential::parameters()
{
    std::vector<Tensor> all;
    for (auto& layer : layers_) {
        auto p = layer->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

void Sequential::train(bool mode)
{
    training_ = mode;
    for (auto& layer : layers_) layer->train(mode);
}

void Sequential::summary() const
{
    std::cout << "Sequential(\n";
    long long total = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        long long p = 0;
        for (const Tensor& t : const_cast<Layer*>(layers_[i].get())->parameters())
            p += t.numel();
        total += p;
        std::cout << "  (" << i << ") " << layers_[i]->name()
                  << "  params=" << p << "\n";
    }
    std::cout << ")  total_params=" << total << "\n";
}
