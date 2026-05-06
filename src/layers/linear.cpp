#include "linear.h"
#include "autograd.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Linear
// ─────────────────────────────────────────────────────────────────────────────

Linear::Linear(int in_features, int out_features, bool use_bias, Device device)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_({out_features, in_features}, device, /*requires_grad=*/true),
      bias_({out_features},               device, /*requires_grad=*/true)
{
    init_weights();
    if (device != Device::CPU) to(device);
}

void Linear::init_weights()
{
    // Kaiming uniform initialisation for weights:
    //   fan_in = in_features
    //   bound  = sqrt(1 / fan_in)   (He uniform variant)
    float bound = std::sqrt(1.0f / static_cast<float>(in_features_));

    // Fill weight_ with uniform(-bound, bound)
    // We use randn then scale+shift as a cheap alternative;
    // a proper uniform fill uses the same RNG path as Tensor::randn.
    Tensor w = Tensor::randn({out_features_, in_features_}, Device::CPU, false);
    float* wp = w.data_ptr();
    for (int i = 0; i < w.numel(); ++i) {
        // randn ~ N(0,1); map to U(-bound, bound) via tanh-like trick:
        // We'll just clamp to ±3σ and rescale.  Proper impl: use uniform dist.
        wp[i] = std::tanh(wp[i]) * bound;
    }
    // Copy into weight_ (same shape, CPU → CPU)
    float* dst = weight_.data_ptr();
    const float* src = w.data_ptr();
    for (int i = 0; i < weight_.numel(); ++i) dst[i] = src[i];

    // Bias = 0
    if (use_bias_) bias_.zero_();
}

Tensor Linear::forward(const Tensor& input)
{
    // input : [N, in_features]
    // weight_: [out_features, in_features]
    // output : [N, out_features]  =  input @ weight_.T  + bias_
    if (input.ndim() != 2)
        throw std::invalid_argument("Linear::forward expects 2-D input [N, in_features]");
    if (input.shape()[1] != in_features_)
        throw std::invalid_argument("Linear::forward: input feature dim mismatch");

    // weight_.transpose(0,1) gives [in_features, out_features]
    Tensor wT = weight_.transpose(0, 1);
    Tensor out = input.matmul(wT);          // [N, out_features]

    if (use_bias_) {
        // Bias broadcast: [out_features] → [N, out_features]
        // Reuse Tensor::operator+ with broadcasting via autograd add
        // For now, add bias row-by-row on CPU (simple, correct)
        if (out.device() == Device::CPU) {
            float* op = out.data_ptr();
            const float* bp = bias_.data_ptr();
            int N = out.shape()[0];
            int C = out.shape()[1];
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    op[n * C + c] += bp[c];
        } else {
            throw std::runtime_error("Linear::forward bias add: CUDA path not yet implemented");
        }
    }

    return out;
}

std::vector<Tensor*> Linear::parameters()
{
    if (use_bias_) return { &weight_, &bias_ };
    return { &weight_ };
}

std::string Linear::name() const
{
    return "Linear(" + std::to_string(in_features_) +
           " → " + std::to_string(out_features_) + ")";
}

void Linear::to(Device device)
{
    weight_ = weight_.to(device);
    bias_   = bias_.to(device);
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

std::vector<Tensor*> Sequential::parameters()
{
    std::vector<Tensor*> all;
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
    long long total_params = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& l = layers_[i];
        long long p = 0;
        for (const Tensor* t : const_cast<Layer*>(l.get())->parameters())
            p += t->numel();
        total_params += p;
        std::cout << "  (" << i << ") " << l->name()
                  << "  params=" << p << "\n";
    }
    std::cout << ")  total_params=" << total_params << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// SGD
// ─────────────────────────────────────────────────────────────────────────────

void SGD::step()
{
    for (Tensor* p : params_) {
        if (!p || !p->grad) continue;
        // p = p - lr * grad  (in-place)
        p->add_(*p->grad * (-lr_));
    }
}

void SGD::zero_grad()
{
    for (Tensor* p : params_) {
        if (p && p->grad) p->grad->zero_();
    }
}
