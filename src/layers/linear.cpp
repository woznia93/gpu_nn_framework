#include "linear.h"
#include "autograd.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

static std::shared_ptr<Tensor> ref_ptr(const Tensor& t) 
{
	return std::shared_ptr<Tensor>(const_cast<Tensor*>(&t), [](Tensor*){});
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear
// ─────────────────────────────────────────────────────────────────────────────

Linear::Linear(int in_features, int out_features, bool use_bias, Device device)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(std::make_shared<Tensor>(std::vector<int>{out_features, in_features}, device, true)),
      bias_(std::make_shared<Tensor>(std::vector<int>{out_features}, device, true))
{
    init_weights();
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
    float* dst = weight_->data_ptr();
	for (int i = 0; i < w.numel(); ++i)
		dst[i] = std::tanh(wp[i]) * bound;
	bias_->zero_();
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
	
	int N = input.shape()[0];
	int in = in_features_;
	int out = out_features_;

	Tensor result({N, out}, input.device());
	result.zero_();

	const float* x = input.data_ptr();
	const float* w = weight_->data_ptr();
	float* r = result.data_ptr();

	for (int n = 0; n < N; ++n)
		for (int o = 0; o < out; ++o)
			for (int i = 0; i < in; ++i)
				r[n * out + o] += x[n * in + i] * w[o * in + i];

	if (use_bias_) {
		const float* b = bias_->data_ptr();
		for (int n = 0; n < N; ++n)
			for (int o = 0; o < out; ++o)
				r[n * out + o] += b[o];

	}

	bool needs_grad = input.requires_grad() || weight_->requires_grad();
	if (needs_grad) {
		result.set_requires_grad(true);
		auto fn = std::make_shared<LinearBackward>();
		fn->saved_input = input.detach();
		fn->saved_weight = weight_->detach();
		fn->N = N;
		fn->in_f = in;
		fn->out_f = out;
		fn->has_bias = use_bias_;
		fn->inputs = { ref_ptr(input), weight_ };
		if (use_bias_) fn->inputs.push_back(bias_);

		result.grad_fn = fn;
	}

	return result;
}


std::vector<Tensor*> Linear::parameters()
{
    if (use_bias_) return { weight_.get(), bias_.get() };
    return { weight_.get() };
}

std::string Linear::name() const
{
    return "Linear(" + std::to_string(in_features_) +
           " → " + std::to_string(out_features_) + ")";
}

void Linear::to(Device device)
{
    *weight_ = weight_->to(device);
    *bias_   = bias_->to(device);
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
    long long total = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        long long p = 0;
        for (const Tensor* t : const_cast<Layer*>(layers_[i].get())->parameters())
            p += t->numel();
        total += p;
        std::cout << "  (" << i << ") " << layers_[i]->name()
                  << "  params=" << p << "\n";
    }
    std::cout << ")  total_params=" << total << "\n";
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
