#pragma once
//
// linear.h
//
// Layer base class + concrete layers:
//   Linear       — fully-connected (weights + optional bias), fused backward
//   ReLULayer    — stateless activation
//   SigmoidLayer
//   TanhLayer
//   Dropout      — inverted dropout (active only in training mode)
//   Sequential   — ordered container that chains layers
//
// parameters() returns Tensor handles (cheap copies that share the underlying
// parameter), so optimizers mutate the real weights — no raw pointers needed.
//
// Usage:
//   Sequential net;
//   net.add(std::make_shared<Linear>(784, 256));
//   net.add(std::make_shared<ReLULayer>());
//   net.add(std::make_shared<Linear>(256, 10));
//
//   Tensor logits = net.forward(x);
//   Tensor loss   = cross_entropy_loss(logits, targets);
//   loss.backward();
//

#include "tensor.h"
#include "autograd.h"

#include <memory>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Layer — abstract base
// ─────────────────────────────────────────────────────────────────────────────
class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    // Trainable parameters as shared handles (empty by default).
    virtual std::vector<Tensor> parameters() { return {}; }

    virtual std::string name() const = 0;

    // Switch between training / inference mode (affects Dropout etc.)
    virtual void train(bool mode = true) { training_ = mode; }
    void eval() { train(false); }
    bool is_training() const { return training_; }

protected:
    bool training_ = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// Linear — y = x Wᵀ + b
//
//   weight : [out_features, in_features]
//   bias   : [out_features]
//
// Init matches PyTorch's nn.Linear default: Kaiming-uniform, i.e. both weight
// and bias ~ U(-1/√fan_in, +1/√fan_in). (The original filled the weight with
// tanh(randn)·bound, which is neither uniform nor normal.)
// ─────────────────────────────────────────────────────────────────────────────
class Linear : public Layer {
public:
    Linear(int in_features, int out_features, bool use_bias = true,
           Device device = Device::CPU);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;

    void to(Device device);

    Tensor&       weight()       { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor&       bias()         { return bias_;   }
    const Tensor& bias()   const { return bias_;   }

    int in_features()  const { return in_features_;  }
    int out_features() const { return out_features_; }

private:
    int  in_features_;
    int  out_features_;
    bool use_bias_;

    Tensor weight_;   // [out, in]
    Tensor bias_;     // [out]

    void init_weights();
};

// ─────────────────────────────────────────────────────────────────────────────
// Stateless activation layers
// ─────────────────────────────────────────────────────────────────────────────
class ReLULayer : public Layer {
public:
    Tensor forward(const Tensor& input) override { return input.relu(); }
    std::string name() const override { return "ReLU"; }
};

class SigmoidLayer : public Layer {
public:
    Tensor forward(const Tensor& input) override { return input.sigmoid(); }
    std::string name() const override { return "Sigmoid"; }
};

class TanhLayer : public Layer {
public:
    Tensor forward(const Tensor& input) override { return input.tanh(); }
    std::string name() const override { return "Tanh"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Dropout — inverted dropout: in training, each element is zeroed with
// probability p and survivors are scaled by 1/(1-p); in eval it's a no-op.
// Implemented as multiplication by a no-grad mask, so autograd handles the
// backward automatically.
// ─────────────────────────────────────────────────────────────────────────────
class Dropout : public Layer {
public:
    explicit Dropout(float p = 0.5f);
    Tensor forward(const Tensor& input) override;
    std::string name() const override;
    float p() const { return p_; }

private:
    float p_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Sequential — chains layers in order
// ─────────────────────────────────────────────────────────────────────────────
class Sequential : public Layer {
public:
    void add(std::shared_ptr<Layer> layer);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override { return "Sequential"; }

    void train(bool mode = true) override;

    void summary() const;

    size_t size() const { return layers_.size(); }
    std::shared_ptr<Layer> operator[](size_t i) { return layers_[i]; }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};
