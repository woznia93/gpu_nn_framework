#pragma once
//
// layers/linear.h
//
// Layer base class + concrete layers:
//   Linear     — fully-connected (weights + optional bias)
//   ReLULayer  — stateless activation
//   SigmoidLayer
//   TanhLayer
//   Sequential — ordered container that chains layers
//
// Usage:
//   Sequential net;
//   net.add(std::make_shared<Linear>(784, 256));
//   net.add(std::make_shared<ReLULayer>());
//   net.add(std::make_shared<Linear>(256, 10));
//
//   Tensor logits = net.forward(x);
//   Tensor loss   = cross_entropy_loss(logits, targets);
//   AutogradEngine::backward(loss);
//   // SGD step: p->add_(-lr * *p->grad) for p in net.parameters()
//

#include "tensor.h"
#include <memory>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Layer — abstract base
// ─────────────────────────────────────────────────────────────────────────────
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass
    virtual Tensor forward(const Tensor& input) = 0;

    // Returns raw pointers to all trainable parameters.
    // The optimizer / zero_grad loop uses these.
    virtual std::vector<Tensor*> parameters() { return {}; }

    // Human-readable name for printing
    virtual std::string name() const = 0;

    // Switch between training / inference mode (affects dropout etc.)
    virtual void train(bool mode = true) { training_ = mode; }
    bool is_training() const { return training_; }

protected:
    bool training_ = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// Linear — y = x W^T + b
//
//   weight : [out_features, in_features]   (so y = x @ W^T)
//   bias   : [out_features]                (broadcast over batch)
//
// Kaiming uniform init for weight, zeros for bias.
// ─────────────────────────────────────────────────────────────────────────────
class Linear : public Layer {
public:
    Linear(int in_features, int out_features, bool use_bias = true, Device device = Device::CPU);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override;

    // Move layer to device (CPU ↔ CUDA)
    void to(Device device);

    // Direct access for inspection / custom init
    Tensor& weight()       { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor& bias()         { return bias_; }
    const Tensor& bias()   const { return bias_; }

    int in_features()  const { return in_features_; }
    int out_features() const { return out_features_; }

private:
    int in_features_;
    int out_features_;
    bool use_bias_;

    Tensor weight_;   // [out, in]
    Tensor bias_;     // [out]

    // Kaiming uniform: uniform(-√(1/in), √(1/in)) * √(2)
    // Good default for ReLU networks
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
// Sequential — chains layers in order
// ─────────────────────────────────────────────────────────────────────────────
class Sequential : public Layer {
public:
    // Append a layer to the end of the pipeline
    void add(std::shared_ptr<Layer> layer);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "Sequential"; }

    // Propagate train/eval mode to all child layers
    void train(bool mode = true) override;

    // Print a summary of the network
    void summary() const;

    // Number of child layers
    size_t size() const { return layers_.size(); }

    // Layer access
    std::shared_ptr<Layer> operator[](size_t i) { return layers_[i]; }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SGD optimizer (simple stateless helper — no momentum)
//
//   for (Tensor* p : params) p->add_(-lr * *p->grad)
// ─────────────────────────────────────────────────────────────────────────────
class SGD {
public:
    explicit SGD(std::vector<Tensor*> params, float lr = 1e-3f)
        : params_(std::move(params)), lr_(lr) {}

    // Apply gradient update to all parameters
    void step();

    // Zero gradients (call before each backward pass)
    void zero_grad();

    float lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }

private:
    std::vector<Tensor*> params_;
    float lr_;
};
