#include "optim.h"

#include <cmath>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer
// ─────────────────────────────────────────────────────────────────────────────
void Optimizer::zero_grad()
{
    for (Tensor& p : params_)
        if (p.defined()) p.zero_grad();
}

static void check_cpu(const Tensor& p, const char* who)
{
    if (p.device() != Device::CPU)
        throw std::runtime_error(std::string(who) + ": CPU parameters only for now");
}

// ─────────────────────────────────────────────────────────────────────────────
// SGD
// ─────────────────────────────────────────────────────────────────────────────
SGD::SGD(std::vector<Tensor> params, float lr, float momentum,
         float weight_decay, bool nesterov)
    : Optimizer(std::move(params)),
      lr_(lr), momentum_(momentum), weight_decay_(weight_decay), nesterov_(nesterov)
{
    if (nesterov_ && momentum_ <= 0.f)
        throw std::invalid_argument("SGD: Nesterov requires momentum > 0");
    velocity_.resize(params_.size());     // undefined until first use
}

void SGD::step()
{
    for (size_t idx = 0; idx < params_.size(); ++idx) {
        Tensor& p = params_[idx];
        if (!p.defined() || !p.grad().defined()) continue;
        check_cpu(p, "SGD");

        float*       pd = p.data_ptr();
        const float* gd = p.grad().data_ptr();
        const int n = p.numel();

        if (momentum_ == 0.f) {
            // p ← p − lr·(g + λ·p), fused in one pass (no temporaries — the
            // original allocated a whole tensor per parameter per step).
            for (int i = 0; i < n; ++i) {
                float g = gd[i] + weight_decay_ * pd[i];
                pd[i] -= lr_ * g;
            }
        } else {
            if (!velocity_[idx].defined())
                velocity_[idx] = Tensor::zeros(p.shape(), Device::CPU);
            float* vd = velocity_[idx].data_ptr();
            for (int i = 0; i < n; ++i) {
                float g = gd[i] + weight_decay_ * pd[i];
                vd[i] = momentum_ * vd[i] + g;
                float update = nesterov_ ? (g + momentum_ * vd[i]) : vd[i];
                pd[i] -= lr_ * update;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────
Adam::Adam(std::vector<Tensor> params, float lr, float beta1, float beta2,
           float eps, float weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
{
    m_.resize(params_.size());
    v_.resize(params_.size());
}

void Adam::step()
{
    ++t_;
    const float bc1 = 1.f - std::pow(beta1_, static_cast<float>(t_));
    const float bc2 = 1.f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t idx = 0; idx < params_.size(); ++idx) {
        Tensor& p = params_[idx];
        if (!p.defined() || !p.grad().defined()) continue;
        check_cpu(p, "Adam");

        if (!m_[idx].defined()) {
            m_[idx] = Tensor::zeros(p.shape(), Device::CPU);
            v_[idx] = Tensor::zeros(p.shape(), Device::CPU);
        }

        float*       pd = p.data_ptr();
        const float* gd = p.grad().data_ptr();
        float*       md = m_[idx].data_ptr();
        float*       vd = v_[idx].data_ptr();
        const int n = p.numel();

        for (int i = 0; i < n; ++i) {
            float g = gd[i] + weight_decay_ * pd[i];
            md[i] = beta1_ * md[i] + (1.f - beta1_) * g;
            vd[i] = beta2_ * vd[i] + (1.f - beta2_) * g * g;
            float m_hat = md[i] / bc1;
            float v_hat = vd[i] / bc2;
            pd[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}
