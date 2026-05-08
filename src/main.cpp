//
// main.cpp
//
// Demonstrates and tests the GPU NN framework.
//
// Tests run (all CPU, no GPU required to build/run):
//   1. Tensor basics          — construction, indexing, device transfer
//   2. Autograd primitives    — scalar ops, backward, grad check
//   3. Loss functions         — MSE, cross-entropy sanity values
//   4. Linear layer           — forward shape, parameter count
//   5. Training loop          — 2-layer MLP on synthetic linearly-separable data
//   6. Numerical gradient check — finite differences vs autograd
//
// Build:
//   cmake -B build && cmake --build build
//   ./build/nn_framework
//
//

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "autograd.h"
#include "tensor.h"
#include "linear.h"

// Small test harness
static int tests_run = 0;
static int test_passed = 0;

#define CHECK(cond)
	do { 
		++tests_run;
		if (cond) {
			++test_passed;
		} else {
			std::cerr << "  FAIL  " << #cond << "  (" << __FILE__ << ":" << __LINE__ << ")\n";
		}
	} while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// ─────────────────────────────────────────────────────────────────────────────
// 1. Tensor basics
// ─────────────────────────────────────────────────────────────────────────────
static void test_tensor_basics()
{
    std::cout << "\n[1] Tensor basics\n";

    // zeros / ones
    Tensor z = Tensor::zeros({2, 3});
    CHECK(z.shape()[0] == 2 && z.shape()[1] == 3);
    CHECK(z.numel() == 6);
    CHECK_NEAR(z.at({0, 0}), 0.f, 1e-6f);

    Tensor o = Tensor::ones({4});
    CHECK(o.numel() == 4);
    CHECK_NEAR(o.at({3}), 1.f, 1e-6f);

    // from data
    Tensor t({1.f, 2.f, 3.f, 4.f}, {2, 2});
    CHECK_NEAR(t.at({0, 0}), 1.f, 1e-6f);
    CHECK_NEAR(t.at({1, 1}), 4.f, 1e-6f);

    // reshape
    Tensor r = t.reshape({4});
    CHECK(r.numel() == 4);

    // transpose
    Tensor tT = t.transpose(0, 1);
    CHECK(tT.shape()[0] == 2 && tT.shape()[1] == 2);
    CHECK_NEAR(tT.at({0, 1}), t.at({1, 0}), 1e-6f);

    // elementwise ops
    Tensor a({1.f, 2.f, 3.f}, {3});
    Tensor b({4.f, 5.f, 6.f}, {3});
    Tensor c = a + b;
    CHECK_NEAR(c.at({2}), 9.f, 1e-6f);

    Tensor d = a * b;
    CHECK_NEAR(d.at({0}), 4.f, 1e-6f);

    // sum / mean
    Tensor s = a.sum();
    CHECK_NEAR(s.at({0}), 6.f, 1e-6f);

    Tensor m = a.mean();
    CHECK_NEAR(m.at({0}), 2.f, 1e-6f);

    // softmax sums to 1
    Tensor logits({1.f, 2.f, 3.f}, {1, 3});
    Tensor probs = logits.softmax(1);
    float sum = probs.at({0,0}) + probs.at({0,1}) + probs.at({0,2});
    CHECK_NEAR(sum, 1.f, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Autograd primitives
// ─────────────────────────────────────────────────────────────────────────────
static void test_autograd_primitives()
{
    std::cout << "\n[2] Autograd primitives\n";

    // d/dx (x^2) at x=3 should be 6
    {
        Tensor x({3.f}, {1}, Device::CPU, /*requires_grad=*/true);
        Tensor y = x * x;
        y.backward();
        CHECK(x.grad != nullptr);
        CHECK_NEAR(x.grad->at({0}), 6.f, 1e-4f);
    }

    // d/dx (x * y) at x=2, y=5 → dx=5, dy=2
    {
        Tensor x({2.f}, {1}, Device::CPU, true);
        Tensor y_t({5.f}, {1}, Device::CPU, true);
        Tensor z = x * y_t;
        z.backward();
        CHECK(x.grad != nullptr);
        CHECK_NEAR(x.grad->at({0}), 5.f, 1e-4f);
    }

    // d/dx relu(x) at x=-1 → 0, at x=2 → 1
    {
        Tensor neg({-1.f}, {1}, Device::CPU, true);
        Tensor rn = neg.relu();
        rn.backward();
        CHECK(neg.grad != nullptr);
        CHECK_NEAR(neg.grad->at({0}), 0.f, 1e-4f);

        Tensor pos({2.f}, {1}, Device::CPU, true);
        Tensor rp = pos.relu();
        rp.backward();
        CHECK(pos.grad != nullptr);
        CHECK_NEAR(pos.grad->at({0}), 1.f, 1e-4f);
    }

    // sigmoid: d/dx σ(0) = σ(0)*(1-σ(0)) = 0.25
    {
        Tensor x({0.f}, {1}, Device::CPU, true);
        Tensor s = x.sigmoid();
        s.backward();
        CHECK(x.grad != nullptr);
        CHECK_NEAR(x.grad->at({0}), 0.25f, 1e-4f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Loss functions
// ─────────────────────────────────────────────────────────────────────────────
static void test_losses()
{
    std::cout << "\n[3] Loss functions\n";

    // MSE: pred == target → loss = 0
    {
        Tensor pred({1.f, 2.f, 3.f}, {3}, Device::CPU, true);
        Tensor target({1.f, 2.f, 3.f}, {3});
        Tensor loss = mse_loss(pred, target);
        CHECK_NEAR(loss.at({0}), 0.f, 1e-6f);
    }

    // MSE: pred=[0], target=[1] → loss = 1
    {
        Tensor pred({0.f}, {1}, Device::CPU, true);
        Tensor target({1.f}, {1});
        Tensor loss = mse_loss(pred, target);
        CHECK_NEAR(loss.at({0}), 1.f, 1e-5f);
    }

    // Cross-entropy: uniform logits → loss ≈ log(C)
    {
        int C = 4;
        std::vector<float> logit_data(C, 0.f);          // [1, C] uniform
        Tensor logits(logit_data, {1, C}, Device::CPU, true);
        Tensor target({0.f}, {1});                       // class 0
        Tensor loss = cross_entropy_loss(logits, target);
        float expected = std::log(static_cast<float>(C));
        CHECK_NEAR(loss.at({0}), expected, 1e-4f);
    }

    // NLL: perfect prediction → loss ≈ 0
    {
        // log_probs: class 1 gets -0.001 (nearly 1 probability)
        Tensor lp({-5.f, -0.001f, -5.f}, {1, 3});
        Tensor target({1.f}, {1});
        Tensor loss = nll_loss(lp, target);
        CHECK_NEAR(loss.at({0}), 0.001f, 1e-3f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Linear layer
// ─────────────────────────────────────────────────────────────────────────────
static void test_linear_layer()
{
    std::cout << "\n[4] Linear layer\n";

    Linear lin(4, 8);
    CHECK(lin.weight().shape()[0] == 8);
    CHECK(lin.weight().shape()[1] == 4);
    CHECK(lin.bias().numel() == 8);
    CHECK(lin.parameters().size() == 2);

    // forward shape
    Tensor x = Tensor::randn({5, 4});   // batch=5
    Tensor y = lin.forward(x);
    CHECK(y.shape()[0] == 5);
    CHECK(y.shape()[1] == 8);

    // Sequential
    Sequential net;
    net.add(std::make_shared<Linear>(4, 16));
    net.add(std::make_shared<ReLULayer>());
    net.add(std::make_shared<Linear>(16, 3));
    CHECK(net.parameters().size() == 4);   // W,b for each Linear

    Tensor out = net.forward(x);
    CHECK(out.shape()[0] == 5);
    CHECK(out.shape()[1] == 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Training loop — 2-layer MLP on synthetic data
//
// Task: binary classification of 2-D points.
//   class 0: points around (-1, -1)
//   class 1: points around (+1, +1)
//
// We train for 200 epochs with SGD and expect loss to fall and accuracy to
// rise above 90 %.
// ─────────────────────────────────────────────────────────────────────────────
static void test_training_loop()
{
    std::cout << "\n[5] Training loop (2-layer MLP, synthetic data)\n";

    // ── Synthetic dataset ──────────────────────────────────────────────────
    const int N = 40;  // total samples (balanced)
    std::vector<float> X_data, Y_data;
    X_data.reserve(N * 2);
    Y_data.reserve(N);

    // Simple pattern: no randomness so the test is deterministic
    for (int i = 0; i < N / 2; ++i) {
        X_data.push_back(-1.0f - 0.1f * (i % 5));
        X_data.push_back(-1.0f - 0.1f * ((i / 5) % 5));
        Y_data.push_back(0.f);
    }
    for (int i = 0; i < N / 2; ++i) {
        X_data.push_back(+1.0f + 0.1f * (i % 5));
        X_data.push_back(+1.0f + 0.1f * ((i / 5) % 5));
        Y_data.push_back(1.f);
    }

    Tensor X(X_data, {N, 2});
    Tensor Y(Y_data, {N});

    // ── Network ────────────────────────────────────────────────────────────
    Sequential net;
    net.add(std::make_shared<Linear>(2, 16));
    net.add(std::make_shared<ReLULayer>());
    net.add(std::make_shared<Linear>(16, 2));

    SGD optimizer(net.parameters(), /*lr=*/0.05f);

    // ── Training ───────────────────────────────────────────────────────────
    const int epochs = 200;
    float final_loss = 0.f;

    std::cout << std::fixed << std::setprecision(4);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();

        Tensor logits = net.forward(X);               // [N, 2]
        Tensor loss   = cross_entropy_loss(logits, Y);

        AutogradEngine::backward(loss);
        optimizer.step();

        final_loss = loss.at({0});
        if (epoch == 0 || (epoch + 1) % 50 == 0) {
            std::cout << "  epoch " << std::setw(3) << epoch + 1
                      << "  loss=" << final_loss << "\n";
        }
    }

    // ── Accuracy check ─────────────────────────────────────────────────────
    Tensor logits = net.forward(X);
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        int pred = (logits.at({i, 0}) > logits.at({i, 1})) ? 0 : 1;
        if (pred == static_cast<int>(Y.at({i}))) ++correct;
    }
    float acc = static_cast<float>(correct) / N;
    std::cout << "  final accuracy: " << acc * 100.f << "%\n";

    CHECK(final_loss < 0.5f);    // loss should have come down substantially
    CHECK(acc > 0.85f);          // should classify well on this easy task
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Numerical gradient check
//
// For each parameter p_i, estimate dL/dp_i by finite differences:
//   (L(p_i + ε) - L(p_i - ε)) / (2ε)
// and compare to p_i.grad from autograd.
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_check()
{
    std::cout << "\n[6] Numerical gradient check\n";

    const float eps = 1e-3f;
    const float tol = 1e-2f;   // loose: we're comparing floats with finite diff

    // Tiny net: 2→4→2, one sample
    Sequential net;
    net.add(std::make_shared<Linear>(2, 4));
    net.add(std::make_shared<ReLULayer>());
    net.add(std::make_shared<Linear>(4, 2));

    Tensor X(std::vector<float>{0.5f, -0.3f}, {1, 2});
    Tensor Y(std::vector<float>{0.f}, {1});   // class 0

    // ── Autograd gradients ─────────────────────────────────────────────────
    {
        auto params = net.parameters();
        zero_grad(params);
        Tensor logits = net.forward(X);
        Tensor loss   = cross_entropy_loss(logits, Y);
        AutogradEngine::backward(loss);
    }

    // ── Finite difference check on first 6 elements of weight[0] ──────────
    Tensor& W = static_cast<Linear*>(net[0].get())->weight();
    int checks = std::min(W.numel(), 6);
    bool all_ok = true;

    for (int i = 0; i < checks; ++i) {
        float w_orig = W.data_ptr()[i];
        float ag_grad = W.grad ? W.grad->data_ptr()[i] : 0.f;

        // L(w + ε)
        W.data_ptr()[i] = w_orig + eps;
        float lp = cross_entropy_loss(net.forward(X), Y).at({0});

        // L(w - ε)
        W.data_ptr()[i] = w_orig - eps;
        float lm = cross_entropy_loss(net.forward(X), Y).at({0});

        W.data_ptr()[i] = w_orig;   // restore

        float fd_grad = (lp - lm) / (2.f * eps);
        float diff    = std::fabs(ag_grad - fd_grad);

        if (diff > tol) {
            std::cerr << "  GRAD MISMATCH  W[" << i << "]"
                      << "  ag=" << ag_grad
                      << "  fd=" << fd_grad
                      << "  diff=" << diff << "\n";
            all_ok = false;
        }
    }

    CHECK(all_ok);
    if (all_ok) std::cout << "  All " << checks << " finite-diff checks passed.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  gpu_nn_framework  —  test suite\n";
    std::cout << "══════════════════════════════════════════\n";

    test_tensor_basics();
    test_autograd_primitives();
    test_losses();
    test_linear_layer();
    test_training_loop();
    test_gradient_check();

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  Results: " << tests_passed << " / " << tests_run << " passed";
    if (tests_passed == tests_run)
        std::cout << "  ✓ all good\n";
    else
        std::cout << "  ← " << (tests_run - tests_passed) << " failures\n";
    std::cout << "══════════════════════════════════════════\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
