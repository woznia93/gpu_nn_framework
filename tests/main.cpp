//
// tests/main.cpp
//
// Test suite for the framework. All tests are CPU-only, deterministic
// (fixed seed), and include regression tests for every bug fixed in the
// autograd rewrite:
//
//   1. Tensor basics       — construction, indexing, broadcasting, views
//   2. Autograd primitives — scalar ops, chains through temporaries,
//                            fan-out/diamond graphs (the big one)
//   3. Loss functions      — values + numerical gradient checks
//   4. Layers              — Linear, Sequential, Dropout
//   5. Training loop       — 2-layer MLP must actually learn
//   6. Gradient check      — finite differences vs autograd over ALL params
//   7. Optimizers          — SGD momentum, Adam convergence
//
// Build & run:
//   cmake -B build && cmake --build build -j && ./build/nn_framework
//

#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "autograd.h"
#include "linear.h"
#include "optim.h"
#include "tensor.h"

// ── Tiny test harness ────────────────────────────────────────────────────────
static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++tests_run;                                                           \
        if (cond) {                                                            \
            ++tests_passed;                                                    \
        } else {                                                               \
            std::cerr << "  FAIL  " << #cond                                   \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";        \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// Central-difference gradient check of `loss_fn` w.r.t. every element of every
// tensor in `params`. Returns the number of mismatches.
static int numeric_grad_check(std::vector<Tensor> params,
                              const std::function<Tensor()>& loss_fn,
                              float eps = 1e-2f, float tol = 2e-2f,
                              bool verbose = true)
{
    // Autograd gradients.
    zero_grad(params);
    Tensor loss = loss_fn();
    loss.backward();

    int bad = 0;
    for (size_t pi = 0; pi < params.size(); ++pi) {
        Tensor& P = params[pi];
        float* pd = P.data_ptr();
        const Tensor G = P.grad();
        for (int i = 0; i < P.numel(); ++i) {
            float ag = G.defined() ? G.data_ptr()[i] : 0.f;
            float orig = pd[i];

            NoGradGuard ng;                          // finite-diff evals build no graph
            pd[i] = orig + eps;
            float lp = loss_fn().at({0});
            pd[i] = orig - eps;
            float lm = loss_fn().at({0});
            pd[i] = orig;

            float fd   = (lp - lm) / (2.f * eps);
            float diff = std::fabs(ag - fd);
            float rel  = diff / std::max({1.f, std::fabs(ag), std::fabs(fd)});
            if (diff > tol && rel > tol) {
                ++bad;
                if (verbose)
                    std::cerr << "  GRAD MISMATCH param " << pi << "[" << i << "]"
                              << "  autograd=" << ag << "  finite-diff=" << fd
                              << "  |diff|=" << diff << "\n";
            }
        }
    }
    return bad;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Tensor basics
// ─────────────────────────────────────────────────────────────────────────────
static void test_tensor_basics()
{
    std::cout << "\n[1] Tensor basics\n";

    Tensor z = Tensor::zeros({2, 3});
    CHECK(z.shape()[0] == 2 && z.shape()[1] == 3);
    CHECK(z.numel() == 6);
    CHECK_NEAR(z.at({0, 0}), 0.f, 1e-6f);

    Tensor o = Tensor::ones({4});
    CHECK(o.numel() == 4);
    CHECK_NEAR(o.at({3}), 1.f, 1e-6f);

    Tensor t({1.f, 2.f, 3.f, 4.f}, {2, 2});
    CHECK_NEAR(t.at({0, 0}), 1.f, 1e-6f);
    CHECK_NEAR(t.at({1, 1}), 4.f, 1e-6f);

    // Handle semantics: copies share data; clone() deep-copies.
    Tensor alias = t;
    alias.at({0, 0}) = 9.f;
    CHECK_NEAR(t.at({0, 0}), 9.f, 1e-6f);
    Tensor deep = t.clone();
    deep.at({0, 0}) = 1.f;
    CHECK_NEAR(t.at({0, 0}), 9.f, 1e-6f);
    t.at({0, 0}) = 1.f;

    // reshape is a zero-copy view.
    Tensor r = t.reshape({4});
    CHECK(r.numel() == 4);
    r.at({0}) = 5.f;
    CHECK_NEAR(t.at({0, 0}), 5.f, 1e-6f);
    t.at({0, 0}) = 1.f;

    // transpose materializes a correct contiguous copy.
    Tensor m({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3});
    Tensor mT = m.transpose(0, 1);
    CHECK(mT.shape()[0] == 3 && mT.shape()[1] == 2);
    CHECK(mT.is_contiguous());
    CHECK_NEAR(mT.at({0, 1}), 4.f, 1e-6f);
    CHECK_NEAR(mT.at({2, 0}), 3.f, 1e-6f);
    // ...and matmul on the transposed result is numerically right:
    // (2x3)ᵀ @ (2x3) = 3x3, entry [0][0] = 1*1 + 4*4 = 17
    Tensor mm = mT.matmul(m);
    CHECK_NEAR(mm.at({0, 0}), 17.f, 1e-5f);

    // elementwise + broadcasting
    Tensor a({1.f, 2.f, 3.f}, {3});
    Tensor b({4.f, 5.f, 6.f}, {3});
    CHECK_NEAR((a + b).at({2}), 9.f, 1e-6f);
    CHECK_NEAR((a * b).at({0}), 4.f, 1e-6f);

    Tensor mat({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3});
    Tensor row({10.f, 20.f, 30.f}, {1, 3});
    Tensor mr = mat + row;                        // [2,3] + [1,3]
    CHECK_NEAR(mr.at({0, 0}), 11.f, 1e-6f);
    CHECK_NEAR(mr.at({1, 2}), 36.f, 1e-6f);

    Tensor col({100.f, 200.f}, {2, 1});
    Tensor mc = mat * col;                        // [2,3] * [2,1]
    CHECK_NEAR(mc.at({0, 2}), 300.f, 1e-6f);
    CHECK_NEAR(mc.at({1, 0}), 800.f, 1e-6f);

    Tensor vec1d = mat + a;                       // [2,3] + [3] (right-aligned)
    CHECK_NEAR(vec1d.at({1, 1}), 7.f, 1e-6f);

    // reductions
    CHECK_NEAR(a.sum().at({0}), 6.f, 1e-6f);
    CHECK_NEAR(a.mean().at({0}), 2.f, 1e-6f);
    Tensor rows = mat.sum(1);                     // [2]
    CHECK(rows.ndim() == 1 && rows.shape()[0] == 2);
    CHECK_NEAR(rows.at({0}), 6.f, 1e-6f);
    CHECK_NEAR(rows.at({1}), 15.f, 1e-6f);
    Tensor cols = mat.sum(0, /*keepdim=*/true);   // [1,3]
    CHECK(cols.shape()[0] == 1 && cols.shape()[1] == 3);
    CHECK_NEAR(cols.at({0, 1}), 7.f, 1e-6f);

    // softmax sums to 1; log_softmax matches log(softmax)
    Tensor logits({1.f, 2.f, 3.f}, {1, 3});
    Tensor probs = logits.softmax(1);
    CHECK_NEAR(probs.at({0,0}) + probs.at({0,1}) + probs.at({0,2}), 1.f, 1e-5f);
    Tensor lsm = logits.log_softmax(1);
    CHECK_NEAR(lsm.at({0, 2}), std::log(probs.at({0, 2})), 1e-5f);

    // matmul value check: [2,3] @ [3,2]
    Tensor B({7.f, 8.f, 9.f, 10.f, 11.f, 12.f}, {3, 2});
    Tensor C = mat.matmul(B);
    CHECK_NEAR(C.at({0, 0}), 58.f, 1e-5f);        // 1*7 + 2*9 + 3*11
    CHECK_NEAR(C.at({1, 1}), 154.f, 1e-5f);       // 4*8 + 5*10 + 6*12
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Autograd primitives
// ─────────────────────────────────────────────────────────────────────────────
static void test_autograd_primitives()
{
    std::cout << "\n[2] Autograd primitives\n";

    // d/dx (x²) at x=3 → 6
    {
        Tensor x({3.f}, {1}, Device::CPU, true);
        Tensor y = x * x;
        y.backward();
        CHECK(x.grad().defined());
        CHECK_NEAR(x.grad().at({0}), 6.f, 1e-4f);
    }

    // d/dx (x·y) at x=2, y=5 → dx=5, dy=2
    {
        Tensor x({2.f}, {1}, Device::CPU, true);
        Tensor y({5.f}, {1}, Device::CPU, true);
        Tensor z = x * y;
        z.backward();
        CHECK_NEAR(x.grad().at({0}), 5.f, 1e-4f);
        CHECK_NEAR(y.grad().at({0}), 2.f, 1e-4f);
    }

    // Chain through TEMPORARIES: y = (a+b)*c. The original severed the graph
    // here (its copy constructor dropped grad_fn), so a and b got no grads.
    {
        Tensor a({2.f}, {1}, Device::CPU, true);
        Tensor b({3.f}, {1}, Device::CPU, true);
        Tensor c({4.f}, {1}, Device::CPU, true);
        Tensor y = (a + b) * c;
        y.backward();
        CHECK(a.grad().defined() && b.grad().defined() && c.grad().defined());
        CHECK_NEAR(a.grad().at({0}), 4.f, 1e-4f);    // dy/da = c
        CHECK_NEAR(b.grad().at({0}), 4.f, 1e-4f);    // dy/db = c
        CHECK_NEAR(c.grad().at({0}), 5.f, 1e-4f);    // dy/dc = a+b
    }

    // FAN-OUT / diamond: u = 2x; y = u·u + 3u.
    // dy/dx = (2u + 3)·2 = 4·(2x) + 6. At x=1.5 → u=3 → dy/dx = 18.
    // The original engine iterated the topo list AND recursed inside each
    // backward, over-counting gradients on exactly this kind of graph.
    {
        Tensor x({1.5f}, {1}, Device::CPU, true);
        Tensor u = x * 2.0f;
        Tensor y = u * u + u * 3.0f;
        y.backward();
        CHECK_NEAR(x.grad().at({0}), 18.f, 1e-3f);
    }

    // Scalar ops must keep the graph alive (original: +, -, / detached it).
    {
        Tensor x({4.f}, {1}, Device::CPU, true);
        Tensor y = ((x + 1.0f) - 2.0f) / 4.0f;       // y = (x-1)/4, dy/dx = 0.25
        y.backward();
        CHECK(x.grad().defined());
        CHECK_NEAR(x.grad().at({0}), 0.25f, 1e-4f);
    }

    // relu grads
    {
        Tensor neg({-1.f}, {1}, Device::CPU, true);
        neg.relu().backward();
        CHECK_NEAR(neg.grad().at({0}), 0.f, 1e-4f);

        Tensor pos({2.f}, {1}, Device::CPU, true);
        pos.relu().backward();
        CHECK_NEAR(pos.grad().at({0}), 1.f, 1e-4f);
    }

    // sigmoid: d/dx σ(0) = 0.25;  tanh: d/dx tanh(0) = 1
    {
        Tensor x({0.f}, {1}, Device::CPU, true);
        x.sigmoid().backward();
        CHECK_NEAR(x.grad().at({0}), 0.25f, 1e-4f);

        Tensor t({0.f}, {1}, Device::CPU, true);
        t.tanh().backward();
        CHECK_NEAR(t.grad().at({0}), 1.f, 1e-4f);
    }

    // exp/log/pow/sqrt
    {
        Tensor x({2.f}, {1}, Device::CPU, true);
        x.exp().backward();
        CHECK_NEAR(x.grad().at({0}), std::exp(2.f), 1e-3f);

        Tensor y({2.f}, {1}, Device::CPU, true);
        y.log().backward();
        CHECK_NEAR(y.grad().at({0}), 0.5f, 1e-4f);

        Tensor p({3.f}, {1}, Device::CPU, true);
        p.pow(2.f).backward();
        CHECK_NEAR(p.grad().at({0}), 6.f, 1e-4f);

        Tensor s({4.f}, {1}, Device::CPU, true);
        s.sqrt().backward();
        CHECK_NEAR(s.grad().at({0}), 0.25f, 1e-4f);  // 1/(2·√4)
    }

    // Broadcast gradients: y = sum(mat + row). d/d(row) sums over the batch.
    {
        Tensor mat = Tensor::ones({4, 3}, Device::CPU, true);
        Tensor row = Tensor::zeros({1, 3}, Device::CPU, true);
        Tensor y = (mat + row).sum();
        y.backward();
        CHECK(row.grad().defined());
        CHECK_NEAR(row.grad().at({0, 0}), 4.f, 1e-4f);   // summed over 4 rows
        CHECK_NEAR(mat.grad().at({2, 1}), 1.f, 1e-4f);
    }

    // sum(dim) grads (regression for the `dim + i` UB bug in the original)
    {
        Tensor m({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, Device::CPU, true);
        Tensor s = m.sum(0);                          // [3]
        CHECK_NEAR(s.at({1}), 7.f, 1e-5f);
        (s * s).sum().backward();                     // d/dm[i,j] = 2·colsum_j
        CHECK_NEAR(m.grad().at({0, 0}), 10.f, 1e-4f); // 2·(1+4)
        CHECK_NEAR(m.grad().at({1, 2}), 18.f, 1e-4f); // 2·(3+6)
    }

    // reshape/transpose grads flow through views.
    {
        Tensor x({1.f, 2.f, 3.f, 4.f}, {2, 2}, Device::CPU, true);
        Tensor y = x.reshape({4}).sum();
        y.backward();
        CHECK_NEAR(x.grad().at({1, 1}), 1.f, 1e-4f);

        Tensor w({1.f, 2.f, 3.f, 4.f}, {2, 2}, Device::CPU, true);
        Tensor c({1.f, 10.f, 100.f, 1000.f}, {2, 2});
        (w.transpose(0, 1) * c).sum().backward();
        // wT[i][j] = w[j][i], so d/dw[0][1] = c[1][0] = 100
        CHECK_NEAR(w.grad().at({0, 1}), 100.f, 1e-4f);
    }

    // matmul grads vs finite differences.
    {
        Tensor A = Tensor::randn({3, 4}, Device::CPU, true);
        Tensor B = Tensor::randn({4, 2}, Device::CPU, true);
        auto loss_fn = [&]() { return (A.matmul(B) * A.matmul(B)).sum(); };
        int bad = numeric_grad_check({A, B}, loss_fn, 1e-2f, 2e-2f);
        CHECK(bad == 0);
    }

    // softmax + log_softmax backward vs finite differences.
    {
        Tensor x = Tensor::randn({2, 5}, Device::CPU, true);
        Tensor w = Tensor::randn({2, 5});             // random projection, no grad
        auto sm_loss  = [&]() { return (x.softmax(1) * w).sum(); };
        CHECK(numeric_grad_check({x}, sm_loss, 1e-2f, 2e-2f) == 0);
        auto lsm_loss = [&]() { return (x.log_softmax(1) * w).sum(); };
        CHECK(numeric_grad_check({x}, lsm_loss, 1e-2f, 2e-2f) == 0);
    }

    // NoGradGuard prevents graph construction.
    {
        Tensor x({3.f}, {1}, Device::CPU, true);
        NoGradGuard ng;
        Tensor y = x * x;
        CHECK(!y.requires_grad());
        CHECK(y.grad_fn() == nullptr);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Loss functions
// ─────────────────────────────────────────────────────────────────────────────
static void test_losses()
{
    std::cout << "\n[3] Loss functions\n";

    // MSE values
    {
        Tensor pred({1.f, 2.f, 3.f}, {3}, Device::CPU, true);
        Tensor target({1.f, 2.f, 3.f}, {3});
        CHECK_NEAR(mse_loss(pred, target).at({0}), 0.f, 1e-6f);
    }
    {
        Tensor pred({0.f}, {1}, Device::CPU, true);
        Tensor target({1.f}, {1});
        Tensor loss = mse_loss(pred, target);
        CHECK_NEAR(loss.at({0}), 1.f, 1e-5f);
        loss.backward();
        CHECK_NEAR(pred.grad().at({0}), -2.f, 1e-4f);    // 2/N·(p−t) = 2·(0−1)
    }

    // Cross-entropy: uniform logits → loss = log(C)
    {
        int C = 4;
        Tensor logits(std::vector<float>(C, 0.f), {1, C}, Device::CPU, true);
        Tensor target({0.f}, {1});
        CHECK_NEAR(cross_entropy_loss(logits, target).at({0}), std::log((float)C), 1e-4f);
    }

    // NLL: near-perfect prediction → loss ≈ 0.001, and backward exists now
    {
        Tensor lp({-5.f, -0.001f, -5.f}, {1, 3}, Device::CPU, true);
        Tensor target({1.f}, {1});
        Tensor loss = nll_loss(lp, target);
        CHECK_NEAR(loss.at({0}), 0.001f, 1e-3f);
        loss.backward();
        CHECK_NEAR(lp.grad().at({0, 1}), -1.f, 1e-4f);
        CHECK_NEAR(lp.grad().at({0, 0}),  0.f, 1e-4f);
    }

    // BCE value + gradient vs finite differences
    {
        Tensor pred({0.8f, 0.3f, 0.6f}, {3}, Device::CPU, true);
        Tensor target({1.f, 0.f, 1.f}, {3});
        float expected = -(std::log(0.8f) + std::log(0.7f) + std::log(0.6f)) / 3.f;
        CHECK_NEAR(binary_cross_entropy_loss(pred, target).at({0}), expected, 1e-5f);
        auto loss_fn = [&]() { return binary_cross_entropy_loss(pred, target); };
        CHECK(numeric_grad_check({pred}, loss_fn, 1e-3f, 2e-2f) == 0);
    }

    // Cross-entropy gradient vs finite differences on random logits
    {
        Tensor logits = Tensor::randn({4, 3}, Device::CPU, true);
        Tensor target({0.f, 2.f, 1.f, 2.f}, {4});
        auto loss_fn = [&]() { return cross_entropy_loss(logits, target); };
        CHECK(numeric_grad_check({logits}, loss_fn, 1e-2f, 2e-2f) == 0);
    }

    // cross_entropy(logits) == nll(log_softmax(logits)) — consistency
    {
        Tensor logits = Tensor::randn({5, 4});
        Tensor target({1.f, 0.f, 3.f, 2.f, 1.f}, {5});
        float a = cross_entropy_loss(logits, target).at({0});
        float b = nll_loss(logits.log_softmax(1), target).at({0});
        CHECK_NEAR(a, b, 1e-4f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Layers
// ─────────────────────────────────────────────────────────────────────────────
static void test_layers()
{
    std::cout << "\n[4] Layers\n";

    Linear lin(4, 8);
    CHECK(lin.weight().shape()[0] == 8);
    CHECK(lin.weight().shape()[1] == 4);
    CHECK(lin.bias().numel() == 8);
    CHECK(lin.parameters().size() == 2);

    Tensor x = Tensor::randn({5, 4});
    Tensor y = lin.forward(x);
    CHECK(y.shape()[0] == 5 && y.shape()[1] == 8);

    // Linear forward matches x @ Wᵀ + b computed via matmul.
    {
        Tensor ref = x.matmul(lin.weight().transpose(0, 1)) + lin.bias().reshape({1, 8});
        float max_diff = 0.f;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 8; ++j)
                max_diff = std::max(max_diff, std::fabs(ref.at({i, j}) - y.at({i, j})));
        CHECK(max_diff < 1e-4f);
    }

    // Sequential
    Sequential net;
    net.add(std::make_shared<Linear>(4, 16));
    net.add(std::make_shared<ReLULayer>());
    net.add(std::make_shared<Linear>(16, 3));
    CHECK(net.parameters().size() == 4);
    Tensor out = net.forward(x);
    CHECK(out.shape()[0] == 5 && out.shape()[1] == 3);

    // Dropout: eval = identity; train zeroes roughly p of the elements and
    // scales survivors by 1/(1-p).
    {
        Dropout drop(0.5f);
        Tensor big = Tensor::ones({1, 4000});

        drop.eval();
        Tensor same = drop.forward(big);
        CHECK(same.same_impl(big));                   // exact pass-through

        drop.train();
        Tensor dropped = drop.forward(big);
        int zeros = 0;
        for (int i = 0; i < dropped.numel(); ++i) {
            float v = dropped.at({0, i});
            if (v == 0.f) ++zeros;
            else CHECK_NEAR(v, 2.f, 1e-5f);           // survivors scaled by 2
        }
        float frac = static_cast<float>(zeros) / dropped.numel();
        CHECK(frac > 0.4f && frac < 0.6f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Training loop — 2-layer MLP on synthetic 2-D blobs
// (with the original autograd, parameters never received gradients, so this
// could not have learned at all)
// ─────────────────────────────────────────────────────────────────────────────
static void test_training_loop()
{
    std::cout << "\n[5] Training loop (2-layer MLP, synthetic data)\n";

    const int N = 40;
    std::vector<float> X_data, Y_data;
    X_data.reserve(N * 2);
    Y_data.reserve(N);
    for (int i = 0; i < N / 2; ++i) {                // class 0 around (-1,-1)
        X_data.push_back(-1.0f - 0.1f * (i % 5));
        X_data.push_back(-1.0f - 0.1f * ((i / 5) % 5));
        Y_data.push_back(0.f);
    }
    for (int i = 0; i < N / 2; ++i) {                // class 1 around (+1,+1)
        X_data.push_back(+1.0f + 0.1f * (i % 5));
        X_data.push_back(+1.0f + 0.1f * ((i / 5) % 5));
        Y_data.push_back(1.f);
    }
    Tensor X(X_data, {N, 2});
    Tensor Y(Y_data, {N});

    Sequential net;
    net.add(std::make_shared<Linear>(2, 16));
    net.add(std::make_shared<ReLULayer>());
    net.add(std::make_shared<Linear>(16, 2));

    SGD optimizer(net.parameters(), /*lr=*/0.1f, /*momentum=*/0.9f);

    const int epochs = 200;
    float first_loss = 0.f, final_loss = 0.f;

    std::cout << std::fixed << std::setprecision(4);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        Tensor loss = cross_entropy_loss(net.forward(X), Y);
        loss.backward();
        optimizer.step();

        final_loss = loss.at({0});
        if (epoch == 0) first_loss = final_loss;
        if (epoch == 0 || (epoch + 1) % 50 == 0)
            std::cout << "  epoch " << std::setw(3) << epoch + 1
                      << "  loss=" << final_loss << "\n";
    }

    net.eval();
    NoGradGuard ng;
    Tensor logits = net.forward(X);
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        int pred = (logits.at({i, 0}) > logits.at({i, 1})) ? 0 : 1;
        if (pred == static_cast<int>(Y.at({i}))) ++correct;
    }
    float acc = static_cast<float>(correct) / N;
    std::cout << "  final accuracy: " << acc * 100.f << "%\n";

    CHECK(final_loss < 0.05f);
    CHECK(final_loss < first_loss * 0.2f);           // loss actually fell
    CHECK(acc > 0.95f);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Full numerical gradient check — EVERY parameter of a small MLP
// (the original checked 6 entries of one weight matrix)
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_check()
{
    std::cout << "\n[6] Numerical gradient check (all parameters)\n";

    Sequential net;
    net.add(std::make_shared<Linear>(3, 5));
    net.add(std::make_shared<TanhLayer>());          // smooth — friendlier to finite diffs
    net.add(std::make_shared<Linear>(5, 3));

    Tensor X = Tensor::randn({4, 3});
    Tensor Y({0.f, 2.f, 1.f, 0.f}, {4});

    auto params = net.parameters();
    int total = 0;
    for (auto& p : params) total += p.numel();

    auto loss_fn = [&]() { return cross_entropy_loss(net.forward(X), Y); };
    int bad = numeric_grad_check(params, loss_fn, 1e-2f, 2e-2f);
    CHECK(bad == 0);
    if (bad == 0)
        std::cout << "  all " << total << " parameter gradients match finite differences\n";

    // Same, with a ReLU net and MSE-style loss through softmax.
    Sequential net2;
    net2.add(std::make_shared<Linear>(3, 4));
    net2.add(std::make_shared<ReLULayer>());
    net2.add(std::make_shared<Linear>(4, 2));
    auto params2 = net2.parameters();
    Tensor T = Tensor::randn({4, 2});
    auto loss_fn2 = [&]() { return mse_loss(net2.forward(X), T); };
    CHECK(numeric_grad_check(params2, loss_fn2, 1e-2f, 2e-2f) == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Optimizers
// ─────────────────────────────────────────────────────────────────────────────
static void test_optimizers()
{
    std::cout << "\n[7] Optimizers\n";

    // Adam minimizes f(w) = mean((w − 3)²) from w = 0.
    {
        Tensor w = Tensor::zeros({4}, Device::CPU, true);
        Tensor target = Tensor::full({4}, 3.f);
        Adam opt({w}, /*lr=*/0.1f);
        for (int i = 0; i < 300; ++i) {
            opt.zero_grad();
            Tensor loss = mse_loss(w, target);
            loss.backward();
            opt.step();
        }
        CHECK_NEAR(w.at({0}), 3.f, 5e-2f);
        CHECK_NEAR(w.at({3}), 3.f, 5e-2f);
    }

    // SGD + momentum on the same problem converges too.
    {
        Tensor w = Tensor::zeros({4}, Device::CPU, true);
        Tensor target = Tensor::full({4}, 3.f);
        SGD opt({w}, /*lr=*/0.1f, /*momentum=*/0.9f);
        for (int i = 0; i < 200; ++i) {
            opt.zero_grad();
            Tensor loss = mse_loss(w, target);
            loss.backward();
            opt.step();
        }
        CHECK_NEAR(w.at({0}), 3.f, 5e-2f);
    }

    // Weight decay shrinks weights when gradients are zero-ish.
    {
        Tensor w = Tensor::full({2}, 1.f, Device::CPU, true);
        SGD opt({w}, /*lr=*/0.1f, /*momentum=*/0.f, /*weight_decay=*/0.5f);
        // loss = mean((w − w₀)²) with w₀ = current w → gradient 0, only decay acts.
        Tensor frozen = w.clone();
        opt.zero_grad();
        mse_loss(w, frozen).backward();
        opt.step();
        CHECK_NEAR(w.at({0}), 0.95f, 1e-4f);         // 1 − 0.1·0.5·1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::cout << "==========================================\n";
    std::cout << "  gpu_nn_framework  —  test suite\n";
    std::cout << "==========================================\n";

    manual_seed(42);                                 // fully deterministic run

    test_tensor_basics();
    test_autograd_primitives();
    test_losses();
    test_layers();
    test_training_loop();
    test_gradient_check();
    test_optimizers();

    std::cout << "\n==========================================\n";
    std::cout << "  Results: " << tests_passed << " / " << tests_run << " passed";
    if (tests_passed == tests_run)
        std::cout << "  :) all good\n";
    else
        std::cout << "  ! " << (tests_run - tests_passed) << " failures\n";
    std::cout << "==========================================\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
