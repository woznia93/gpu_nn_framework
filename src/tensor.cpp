#include "tensor.h"
#include "autograd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef USE_CUDA
extern "C" void cuda_matmul    (const float* A, const float* B, float* C, int M, int K, int N);
extern "C" void cuda_relu      (const float* x, float* y, int n);
extern "C" void cuda_sigmoid   (const float* x, float* y, int n);
extern "C" void cuda_tanh_act  (const float* x, float* y, int n);
extern "C" void cuda_add_scalar(float* x, float s, int n);
extern "C" void cuda_mul_scalar(float* x, float s, int n);
extern "C" void cuda_axpy      (float* y, const float* x, float alpha, int n);
extern "C" void cuda_fill      (float* x, float value, int n);
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Grad mode & RNG
// ─────────────────────────────────────────────────────────────────────────────
static thread_local bool g_grad_enabled = true;

bool GradMode::is_enabled()        { return g_grad_enabled; }
void GradMode::set_enabled(bool e) { g_grad_enabled = e;    }

static std::mt19937 g_rng{0x5EEDu};
std::mt19937& global_rng()      { return g_rng; }
void manual_seed(uint64_t seed) { g_rng.seed(static_cast<std::mt19937::result_type>(seed)); }

// ─────────────────────────────────────────────────────────────────────────────
// Storage
// ─────────────────────────────────────────────────────────────────────────────
Storage::Storage(size_t n, Device dev) : numel(n), device(dev)
{
    if (n == 0) return;
    if (dev == Device::CPU) {
        ptr = new float[n]();                       // zero-initialized
    } else {
#ifdef USE_CUDA
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(float)));
        CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float)));
#else
        throw std::runtime_error("Built without CUDA support (configure with -DUSE_CUDA=ON)");
#endif
    }
}

Storage::~Storage()
{
    if (!ptr) return;
    if (device == Device::CPU) {
        delete[] ptr;
    } else {
#ifdef USE_CUDA
        cudaFree(ptr);
#endif
    }
    ptr = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Static helpers
// ─────────────────────────────────────────────────────────────────────────────
int Tensor::compute_numel(const std::vector<int>& shape)
{
    if (shape.empty()) return 0;
    long long n = 1;
    for (int s : shape) {
        if (s <= 0) throw std::invalid_argument("Tensor: all dimensions must be >= 1");
        n *= s;
    }
    if (n > std::numeric_limits<int>::max())
        throw std::invalid_argument("Tensor: too many elements for int indexing");
    return static_cast<int>(n);
}

std::vector<int> Tensor::compute_strides(const std::vector<int>& shape)
{
    // Row-major (C-contiguous): last dim has stride 1.
    std::vector<int> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

std::shared_ptr<TensorImpl> Tensor::make_impl(const std::vector<int>& shape,
                                              Device device, bool requires_grad)
{
    if (shape.empty())
        throw std::invalid_argument("Tensor: shape must have at least one dimension");
    auto impl = std::make_shared<TensorImpl>();
    impl->shape         = shape;
    impl->strides       = compute_strides(shape);
    impl->numel         = compute_numel(shape);
    impl->device        = device;
    impl->requires_grad = requires_grad;
    impl->storage       = std::make_shared<Storage>(static_cast<size_t>(impl->numel), device);
    impl->offset        = 0;
    return impl;
}

void Tensor::check_defined(const char* what) const
{
    if (!impl_)
        throw std::runtime_error(std::string(what) + " called on an undefined tensor");
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructors & factories
// ─────────────────────────────────────────────────────────────────────────────
Tensor::Tensor(const std::vector<int>& shape, Device device, bool requires_grad)
    : impl_(make_impl(shape, device, requires_grad))
{
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape,
               Device device, bool requires_grad)
    : impl_(make_impl(shape, device, requires_grad))
{
    if (static_cast<int>(data.size()) != impl_->numel)
        throw std::invalid_argument("Tensor: data size (" + std::to_string(data.size()) +
                                    ") does not match shape " + shape_str());
    if (device == Device::CPU) {
        std::memcpy(impl_->storage->ptr, data.data(), impl_->numel * sizeof(float));
    } else {
#ifdef USE_CUDA
        CUDA_CHECK(cudaMemcpy(impl_->storage->ptr, data.data(),
                              impl_->numel * sizeof(float), cudaMemcpyHostToDevice));
#else
        throw std::runtime_error("Built without CUDA support");
#endif
    }
}

Tensor Tensor::zeros(const std::vector<int>& shape, Device d, bool rg)
{
    return Tensor(shape, d, rg);                    // Storage zero-initializes
}

Tensor Tensor::ones(const std::vector<int>& shape, Device d, bool rg)
{
    return full(shape, 1.0f, d, rg);
}

Tensor Tensor::full(const std::vector<int>& shape, float value, Device d, bool rg)
{
    Tensor t(shape, Device::CPU, rg);
    std::fill(t.data_ptr(), t.data_ptr() + t.numel(), value);
    if (d == Device::CUDA) return t.cuda();
    return t;
}

Tensor Tensor::randn(const std::vector<int>& shape, Device d, bool rg)
{
    Tensor t(shape, Device::CPU, rg);
    t.normal_(0.f, 1.f);
    if (d == Device::CUDA) return t.cuda();
    return t;
}

Tensor Tensor::uniform(const std::vector<int>& shape, float lo, float hi, Device d, bool rg)
{
    Tensor t(shape, Device::CPU, rg);
    t.uniform_(lo, hi);
    if (d == Device::CUDA) return t.cuda();
    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Introspection
// ─────────────────────────────────────────────────────────────────────────────
const std::vector<int>& Tensor::shape()   const { check_defined("shape()");   return impl_->shape; }
const std::vector<int>& Tensor::strides() const { check_defined("strides()"); return impl_->strides; }
int    Tensor::ndim()          const { check_defined("ndim()");   return static_cast<int>(impl_->shape.size()); }
int    Tensor::numel()         const { check_defined("numel()");  return impl_->numel; }
Device Tensor::device()        const { check_defined("device()"); return impl_->device; }
bool   Tensor::requires_grad() const { check_defined("requires_grad()"); return impl_->requires_grad; }
void   Tensor::set_requires_grad(bool v) { check_defined("set_requires_grad()"); impl_->requires_grad = v; }

bool Tensor::is_contiguous() const
{
    check_defined("is_contiguous()");
    return impl_->strides == compute_strides(impl_->shape);
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw access
// ─────────────────────────────────────────────────────────────────────────────
float* Tensor::data_ptr()
{
    check_defined("data_ptr()");
    if (impl_->device != Device::CPU)
        throw std::runtime_error("data_ptr() called on a CUDA tensor; use cuda_ptr()");
    return impl_->storage->ptr + impl_->offset;
}

const float* Tensor::data_ptr() const
{
    check_defined("data_ptr()");
    if (impl_->device != Device::CPU)
        throw std::runtime_error("data_ptr() called on a CUDA tensor; use cuda_ptr()");
    return impl_->storage->ptr + impl_->offset;
}

float* Tensor::cuda_ptr()
{
    check_defined("cuda_ptr()");
    if (impl_->device != Device::CUDA)
        throw std::runtime_error("cuda_ptr() called on a CPU tensor; use data_ptr()");
    return impl_->storage->ptr + impl_->offset;
}

const float* Tensor::cuda_ptr() const
{
    check_defined("cuda_ptr()");
    if (impl_->device != Device::CUDA)
        throw std::runtime_error("cuda_ptr() called on a CPU tensor; use data_ptr()");
    return impl_->storage->ptr + impl_->offset;
}

// ─────────────────────────────────────────────────────────────────────────────
// Device transfer (result is a leaf on the target device)
// ─────────────────────────────────────────────────────────────────────────────
Tensor Tensor::to(Device target) const
{
    check_defined("to()");
    if (impl_->device == target) return *this;

    Tensor out(impl_->shape, target, impl_->requires_grad);
    size_t bytes = static_cast<size_t>(impl_->numel) * sizeof(float);
    const float* src = impl_->storage->ptr + impl_->offset;
    float* dst = out.impl_->storage->ptr;

#ifdef USE_CUDA
    if (impl_->device == Device::CPU && target == Device::CUDA)
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#else
    (void)bytes; (void)src; (void)dst;
    throw std::runtime_error("Built without CUDA support");
#endif
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Indexing
// ─────────────────────────────────────────────────────────────────────────────
int Tensor::flat_index(const std::vector<int>& idx) const
{
    if (idx.size() != impl_->shape.size())
        throw std::out_of_range("at(): index rank doesn't match tensor rank");
    int flat = static_cast<int>(impl_->offset);
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= impl_->shape[i])
            throw std::out_of_range("at(): index out of bounds");
        flat += idx[i] * impl_->strides[i];
    }
    return flat;
}

float Tensor::at(const std::vector<int>& idx) const
{
    check_defined("at()");
    if (impl_->device != Device::CPU)
        throw std::runtime_error("at() only supported for CPU tensors");
    return impl_->storage->ptr[flat_index(idx)];
}

float& Tensor::at(const std::vector<int>& idx)
{
    check_defined("at()");
    if (impl_->device != Device::CPU)
        throw std::runtime_error("at() only supported for CPU tensors");
    return impl_->storage->ptr[flat_index(idx)];
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape ops
// ─────────────────────────────────────────────────────────────────────────────
Tensor Tensor::reshape(const std::vector<int>& new_shape) const
{
    check_defined("reshape()");
    if (!is_contiguous())
        throw std::runtime_error("reshape() requires a contiguous tensor; call contiguous() first");

    // Resolve a single -1 dimension.
    int inferred = -1;
    long long known = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (inferred != -1) throw std::invalid_argument("reshape(): only one -1 allowed");
            inferred = static_cast<int>(i);
        } else {
            if (new_shape[i] <= 0) throw std::invalid_argument("reshape(): dims must be >= 1 (or -1)");
            known *= new_shape[i];
        }
    }
    std::vector<int> resolved = new_shape;
    if (inferred != -1) {
        if (known == 0 || impl_->numel % known != 0)
            throw std::invalid_argument("reshape(): cannot infer -1 dimension");
        resolved[inferred] = static_cast<int>(impl_->numel / known);
    }
    if (compute_numel(resolved) != impl_->numel)
        throw std::invalid_argument("reshape(): total number of elements must match");

    // Zero-copy view sharing storage.
    auto view = std::make_shared<TensorImpl>();
    view->storage = impl_->storage;
    view->offset  = impl_->offset;
    view->shape   = resolved;
    view->strides = compute_strides(resolved);
    view->numel   = impl_->numel;
    view->device  = impl_->device;

    Tensor out(view);
    if (GradMode::is_enabled() && impl_->requires_grad) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<ReshapeBackward>();
        fn->input_shape = impl_->shape;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::transpose(int dim0, int dim1) const
{
    check_defined("transpose()");
    if (impl_->device != Device::CPU)
        throw std::runtime_error("transpose(): CUDA path not implemented yet");
    int nd = ndim();
    if (dim0 < 0) dim0 += nd;
    if (dim1 < 0) dim1 += nd;
    if (dim0 < 0 || dim0 >= nd || dim1 < 0 || dim1 >= nd)
        throw std::out_of_range("transpose(): dim out of range");

    std::vector<int> out_shape = impl_->shape;
    std::swap(out_shape[dim0], out_shape[dim1]);

    // Materialize a contiguous copy. This keeps the "all tensors contiguous"
    // invariant so every raw-pointer kernel downstream stays correct — the
    // original returned a strided view that matmul() then read as if it were
    // contiguous, producing silently wrong gradients in MatMulBackward.
    Tensor out(out_shape, Device::CPU);
    const float* src = data_ptr();
    float* dst = out.data_ptr();

    if (nd == 2 && dim0 != dim1) {
        int R = impl_->shape[0], C = impl_->shape[1];
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                dst[static_cast<size_t>(c) * R + r] = src[static_cast<size_t>(r) * C + c];
    } else if (dim0 == dim1) {
        std::memcpy(dst, src, static_cast<size_t>(numel()) * sizeof(float));
    } else {
        const auto& in_strides  = impl_->strides;
        const auto  out_strides = compute_strides(out_shape);
        int n = numel();
        std::vector<int> coord(nd);
        for (int i = 0; i < n; ++i) {
            int rem = i;
            for (int d = 0; d < nd; ++d) { coord[d] = rem / out_strides[d]; rem %= out_strides[d]; }
            std::swap(coord[dim0], coord[dim1]);       // out coord -> in coord
            int in_off = 0;
            for (int d = 0; d < nd; ++d) in_off += coord[d] * in_strides[d];
            dst[i] = src[in_off];
        }
    }

    if (GradMode::is_enabled() && impl_->requires_grad) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<TransposeBackward>();
        fn->dim0 = dim0; fn->dim1 = dim1;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::squeeze(int dim) const
{
    check_defined("squeeze()");
    std::vector<int> new_shape;
    if (dim == -1) {
        for (int s : impl_->shape) if (s != 1) new_shape.push_back(s);
    } else {
        if (dim < 0) dim += ndim();
        if (dim < 0 || dim >= ndim()) throw std::out_of_range("squeeze(): dim out of range");
        for (int i = 0; i < ndim(); ++i)
            if (!(i == dim && impl_->shape[i] == 1)) new_shape.push_back(impl_->shape[i]);
    }
    if (new_shape.empty()) new_shape.push_back(1);
    return reshape(new_shape);
}

Tensor Tensor::unsqueeze(int dim) const
{
    check_defined("unsqueeze()");
    int nd = ndim();
    if (dim < 0) dim += nd + 1;
    if (dim < 0 || dim > nd) throw std::out_of_range("unsqueeze(): dim out of range");
    std::vector<int> new_shape = impl_->shape;
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}

Tensor Tensor::contiguous() const
{
    check_defined("contiguous()");
    if (is_contiguous()) return *this;
    return clone();
}

Tensor Tensor::clone() const
{
    check_defined("clone()");
    Tensor out(impl_->shape, impl_->device, /*requires_grad=*/false);
    size_t bytes = static_cast<size_t>(impl_->numel) * sizeof(float);
    if (impl_->device == Device::CPU) {
        if (is_contiguous()) {
            std::memcpy(out.impl()->storage->ptr, impl_->storage->ptr + impl_->offset, bytes);
        } else {
            // Element-wise copy through strides (handles any layout).
            int nd = ndim(), n = numel();
            const auto out_strides = compute_strides(impl_->shape);
            const float* src = impl_->storage->ptr + impl_->offset;
            float* dst = out.impl()->storage->ptr;
            for (int i = 0; i < n; ++i) {
                int rem = i, off = 0;
                for (int d = 0; d < nd; ++d) {
                    int c = rem / out_strides[d]; rem %= out_strides[d];
                    off += c * impl_->strides[d];
                }
                dst[i] = src[off];
            }
        }
    } else {
#ifdef USE_CUDA
        CUDA_CHECK(cudaMemcpy(out.impl()->storage->ptr,
                              impl_->storage->ptr + impl_->offset, bytes,
                              cudaMemcpyDeviceToDevice));
#endif
    }
    return out;
}

Tensor Tensor::detach() const
{
    check_defined("detach()");
    auto d = std::make_shared<TensorImpl>();
    d->storage = impl_->storage;             // shares data
    d->offset  = impl_->offset;
    d->shape   = impl_->shape;
    d->strides = impl_->strides;
    d->numel   = impl_->numel;
    d->device  = impl_->device;
    d->requires_grad = false;                // no grad, no grad_fn
    return Tensor(d);
}

// ─────────────────────────────────────────────────────────────────────────────
// Broadcasting machinery (NumPy/PyTorch rules)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> broadcast_shape(const std::vector<int>& a,
                                        const std::vector<int>& b,
                                        const char* opname)
{
    size_t n = std::max(a.size(), b.size());
    std::vector<int> out(n);
    for (size_t i = 0; i < n; ++i) {
        int ad = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
        int bd = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
        if (ad != bd && ad != 1 && bd != 1)
            throw std::invalid_argument(std::string(opname) + ": shapes are not broadcastable");
        out[i] = std::max(ad, bd);
    }
    return out;
}

// Strides of `t` padded to out_ndim, with 0 where the dimension broadcasts.
static std::vector<int> padded_bcast_strides(const Tensor& t, size_t out_ndim)
{
    std::vector<int> s(out_ndim, 0);
    size_t off = out_ndim - t.shape().size();
    for (size_t i = 0; i < t.shape().size(); ++i)
        s[off + i] = (t.shape()[i] == 1) ? 0 : t.strides()[i];
    return s;
}

template <class F>
static Tensor ew_binary(const Tensor& a, const Tensor& b, F f, const char* opname)
{
    if (a.device() != b.device())
        throw std::runtime_error(std::string(opname) + ": tensors are on different devices");
    if (a.device() != Device::CPU)
        throw std::runtime_error(std::string(opname) +
            ": element-wise binary ops are CPU-only for now (move tensors with .cpu())");

    // Fast path: identical shapes.
    if (a.shape() == b.shape()) {
        Tensor out(a.shape(), Device::CPU);
        const float* pa = a.data_ptr();
        const float* pb = b.data_ptr();
        float* po = out.data_ptr();
        int n = a.numel();
        for (int i = 0; i < n; ++i) po[i] = f(pa[i], pb[i]);
        return out;
    }

    // General broadcast path.
    auto oshape = broadcast_shape(a.shape(), b.shape(), opname);
    Tensor out(oshape, Device::CPU);
    size_t nd = oshape.size();
    std::vector<int> ost(nd, 1);
    for (int i = static_cast<int>(nd) - 2; i >= 0; --i) ost[i] = ost[i + 1] * oshape[i + 1];
    auto as = padded_bcast_strides(a, nd);
    auto bs = padded_bcast_strides(b, nd);

    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    float* po = out.data_ptr();
    int n = out.numel();
    for (int i = 0; i < n; ++i) {
        int rem = i, ao = 0, bo = 0;
        for (size_t d = 0; d < nd; ++d) {
            int c = rem / ost[d]; rem %= ost[d];
            ao += c * as[d]; bo += c * bs[d];
        }
        po[i] = f(pa[ao], pb[bo]);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary ops (autograd-aware)
// ─────────────────────────────────────────────────────────────────────────────
Tensor Tensor::operator+(const Tensor& other) const
{
    Tensor out = ew_binary(*this, other, [](float x, float y) { return x + y; }, "operator+");
    if (GradMode::is_enabled() && (requires_grad() || other.requires_grad())) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<AddBackward>();
        fn->inputs = { *this, other };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const
{
    Tensor out = ew_binary(*this, other, [](float x, float y) { return x - y; }, "operator-");
    if (GradMode::is_enabled() && (requires_grad() || other.requires_grad())) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<SubBackward>();
        fn->inputs = { *this, other };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    Tensor out = ew_binary(*this, other, [](float x, float y) { return x * y; }, "operator*");
    if (GradMode::is_enabled() && (requires_grad() || other.requires_grad())) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<MulBackward>();
        fn->saved_a = detach();
        fn->saved_b = other.detach();
        fn->inputs = { *this, other };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const
{
    Tensor out = ew_binary(*this, other, [](float x, float y) {
        if (y == 0.f) throw std::runtime_error("operator/: division by zero");
        return x / y;
    }, "operator/");
    if (GradMode::is_enabled() && (requires_grad() || other.requires_grad())) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<DivBackward>();
        fn->saved_a = detach();
        fn->saved_b = other.detach();
        fn->inputs = { *this, other };
        out.impl()->grad_fn = fn;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar ops.
// Autograd-aware — the original silently detached the graph on +, - and /
// with a float (only * had a backward).
// ─────────────────────────────────────────────────────────────────────────────
template <class F>
static Tensor ew_scalar_cpu(const Tensor& a, F f, const char* opname)
{
    if (a.device() != Device::CPU)
        throw std::runtime_error(std::string(opname) + ": CPU-only for now");
    Tensor out(a.shape(), Device::CPU);
    const float* pa = a.data_ptr();
    float* po = out.data_ptr();
    int n = a.numel();
    for (int i = 0; i < n; ++i) po[i] = f(pa[i]);
    return out;
}

Tensor Tensor::operator+(float s) const
{
    Tensor out = ew_scalar_cpu(*this, [s](float a) { return a + s; }, "operator+(float)");
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<AddScalarBackward>();
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::operator-(float s) const
{
    Tensor out = ew_scalar_cpu(*this, [s](float a) { return a - s; }, "operator-(float)");
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<AddScalarBackward>();   // d(a - s)/da = 1, same as add
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::operator*(float s) const
{
    Tensor out = ew_scalar_cpu(*this, [s](float a) { return a * s; }, "operator*(float)");
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<MulScalarBackward>();
        fn->scalar = s;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::operator/(float s) const
{
    if (s == 0.f) throw std::runtime_error("operator/(float): division by zero");
    Tensor out = ew_scalar_cpu(*this, [s](float a) { return a / s; }, "operator/(float)");
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<DivScalarBackward>();
        fn->scalar = s;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix multiply — C[M,N] = A[M,K] @ B[K,N]
// CPU: i-k-j loop order (unit-stride inner loop, auto-vectorizes) + OpenMP.
// ─────────────────────────────────────────────────────────────────────────────
Tensor Tensor::matmul(const Tensor& other) const
{
    check_defined("matmul()");
    if (device() != other.device())
        throw std::runtime_error("matmul: tensors are on different devices");
    if (ndim() != 2 || other.ndim() != 2)
        throw std::invalid_argument("matmul: only 2-D tensors supported");
    if (!is_contiguous() || !other.is_contiguous())
        throw std::runtime_error("matmul: inputs must be contiguous");

    int M = impl_->shape[0], K = impl_->shape[1];
    int K2 = other.impl()->shape[0], N = other.impl()->shape[1];
    if (K != K2)
        throw std::invalid_argument("matmul: inner dimensions don't match (" +
                                    shape_str() + " @ " + other.shape_str() + ")");

    Tensor out({M, N}, device());                   // zero-initialized

    if (device() == Device::CPU) {
        const float* A = data_ptr();
        const float* B = other.data_ptr();
        float* C = out.data_ptr();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            const float* arow = A + static_cast<size_t>(i) * K;
            float* crow = C + static_cast<size_t>(i) * N;
            for (int k = 0; k < K; ++k) {
                const float aik = arow[k];
                const float* brow = B + static_cast<size_t>(k) * N;
                for (int j = 0; j < N; ++j)
                    crow[j] += aik * brow[j];
            }
        }
    } else {
#ifdef USE_CUDA
        cuda_matmul(cuda_ptr(), other.cuda_ptr(), out.cuda_ptr(), M, K, N);
#else
        throw std::runtime_error("matmul: built without CUDA support");
#endif
    }

    if (GradMode::is_enabled() && (requires_grad() || other.requires_grad())) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<MatMulBackward>();
        fn->saved_a = detach();
        fn->saved_b = other.detach();
        fn->inputs = { *this, other };
        out.impl()->grad_fn = fn;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reductions
// ─────────────────────────────────────────────────────────────────────────────
Tensor Tensor::sum(int dim, bool keepdim) const
{
    check_defined("sum()");
    if (device() != Device::CPU)
        throw std::runtime_error("sum: CUDA path not implemented yet");

    if (dim == -1) {
        // Full reduction -> shape {1}. Double accumulator for accuracy.
        double total = 0.0;
        const float* p = data_ptr();
        for (int i = 0; i < impl_->numel; ++i) total += p[i];
        Tensor out({1}, Device::CPU);
        out.data_ptr()[0] = static_cast<float>(total);
        if (GradMode::is_enabled() && requires_grad()) {
            out.impl()->requires_grad = true;
            auto fn = std::make_shared<SumBackward>();
            fn->input_shape = impl_->shape;
            fn->dim = -1;
            fn->inputs = { *this };
            out.impl()->grad_fn = fn;
        }
        return out;
    }

    int d = dim < 0 ? dim + ndim() : dim;
    if (d < 0 || d >= ndim()) throw std::out_of_range("sum: dim out of range");

    int outer = 1, inner = 1, reduce = impl_->shape[d];
    for (int i = 0; i < d; ++i)          outer *= impl_->shape[i];
    for (int i = d + 1; i < ndim(); ++i) inner *= impl_->shape[i];   // fixed: original had `dim + i` (UB)

    std::vector<int> out_shape;
    for (int i = 0; i < ndim(); ++i) {
        if (i == d) { if (keepdim) out_shape.push_back(1); }
        else          out_shape.push_back(impl_->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor out(out_shape, Device::CPU);
    const float* src = data_ptr();
    float* dst = out.data_ptr();

    for (int o = 0; o < outer; ++o)
        for (int i = 0; i < inner; ++i) {
            double acc = 0.0;
            for (int r = 0; r < reduce; ++r)
                acc += src[(static_cast<size_t>(o) * reduce + r) * inner + i];
            dst[static_cast<size_t>(o) * inner + i] = static_cast<float>(acc);
        }

    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<SumBackward>();
        fn->input_shape = impl_->shape;
        fn->dim = d;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::mean(int dim, bool keepdim) const
{
    check_defined("mean()");
    Tensor s = sum(dim, keepdim);
    int divisor = (dim == -1) ? numel() : shape()[dim < 0 ? dim + ndim() : dim];
    return s * (1.0f / static_cast<float>(divisor));   // MulScalarBackward keeps grads flowing
}

// ─────────────────────────────────────────────────────────────────────────────
// Activations & pointwise math
// ─────────────────────────────────────────────────────────────────────────────
Tensor Tensor::relu() const
{
    check_defined("relu()");
    Tensor out(impl_->shape, device());
    if (device() == Device::CPU) {
        const float* p = data_ptr();
        float* q = out.data_ptr();
        for (int i = 0; i < impl_->numel; ++i) q[i] = p[i] > 0.f ? p[i] : 0.f;
    } else {
#ifdef USE_CUDA
        cuda_relu(cuda_ptr(), out.cuda_ptr(), impl_->numel);
#else
        throw std::runtime_error("relu: built without CUDA support");
#endif
    }
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<ReLUBackward>();
        fn->saved_input = detach();
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::sigmoid() const
{
    check_defined("sigmoid()");
    Tensor out(impl_->shape, device());
    if (device() == Device::CPU) {
        const float* p = data_ptr();
        float* q = out.data_ptr();
        for (int i = 0; i < impl_->numel; ++i) q[i] = 1.f / (1.f + std::exp(-p[i]));
    } else {
#ifdef USE_CUDA
        cuda_sigmoid(cuda_ptr(), out.cuda_ptr(), impl_->numel);  // fixed: original launched cuda_relu here
#else
        throw std::runtime_error("sigmoid: built without CUDA support");
#endif
    }
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<SigmoidBackward>();
        fn->saved_output = out.detach();
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::tanh() const
{
    check_defined("tanh()");
    Tensor out(impl_->shape, device());
    if (device() == Device::CPU) {
        const float* p = data_ptr();
        float* q = out.data_ptr();
        for (int i = 0; i < impl_->numel; ++i) q[i] = std::tanh(p[i]);
    } else {
#ifdef USE_CUDA
        cuda_tanh_act(cuda_ptr(), out.cuda_ptr(), impl_->numel);
#else
        throw std::runtime_error("tanh: built without CUDA support");
#endif
    }
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<TanhBackward>();
        fn->saved_output = out.detach();
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

// Shared iteration pattern for softmax/log_softmax over an arbitrary dim.
// Calls fn(base, stride, size) for every 1-D slice along `dim`.
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

Tensor Tensor::softmax(int dim) const
{
    check_defined("softmax()");
    if (device() != Device::CPU)
        throw std::runtime_error("softmax: CUDA path not implemented yet");
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) throw std::out_of_range("softmax: dim out of range");

    Tensor out(impl_->shape, Device::CPU);
    const float* src = data_ptr();
    float* dst = out.data_ptr();

    for_each_slice(impl_->shape, dim, [&](size_t base, size_t stride, int size) {
        float mx = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < size; ++s) mx = std::max(mx, src[base + s * stride]);
        float sum = 0.f;
        for (int s = 0; s < size; ++s) {
            float v = std::exp(src[base + s * stride] - mx);
            dst[base + s * stride] = v;
            sum += v;
        }
        for (int s = 0; s < size; ++s) dst[base + s * stride] /= sum;
    });

    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<SoftmaxBackward>();
        fn->saved_output = out.detach();
        fn->dim = dim;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::log_softmax(int dim) const
{
    check_defined("log_softmax()");
    if (device() != Device::CPU)
        throw std::runtime_error("log_softmax: CUDA path not implemented yet");
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) throw std::out_of_range("log_softmax: dim out of range");

    Tensor out(impl_->shape, Device::CPU);
    const float* src = data_ptr();
    float* dst = out.data_ptr();

    for_each_slice(impl_->shape, dim, [&](size_t base, size_t stride, int size) {
        float mx = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < size; ++s) mx = std::max(mx, src[base + s * stride]);
        float sum = 0.f;
        for (int s = 0; s < size; ++s) sum += std::exp(src[base + s * stride] - mx);
        float lse = mx + std::log(sum);
        for (int s = 0; s < size; ++s) dst[base + s * stride] = src[base + s * stride] - lse;
    });

    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<LogSoftmaxBackward>();
        fn->saved_output = out.detach();
        fn->dim = dim;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::exp() const
{
    check_defined("exp()");
    if (device() != Device::CPU) throw std::runtime_error("exp: CUDA path not implemented yet");
    Tensor out(impl_->shape, Device::CPU);
    const float* p = data_ptr();
    float* q = out.data_ptr();
    for (int i = 0; i < impl_->numel; ++i) q[i] = std::exp(p[i]);
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<ExpBackward>();
        fn->saved_output = out.detach();
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::log() const
{
    check_defined("log()");
    if (device() != Device::CPU) throw std::runtime_error("log: CUDA path not implemented yet");
    Tensor out(impl_->shape, Device::CPU);
    const float* p = data_ptr();
    float* q = out.data_ptr();
    for (int i = 0; i < impl_->numel; ++i) {
        if (p[i] <= 0.f) throw std::domain_error("log: input must be > 0");
        q[i] = std::log(p[i]);
    }
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<LogBackward>();
        fn->saved_input = detach();
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::pow(float e) const
{
    check_defined("pow()");
    if (device() != Device::CPU) throw std::runtime_error("pow: CUDA path not implemented yet");
    Tensor out(impl_->shape, Device::CPU);
    const float* p = data_ptr();
    float* q = out.data_ptr();
    for (int i = 0; i < impl_->numel; ++i) q[i] = std::pow(p[i], e);
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<PowBackward>();
        fn->saved_input = detach();
        fn->exponent = e;
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

Tensor Tensor::sqrt() const
{
    check_defined("sqrt()");
    if (device() != Device::CPU) throw std::runtime_error("sqrt: CUDA path not implemented yet");
    Tensor out(impl_->shape, Device::CPU);
    const float* p = data_ptr();
    float* q = out.data_ptr();
    for (int i = 0; i < impl_->numel; ++i) {
        if (p[i] < 0.f) throw std::domain_error("sqrt: input must be >= 0");
        q[i] = std::sqrt(p[i]);
    }
    if (GradMode::is_enabled() && requires_grad()) {
        out.impl()->requires_grad = true;
        auto fn = std::make_shared<SqrtBackward>();
        fn->saved_output = out.detach();
        fn->inputs = { *this };
        out.impl()->grad_fn = fn;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// In-place ops (for parameters/buffers; don't apply to graph intermediates)
// ─────────────────────────────────────────────────────────────────────────────
void Tensor::zero_()
{
    check_defined("zero_()");
    if (device() == Device::CPU) {
        std::memset(impl_->storage->ptr + impl_->offset, 0, impl_->numel * sizeof(float));
    } else {
#ifdef USE_CUDA
        CUDA_CHECK(cudaMemset(impl_->storage->ptr + impl_->offset, 0, impl_->numel * sizeof(float)));
#endif
    }
}

void Tensor::fill_(float value)
{
    check_defined("fill_()");
    if (device() == Device::CPU) {
        float* p = data_ptr();
        std::fill(p, p + impl_->numel, value);
    } else {
#ifdef USE_CUDA
        cuda_fill(cuda_ptr(), value, impl_->numel);
#else
        throw std::runtime_error("fill_: built without CUDA support");
#endif
    }
}

void Tensor::add_(const Tensor& other, float alpha)
{
    check_defined("add_()");
    if (!other.defined() || other.numel() != numel())
        throw std::invalid_argument("add_: size mismatch");
    if (device() != other.device())
        throw std::invalid_argument("add_: tensors on different devices");
    if (device() == Device::CPU) {
        float* a = data_ptr();
        const float* b = other.data_ptr();
        for (int i = 0; i < impl_->numel; ++i) a[i] += alpha * b[i];
    } else {
#ifdef USE_CUDA
        cuda_axpy(cuda_ptr(), other.cuda_ptr(), alpha, impl_->numel);
#else
        throw std::runtime_error("add_: built without CUDA support");
#endif
    }
}

void Tensor::add_(float scalar)
{
    check_defined("add_()");
    if (device() == Device::CPU) {
        float* p = data_ptr();
        for (int i = 0; i < impl_->numel; ++i) p[i] += scalar;
    } else {
#ifdef USE_CUDA
        cuda_add_scalar(cuda_ptr(), scalar, impl_->numel);
#else
        throw std::runtime_error("add_: built without CUDA support");
#endif
    }
}

void Tensor::mul_(float scalar)
{
    check_defined("mul_()");
    if (device() == Device::CPU) {
        float* p = data_ptr();
        for (int i = 0; i < impl_->numel; ++i) p[i] *= scalar;
    } else {
#ifdef USE_CUDA
        cuda_mul_scalar(cuda_ptr(), scalar, impl_->numel);
#else
        throw std::runtime_error("mul_: built without CUDA support");
#endif
    }
}

void Tensor::uniform_(float lo, float hi)
{
    check_defined("uniform_()");
    if (device() != Device::CPU) throw std::runtime_error("uniform_: CPU tensors only");
    std::uniform_real_distribution<float> dist(lo, hi);
    float* p = data_ptr();
    for (int i = 0; i < impl_->numel; ++i) p[i] = dist(g_rng);
}

void Tensor::normal_(float mean, float stddev)
{
    check_defined("normal_()");
    if (device() != Device::CPU) throw std::runtime_error("normal_: CPU tensors only");
    std::normal_distribution<float> dist(mean, stddev);
    float* p = data_ptr();
    for (int i = 0; i < impl_->numel; ++i) p[i] = dist(g_rng);
}

// ─────────────────────────────────────────────────────────────────────────────
// Autograd plumbing
// ─────────────────────────────────────────────────────────────────────────────
Tensor Tensor::grad() const
{
    check_defined("grad()");
    if (!impl_->grad) return Tensor();               // undefined
    return Tensor(impl_->grad);
}

std::shared_ptr<GradFn> Tensor::grad_fn() const
{
    check_defined("grad_fn()");
    return impl_->grad_fn;
}

void Tensor::accumulate_grad(const Tensor& grad_update)
{
    check_defined("accumulate_grad()");
    if (!grad_update.defined())
        throw std::invalid_argument("accumulate_grad: undefined gradient");
    if (grad_update.numel() != impl_->numel)
        throw std::invalid_argument("accumulate_grad: shape mismatch (tensor " + shape_str() +
                                    " vs grad " + grad_update.shape_str() + ")");
    if (!impl_->grad) {
        Tensor g(impl_->shape, impl_->device);       // zero-initialized
        impl_->grad = g.impl();
    }
    Tensor g(impl_->grad);
    g.add_(grad_update);
}

void Tensor::zero_grad()
{
    check_defined("zero_grad()");
    if (impl_->grad) Tensor(impl_->grad).zero_();
}

void Tensor::backward()
{
    check_defined("backward()");
    AutogradEngine::backward(*this);
}

void Tensor::backward(const Tensor& grad_seed)
{
    check_defined("backward()");
    AutogradEngine::backward(*this, grad_seed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────
std::string Tensor::shape_str() const
{
    if (!impl_) return "[undefined]";
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < impl_->shape.size(); ++i) {
        if (i) oss << ", ";
        oss << impl_->shape[i];
    }
    oss << "]";
    return oss.str();
}

void Tensor::print(const std::string& name) const
{
    if (!name.empty()) std::cout << name << " ";
    if (!impl_) { std::cout << "Tensor(undefined)\n"; return; }
    std::cout << "Tensor(shape=" << shape_str()
              << ", device=" << (impl_->device == Device::CPU ? "CPU" : "CUDA")
              << ", requires_grad=" << (impl_->requires_grad ? "true" : "false")
              << ")\n";
    if (impl_->device != Device::CPU) {
        std::cout << "  [data on GPU — call .cpu().print() to inspect]\n";
        return;
    }
    const float* p = impl_->storage->ptr + impl_->offset;
    if (ndim() == 1) {
        std::cout << "  [ ";
        for (int i = 0; i < std::min(numel(), 8); ++i)
            std::cout << std::setw(9) << std::setprecision(4) << p[i * impl_->strides[0]] << " ";
        if (numel() > 8) std::cout << "...";
        std::cout << "]\n";
    } else if (ndim() == 2) {
        int rows = std::min(impl_->shape[0], 4), cols = std::min(impl_->shape[1], 8);
        for (int r = 0; r < rows; ++r) {
            std::cout << "  [ ";
            for (int c = 0; c < cols; ++c)
                std::cout << std::setw(9) << std::setprecision(4)
                          << p[r * impl_->strides[0] + c * impl_->strides[1]] << " ";
            if (impl_->shape[1] > 8) std::cout << "...";
            std::cout << "]\n";
        }
        if (impl_->shape[0] > 4) std::cout << "  ...\n";
    } else {
        std::cout << "  (ndim > 2: use at() to inspect individual elements)\n";
    }
}
