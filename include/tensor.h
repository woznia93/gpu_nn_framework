#pragma once
//
// tensor.h
//
// Tensor is a *handle* (cheap to copy) over a shared TensorImpl, exactly like
// PyTorch's Tensor/TensorImpl split. This is the key design decision that
// makes autograd correct:
//
//   * Copying a Tensor copies a pointer, not the data. Every consumer of a
//     tensor therefore refers to the SAME graph node, so gradients from
//     multiple consumers accumulate on one object (required for fan-out /
//     diamond graphs).
//   * GradFn::inputs holds owning Tensor handles, so the graph keeps all of
//     its nodes alive. No non-owning aliases, no dangling pointers.
//   * Use clone() when you actually want a deep copy of the data.
//
// Invariant: every tensor produced by an op is contiguous (row-major).
// reshape() returns a zero-copy view (safe: layout is identical);
// transpose() returns a materialized contiguous copy.
//

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            throw std::runtime_error(                                          \
                std::string("CUDA error: ") + cudaGetErrorString(_err) +       \
                " at " __FILE__ ":" + std::to_string(__LINE__));               \
        }                                                                      \
    } while (0)
#endif

enum class Device { CPU, CUDA };

struct GradFn;

// ─────────────────────────────────────────────────────────────────────────────
// Storage — raw memory owner (one per allocation, shared by views)
// ─────────────────────────────────────────────────────────────────────────────
struct Storage {
    float* ptr    = nullptr;
    size_t numel  = 0;
    Device device = Device::CPU;

    Storage() = default;
    Storage(size_t n, Device dev);          // zero-initialized
    ~Storage();
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;
};

// ─────────────────────────────────────────────────────────────────────────────
// TensorImpl — the actual graph node (shape + storage + autograd metadata)
// ─────────────────────────────────────────────────────────────────────────────
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    size_t offset = 0;                       // element offset into storage

    std::vector<int> shape;
    std::vector<int> strides;                // in elements, row-major
    int    numel         = 0;
    Device device        = Device::CPU;
    bool   requires_grad = false;

    std::shared_ptr<TensorImpl> grad;        // lazily allocated, same shape
    std::shared_ptr<GradFn>     grad_fn;     // null for leaves
};

// ─────────────────────────────────────────────────────────────────────────────
// Grad mode — RAII switch to disable graph construction (inference)
// ─────────────────────────────────────────────────────────────────────────────
struct GradMode {
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

struct NoGradGuard {
    NoGradGuard()  : prev_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
    ~NoGradGuard() { GradMode::set_enabled(prev_); }
private:
    bool prev_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — user-facing handle
// ─────────────────────────────────────────────────────────────────────────────
class Tensor {
public:
    // An undefined tensor (impl == nullptr). defined() is false.
    Tensor() = default;

    explicit Tensor(const std::vector<int>& shape,
                    Device device        = Device::CPU,
                    bool   requires_grad = false);

    Tensor(const std::vector<float>& data,
           const std::vector<int>&   shape,
           Device device        = Device::CPU,
           bool   requires_grad = false);

    // Wrap an existing impl (used by autograd internals)
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

    // Copies/moves share the impl — this is intentional (handle semantics).

    // ── Factories ────────────────────────────────────────────────────────────
    static Tensor zeros  (const std::vector<int>& shape, Device d = Device::CPU, bool rg = false);
    static Tensor ones   (const std::vector<int>& shape, Device d = Device::CPU, bool rg = false);
    static Tensor full   (const std::vector<int>& shape, float value, Device d = Device::CPU, bool rg = false);
    static Tensor randn  (const std::vector<int>& shape, Device d = Device::CPU, bool rg = false);
    static Tensor uniform(const std::vector<int>& shape, float lo, float hi, Device d = Device::CPU, bool rg = false);

    // ── Introspection ────────────────────────────────────────────────────────
    bool defined() const { return impl_ != nullptr; }
    const std::vector<int>& shape()   const;
    const std::vector<int>& strides() const;
    int    ndim()   const;
    int    numel()  const;
    Device device() const;
    bool   requires_grad() const;
    void   set_requires_grad(bool val);
    bool   is_contiguous() const;
    bool   same_impl(const Tensor& other) const { return impl_ == other.impl_; }

    // ── Raw access ───────────────────────────────────────────────────────────
    float*       data_ptr();
    const float* data_ptr() const;
    float*       cuda_ptr();
    const float* cuda_ptr() const;

    // ── Device transfer ──────────────────────────────────────────────────────
    Tensor to(Device target) const;
    Tensor cpu()  const { return to(Device::CPU);  }
    Tensor cuda() const { return to(Device::CUDA); }

    // ── Indexing (CPU) ───────────────────────────────────────────────────────
    float  at(const std::vector<int>& idx) const;
    float& at(const std::vector<int>& idx);

    // ── Shape ops ────────────────────────────────────────────────────────────
    Tensor reshape(const std::vector<int>& new_shape) const;  // zero-copy view
    Tensor transpose(int dim0, int dim1) const;               // materialized copy
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor contiguous() const;
    Tensor clone()  const;   // deep copy, detached from the graph
    Tensor detach() const;   // shares data, requires_grad = false, no grad_fn

    // ── Math (autograd-aware; +,-,*,/ broadcast like NumPy/PyTorch) ─────────
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float s) const;
    Tensor operator-(float s) const;
    Tensor operator*(float s) const;
    Tensor operator/(float s) const;
    Tensor operator-() const { return (*this) * (-1.0f); }

    Tensor matmul(const Tensor& other) const;         // 2-D only
    Tensor sum (int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;

    Tensor relu()        const;
    Tensor sigmoid()     const;
    Tensor tanh()        const;
    Tensor softmax    (int dim = -1) const;
    Tensor log_softmax(int dim = -1) const;
    Tensor exp()         const;
    Tensor log()         const;
    Tensor pow(float e)  const;
    Tensor sqrt()        const;

    // ── In-place ops (leaf/optimizer use; do not apply to graph internals) ──
    void zero_();
    void fill_(float value);
    void add_(const Tensor& other, float alpha = 1.0f);   // this += alpha * other
    void add_(float scalar);
    void mul_(float scalar);
    void uniform_(float lo, float hi);
    void normal_(float mean = 0.f, float stddev = 1.f);

    // ── Autograd ─────────────────────────────────────────────────────────────
    Tensor grad() const;                       // undefined if no grad yet
    std::shared_ptr<GradFn> grad_fn() const;
    void accumulate_grad(const Tensor& grad_update);
    void zero_grad();                          // zero (keep buffer) if present
    void backward();                           // scalar outputs only
    void backward(const Tensor& grad_seed);    // explicit vector-Jacobian seed

    // ── Utilities ────────────────────────────────────────────────────────────
    void        print(const std::string& name = "") const;
    std::string shape_str() const;

    std::shared_ptr<TensorImpl>&       impl()       { return impl_; }
    const std::shared_ptr<TensorImpl>& impl() const { return impl_; }

private:
    std::shared_ptr<TensorImpl> impl_;

    static std::vector<int> compute_strides(const std::vector<int>& shape);
    static int              compute_numel  (const std::vector<int>& shape);
    static std::shared_ptr<TensorImpl> make_impl(const std::vector<int>& shape,
                                                 Device device, bool requires_grad);
    int  flat_index(const std::vector<int>& idx) const;
    void check_defined(const char* what) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// GradFn — a node's backward function.
//
// apply() must ONLY compute the gradients of `inputs` w.r.t. grad_output and
// accumulate them (Tensor::accumulate_grad). It must NOT recurse into the
// graph — the AutogradEngine drives propagation in topological order so that
// each node fires exactly once with its fully accumulated upstream gradient.
// ─────────────────────────────────────────────────────────────────────────────
struct GradFn {
    std::vector<Tensor> inputs;   // owning handles — keep the graph alive
    virtual void apply(const Tensor& grad_output) = 0;
    virtual const char* name() const { return "GradFn"; }
    virtual ~GradFn() = default;
};

// ─────────────────────────────────────────────────────────────────────────────
// Free operators & RNG
// ─────────────────────────────────────────────────────────────────────────────
inline Tensor operator+(float s, const Tensor& t) { return t + s; }
inline Tensor operator*(float s, const Tensor& t) { return t * s; }
inline Tensor operator-(float s, const Tensor& t) { return (t * -1.0f) + s; }

// Global RNG shared by randn/uniform/dropout. manual_seed() makes runs
// reproducible (like torch.manual_seed).
std::mt19937& global_rng();
void manual_seed(uint64_t seed);
