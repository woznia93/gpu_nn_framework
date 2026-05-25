#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(_err) +       \
                " at " __FILE__ ":" + std::to_string(__LINE__));               \
        }                                                                       \
    } while (0)
#else
#define CUDA_CHECK(call) (void)0
#endif

enum class Device { CPU, CUDA };

struct GradFn;


class Tensor {
public:
    explicit Tensor(const std::vector<int>& shape,
                    Device device        = Device::CPU,
                    bool   requires_grad = false);

    Tensor(const std::vector<float>& data,
           const std::vector<int>&   shape,
           Device device        = Device::CPU,
           bool   requires_grad = false);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    static Tensor zeros(const std::vector<int>& shape,
                        Device device        = Device::CPU,
                        bool   requires_grad = false);
    static Tensor ones (const std::vector<int>& shape,
                        Device device        = Device::CPU,
                        bool   requires_grad = false);
    static Tensor randn(const std::vector<int>& shape,
                        Device device        = Device::CPU,
                        bool   requires_grad = false);

    const std::vector<int>& shape()   const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int    ndim()   const { return static_cast<int>(shape_.size()); }
    int    numel()  const { return numel_; }
    Device device() const { return device_; }
    bool   requires_grad() const { return requires_grad_; }
    void   set_requires_grad(bool val) { requires_grad_ = val; }

    float*       data_ptr();
    const float* data_ptr() const;
    float*       cuda_ptr();
    const float* cuda_ptr() const;

    Tensor to(Device target) const;
    Tensor cpu()  const { return to(Device::CPU);  }
    Tensor cuda() const { return to(Device::CUDA); }

    float  at(const std::vector<int>& idx) const;
    float& at(const std::vector<int>& idx);

    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float s) const;
    Tensor operator-(float s) const;
    Tensor operator*(float s) const;
    Tensor operator/(float s) const;

    Tensor matmul(const Tensor& other) const;
    Tensor sum (int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor relu()            const;
    Tensor sigmoid()         const;
    Tensor tanh()            const;
    Tensor softmax(int dim = -1) const;
    Tensor log()             const;
    Tensor exp()             const;
    Tensor pow(float e)      const;
    Tensor sqrt()            const;

    void zero_();
    void fill_(float value);
    void add_(const Tensor& other);
    void add_(float scalar);
    void mul_(float scalar);

    std::shared_ptr<Tensor> grad;
    std::shared_ptr<GradFn> grad_fn;

    void   accumulate_grad(const Tensor& grad_update);
    void   backward();
    void   backward(const Tensor& upstream_grad);
    Tensor detach() const;

    void        print(const std::string& name = "") const;
    std::string shape_str() const;

    // public so GradFn structs with Tensor members can default-construct
    Tensor() = default;

private:
    struct Storage {
        float*  ptr    = nullptr;
        size_t  bytes  = 0;
        Device  device = Device::CPU;

        Storage() = default;
        Storage(size_t bytes, Device device);
        ~Storage();
        Storage(const Storage&)            = delete;
        Storage& operator=(const Storage&) = delete;
    };

    std::shared_ptr<Storage> storage_;
    size_t offset_ = 0;

    std::vector<int> shape_;
    std::vector<int> strides_;
    int    numel_         = 0;
    Device device_        = Device::CPU;
    bool   requires_grad_ = false;
    bool   is_view_       = false;

    static std::vector<int> compute_strides(const std::vector<int>& shape);
    static int              compute_numel  (const std::vector<int>& shape);
    int  flat_index(const std::vector<int>& idx) const;
    void allocate();
    void copy_from(const Tensor& src);
};

// ── GradFn base ───────────────────────────────────────────────────────────────
struct GradFn {
    std::vector<std::shared_ptr<Tensor>> inputs;
    virtual void backward(const Tensor& grad_output) = 0;
    virtual ~GradFn() = default;
};

// ── All GradFn subclasses live here so both tensor.cpp and autograd.cpp see them
struct AddBackward : GradFn {
    void backward(const Tensor& grad_output) override;
};
struct SubBackward : GradFn {
    void backward(const Tensor& grad_output) override;
};
struct MulBackward : GradFn {
    Tensor saved_a, saved_b;
    void backward(const Tensor& grad_output) override;
};
struct MulScalarBackward : GradFn {
    float scalar = 0.f;
    void backward(const Tensor& grad_output) override;
};
struct DivBackward : GradFn {
    Tensor saved_a, saved_b;
    void backward(const Tensor& grad_output) override;
};
struct MatMulBackward : GradFn {
    Tensor saved_a, saved_b;
    void backward(const Tensor& grad_output) override;
};
struct ReLUBackward : GradFn {
    Tensor saved_input;
    void backward(const Tensor& grad_output) override;
};
struct SigmoidBackward : GradFn {
    Tensor saved_output;
    void backward(const Tensor& grad_output) override;
};
struct TanhBackward : GradFn {
    Tensor saved_output;
    void backward(const Tensor& grad_output) override;
};
struct ExpBackward : GradFn {
    Tensor saved_output;
    void backward(const Tensor& grad_output) override;
};
struct LogBackward : GradFn {
    Tensor saved_input;
    void backward(const Tensor& grad_output) override;
};
struct PowBackward : GradFn {
    Tensor saved_input;
    float  exponent = 1.f;
    void backward(const Tensor& grad_output) override;
};
struct SumBackward : GradFn {
    std::vector<int> input_shape;
    int  dim     = -1;
    bool keepdim = false;
    void backward(const Tensor& grad_output) override;
};
struct MSEBackward : GradFn {
    Tensor saved_pred, saved_target;
    void backward(const Tensor& grad_output) override;
};
struct CrossEntropyBackward : GradFn {
    Tensor saved_softmax, saved_target;
    void backward(const Tensor& grad_output) override;
};

inline Tensor operator+(float s, const Tensor& t) { return t + s; }
inline Tensor operator*(float s, const Tensor& t) { return t * s; }
