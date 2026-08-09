//
// cuda/elementwise.cu   (replaces the old cuda/relu.cu)
//
// Elementwise kernels for activations and simple math, exposed as plain-C.
//
// Kernels provided:
//   cuda_relu          — max(x, 0)
//   cuda_relu_backward — grad * (x > 0)
//   cuda_sigmoid       — 1 / (1 + exp(-x))
//   cuda_tanh_act      — tanh(x)
//   cuda_add_scalar    — x += s
//   cuda_mul_scalar    — x *= s
//   cuda_axpy          — y += alpha * x        (new: powers Tensor::add_)
//   cuda_fill          — x = value             (new: powers Tensor::fill_)
//

#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <string>

#define CU(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA elementwise error: ") + \
                                     cudaGetErrorString(_e));                 \
    } while (0)

static inline dim3 make_grid(int n, int block = 256)
{
    return dim3((n + block - 1) / block);
}

// ── ReLU ─────────────────────────────────────────────────────────────────────
__global__ void relu_kernel(const float* __restrict__ x,
                            float*       __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

extern "C" void cuda_relu(const float* x, float* y, int n)
{
    relu_kernel<<<make_grid(n), 256>>>(x, y, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}

// ── ReLU backward: grad_in = grad_out * (x > 0) ─────────────────────────────
__global__ void relu_backward_kernel(const float* __restrict__ grad_out,
                                     const float* __restrict__ x,
                                     float*       __restrict__ grad_in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = (x[i] > 0.0f) ? grad_out[i] : 0.0f;
}

extern "C" void cuda_relu_backward(const float* grad_out, const float* x,
                                   float* grad_in, int n)
{
    relu_backward_kernel<<<make_grid(n), 256>>>(grad_out, x, grad_in, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────
__global__ void sigmoid_kernel(const float* __restrict__ x,
                               float*       __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

extern "C" void cuda_sigmoid(const float* x, float* y, int n)
{
    sigmoid_kernel<<<make_grid(n), 256>>>(x, y, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}

// ── Tanh ─────────────────────────────────────────────────────────────────────
__global__ void tanh_kernel(const float* __restrict__ x,
                            float*       __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

extern "C" void cuda_tanh_act(const float* x, float* y, int n)
{
    tanh_kernel<<<make_grid(n), 256>>>(x, y, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}

// ── Scalar ops ───────────────────────────────────────────────────────────────
__global__ void add_scalar_kernel(float* __restrict__ x, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += s;
}

extern "C" void cuda_add_scalar(float* x, float s, int n)
{
    add_scalar_kernel<<<make_grid(n), 256>>>(x, s, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}

__global__ void mul_scalar_kernel(float* __restrict__ x, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

extern "C" void cuda_mul_scalar(float* x, float s, int n)
{
    mul_scalar_kernel<<<make_grid(n), 256>>>(x, s, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}

// ── axpy: y += alpha * x ─────────────────────────────────────────────────────
__global__ void axpy_kernel(float* __restrict__ y, const float* __restrict__ x,
                            float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

extern "C" void cuda_axpy(float* y, const float* x, float alpha, int n)
{
    axpy_kernel<<<make_grid(n), 256>>>(y, x, alpha, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}

// ── fill ─────────────────────────────────────────────────────────────────────
__global__ void fill_kernel(float* __restrict__ x, float value, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = value;
}

extern "C" void cuda_fill(float* x, float value, int n)
{
    fill_kernel<<<make_grid(n), 256>>>(x, value, n);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
}
