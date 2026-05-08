#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef USE_CUDA
extern "C" void cuda_matmul(const float* A, const float* B, float* C,
                             int M, int K, int N);
extern "C" void cuda_relu    (const float* x, float* y, int n);
extern "C" void cuda_sigmoid (const float* x, float* y, int n);
extern "C" void cuda_tanh_act(const float* x, float* y, int n);
#endif

//
// Storage - raw memory owner
//

Tensor::Storage::Storage(size_t bytes, Device dev) : bytes(bytes), device(dev)
{
	if (bytes == 0) return;
	if (dev == Device::CPU) {
		ptr = new float[bytes / sizeof(float)]();	// zero-initialized
	} else {
		CUDA_CHECK(cudaMalloc(&ptr, bytes));
		CUDA_CHECK(cudaMemset(ptr, 0, bytes));
	}
}

Tensor::Storage::~Storage()
{
	if (!ptr) return;
	if (device == Device::CPU) {
		delete[] ptr;
	} else {
		cudaFree(ptr);
	}
	ptr = nullptr;
}


//
// Static helpers
//

/*static*/
int Tensor::compute_numel(const std::vector<int>& shape)
{
	if (shape.empty()) return 0;
	return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

/*static*/
std::vector<int> Tensor::compute_strides(const std::vector<int>& shape)
{
	// Row major (C-contiguous): last dim has stride 1
	// e.g. shape [2,3,4] -> strides [12,4,1]
	std::vector<int> strides(shape.size(),1);
	for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
		strides[i] = strides[i + 1] * shape[i + 1];
	}
	return strides;
}

int Tensor::flat_index(const std::vector<int>& idx) const
{
	if (idx.size() != shape_.size())
		throw std::out_of_range("Index rank doesn't match tensor rank");
	int flat = static_cast<int>(offset_);
	for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
		if (idx[i] < 0 || idx[i] >= shape_[i])
			throw std::out_of_range("Index out of bounds");
		flat += idx[i] * strides_[i];
	}
	return flat;
}


//
// Allocate Helper
//

void Tensor::allocate()
{
	size_t bytes = static_cast<size_t>(numel_) * sizeof(float);
	storage_ = std::make_shared<Storage>(bytes, device_);
	offset_ = 0;
}


//
// Constructors
//

Tensor::Tensor(const std::vector<int>& shape, Device device, bool requires_grad) 
	: shape_(shape),
	strides_(compute_strides(shape)),
	numel_(compute_numel(shape)),
	device_(device),
	requires_grad_(requires_grad)
{
	allocate();
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, Device device, bool requires_grad)
	: shape_(shape),
	strides_(compute_strides(shape)),
	numel_(compute_numel(shape)),
	device_(device),
	requires_grad_(requires_grad)
{
	if (static_cast<int>(data.size()) != numel_)	
		throw std::invalid_argument( 
				"Data size does not match shape: got " + std::to_string(data.size()) + 
				" elements for shape " + Tensor(shape, Device::CPU).shape_str());
	allocate();
	if (device == Device::CPU) {
		std::memcpy(storage_->ptr + offset_, data.data(), numel_ * sizeof(float));
	} else {
		// Copy host vector -> device
		CUDA_CHECK(cudaMemcpy(storage_->ptr + offset_, data.data(), numel_ * sizeof(float), cudaMemcpyHostToDevice));
	}
}


//
// Copy constructor - deep copy
//

Tensor::Tensor(const Tensor& other) 
	: shape_(other.shape_),
	strides_(other.strides_),
	numel_(other.numel_),
	device_(other.device_), 
	requires_grad_(other.requires_grad_),
	offset(0),
	is_view_(false)
{
	allocate();
	copy_from(other);
}

Tensor& Tensor::operator=(const Tensor& other)
{
	if (this == &other) return *this;
	shape_ = other.shape_;
	strides_ = other.strides_;
	numel_ = other.numel_;
	device_ = other.device_;
	requires_grad_ = other.requires_grad_;
	offset_ = 0;
	is_view_ = false;
	grad = nullptr;
	grad_fn = nullptr;
	allocate();
	copy_from(other);
	return *this;
}

//
// Move Constructor - steal resources
//

Tensor::Tensor(Tensor&& other) noexcept
	: storage_(std::move(other.storage_)),
	offset_(other.offset_),
	shape_(std::move(other.shape_)),
	strides_(std::move(other.strides_)),
	numel_(other.numel_),
	device_(other.device_),
	requires_grad_(other.requires_grad_),
	is_view_(other.is_view_),
	grad(std::move(other.grad)),
	grad_fn(std::move(other.grad_fn))
{
	other.numel_ = 0;
	other.offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
	if (this == &other) return *this;
	storage_ = std::move(other.storage_);
	offset_ = other.offset_;
	shape_ = std::move(other.shape_);
	strides_ = std::move(other.strides_);
	numel_ = other.numel_;
	device_ = other.device_;
	requires_grad_ = other.requires_grad_;
	is_view_ = other.is_view_;
	grad = std::move(other.grad);
	grad_fn = std::move(other.grad_fn);
	other.numel_ = 0;
	other.offset_ = 0;
	return *this;
}

Tensor::~Tensor() = default;


//
// Copy_from - memcpy respecting device combinations
//

void Tensor::copy_from(const Tensor& src)
{
	size_t bytes = static_cast<size_t>(numel_) * sizeof(float);
	const float* src_ptr = 
		(src.device_ = Device::CPU)
		? (src.storage_->ptr + src.offset_)
		: (src.storage_->ptr + src.offset_); // same expr, keep explicit

	float* dst_ptr = storage_->ptr + offset_;

	if (device == Device::CPU && src.device_ == Device::CPU) {
		std::memcpy(dst_ptr, src_ptr, bytes);
	} else if (device_ == Device::CUDA && src.device_ == Device::CUDA) {
		CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice));
	} else if (device_ == Device::CPU && src.device_ == Device::CUDA) {
		CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost));
	} else {
		CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice));
	}
}


//
// Factory Helpers
//

/*static*/
Tensor Tensor::zeros(const std::vector<int>& shape, Device device, bool requires_grad)
{
	// Storage constructor : zero-initilaizes, so just construct
	return Tensor(shape, device, requires_grad);
}
	
/*static*/
Tensor Tensor::ones(const std::vector<int>& shape, Device device, bool requires_grad)
{
	Tensor t(shape, Device::CPU, requires_grad);	// fill on CPU first
	std::fill(t.storage_->ptr, t.storage_->ptr + t.numel_, 1.0f);
	if (device == Device::CUDA) return t.cuda();
	return t;
}

/*static*/
Tensor Tensor::randn(const std::vector<int>& shape, Device device, bool requires_grad)
{
	Tensor t(shape, Device::CPU, requires_grad);
	std::mt19937 rng(std::random_device{}());
	std::normal_distribution<float> dist(0.f, 1.f);
	for (int i = 0; i < t.numel_; ++i) 
		t.storage_->ptr[i] = dist(rng);
	if (device == Device::CUDA) return t.cuda();
	return t;
}


//
// Raw data access 
//

float* Tensor::data_ptr()
{
	if (device_ != Device::CPU)
		throw std::runtime_error("data_ptr() called on CUDA tensor. Use cuda_ptr().");
	return storage_->ptr + offset_;
}

const float* Tensor::data_ptr() const 
{
	if (device_ != Device::CPU)
		throw std::runtime_error("data_ptr() called on CUDA tensor. Use cuda_ptr.");
	return storage_->ptr + offset_;
}

float* Tensor::cuda_ptr() 
{
	if (device_ != Device::CUDA)
		throw std::runtime_error("cuda_ptr() called on CPU tensor. Use data_ptr().");
	return storage_->ptr + offset_;
}

const float* Tensor::cuda_ptr() const
{
	if (device_ != Device::CUDA)
		throw std::runtime_error("cuda_ptr() call on CPU tensor. Use data_ptr().");
	return storage_->ptr + offset_;
}


//
// Device Transfer
//

Tensor Tensor::to(Device target) const
{
	if (device_ == target) return *this;	// already exists, return copy
	Tensor out(shape_, target, requires_grad_);
	out.copy_from(*this);
	return out;
}


//
// Indexxing
//

float Tensor::at(const std::vector<int>& idx) const 
{
	if (device_ != Device::CPU)
		throw std::runtime_error("at() only supported for CPU tensors");
	return storage_->ptr[flat_index(idx)];
}

float& Tensor::at(const std::vector<int>& idx) 
{
	if (device_ != Device::CPU)
		throw std::runtime_error("at() only supported for CPU tensors");
	return storage_->ptr[flat_index(idx)];
}


//
// Reshaping
//

Tensor Tensor::reshape(const std::vector<int>& new_shape) const
{
	// Resovle a single -1 dimesion
	int inferred = -1, known = 1;
	for (int i = 0; i < static_cast<int>(new_shape.size()); ++i) {
		if (new_shape[i] == -1){
			if (inferred != -1) 
				throw std::invalid_argument("Only one -1 allowed in reshape")
			inferred = i;

		} else {
			known *= new_shape[i];
		}
	}
	std::vector<int> resolved = new_shape;
	if (inferred != -1) {
		if (numel_ % known != 0) 
			throw std::invalid_argument("Cannot reshape: sizes incompatible");
		resolved[inferred] = numel_ / known;
	}
	if (compute_numel(resolved) != numel_)
		throw std::invalid_argument("Reshape: total number of elements must match!");

	// Create a view that shares the same storage
	Tensor view;
	view.storage_ = storage_;
	view.offset_ = offset_;
	view.shape_ = resolved;
	view.strides_ = compute_strides(resolved);
	view.numel_ = numel_;
	view.device_ = device_;
	view.requires_grad_ = requires_grad_;
	view.is_view_ = true;
	// grad/grad_fn is intentionally not carried over to view
	return view;
}

Tensor Tensor::transpose(int dim0, int dim1) const
{
	int nd = ndim();
	if (dim0 < 0) dim0 += nd;
	if (dim1 < 0) dim1 += nd;
	if (dim0 < 0 || dim0 >= nd || dim1 < 0 || dim1 >= nd)
		throw std::out_of_range("transpose: out of range");
	
	Tensor view;
	view.storage_ = storage_;
	view.offset_ = offset_;
	view.shape_ = shape_;
	view.strides_ = strides_;
	view.numel_ = numel_;
	view.device_ = device_;
	view.requires_grad_ = requires_grad_;
	view.is_view_ = true;

	std::swap(view.shape_[dim0], view.shape_[dim1]);
	std::swap(view.strides_[dim0], view.strides_[dim1]);
	return view;
}

Tensor Tensor::squeeze(int dim) const
{
	std::vector<int> new_shape;
	if (dim == -1) {
		for (int s : shape_) if (s != 1) new_shape.push_back(s);
	} else {
		if (dim < 0 || dim >= ndim())
			throw std::out_of_range("squeeze: dim out of range");
		for (int i = 0; i < ndim(); ++i)
			if (!(i == dim && shape_[i] == 1)) new_shape.push_back(shape_[i]);
	}
	if (new_shape.empty()) new_shape.push_back(1);
	return reshape(new_shape);
}

Tensor Tensor::unsqueeze(int dim) const
{
	int nd = ndim();
	if (dim < 0) dim += nd + 1;
	if (dim < 0 || dim > nd)
		throw std::out_of_range("unsqueeze: dim out of range");
	std::vector<int> new_shape = shape_;
	new_shape.insert(new_shape.begin() + dim, 1);
	return reshape(new_shape);
}


//
// In-place ops (CPU only for now, CUDA variants added later)
//

void Tensor::zero_()
{
	if (device_ == Device::CPU) {
		std::memset(storage_->ptr + offset_, 0, numel_ * sizeof(float));
	} else {
		CUDA_CHECK(cudaMemset(storage_->ptr + offset_, 0, numel_ * sizeof(float)));
	}
}

void Tensor::fill_(float value) 
{
	if (device_ != Device::CPU)
		throw std::runtime_error("fill_() not yet implemented for CUDA tensors");
	float* p = storage_->ptr + offset_;
	std::fill(p, p + numel_, value);
}

void Tensor::add_(const Tensor& other)
{
	if (other.numel_ != numel_)
		throw std::invalid_argument("add_: size mismatch");
	if (device_ != Device::CPU)
		throw std::runtime_error("add_() not yet implemented for CUDA tensors");
	float* a = storage_->ptr + offset_;
	const float* b = other.storage_->ptr + other.offset_;
	for (int i = 0; i < numel_ ; ++i) a[i] += b[i];
}

void Tensor::add_(float scalar)
{
	if (device_ != Device::CPU)
		throw std::runtime_error("add_() not yet implemented for CUDA tensors");
	float* p = storage_->ptr + offset_;
	for (int i = 0; i < numel_; ++i) p[i] += scalar;
}

void Tensor::mul_(float scalar)
{
	if (device_ != Device::CPU)
		throw std::runtime_error("mul_() not yet implemented for CUDA tensors");
	float* p = storage_->ptr + offset_;
	for (int i = 0; i < numel_; ++i) p[i] *= scalar;
}


//
// CPU elementwise helpers
// All math ops use this same pattern:
//		1. Assert same device / compatible shapes
//		2. Allocate output tensor
//		3. Apply the operation element by element (CPU) or launch kernel (CUDA)
//		4. Wire up grad_fn if either input requires_grad
//

// Internal helper - elementwise binary op on CPU
static Tensor cpu_elementwise_binary(const Tensor& a, const Tensor& b, std::function<float(float,float)> op)
{
	if (a.numel() != b.numel())
		throw std::invalid_argument("Elementwise op: size mismatch");
	Tensor out(a.shape(), Device::CPU);
	const float* pa = a.data_ptr();
	const float* pb = b.data_ptr();
	float* po = out.data_ptr();
	for (int i = 0; i < a.numel(); ++i) po[i] = op(pa[i], pb[i]);
	return out;
}

static Tensor cpu_elementwise_unary(const Tensor& a, std::function<float(float)> op)
{
	Tensor out(a.shape(), Device::CPU);
	const float* pa = a.data_ptr();
	float* po = out.data_ptr();
	for (int i = 0; i < a.numel(); ++i) po[i] = op(pa[i]);
	return out;
}


//
// Binary ops
//

Tensor Tensor::operator+(const Tensor& other) const
{
	if (device_ != other.device_)
		throw std::runtime_error("operator+: tensors are on different devices");
	Tensor out = cpu_elementwise_binary(*this, other, [](float a, float b) { return a + b; });
	if (requires_grad_ || other.requires_grad_) {
		out.requires_grad_ = true;
		auto fn = std::make_shared<AddBackward>();
		fn->inputs = { std::weak_ptr<Tensor>(), std::weak_ptr<Tensor>() };
		// Note: full auto grad writing in autograd.cpp; grad_fn set there
		out.grad_fn = fn;
	}
	return out;
}

Tensor Tensor::operator-(const Tensor& other) const
{
	if (device_ != other.device_) 
		throw std::runtime_error("operator-: tensors on different devices");
	Tensor out = cpu_elementwise_binary(*this, other, [](float a, float b) { return a - b; });
	if (requires_grad_ || other.requires_grad_)
		out.requires_grad_ = true;
	return out;
}

Tensor Tensor::operator*(const Tensor& other) const 
{
	if (device_ != other.device_)
		throw std::runtime_error("operator*: tensors on different devices");
	Tensor out = cpu_elementwise_binary(*this, other, [](float a, float b) { return a * b; });
	if (requires_grad_ || other.requires_grad_) {
		out.requires_grad_ = true;
		auto fn = std::make_shared<MulBackward>();
		fn->saved_a = this->detach();
		fn->saved_b = other.detach();
		out.grad_fn = fn;
	}
	return out;
}

Tensor Tensor::operator/(const Tensor& other) const 
{
	if (device_ != other.device_)
		throw std::runtime_error("operator/: tensors on different devices");
	Tensor out = cpu_elementwise_binary(*this, other, 
			[](float a, float b) { 
			if (b == 0.f) 
				throw std::runtime_error("Divison by zero");
			return a / b;
		}):
	if (requires_grad_ || other.requires_grad_)
		out.requires_grad_ = true;
	return out;
}


//
// scalar overload
//						
	
Tensor Tensor::operator+(float s) const
{
	return cpu_elementwise_unary(*this, [s](float a) { return a + s; });
}

Tensor Tensor::operator-(float s) const
{
	return cpu_elementwise_unary(*this, [s](float a) { return a - s; });
}

Tensor Tensor::operator*(float s) const
{
	Tensor out = cpu_elementwise_unary(*this, [s](float a) { return a * s; });
	if (requires_grad_) out.requires_grad_ = true;
	return out;
}

Tensor Tensor::operator/(float s) const
{
	if (s == 0.f) throw std::runtime_error("Division by zero (scalar)");
	return cpu_elementwise_unary(*this, [s](float a) { return a / s; });
}


//
// Matrix multiply
//

Tensor Tensor::matmul(const Tensor& other) const
{
	if (device_ != other.device_)
		throw std::runtime_error("matmul: tensor on different devices");
	if (ndim() != 2 || other.ndim() != 2)
		throw std::invalid_argument("matmul: only 2-D tensors supported");
	int M = shape_[0], K = shape_[1];
	int K2 = other.shape_[0], N = other.shape_[1];
	if (K != K2)
		throw std::invalid_argument("matmul: inner dimesions don't match");
	
	Tensor out({M, N}, device_);

	if (device_ == Device::CPU) {
		const float* A = data_ptr();
		const float* B = other.data_ptr();
		float* C = out.data_ptr();
		for (int i = 0; i < M; ++i)
			for (int k = 0; k < K; ++k) {
				float aik = A[i * K + k];
				for (int j = 0; j < N; ++j)
					C[i * N + j] += aik * B[k * N + j];
		}
	} else {
		throw std::runtime_error("matmul: CUDA path not yet implemented. See cuda/matmul.cu");
	}

	if (requires_grad_ || other.requires_grad_) {
		out.requires_grad_ = true;
		auto fn = std::make_shared<MatMulBackward>();
		fn->saved_a = this->detach();
		fn->saved_b = other.detach();
		out.grad_fn = fn;
	}
	return out;
}


//
// Reduction ops
//

Tensor Tensor::sum(int dim, bool keepdim) cosnt
{
	if (device_ != Device::CPU)
		throw std::runtime_error("sum: CUDA path not yet implemented");

	if (dim == -1) {
		// Global sum -> scalar tensor
		float total = 0.f;
		const float* p = data_ptr();
		for (int i = 0; i < numel_; ++i) total += p[i];
		Tensor out({1}, Device::CPU, requires_grad_);
		out.data_ptr()[0] = total;
		if (requires_grad_) {
			out.requires_grad_ = true;
			auto fn = std::make_shared<SumBackward>();
			fn->input_shape = shape_;
			fn->dim = dim;
			fn->keepdim = keepdim;
			out.grad_fn = fn;
		}
		return out;
	}

	if (dim < 0) dim += ndim();
	if (dim < 0 || dim >= ndim())
		throw std::out_of_range("sum: dim out of range");

	// Build output shape
	std::vector<int> out_shape = shape_;
	out_shape[dim] = 1;
	Tensor out(out_shape, Device::CPU);

	const float* src = data_ptr();
	float* dst = out.data_ptr();

	int outer = 1, inner = 1;
	for (int i = 0; i < dim; ++i) outer *= shape_[i];
	for (int i = dim + 1; i < ndim(); ++i) inner *= shape_[i];
	int reduce = shape_[dim];

	for (int o = 0; o < outer; ++o)
		for (int i = 0; i < inner; ++i) {
			float acc = 0.f;
			for (int r = 0; r < reduce; ++r)
				acc += src[(o * reduce + r) * inner + i];
			dst[o * inner + i] = acc;
		}

	if (!keepdim) out = out.squeeze(dim);

	if (requires_grad_) {
		out.requires_grad_ = true;
		auto fn = std::make_shared<SumBackward>();
		fn->input_shape = shape_;
		fn->dim = dim;
		fn->keepdim = keepdim;
		out.grad_fn = fn;
	}
	return out;
}

Tensor Tensor::mean(int dim, bool keepdim) const
{
	Tensor s = sum(dim, keepdim);
	int divisor = (dim == -1) ? numel_ : shape_[dim < 0 ? dim + ndim() : dim];
	return s * (1.0f / static_cast<float>(divisor));
}


//
// Activation Functions
//

Tensor Tensor::relu() const
{
	Tensor out = cpu_elementwise_unary(*this, [](float x) { return x > 0.f ? x : 0.f; });
	if (requires_grad_) {
		out.requires_grad_ = true;
		auto fn = std::make_shared<ReLUBackward>();
		fn->saved_input = this->detach();
		out.grad_fn = fn;
	}
	return out;
}

Tensor Tensor::sigmoid() const
{
	Tensor out  = cpu_elementwise_unary(*this, [](float x) { return 1.f / (1.f + std::exp(-x)); });
	if (requires_grad_) {
		out.requires_grad_ = true;
		auto fn = std::make_shared<SigmoidBackward>();
		fn->saved_output = out.detach();
		out.grad_fn = fn;
	}
	return out;
}

Tensor Tensor::tanh() const
{
	return cpu_elementwise_unary(*this, [](float x) { return std::tanh(x); });
}

Tensor Tensor::softmax(int dim) const
{
	if (dim < 0) dim += ndim();
	// Numerically stable: subtract row max before exp
	// For simplicity, implemented for last dim only here
	Tensor out(shape_, device_);
	if (device_ != Device::CPU)
		throw std::runtime_error("softmax: CUDA path not yet implemented");

	int outer = 1;
	for (int i = 0; i < dim; ++i) outer *= shape_[i];
	int size = shape_[dim];
	int inner = 1;
	for (int i = dim + 1; i < ndim(); ++i) inner *= shape_[i];

	const float* src = data_ptr();
	float* dst = out.data_ptr();

	for (int o = 0; o < outer; ++o)
		for (int i = 0; i < inner; ++i) {
			// find max for numerical stability
			float mx = -std::numeric_limits<float>::infinity();
			for (int s = 0; s < size; ++s)
				mx = std::max(mx, src[(o * size + s) * inner + i]);
			float sum = 0.f;
			for (int s = 0; s < size; ++s) {
				float v = std::exp(src[(o * size + s) * inner + i] - mx);
				dst[(o * size + s) * inner + i] = v;
				sum += v;
			}
			for (int s = 0; s < size; ++s)
				dst[(o * size + s) * inner + i] /= sum;
		}
	return out;
}

Tensor Tensor::log() const
{
	Tensor out = cpu_elementwise_unary(*this, [](float x) {
			if (x <= 0.f)
				throw std::domain_error("log of non-positive number");
			return std::log(x);
		});
	if (requires_grad_) {
		out.requires_grad_ = true;
		auto fn = std::make_shared<LogBackward>();
		fn->saved_input = this->detach();
		out.grad_fn = fn;
	}
	return out;
}

Tensor Tensor::exp() const
{
	return cpu_elementwise_unary(*this, [](float x) { return std::exp(x); });
}

Tensor Tensor::pow(float exponent) const
{
	return cpu_elementwise_unary(*this, [exponent](float x) { return std::pow(x, exponent); });
}

Tensor Tensor::sqrt() const
{
	return cpu_elementwise_unary(*this, [](float x) {
			if ( x < 0.f) throw std::domain_error("sqrt of negative number");
			return std::sqrt(x);
			});
}


//
// Autograd
//

void Tensor::accumulate_grad(const Tensor& grad_update)
{
	if (!grad) {
		grad = std::make_shared<Tensor>(shape_, device_);
		grad->zero_();
	}
	grad->add_(grad_update);
}

void Tensor::backward()
{
	// Default upstream gradiant = all-ones (for scalar loss)
	Tensor ones_grad = Tensor::ones(shape_, device_);
	backward(ones_grad);
}

// Topological sort + backprop
// Simple recusive DFS for time being
// explicit stack would be good to implement to avoid stack overflow
void Tensor::backward(const Tensor& upstream_grad)
{
	if (!requires_grad_)
		throw std::runtime_error("backward() called on tensor with requires_grad=false");

	accumulate_grad(upstream_grad);

	if (grad_fn) {
		grad_fn->backward(upstream_grad);
	}
}

Tensor Tensor::detach() const
{
	Tensor out;
	out.storage_ = storage_;
	out.offset_ = offset_;
	out.shape_ = shape_;
	out.strides_ = strides_;
	out.numel_ = numel_;
	out.device_ = device_;
	out.requires_grad_ = false;
	out.is_view_ = true;
	// grad and grad_fn intentioanlly null
	return out;
}


//
// Utilities 
//

std::string Tensor::shape_str() const
{
	std::ostringstream oss;
	oss << "["
	for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
		if (i) oss << ", ";
		oss << shape_[i];
	}
	oss << "]";
	return oss.str();
}

void Tensor::print(const std::string& name) const
{
	if (!name.empty()) std::cout << name << " ";
	std::cout << "Tensor(shape=" << shape_str()
			  << ", device=" << (device_ == Device::CPU ? "CPU" : "CUDA")
			  << ", requires_grad=" << (requires_grad_ ? "true" : "false")
			  << ")\n";

	if (device_ != Device::CPU) {
		std::cout << " [data on GPU - call .cpu().print() to inspect]\n";
		return;
	}

	const float* p = data_ptr();
	if (ndim() == 1) {
		std::cout << "  [ ";
		for (int i = 0; i < std::min(numel_, 8); ++i)
			std::cout << std::setw(8) << std::setprecision(4) << p[i] << " ";
		if (numel_ > 8) std::cout << "...";
		std::cout << "]\n";
	} else if (ndim() == 2) {
		int rows = std::min(shape_[0], 4), cols = std::min(shape_[1], 8);
		for (int r = 0; r < rows; ++r) {
			std::cout << "  [ ";
			for (int c = 0; c < cols; ++c)
				std::cout << std::setw(8) << std::setprecision(4)
						  << p[r * strides_[0] + c] << " ";
			if (shape_[1] > 8) std::cout << "...";
			std::cout << "]\n";
		}
		if (shape_[0] > 4) std::cout << "  ...\n";
	} else {
		std::cout << "  (ndim > 2: use at() to inspect individual elements)\n";
	}
}


// 
// Need to add 
//
// Private default constructor used by detach()
// and reshape() to build view tensors without 
// going through the normal allocation path
//
// 


