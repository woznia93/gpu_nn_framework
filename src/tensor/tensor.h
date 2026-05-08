#pragma once

#include <vector>
#include <memory> 
#include <stdexcept>
#include <numeric>
#include <functional>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>


// Device Enum
//

enum class Device { CPU, CUDA }; 

// CUDA error-check helper

#define CUDA_CHECK(call)
do {
	cudaError_t err = (call);
	if (err != cudaSuccess){
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " __FILE__ ":" + std::to_string(__LINE__)):
	}
} while (0)


// Forward declaration (autograd)
struct GradFn;

// Tensor
class Tensor{
	public:
		// constructors

		// Create an uninitialized tensor on CPU
		explicit Tensor(const std::vector<int>& shape, Device device = Device::CPU, bool requires_grad = false);

		// Create a tensor from existing host data
		Tensor(const std::vector<float>& data, const std::vector<int>& shape, Device device = Device::CPU, bool requires_grad = false);

		// Copy constructor / assignment
		Tensor(const Tensor& other);
		Tensor& operator=(const Tensor& other);

		// Move constructor / assignment
		Tensor(Tensor&& other) noexcept;
		Tensor& operator=(const Tensor& other) noexcept;

		~Tensor();

		// helpers
		static Tensor zeros(const std::vector<int>& shape, Device device = Device::CPU,	bool requires_grad = false);

		static Tensor ones(const std::vector<int>& shape, Device device = Device::CPU, bool requires_grad = false);

		static Tensor randn(const std::vector<int>& shape, Device device = Device::CPU, bool requires_grad = false);

		// metadata
		const std::vector<int>& shape() const { return shape_; }
		const std::vector<int>& strides() const { return strides_; }
		int ndim()		const { return static_cast<int>(shape_.size()); }
		int numel()		const { return numel_; } 
		Device device() const { return device_; } 
		bool requires_grad() const { return requires_grad_; }

		// Raw data access
		// CPU pointer (throws if on CUDA)
		float*	data_ptr();
		const float* data_ptr() const;

		// GPU pointer (throws if on CPU)
		float* cuda_ptr();
		const float* cuda_ptr() const;

		// Device Transfer 
		Tensor to(Device device) const;
		Tensor cpu() const { return to(Device::CPU); } 
		Tensor cuda() const { return to(Device::CUDA); }

		// indexing/ slicing 
		// Flat index into data for CPU only/testing
		float at(const std::vector<int>& idx) const;
		float& at(const std::vector<int>& idx);

		// Reshaping 
		// return view (shares data) when possible, else copy
		Tensor reshape(const std::vector<int>& new_shape) const;
		Tensor transpose(int dim0, int dim1) const;
		Tensor squeeze(int dim = -1) const;
		Tensor unsqueeze(int dim) const;

		// basic math ops 
		// Dispach to CPU or CUDA implementations
		Tensor operator+(const Tensor& other) const;
		Tensor operator-(const Tensor& other) const;
		Tensor operator*(const Tensor& other) const; // elementwise
		Tensor operator/(const Tensor& other) const;
		
		Tensor operator+(float scalar) const;
		Tensor operator-(float scalar) const;
		Tensor operator*(float scalar) const;
		Tensor operator/(float scalar) const;

		Tensor matmul(const Tensor& other) const;
		Tensor sum(int dim = -1, bool keepdim = false) const;
		Tensor mean(int dim = -1, bool keepdim = false) const;
		Tensor relu() const;
		Tensor sigmoid() const;
		Tensor tanh() const;
		Tensor softmax(int dim = -1) const;
		Tensor log() const;
		Tensor exp() const;
		Tensor pow(float exponent) const;
		Tensor sqrt() const;

		// in place ops
		void zero_();
		void fill_(float value);
		void add_(const Tensor& other);
		void add_(float scalar);
		void mul_(float scalar);
		

		// autograd 
		void set_requires_grad(bool val) { requires_grad_ = val; }

		// gradiant tensor (same shape) allocate on first backward pass
		std::shared_ptr<Tensor> grad;

		// Node in computational graph; set by ops that produce this tensor
		std::shared_ptr<GradFn> grad_fn;

		// accumulate grad into this -> grad (allocates if null)
		void accumulate_grad(const Tensor& grad_update);

		// kick off backprop from this scalar tensor
		void backward(const Tensor& upstream_grad);
		void backward();		// convience: upstream_grad = ones (for scalar loss)

		// Detach from computational graph (returns a copy with no grad_fn)
		Tensor detach() const;

		// utilities
		void print(const std::string& name = "") const;
		std::string shape_str() const;


	private:

		Tensor() = default;

		// internal storage
		// use shared_ptr to a raw buffer so that views can share data
		// without extra copies
		struct Storage{
			float* ptr = nullptr;
			size_t bytes = 0;
			Device device = Device::CPU;

			Storage() = default;
			Storage(size_t bytes, Device device);
			~Storage();

			// Non-copyable: use shared_ptr
			Storage(const Storage&) = delete;
			Storage& operator=(const Storage&) = delete;
		};

		std::shared_ptr<Storage> storage_;		// underlying memory (possibly shared)
		size_t offset_ = 0;				// byte offset onto storage (for views)

		std::vector<int> shape_;
		std::vector<int> strides_;	// in *elements*  (row major / C-contiguous)
		int numel_ = 0;
		Device device_ = Device::CPU;
		bool requires_grad_ = false;
		bool is_view_ = false;

		// Helpers
		static std::vector<int> compute_strides(const std::vector<int>& shape);
		static int compute_numel(const std::vector<int>& shape);
		int flat_index(const std::vector<int>& idx) const;

		// Allocate fresh storage for this tensor
		void allocate();

		// Deep copy src storage into this tensor's storage
		void copy_from(const Tensor& src);
};

// GradFn - a node in the autograd graph
//
// Every differential op creates a subclass of GradFn,
// stores the inputs it needs for the backward pass,
// and registers it as the grad_fn of its output tensor
//
struct GradFn {
	// inputs to the forward op that created this node
	// (stored as weak_ptr to avoid refernce cycles)
	std::vector<std::weak_ptr<Tensor>> inputs;

	// Called during backward() with the upstream gradiant
	// Must push gradiants back to each input
	virtual void backward(const Tensor& grad_output) = 0;

	virtual ~GradFn() = default;
}

//  Concrete GradFn subclasses (declared here,
//  defined in autograd.cpp)
struct AddBackward : public GradFn {
		void backward(const Tensor& grad_output) override;
};

struct MulBackward : public GradFn {
	// need the forward inputs to compute grad
	Tensor saved_a, saved_b;
	void backward(const Tensor& grad_ouput) override;
};

struct MatMulBackward : public GradFn {
	Tensor saved_a, saved_b;
	void backward(const Tensor& grad_ouput) override;
};

struct ReLUBackward : public GradFn {
	Tensor saved_input;		// needed for mask
	void backward(const Tensor& grad_ouput) override;
};

struct SigmoidBackward : public GradFn {
	Tensor saved_output;	// sigmoid(x) is cheaper to use
	void backward(const Tensor& grad_ouput) override;
};

struct SumBackward : public GradFn {
	std::vector<int> input_shape;
	int dim;
	bool keepdim;
	void backward(const Tensor& grad_output) override;
};

struct LogBackward : public GradFn {
	Tensor saved_input;
	void backward(const Tensor& grad_output) override;
};

// Free function operation overloads
// (scalar op tensor) 
inline Tensor operator+(float s, const Tensor& t) { return t + s; }
inline Tensor operator*(float s, const Tensor& t) { return t * s; } 

