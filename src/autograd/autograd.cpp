#include "autograd.h"

#include <algorithm> 
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>


// Utility: broadcast aware grad accumulation
//
//
//  When op c = a + b fires and a.shape != b.shape (broadcasting happened),
//  the upstream gradient dL/dc must be SUMMED over the broadcast dimensions
//  before being added to a.grad / b.grad.
//
//  This helper takes the raw upstream grad (shape == c.shape) and reduces it back down to target_shape
//


static Tensor reduce_to_shape(const Tensor& grad, const std::vector<int>& target_shape)
{
	if (grad.shape() == target_shape) return grad;

	Tensor result = grad;


	// If target is scalar-like {1}, sum everything
	int target_numel = 1;
	for (int s : target_shape) target_numel *= s;

	if (target_numel == 1) {
		return result.sum();
	}

	// Sum over any leading dimensions that target doesn't have
	int ndiff = result.ndim() - static_cast<int>(target_shape.size());
	for (int i = 0; i < ndiff; ++i) {
		result = result.sum(0, false);
	}

	// Sum over dimensions where target has size 1 (broadcast dims)
	for (int i = 0; i < static_cast<int>(target_shape.size()); ++i) {
		if (target_shape[i] == 1 && result.shape()[i] != 1) {
			result = result.sum(i, true);
		}
	}
	
	return result.reshape(target_shape);
}



//
// AddBackward: c = a + b
//
// Forward: c = a + b
// Backward: dL / da = dL / dc * 1 = grad_output
//			 dL / db = dL / dc * 1 = grad_output
//	(reduced to input shape if braodcast happened)
//

void AddBackward::backward(const Tensor& grad_output) 
{
	// inputs[0] = a, inputs[1] = b
	if (auto a = inputs[0].lock()) {
		if (a->requires_grad()) {
			Tensor ga = reduce_to_shape(grad_output, a->shape());
			a->accumulate_grad(ga);
			if (a->grad_fn) a->grad_fn->backward(ga);
		}
	}

	if (auto b = inputs[1].lock()) {
		if (b->requires_grad()) {
			Tensor gb = reduce_to_shape(grad_output, b->shape());
			b->accumulate_grad(gb);
			if (b->grad_fn) b->grad_fn->backward(gb);
		}
	}

}


// 
// SubBackward: c = a - b
//
// Forward: c = a - b
// Backward: dL/da = grad_output
//			 dL/db = -grad_output
//


void SubBackward::backward(const Tensor& grad_output)
{
	if (auto a = inputs[0].lock()) {
		if (a->requires_grad()) {
			Tensor ga = reduce_to_shape(grad_output, a->shape());
			a->accumulate_grad(ga);
			if (a->grad_fn) a->grad_fn->backward(ga);
		}
	}

	if (auto b = inpiuts[1].lock()) {
		if (b->requires_grad()) {
			// negate: grad * -1
			Tensor neg_grad = grad_output * (-1.0f);
			Tensor gb = reduce_to_shape(neg_grad, b->shape());
			b->accumulate_grad(gb);
			if (b->grad_fn) b->grad_fn->backward(gb);
		}
	}
}


// 
// MulBackward: c = a * b (elementwise)
//
// Forward: c = a * b 
//
// Backward: dL/da = dL/dc * b
//			 dL/db = dL/dc * a
//


void MulBackward::backward(const Tensor& grad_output)
{
	if (auto a = inputs[0].lock()) {
		if (a->requires_grad()) {
			// dL/da = grad_output * b
			Tensor ga_raw = grad_output * saved_b;
			Tensor ga = reduce_to_shape(ga_raw, a->shape());
			a->accumulate_grad(ga);
			if (a->grad_fn) a->grad_fn->backward(ga);
		}
	}

	if (auto b = inputs[1].lock()) {
		if (b->requires_grad()) {
			// dL/db = grad_output * a
			Tensor gb_raw = grad_output * saved_a;
			Tensor gb = reduce_to_shape(gb_raw, b->shape());
			b->accumulate_grad(gb);
			if (b->grad_fn) b->grad_fn->backward(gb);
		}
	}
}


//
// MulScalarBackward: c = a * scalar
//
// Foward: c = a * s
//
// Backward: dL/da = dL/dc * s
//

void MulScalarBackward::backward(const Tensor& grad_output) 
{
	if (auto a = inputs[0].lock()) {
		if (a->requires_grad()) {
			Tensor ga = grad_output * scalar;
			a->accumulate_grad(ga);
			if (a->grad_fn) a->grad_fn->backward(ga);
		}
	}
}


//
// DivBackward: c = a / b
//
// Forward: c = a / b
//
// Backward: dL/da = dL/dc / b
//			 dL/db = dL/dc * (-a / b^2)
//

void DivBackward::backward(const Tensor& grad_ouput) 
{
	if (auto a = inputs[0].lock()) {
		if (a->requires_grad()) {
			// dL/da = grad / b
			Tensor ga_raw = grad_output / saved_b;
			Tensor ga = reduce_to_shape(ga_raw, a->shape());
			a->accumulate_grad(ga);
			if (a->grad_fn) a->grad_fn->backward(ga);
		}
	}

	if (auto b = inputs[1].lock()) {
		if (b->requires_grad()) {
			// dL/db  = grad * (-a / b^2)
			//		  = -grad * a / (b * b)
			Tensor b_sq = saved_b * saved_b;
			Tensor gb_raw = grad_output * (saved_a * (-1.0f)) / b_sq;
			Tensor gb = reduce_to_shape(gb_raw, b->shape());
			b->accumulate_grad(gb);
			if (b->grad_fn) b->grad_fn->backward(gb);
		}
	}
}


// 
// MatMulBackward: C = A @ B
//
// Forward: C = A @ B			shapes: [M, K] @ [K, N] -> [M, N]
//
// Backward: dL/dA = dL/dC @ B^T		shapes: [M, N] @ [N,K] -> [M, K]
//			 dL/dB = A^T @ dL/dC		shapes: [K, M] @ [M,N] -> [K, N]
//

void MatMulBackward::backward(const Tensor& grad_output)
{
	if (auto a = inputs[0].lock()) {
		if (a->requires_grad()) {
			// B^T has shape [N, K]
			Tensor Bt = saved_b.transpose(0,1);
			Tensor ga = grad_output.matmul(Bt);		// [M, N] @ [N, K]  -> [M, K]
			a->accumulate_grad(ga);
			if (a->grad_fn) a->grad_fn->backward(ga);
		}
	}

	if (auto b = inputs[1].lock()) {
		if (b->requires_grad()) {
			// A^T has shape [K, M]
			Tensor At = saved_a.transpose(0,1);
			Tensor gb = At.matmul(grad_output);		// [K, M] @ [M, N] -> [K, N]
			b->accumulate_grad(gb);
			if (b->grad_fn) b->grad_fn->backward(gb);
		}
	}
}


//
// ReLUBackward: y = max(x,0)
//
// Forward: y = reLu(x)
//
// Backward: dL/dx = dL/dy * ( x > 0 ? 1 : 0)
//
// The mask is recomputed from saved_input rather than stored explicitly (lower mem usage)
//

void RelUBackward::backward(const Tensor& grad_output)
{
	if (auto x = inputs[0].lock()) {
		if (!x->requires_grad()) return;

		// Build them mask, 1 where input > 0, else 0
		Tensor mask(saved_input.shape(), Device::CPU);
		const float* in_ptr = saved_input.data_ptr();
		float* mask_ptr = mask.data_ptr();
		for (int i = 0; i < saved_input.numel(); ++i)
			mask_ptr[i] = in_ptr[i] > 0.f ? 1.f : 0.f;

		Tensor gx = grad_output * mask;
		x->accumulate_grad(gx);
		if (x->grad_fn) x->grad_fn->backward(gx);
	}
}


//
// SigmoidBackward: y = 1 / (1 + e^{-x}) 
//
// Forward: y = sigmoid(x)
//
// Backward: dL/dx = dL/dy * y  * (1-y)
//
// Saved output (y) not input, bc y*(y-1) is cheap to compute from output, avoids sharing both 
//

void SigmoidBackward::backward(const Tensor& grad_output) 
{
	if (auto x = inputs[0].lock()) {
		if (!x->requries_grad()) return;

		// sigmoid' (x) = y * (1-y)
		// Compute (1 - y) as ones - saved_output
		Tensor ones = Tensor::ones(saved_output.shape(), saved_output.device());
		Tensor one_minus_y = ones - saved_output;		// 1 - sigmoid(x)
		Tensor local_grad = saved_output * one_minus_y; // y * (y-1)
		Tensor gx = grad_output * local_grad;

		x->accumulate_grad(gx);
		if (x->grad_fn) x->grad_fn->backward(gx);
	}
}


//
// TanhBackward: y = tanh(x)
//
// Forward: y = tanh(x)
//
// Backward: dL/dx = dL.dy * (1 - y^2)
//

void TanhBackward::backward(const Tensor& grad_output)
{
	if (auto x = inputs[0].lock()) {
		if (!x->requires_grad()) return;

		// 1 - tanh^2(x)
		Tensor y_sq = saved_output * saved_output;	// y^2
		Tensor ones = Tensor::ones(y.sq.shape(), y_sq.device());
		Tensor local_grad = ones - y_sq;			// 1 - y^2
		Tensor gx = grad_output * local_grad; 

		x->accumulate_grad(gx);
		if (x->grad_fn) x->grad_fn->backward(gx);
	}
}


//
// ExpBackward: y = e^x
//
// Forward: y = exp(x)
//
// Backward: dL/dx = dL/dy * e^x = dL/dy * y
//

void ExpBackward::backward(const Tensor& grad_output) 
{
	if (auto x = inputs[0].lock()) {
		if (!x->requires_grad()) return;

		// saved_output = exp(x)
		Tensor gx = grad_output * saved_output;
		x->accumulate_grad(gx);
		if (x->grad_fn) x->grad_fn->backward(gx);
	}
}


//
// LogBackward: y = ln(x)
//
// Forward: y = log(x)
//
// Backward: dL/dx = dL/dy / x
//

void LogBackward::backward(const Tensor& grad_output) 
{
	if (auto x = inputs[0].lock()) {
		if (!x->requires_grad()) return;

		// Guard against divide by zero by addint small epsilon
		const float eps = 1e-8f;
		Tensor safe_input(saved_input.shape(), Device::CPU);
		const float* sp = saved_input.data_ptr();
		float* dp = safe_input.data_ptr();
		for (int i = 0; i < saved_input.numel(); ++i)
			dp[i] = sp[i] + eps;

		Tensor gx = grad_output / safe_input;
		x->accumulate_grad(gx);
		if (x->grad_fn) x->grad_fn->backward(gx);
	}
}


//
// PowBackward: y = x^n
//
// Forward: y = x^n
//
// Backward: dL/dx = dL/dy * n * x^{n-1}
//

void PowBackward::backward(const Tensor& grad_output)
{
	if (auto x = inputs[0].lock()) {
		if (!x->requires_grad()) return;

		// n * x^{n-1}
		Tensor base_grad = saved_input.pow(exponent - 1.0f) * exponent;
		Tensor gx = grad_output * base_grad;

		x->accumulate_grad(gx);
		if (x->grad_fn) x->grad_fn->backward(gx);
	}
}


//
// SumBackward: y = sum(x, dim)
//
// Forward: y = x.sum(dim)
//
// Backward: dL/dx = broadcast(dL/dy) back to x.shape
//
// Summation collapses a dim: its grad is a broadcast (expanded)
// Implmenet expand by tiling grad_output along the summed dim
//

void SumBackward::backward(const Tensor& grad_output)
{
	if (inputs.empty()) return;
	auto x = inputs[0].lock();
	if (!x || !x->requires_grad()) return;

	Tensor expanded(input_shape, Device::CPU);
	expanded._zero();

	if (dim == -1) {
		float g_val = grad_output.data_ptr()[0];
		float* ep = expanded.data_pt();
		int total = 1;
		for (int s : input_shape) total *= s;
		for (int i = 0; i < total; ++i) ep[i] = g_val;
	} else {
		int actual_dim = dim;
		if (actual_dim < 0) actual_dim += static_cast<int>(input_shape.size());

		int outer = 1, inner = 1;
		int reduced_size = input_shape[actual_dim];
		for (int i = 0; i < actual_dim; ++i)
			outer *= input_shape[i];
		for (int i = actual_dim + 1; i < static_cast<int>(input_shape.size()); ++i)
			inner *= input_shape[i];

		const float* gp = grad_output.data_ptr();
		float* ep = expanded.data_ptr();

		for (int o = 0; o < outer; ++o)
			for (int r = 0; r < reduced_size; ++r)
				for (int i = 0; i < inner; ++i)
					ep[(o * reduced_size + r) * inner  + i] = gp[o * inner + i];
	}

	x->accumulate_grad(expanded);
	if (x->grad_fn) x->grad_fn->backward(expanded);
}

