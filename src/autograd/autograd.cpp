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

	if (auto b = inputs[1].lock()) {
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

void DivBackward::backward(const Tensor& grad_output) 
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
		if (!x->requires_grad()) return;

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
		Tensor ones = Tensor::ones(y_sq.shape(), y_sq.device());
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
	expanded.zero_();

	if (dim == -1) {
		float g_val = grad_output.data_ptr()[0];
		float* ep = expanded.data_ptr();
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


// 
// MSE Backward
//
// Forward: diff = pred - target 
//			loss = mean(diff^2)
// 
// Backward: dL/d_pred = (2 / N) * (pred - target) * upstream_grad
//
void MSEBackward::backward(const Tensor& grad_output)
{
	if (auto pred = inputs[0].lock()) {
		if (!pred->requires_grad()) return;

		float scale = 2.0f / static_cast<float>(saved_pred.numel());
		Tensor diff = saved_pred - saved_target;
		Tensor gx = diff * scale * grad_output.data_ptr()[0];

		pred->accumulate_grad(gx);
		if (pred->grad_fn) pred->grad_fn->backward(gx);
	}
}


// 
// CrossEntropyBackward - loss = cross_entroy(logits, target)
//
// Softmax + NLL backward 
//
// Forward: p = softmax(logits) 
//			loss = -mean(log(p[i, target[i]]))
//
// Backward: dL/d_logits[i, j] = (1 / N) * (p[i,j] - 1{j == target[i]})
//
// The gradiant is just softmax output minus the one hot target, sclaed by 1/N
//
void CrossEntropyBackward::backward(const Tensor& grad_output)
{
	if (inputs.empty()) return;
	auto logits = inputs[0].lock();
	if (!logits || !logits->requires_grad()) return; 

	int N = saved_softmax.shape()[0];
	int C = saved_softmax.shape()[1];
	float upstream = grad_output.data_ptr()[0];  // scalar upstream grad
	float scale = upstream / static_cast<float>(N);

	// grad = softmax_probs (cpy)
	Tensor gx = saved_softmax.detach();
	float* gp = gx.data_ptr();

	// Subtract 1 from the true class column ex: gx[i, target[i]] -= 1
	const float* tp = saved_target.data_ptr();
	for (int i = 0; i < N; ++i) {
		int cls = static_cast<int>(tp[i]);
		if (cls < 0 || cls >= C)
			throw std::out_of_range("Cross entropy: target class out of range");
		gp[i * C + cls] -= 1.0f;
	}

	// Scale
	for (int i = 0; i < N * C; ++i) gp[i] *= scale;

	logits->accumulate_grad(gx);
	if (logits->grad_fn) logits->grad_fn->backward(gx);

}


// 
// Autograd Engine
//

/*static*/
void AutogradEngine::dfs(Tensor* node, std::unordered_set<Tensor*>& visited, std::vector<Tensor*>& order) 
{
	if (!node || visited.count(node)) return;
	visited.insert(node);

	// visit children first
	if (node->grad_fn) {
		for (auto& weak_in : node->grad_fn->inputs) {
			if (auto in = weak_in.lock()) {
				dfs(in.get(), visited, order);
			}
		}
	}

	// post order: curr node goes after children
	order.push_back(node);
}


/*static*/
std::vector<Tensor*> AutogradEngine::build_topo(Tensor* root)
{
	std::unordered_set<Tensor*> visited;
	std::vector<Tensor*> order;
	dfs(root, visited, order);
	// Reverse so root comes first (reverse topo order for backprop)
	std::reverse(order.begin(), order.end());
	return order;
}


/*static*/
void AutogradEngine::backward(Tensor& root) 
{
	if (!root.requires_grad())
		throw std::runtime_error("AutogradEngine::backward called on a tensor, with requires_grad = false");
	if (root.numel() != 1)
		throw std::runtime_error("AutogradEngine::backward called on a non-scalar tensor, call .sum() or .mean() to reduce first");

	// Seed grad: dL/dL = 1
	Tensor seed = Tensor::ones({1}, root.device());
	root.accumulate_grad(seed);

	// Walk in reverse topo order, propogating grads
	std::vector<Tensor*> topo = build_topo(&root);

	for (Tensor* node: topo) {
		if (!node->grad-fn) continue;	// lead node, nothing to propogate
		if (!node->grad) continue;		// no grad here yet

		// start backward pass for this node
		node->grad_fn->backward(*node->grad);
	}
}


/*static*/
void AutogradEngine::zero_grad(Tensor& root) {
	std::vector<Tensor*> topo = build_topo(&root);
	for (Tensor* node : topo) {
		if (node->grad)	node->grad->zero_();
	}
}


//
// Free helper
//
void zero_grad(std::vector<Tensor*>& params)
{
	for (Tensor* p : params) {
		if (p && p->grad) p->grad->zero_();
	}
}


//
// Loss funtions
//


//
// MSE: mean((pred - target) ^2) 
//
Tensor mse_loss(const Tensor& pred, const Tensor& target) 
{
	if (pred.numel() != target.numel())
		throw std::invalid_argument("mse_loss: pred and target must have same number of elements");
	if (pred.device() != target.device())
		throw std::runtime_error("mse_loss: pred and target must be on same device");

	// diff = pred - target
	Tensor diff(pred.shape(), pred.device());
	const float* pp = pred.data_ptr();
	const float* tp = target.data_ptr();
	float* dp = diff.data_ptr();
	int N = pred.numel();
	for (int i = 0; i < N; i++) dp[i] = pp[i] - tp[i];

	// loss = mean(diff^2) = sum(diff^2) / N 
	float sum_sq = 0.f;
	for (int i = 0; i < N; ++i) sum_sq += dp[i] * dp[i];

	Tensor loss({1}, pred.device(), /*requires_grad*/pred.requires_grad());
	loss.data_ptr()[0] = sum_sq / static_cast<float>(N);

	if (pred.requires_grad()){
		auto fn = std::make_shared<MSEBackward>();
		fn->saved_pred = pred.detach();
		fn->saved_target = target.detach();
		// stored shared_ptr to pred so backward can reach it 
		// Need: to store in shared_ptr<Tensor> for now using weak ptr
		// inputs mechanism to wrapper
		loss.grad_fn = fn;
	}

	return loss;
}


// 
// Binary cross-entropy = -mean(t*log(p) + (1-t)*log(1-p) )
// pred must already be sigmoid output (values int (0,1))
//
Tensor binary_cross_entropy_loss(const Tensor& pred, const Tensor& target) 
{
	if (pred.numel() != target.numel())
		throw std::invalid_argument("binary_cross_entropy_loss: size mismatch");

	const float eps = 1e-7f;
	const float* pp = pred.data_ptr();
	const float* tp = target.data_ptr();
	int N = pred.numel();

	float total = 0.f;
	for (int i = 0; i < N; ++i) {
		float p = std::max(eps, std::min(1.f - eps, pp[i]));
		total += tp[i] * std::log(p) + (1.f - tp[i]) * std::log(1.f - p);
	}

	Tensor loss({1}, pred.device(), pred.requires_grad());
	loss.data_ptr()[0] = -total / static_cast<float>(N);

	// Backward is handled by composing log/ mul/ sum ops on pred
	// Need to add BCEbackward here for a fused version 
	// For now should build BCE from primitives to get auotgrad:
	// loss = - (target * pred.log() + (1-target) * (1.pred).log() ).mean()
	
	return loss;
}


// 
// NLL loss = -mean(log_probs[i, class_i] )
// log_probs : [N, C], target: [N] (float encoded integer class indicies)
//
Tensor nll_loss(const Tensor& log_probs, const Tensor& target) 
{
	if (log_probs.ndim() != 2)
		throw std::invalid_argument("nll_loss: log_prob must be 2-D [N, C]");
	int N = log_probs.shape()[0];
	int C = log_probs.shape()[1];
	if (target.numel() != N)
		throw std::invalid_argument("nll_loss: target size must equal batch size N");

	const float* lp = log_probs.data_ptr();
	const float* tp = target.data_ptr();

	float total = 0.f;
	for (int i = 0; i < N; ++i) {
		int cls = static_cast<int>(tp[i]);
		if	(cls < 0 || cls >= C)
			throw std::out_of_range("nll_loss: class index out of range");
		total += lp[i * C + cls];
	}


	Tensor loss({1}, log_probs.device(), log_probs.requires_grad());
	loss.data_ptr()[0] = -total / static_cast<float>(N);
	return loss;
}


//
// Cross-entropy - numerically stable softmax + NLL
//
// logits: [N, C] raw scores
// target: [N] integer class indicies stored as float
//
// Formula: 
//		log_sum_exp = log(sum_j exp(logits[i,j] - max_i) ) + max_i
//		loss_i = -logits[i, target[i]] + log_sum_exp
//		loss = mean(loss_i)
//
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target)
{
	if (logits.ndim() != 2) 
		throw std::invalid_argument("cross_entropy_loss: logits must be 2-D [N, C]");
	if (logits.device() != Device::CPU)
		throw std::runtime_error("cross_entropy_loss: CUDA path not implemented yet");

	int N = logits.shape()[0];
	int C = logits.shape()[1];
	if (target.numel() != N)
		throw std::invalid_argument("cross_entropy_loss: target size must equal batch size N");

	const float* lp = logits.data_ptr();
	const float* tp = target.data_ptr();

	// Compute softmax possibilities (needed for backward pass)
	Tensor softmax_out({N,C}, Device::CPU);
	float* sp = softmax_out.data_ptr();

	float total_loss = 0.f;
	for (int i = 0; i < N; ++i) {
		// Numericall stable, subtract row max
		float mx = -std::numeric_limits<float>::infinity();
		for (int j = 0; j < C; ++j) mx = std::max(mx, lp[i * C +j]);


		float row_sum = 0.f;
		for (int j = 0; j < C; ++j) {
			sp[i * C + j]  = std::exp(lp[i * C +j] - mx);
			row_sum += sp[i * C + j];
		}

		for (int j = 0; j < C; ++j) sp[i * C + j] /= row_sum;

		// NLL: -log(p[i, target[i]])
		int cls = static_cast<int>(tp[i]);
		if (cls < 0 || cls >= C)
			throw std::out_of_range("cross_entropy_loss: class index out of range");
		total_loss += -std::log(std::max(sp[i * C + cls], 1e-7f));
	}

	Tensor loss({1}, Device::CPU, logits.requires_grad());
	loss.data_ptr()[0] = total_loss / static_cast<float>(N);

	if (logits.requires_grad()) {
		auto fn = std::make_shared<CrossEntropyBackward>();
		fn->saved_softmax = softmax_out;
		fn->saved_target = target.detach();
		loss.grad_fn = fn;
	}

	return loss;

}





