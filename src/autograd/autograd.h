#pragma once

//
// AutoGrad.h
//
// Everything needed to run a full forward + backward pass:
//
//	1. Engine - topological sort & gradiant accumulation
//	2. GraphNode - thin wrapper that holds Tensor* and its grad_fn
//	3. Helper free-fns - build_topo(), zero_grad()
//	4. Loss functions - mse_loss(), cross_entropy_loss(), nll_loss(), binary_cross_entropy_loss()
//
//	The GradFn subclasses (AddBackward, MulBackward, etc) are declared in 
//	tensor.h (because Tensor needs to see them) and defined in autograd.cpp.
//	This file just reincludes tensor.h and adds the engine + losses on top.
//
//


#include "tensor.h"

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>


//
// Autograd Engine
//
// Usage: 
//		Tensor loss = ...;
//		AutoGradEngine::backward(loss);
//		// leaf tensors now have their .grad filled in
//


class AutogradEngine {
public:
	// Run backprop from root.
	// root must be a scalar (numel == 1) with requires_grad == true.
	// After this call every leaf tensor that was created with
	// requires_grad=true will have its .grad tensor filled.
	static void backward(Tensor& root);

	// Zero the .grad of every leaf reachable from root.
	static void zero_grad(Tensor& root);

private:
	// Build a topologically-sorted list of all nodes reachable from root, 
	// in reverse topological order (root first, leaves last).
	static std::vector<Tensor*> build_topo(Tensor* root);

	// DFS helper used by build_topo
	static void dfs(Tensor* node, std::unordered_set<Tensor*>& visited, std::vector<Tensor*>& order);
};


//
// Free helpers
//

// zero the .grad field of all tensors in params
void zero_grad(std::vector<Tensor*>& params);


//
// Loss functions
//
// All return a scalar tensor (shape {1}) with requires_grad=true so that
// loss.backward() can be called directly.
//

// Mean-squared error: ((pred - target) ^2)
Tensor mse_loss(const Tensor& pred, const Tensor& target);

// Binary cross-entropy: -mean( t*log(p) + (1-t)*log(1-p) )
// pred values must be in (0, 1) - apply sigmoid first.
Tensor binary_cross_entropy_loss(const Tensor& pred, const Tensor& target);

// Negative log-likelihood : -mean( log_probs[i, class_i] )
// log_probs shape: [N, C],  target shape: [N] (integer class indicies as float)
Tensor nll_loss(const Tensor& log_probs, const Tensor& target);

// Cross - entropy = softmax + nll in one numerically - stable op.
// logits shape: [N, C], target shape: [N]
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target);


//
// Additional GradFn nodes that need to be visible to callers
// (the rest live inside tensor.h / autograd.cpp)
//

// Backward subtraction (c = b - a)
struct SubBackward : public GradFn {
	void backward(const Tensor& grad_output) override;
};

// Back for element division (c = a / b)
struct DivBackward : public GradFn {
	Tensor saved_a, saved_b;
	void backward(const Tensor& grad_output) override;
};

// Backward tanh
struct TanhBackward : GradFn {
	Tensor saved_ouput;
	void backward(const Tensor& grad_output) override;
};

// Backward exp 
struct ExpBackward : GradFn {
	Tensor saved_ouput;
	void backward(const Tensor& grad_output) override;
};

// Backward pow(x,n)
struct PowBackward : GradFn {
	Tensor saved_output;
	float exponent;
	void backward(const Tensor& grad_output) override;
};

// Backward MSE loss 
struct MSEBackward : GradFn {
	Tensor saved_pred, saved_target;
	void backward(const Tensor& grad_output) override;
};

// Backward for cross-entropy loss
struct CrossEntropyBackward : GradFn {
	Tensor saved_softmax;	// softmax (logits) computed in forward pass
	Tensor saved_target;	// integer class labels 
	void backward(const Tensor& grad_output) override;
};


