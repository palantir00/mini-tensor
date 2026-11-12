#pragma once

#include "minitensor/tensor.hpp"

#include <cstdint>

namespace mt {

// Math on tensors. Every function returns a new contiguous tensor with its own storage;
// the inputs are never modified and may have any layout (views, transposes, slices).
// Autograd: if an input requires grad and no NoGradGuard is active, the result records a node
// (its grad_fn) with the gradient formula, so that backward() can run through it.

// --- Element-wise, with broadcasting ----------------------------------------------
// The result has shape broadcast_shapes(a.shape(), b.shape()), e.g. [3, 1] + [4] -> [3, 4].
// Throws std::invalid_argument if the shapes cannot be broadcast.

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);  // element-wise, not matrix product

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);

// max(x, 0) for every element.
Tensor relu(const Tensor& t);

// --- Matrix product -----------------------------------------------------------------

// [m, k] @ [k, n] -> [m, n]. Only 2-D tensors, no broadcasting.
Tensor matmul(const Tensor& a, const Tensor& b);

// --- Reductions ---------------------------------------------------------------------

// Sum of all elements, as a scalar (shape {}).
Tensor sum(const Tensor& t);

// Sum along one dimension: [2, 3] summed over dim 1 gives [2], or [2, 1] with keepdim.
// keepdim leaves a size-1 dim, so the result broadcasts back against t.
Tensor sum(const Tensor& t, std::int64_t dim, bool keepdim = false);

// Mean of all elements, as a scalar. A tensor with no elements gives NaN (0 / 0), as in PyTorch.
Tensor mean(const Tensor& t);

}  // namespace mt
