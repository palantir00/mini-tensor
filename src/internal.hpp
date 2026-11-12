#pragma once

// Helpers shared by the library's .cpp files. Not part of the public API:
// this header lives in src/, not in include/, so users of the library cannot include it.

#include "minitensor/tensor.hpp"

#include <cstdint>
#include <string>

namespace mt::detail {

// Shape as text, e.g. "[2, 3]"; used in error messages.
std::string shape_to_string(const Shape& shape);

// Maps dim from [-ndim, ndim) to [0, ndim); -1 is the last dimension.
// Throws std::out_of_range otherwise; `where` names the calling function in the message.
std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim, const char* where);

// Reverse of broadcasting, used by the backward formulas: sums g over the dimensions that
// broadcasting added on the left or stretched from 1, so that the result has `shape`.
// Example: g [3, 4] -> [4] sums dim 0; g [3, 4] -> [3, 1] sums dim 1 with keepdim.
// `shape` must broadcast to g.shape(). Defined in ops.cpp.
Tensor sum_to_shape(const Tensor& g, const Shape& shape);

}  // namespace mt::detail
