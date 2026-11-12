#pragma once

// Helpers shared by the library's .cpp files. Not part of the public API:
// this header lives in src/, not in include/, so users of the library cannot include it.

#include "minitensor/tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mt::detail {

// Shape as text, e.g. "[2, 3]"; used in error messages.
std::string shape_to_string(const Shape& shape);

// Maps dim from [-ndim, ndim) to [0, ndim); -1 is the last dimension.
// Throws std::out_of_range otherwise; `where` names the calling function in the message.
std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim, const char* where);

// A row-major shape seen as a block [outer, size, inner] around dimension d:
//   outer = product of the sizes before d, size = shape[d], inner = product of the sizes after d.
// Element (o, r, i) of the block is at (o * size + r) * inner + i in a row-major buffer.
// Example: [2, 3, 4] around d = 1 -> outer 2, size 3, inner 4. Used by sum(dim) and slice's backward.
struct Block {
    std::int64_t outer;
    std::int64_t size;
    std::int64_t inner;
};
Block block_around(const Shape& shape, std::int64_t d);

// Should the result of an operation on `inputs` record a node? Only while recording is on
// (no NoGradGuard alive) and at least one input needs a gradient; otherwise autograd costs nothing.
// Defined in autograd.cpp.
bool should_record(const std::vector<Tensor>& inputs);

// Reverse of broadcasting, used by the backward formulas: sums g over the dimensions that
// broadcasting added on the left or stretched from 1, so that the result has `shape`.
// Example: g [3, 4] -> [4] sums dim 0; g [3, 4] -> [3, 1] sums dim 1 with keepdim.
// `shape` must broadcast to g.shape(). Defined in ops.cpp.
Tensor sum_to_shape(const Tensor& g, const Shape& shape);

}  // namespace mt::detail
