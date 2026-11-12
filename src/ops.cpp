#include "minitensor/ops.hpp"

#include "internal.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mt {

namespace {

// Combines two numbers into one, e.g. x + y. A lambda without captures converts to this type.
using BinaryFn = float (*)(float, float);

// Element-wise loop over two tensors of the same shape, in row-major order.
// Same recursion as gather_elements in tensor.cpp, but it walks two tensors at once,
// each with its own strides. pa and pb are positions (relative to data_ptr()) of the first
// element of the current sub-tensor, the one spanning dimensions d..ndim-1.
void walk_pair(const Tensor& a, const Tensor& b, std::size_t d, std::int64_t pa, std::int64_t pb, BinaryFn fn,
               std::vector<float>& out) {
    if (d == a.shape().size()) {  // all indices fixed: one pair of numbers
        out.push_back(fn(a.data_ptr()[pa], b.data_ptr()[pb]));
        return;
    }
    for (std::int64_t i = 0; i < a.shape()[d]; ++i) {
        walk_pair(a, b, d + 1, pa + i * a.strides()[d], pb + i * b.strides()[d], fn, out);
    }
}

// Broadcasts a and b to a common shape (views with stride 0, nothing is copied),
// then applies fn to every pair of elements.
Tensor elementwise(const Tensor& a, const Tensor& b, BinaryFn fn) {
    const Shape shape = broadcast_shapes(a.shape(), b.shape());
    const Tensor ea = a.expand(shape);
    const Tensor eb = b.expand(shape);
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(numel(shape)));
    walk_pair(ea, eb, 0, 0, 0, fn, out);
    return from_data(std::move(out), shape);
}

}  // namespace

// --- Element-wise -----------------------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](float x, float y) { return x + y; });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](float x, float y) { return x * y; });
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }

Tensor relu(const Tensor& t) {
    std::vector<float> values = t.to_vector();  // this copy becomes the result's buffer
    // std::max(x, 0) returns x when the comparison x < 0 is false, so NaN stays NaN.
    for (float& x : values) x = std::max(x, 0.0f);
    return from_data(std::move(values), t.shape());
}

// --- Matrix product ---------------------------------------------------------------

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::invalid_argument("matmul: expected two 2-D tensors, got shapes " +
                                    detail::shape_to_string(a.shape()) + " and " +
                                    detail::shape_to_string(b.shape()));
    }
    const std::int64_t m = a.shape()[0];
    const std::int64_t k = a.shape()[1];
    const std::int64_t n = b.shape()[1];
    if (b.shape()[0] != k) {
        throw std::invalid_argument("matmul: shapes " + detail::shape_to_string(a.shape()) + " and " +
                                    detail::shape_to_string(b.shape()) + " cannot be multiplied (" +
                                    std::to_string(k) + " != " + std::to_string(b.shape()[0]) + ")");
    }

    // Both inputs are read through their own strides, so a transposed or sliced view
    // needs no copy: A[i][p] = pa[i * a_row + p * a_col].
    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    const std::int64_t a_row = a.strides()[0];
    const std::int64_t a_col = a.strides()[1];
    const std::int64_t b_row = b.strides()[0];
    const std::int64_t b_col = b.strides()[1];

    std::vector<float> out(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;  // C[i][j] = sum over p of A[i][p] * B[p][j]
            for (std::int64_t p = 0; p < k; ++p) {
                acc += pa[i * a_row + p * a_col] * pb[p * b_row + j * b_col];
            }
            out[static_cast<std::size_t>(i * n + j)] = acc;
        }
    }
    return from_data(std::move(out), {m, n});
}

// --- Reductions -------------------------------------------------------------------

Tensor sum(const Tensor& t) {
    float total = 0.0f;
    for (float x : t.to_vector()) total += x;
    return from_data({total}, {});
}

Tensor sum(const Tensor& t, std::int64_t dim, bool keepdim) {
    const std::int64_t d = detail::wrap_dim(dim, t.dim(), "sum");

    // A row-major tensor can be seen as a block [outer, size, inner]:
    //   outer = product of the sizes before d, size = shape[d], inner = product of the sizes after d.
    // Element (o, r, i) is at (o * size + r) * inner + i, and we add up over r.
    // Example: [2, 3, 4] summed over dim 1 -> outer 2, size 3, inner 4, result [2, 4].
    const Tensor c = t.contiguous();  // the formula needs row-major order; copies only if t is not
    const Shape& shape = c.shape();
    std::int64_t outer = 1;
    for (std::int64_t k = 0; k < d; ++k) outer *= shape[k];
    const std::int64_t size = shape[d];
    std::int64_t inner = 1;
    for (std::int64_t k = d + 1; k < c.dim(); ++k) inner *= shape[k];

    const float* src = c.data_ptr();
    std::vector<float> out(static_cast<std::size_t>(outer * inner), 0.0f);
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t r = 0; r < size; ++r) {
            for (std::int64_t i = 0; i < inner; ++i) {
                out[static_cast<std::size_t>(o * inner + i)] += src[(o * size + r) * inner + i];
            }
        }
    }

    Shape out_shape = shape;
    if (keepdim) {
        out_shape[d] = 1;
    } else {
        out_shape.erase(out_shape.begin() + d);
    }
    return from_data(std::move(out), out_shape);
}

Tensor mean(const Tensor& t) {
    const float total = sum(t).at({});
    return from_data({total / static_cast<float>(t.numel())}, {});
}

}  // namespace mt
