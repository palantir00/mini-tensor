#include "minitensor/ops.hpp"

#include "minitensor/autograd.hpp"

#include "internal.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mt {

namespace {

// --- Forward helpers ----------------------------------------------------------------------
// They only compute values; each public operation below records one node with its own formula.
// Views they use internally (broadcast expand, row-major copy) run under NoGradGuard: they are
// implementation details, and recording them would only allocate nodes that die right away.
// (In PyTorch the kernels run "below" autograd for the same reason.)

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
    NoGradGuard no_grad;  // ea and eb are internal; the caller's node already undoes broadcasting
    const Shape shape = broadcast_shapes(a.shape(), b.shape());
    const Tensor ea = a.expand(shape);
    const Tensor eb = b.expand(shape);
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(numel(shape)));
    walk_pair(ea, eb, 0, 0, 0, fn, out);
    return from_data(std::move(out), shape);
}

// Sum of all elements, read in logical order.
float sum_values(const Tensor& t) {
    float total = 0.0f;
    for (float x : t.to_vector()) total += x;
    return total;
}

// Sums t along dimension d (already wrapped); returns the result's values in row-major order.
// Seen as a block [outer, size, inner] around d (detail::block_around), element (o, r, i) of t
// is at (o * size + r) * inner + i, and we add up over r.
// Example: [2, 3, 4] summed over dim 1 -> outer 2, size 3, inner 4, result [2, 4].
std::vector<float> sum_dim_values(const Tensor& t, std::int64_t d) {
    NoGradGuard no_grad;              // c is an internal copy; the public sum records a node for t
    const Tensor c = t.contiguous();  // the formula needs row-major order; copies only if t is not
    const detail::Block b = detail::block_around(c.shape(), d);
    const float* src = c.data_ptr();
    std::vector<float> out(static_cast<std::size_t>(b.outer * b.inner), 0.0f);
    for (std::int64_t o = 0; o < b.outer; ++o) {
        for (std::int64_t r = 0; r < b.size; ++r) {
            for (std::int64_t i = 0; i < b.inner; ++i) {
                out[static_cast<std::size_t>(o * b.inner + i)] += src[(o * b.size + r) * b.inner + i];
            }
        }
    }
    return out;
}

Tensor scalar(float value) { return from_data({value}, {}); }

// --- Backward nodes: one gradient formula per operation -----------------------------------
// backward(grad) gets dC = d loss / d result and returns d loss / d input for every input.
// inputs() are the tensors the operation was called with, saved when the node was recorded.
// `using Node::Node;` reuses Node's constructor, the one that takes the inputs.
// backward() runs under NoGradGuard, so the operations used here record nothing.

// C = A + B:  dA = dC, dB = dC, each summed back to its input's shape (undoes broadcasting).
class AddBackward : public Node {
public:
    using Node::Node;
    const char* name() const override { return "AddBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override {
        return {detail::sum_to_shape(grad, inputs()[0].shape()),
                detail::sum_to_shape(grad, inputs()[1].shape())};
    }
};

// C = A - B:  dA = dC, dB = -dC.
class SubBackward : public Node {
public:
    using Node::Node;
    const char* name() const override { return "SubBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override {
        return {detail::sum_to_shape(grad, inputs()[0].shape()),
                detail::sum_to_shape(mul(grad, scalar(-1.0f)), inputs()[1].shape())};
    }
};

// C = A * B (element-wise):  dA = dC * B, dB = dC * A.
class MulBackward : public Node {
public:
    using Node::Node;
    const char* name() const override { return "MulBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override {
        const Tensor& a = inputs()[0];
        const Tensor& b = inputs()[1];
        return {detail::sum_to_shape(mul(grad, b), a.shape()), detail::sum_to_shape(mul(grad, a), b.shape())};
    }
};

// Y = relu(X):  dX = dY * [X > 0]  (0 at X = 0, as in PyTorch).
// Saves the input X, not the result Y: the result owns this node, so a node holding the
// result would form a cycle (result -> node -> result) that is never freed.
class ReluBackward : public Node {
public:
    using Node::Node;
    const char* name() const override { return "ReluBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override {
        const Tensor& x = inputs()[0];
        std::vector<float> mask = x.to_vector();
        for (float& v : mask) v = v > 0.0f ? 1.0f : 0.0f;
        return {mul(grad, from_data(std::move(mask), x.shape()))};
    }
};

// C = A @ B with A [m, k], B [k, n], dC [m, n]:
//   dA = dC @ B^T   ([m, n] @ [n, k] = [m, k])
//   dB = A^T @ dC   ([k, m] @ [m, n] = [k, n])
// The transposes are views; matmul reads them through strides, nothing is copied.
class MatmulBackward : public Node {
public:
    using Node::Node;
    const char* name() const override { return "MatmulBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override {
        const Tensor& a = inputs()[0];
        const Tensor& b = inputs()[1];
        return {matmul(grad, b.transpose(0, 1)), matmul(a.transpose(0, 1), grad)};
    }
};

// s = sum(X):  every element of X adds to s with weight 1, so dX = ds at every position.
// expand gives that without copying (stride 0); backward() stores a contiguous copy in grad().
class SumBackward : public Node {
public:
    using Node::Node;
    const char* name() const override { return "SumBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override { return {grad.expand(inputs()[0].shape())}; }
};

// S = sum(X, dim):  each element of X adds to one element of S with weight 1, so dS is copied
// back along dim. dS is first brought to keepdim form ([2] -> [2, 1]), then expanded to X's shape.
class SumDimBackward : public Node {
public:
    SumDimBackward(std::vector<Tensor> inputs, std::int64_t dim) : Node(std::move(inputs)), dim_(dim) {}
    const char* name() const override { return "SumDimBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override {
        const Tensor& x = inputs()[0];
        Shape kept = x.shape();
        kept[dim_] = 1;
        return {grad.reshape(kept).expand(x.shape())};
    }

private:
    std::int64_t dim_;  // already wrapped to [0, ndim)
};

// m = mean(X) = sum(X) / n:  dX = dm / n at every position.
class MeanBackward : public Node {
public:
    using Node::Node;
    const char* name() const override { return "MeanBackward"; }
    std::vector<Tensor> backward(const Tensor& grad) const override {
        const Tensor& x = inputs()[0];
        const Tensor scaled = mul(grad, scalar(1.0f / static_cast<float>(x.numel())));
        return {scaled.expand(x.shape())};
    }
};

}  // namespace

namespace detail {

Tensor sum_to_shape(const Tensor& g, const Shape& shape) {
    Tensor result = g;
    // Dimensions that broadcasting added on the left: sum them away (each time the first one).
    const std::int64_t added = result.dim() - static_cast<std::int64_t>(shape.size());
    for (std::int64_t i = 0; i < added; ++i) result = sum(result, 0);
    // Dimensions that broadcasting stretched from 1: sum them back to size 1.
    for (std::int64_t d = 0; d < result.dim(); ++d) {
        if (shape[d] == 1 && result.shape()[d] != 1) result = sum(result, d, true);
    }
    return result;
}

}  // namespace detail

// --- Element-wise -------------------------------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out = elementwise(a, b, [](float x, float y) { return x + y; });
    if (detail::should_record({a, b})) out.set_grad_fn(std::make_shared<AddBackward>(std::vector<Tensor>{a, b}));
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out = elementwise(a, b, [](float x, float y) { return x - y; });
    if (detail::should_record({a, b})) out.set_grad_fn(std::make_shared<SubBackward>(std::vector<Tensor>{a, b}));
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = elementwise(a, b, [](float x, float y) { return x * y; });
    if (detail::should_record({a, b})) out.set_grad_fn(std::make_shared<MulBackward>(std::vector<Tensor>{a, b}));
    return out;
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }

Tensor relu(const Tensor& t) {
    std::vector<float> values = t.to_vector();  // this copy becomes the result's buffer
    // std::max(x, 0) returns x when the comparison x < 0 is false, so NaN stays NaN.
    for (float& x : values) x = std::max(x, 0.0f);
    Tensor out = from_data(std::move(values), t.shape());
    if (detail::should_record({t})) out.set_grad_fn(std::make_shared<ReluBackward>(std::vector<Tensor>{t}));
    return out;
}

// --- Matrix product -----------------------------------------------------------------------

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
    Tensor result = from_data(std::move(out), {m, n});
    if (detail::should_record({a, b})) result.set_grad_fn(std::make_shared<MatmulBackward>(std::vector<Tensor>{a, b}));
    return result;
}

// --- Reductions ---------------------------------------------------------------------------

Tensor sum(const Tensor& t) {
    Tensor out = scalar(sum_values(t));
    if (detail::should_record({t})) out.set_grad_fn(std::make_shared<SumBackward>(std::vector<Tensor>{t}));
    return out;
}

Tensor sum(const Tensor& t, std::int64_t dim, bool keepdim) {
    const std::int64_t d = detail::wrap_dim(dim, t.dim(), "sum");
    Shape out_shape = t.shape();
    if (keepdim) {
        out_shape[d] = 1;
    } else {
        out_shape.erase(out_shape.begin() + d);
    }
    Tensor result = from_data(sum_dim_values(t, d), out_shape);
    if (detail::should_record({t})) result.set_grad_fn(std::make_shared<SumDimBackward>(std::vector<Tensor>{t}, d));
    return result;
}

Tensor mean(const Tensor& t) {
    Tensor out = scalar(sum_values(t) / static_cast<float>(t.numel()));
    if (detail::should_record({t})) out.set_grad_fn(std::make_shared<MeanBackward>(std::vector<Tensor>{t}));
    return out;
}

}  // namespace mt
