#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

namespace mt {

// Sizes, strides and indices are signed 64-bit, like in PyTorch.
using Shape = std::vector<std::int64_t>;    // size of each dimension
using Strides = std::vector<std::int64_t>;  // elements to skip per step in each dimension
using Index = std::vector<std::int64_t>;    // position of one element

// One flat buffer of floats. Several tensors (views) may share it.
struct Storage {
    std::vector<float> data;
};

class Node;  // one recorded operation of the autograd graph, defined in autograd.hpp

// What a tensor really is: a buffer plus the recipe for reading it.
// Element [i0, i1, ...] lives at data[offset + i0*strides[0] + i1*strides[1] + ...].
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape shape;
    Strides strides;          // in elements, not bytes
    std::int64_t offset = 0;  // position of element [0, 0, ...] in storage

    // --- Autograd ---
    bool requires_grad = false;        // does the gradient flow through this tensor?
    std::shared_ptr<TensorImpl> grad;  // summed up by backward(); leaves only, nullptr = none yet
    std::shared_ptr<Node> grad_fn;     // operation that produced this tensor; nullptr = leaf
};

// Lightweight handle: copying a Tensor copies the pointer, not the data,
// so after `Tensor b = a;` a and b are the same tensor (as in PyTorch).
// const applies to the handle, not to the data: a view of a const Tensor can still write to it.
class Tensor {
public:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    const Shape& shape() const;
    const Strides& strides() const;
    std::int64_t offset() const;
    std::int64_t dim() const;
    std::int64_t numel() const;

    // Element access; index must have exactly dim() entries.
    float at(const Index& index) const;
    float& at(const Index& index);

    // Pointer to element [0, 0, ...] (storage start + offset), like torch's data_ptr().
    // Element [i0, i1, ...] is data_ptr()[i0*strides()[0] + i1*strides()[1] + ...].
    const float* data_ptr() const;

    // All elements in logical (row-major) order, whatever the strides are.
    std::vector<float> to_vector() const;

    // True if both handles point to the same tensor.
    bool is_same(const Tensor& other) const;

    // --- Views: new tensors over the same storage, no data is copied ---
    // Negative dims count from the end (-1 is the last dimension), as in PyTorch.

    // Swaps two dimensions (their sizes and strides).
    Tensor transpose(std::int64_t dim0, std::int64_t dim1) const;

    // Elements start, start+step, ... before end along dim, like t[start:end:step] in Python:
    // negative bounds count from the end, out-of-range bounds are clamped. step must be > 0.
    Tensor slice(std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1) const;

    // Same elements in a new shape; one size may be -1 and is then inferred.
    // A view if this tensor is contiguous, otherwise a copy (like torch.reshape).
    Tensor reshape(const Shape& shape) const;

    // True if reading in row-major order visits storage one element after another.
    bool is_contiguous() const;

    // This tensor if it is already contiguous, otherwise a contiguous copy.
    Tensor contiguous() const;

    // Stretches this tensor to a bigger shape without copying (used by broadcasting).
    // Shapes are matched from the right: new dims may be added on the left, and a dim of size 1
    // may become any size. Both get stride 0, so all their indices read the same element.
    // Careful: writing through at() to such a dim changes every position that shares the element.
    Tensor expand(const Shape& shape) const;

    // --- Autograd (graph nodes and NoGradGuard in autograd.hpp) ---

    // True if the gradient flows through this tensor: a leaf you marked, or a result computed from one.
    bool requires_grad() const;

    // Marks a leaf as a parameter whose gradient backward() should compute (false unmarks it).
    // Only leaves: results of operations get requires_grad from their inputs (std::logic_error otherwise).
    // Returns this tensor, so it can be chained: Tensor w = randn({2, 3}, 0).set_requires_grad(true);
    Tensor set_requires_grad(bool value);

    // True if the tensor was not produced by a recorded operation (it has no grad_fn).
    bool is_leaf() const;

    // Gradient summed up by backward(); empty before the first backward(), like None in PyTorch.
    // Only leaves with requires_grad get one.
    std::optional<Tensor> grad() const;

    // Forgets the gradient (grad() is empty again), like zero_grad(set_to_none=True) in PyTorch.
    void zero_grad();

    // Operation that produced this tensor; nullptr for leaves.
    std::shared_ptr<Node> grad_fn() const;

    // Used by operations: attaches the node that produced this tensor and sets requires_grad.
    void set_grad_fn(std::shared_ptr<Node> node);

    // Runs the chain rule from this tensor back to the leaves and adds d this / d leaf to grad()
    // of every leaf with requires_grad. Without an argument this tensor must have exactly one
    // element (its gradient is then 1); otherwise pass d loss / d this, with the shape of this.
    // Defined in autograd.cpp. Known limit: changing a tensor through at() after the graph used it
    // silently breaks the gradients (PyTorch detects that with version counters; we do not).
    void backward() const;
    void backward(const Tensor& grad) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    // Adds g to the gradient of this leaf: an own copy the first time, in place afterwards.
    // Defined in autograd.cpp.
    void accumulate_grad(const Tensor& g) const;

    std::int64_t storage_index(const Index& index) const;

    // New tensor that shares this tensor's storage but reads it with a different layout.
    Tensor make_view(Shape shape, Strides strides, std::int64_t offset) const;

    std::shared_ptr<TensorImpl> impl_;
};

// Product of all sizes; 1 for a scalar (shape {}).
std::int64_t numel(const Shape& shape);

// Row-major strides: the last dimension has stride 1.
Strides contiguous_strides(const Shape& shape);

// Result shape of broadcasting a with b (NumPy rules): shapes are matched from the right,
// a missing dim counts as 1, two sizes fit if they are equal or one of them is 1.
// Example: [3, 1] and [4] give [3, 4]. Throws std::invalid_argument if the shapes do not fit.
Shape broadcast_shapes(const Shape& a, const Shape& b);

// Factories; each returns a new contiguous tensor with its own storage.
Tensor from_data(std::vector<float> data, const Shape& shape);
Tensor zeros(const Shape& shape);
Tensor ones(const Shape& shape);
Tensor randn(const Shape& shape, std::uint64_t seed);  // standard normal N(0, 1)

// PyTorch-style output, e.g. tensor([[1, 2],
//                                    [3, 4]])
std::ostream& operator<<(std::ostream& os, const Tensor& t);

}  // namespace mt
