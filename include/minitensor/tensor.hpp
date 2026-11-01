#pragma once

#include <cstdint>
#include <memory>
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

// What a tensor really is: a buffer plus the recipe for reading it.
// Element [i0, i1, ...] lives at data[offset + i0*strides[0] + i1*strides[1] + ...].
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape shape;
    Strides strides;          // in elements, not bytes
    std::int64_t offset = 0;  // position of element [0, 0, ...] in storage
};

// Lightweight handle: copying a Tensor copies the pointer, not the data,
// so after `Tensor b = a;` a and b are the same tensor (as in PyTorch).
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

    // True if both handles point to the same tensor.
    bool is_same(const Tensor& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    std::int64_t storage_index(const Index& index) const;

    std::shared_ptr<TensorImpl> impl_;
};

// Product of all sizes; 1 for a scalar (shape {}).
std::int64_t numel(const Shape& shape);

// Row-major strides: the last dimension has stride 1.
Strides contiguous_strides(const Shape& shape);

// Factories; each returns a new contiguous tensor with its own storage.
Tensor from_data(std::vector<float> data, const Shape& shape);
Tensor zeros(const Shape& shape);
Tensor ones(const Shape& shape);
Tensor randn(const Shape& shape, std::uint64_t seed);  // standard normal N(0, 1)

// PyTorch-style output, e.g. tensor([[1, 2],
//                                    [3, 4]])
std::ostream& operator<<(std::ostream& os, const Tensor& t);

}  // namespace mt
