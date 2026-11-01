#include "minitensor/tensor.hpp"

#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mt {

namespace {

std::string shape_to_string(const Shape& shape) {
    std::string s = "[";
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (d > 0) s += ", ";
        s += std::to_string(shape[d]);
    }
    return s + "]";
}

void check_shape(const Shape& shape, const char* where) {
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] < 0) {
            throw std::invalid_argument(std::string(where) + ": negative size in shape " +
                                        shape_to_string(shape));
        }
    }
}

// Wraps a row-major buffer into a new tensor that owns it.
Tensor make_contiguous(std::vector<float> data, const Shape& shape) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(Storage{std::move(data)});
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    impl->offset = 0;
    return Tensor(std::move(impl));
}

Tensor full(const Shape& shape, float value, const char* where) {
    check_shape(shape, where);
    return make_contiguous(std::vector<float>(static_cast<std::size_t>(numel(shape)), value), shape);
}

constexpr std::string_view kPrintPrefix = "tensor(";

// Prints the sub-tensor spanning dimensions d..ndim-1 whose first element is at storage[pos].
// Walks with strides, so it works for any layout, not only row-major.
void print_elements(std::ostream& os, const TensorImpl& t, std::size_t d, std::int64_t pos) {
    const std::size_t ndim = t.shape.size();
    if (d == ndim) {  // all indices fixed: a single number
        os << t.storage->data[static_cast<std::size_t>(pos)];
        return;
    }
    os << '[';
    for (std::int64_t i = 0; i < t.shape[d]; ++i) {
        if (i > 0) {
            os << ',';
            if (d + 1 == ndim) {
                os << ' ';  // numbers of the last dimension stay on one line
            } else {
                // Like PyTorch: one newline per remaining dimension, then align under '['.
                os << std::string(ndim - d - 1, '\n') << std::string(kPrintPrefix.size() + d + 1, ' ');
            }
        }
        print_elements(os, t, d + 1, pos + i * t.strides[d]);
    }
    os << ']';
}

}  // namespace

// --- Tensor -------------------------------------------------------------------

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    if (!impl_ || !impl_->storage) {
        throw std::invalid_argument("Tensor: impl and storage must not be null");
    }
    if (impl_->strides.size() != impl_->shape.size()) {
        throw std::invalid_argument("Tensor: strides must have one entry per dimension");
    }
}

const Shape& Tensor::shape() const { return impl_->shape; }
const Strides& Tensor::strides() const { return impl_->strides; }
std::int64_t Tensor::offset() const { return impl_->offset; }
std::int64_t Tensor::dim() const { return static_cast<std::int64_t>(impl_->shape.size()); }
std::int64_t Tensor::numel() const { return mt::numel(impl_->shape); }

// offset + sum of index[d] * strides[d], with bounds checks.
std::int64_t Tensor::storage_index(const Index& index) const {
    if (static_cast<std::int64_t>(index.size()) != dim()) {
        throw std::invalid_argument("at: expected " + std::to_string(dim()) + " indices, got " +
                                    std::to_string(index.size()));
    }
    std::int64_t pos = impl_->offset;
    for (std::int64_t d = 0; d < dim(); ++d) {
        const std::int64_t i = index[d];
        if (i < 0 || i >= impl_->shape[d]) {
            throw std::out_of_range("at: index " + std::to_string(i) + " is out of range for dimension " +
                                    std::to_string(d) + " of size " + std::to_string(impl_->shape[d]));
        }
        pos += i * impl_->strides[d];
    }
    return pos;
}

float Tensor::at(const Index& index) const {
    return impl_->storage->data[static_cast<std::size_t>(storage_index(index))];
}

float& Tensor::at(const Index& index) {
    return impl_->storage->data[static_cast<std::size_t>(storage_index(index))];
}

bool Tensor::is_same(const Tensor& other) const { return impl_ == other.impl_; }

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << kPrintPrefix;
    print_elements(os, *t.impl_, 0, t.impl_->offset);
    return os << ')';
}

// --- Free functions -------------------------------------------------------------

std::int64_t numel(const Shape& shape) {
    std::int64_t n = 1;
    for (std::int64_t size : shape) n *= size;
    return n;
}

Strides contiguous_strides(const Shape& shape) {
    Strides strides(shape.size());
    std::int64_t step = 1;
    // From the last dimension to the first; with a signed index the loop can stop at d >= 0.
    for (std::int64_t d = static_cast<std::int64_t>(shape.size()) - 1; d >= 0; --d) {
        strides[d] = step;
        step *= shape[d];
    }
    return strides;
}

Tensor from_data(std::vector<float> data, const Shape& shape) {
    check_shape(shape, "from_data");
    const std::int64_t n = numel(shape);
    if (static_cast<std::int64_t>(data.size()) != n) {
        throw std::invalid_argument("from_data: shape " + shape_to_string(shape) + " needs " + std::to_string(n) +
                                    " values, got " + std::to_string(data.size()));
    }
    return make_contiguous(std::move(data), shape);
}

Tensor zeros(const Shape& shape) { return full(shape, 0.0f, "zeros"); }

Tensor ones(const Shape& shape) { return full(shape, 1.0f, "ones"); }

Tensor randn(const Shape& shape, std::uint64_t seed) {
    check_shape(shape, "randn");
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(static_cast<std::size_t>(numel(shape)));
    for (float& x : data) x = dist(gen);
    return make_contiguous(std::move(data), shape);
}

}  // namespace mt
