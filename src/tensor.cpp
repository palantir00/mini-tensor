#include "minitensor/tensor.hpp"

#include "internal.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mt {

// --- Helpers shared with ops.cpp (declared in internal.hpp) --------------------------

namespace detail {

std::string shape_to_string(const Shape& shape) {
    std::string s = "[";
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (d > 0) s += ", ";
        s += std::to_string(shape[d]);
    }
    return s + "]";
}

std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim, const char* where) {
    if (dim < -ndim || dim >= ndim) {
        throw std::out_of_range(std::string(where) + ": dim " + std::to_string(dim) +
                                " is out of range for a tensor with " + std::to_string(ndim) + " dimensions");
    }
    return dim < 0 ? dim + ndim : dim;
}

}  // namespace detail

namespace {

void check_shape(const Shape& shape, const char* where) {
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] < 0) {
            throw std::invalid_argument(std::string(where) + ": negative size in shape " +
                                        detail::shape_to_string(shape));
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

// Same walk as print_elements, but appends the numbers to out instead of printing them.
void gather_elements(const TensorImpl& t, std::size_t d, std::int64_t pos, std::vector<float>& out) {
    if (d == t.shape.size()) {
        out.push_back(t.storage->data[static_cast<std::size_t>(pos)]);
        return;
    }
    for (std::int64_t i = 0; i < t.shape[d]; ++i) {
        gather_elements(t, d + 1, pos + i * t.strides[d], out);
    }
}

std::invalid_argument reshape_error(const Shape& shape, std::int64_t n) {
    return std::invalid_argument("reshape: shape " + detail::shape_to_string(shape) + " is invalid for " +
                                 std::to_string(n) + " elements");
}

// Validates a reshape target for n elements and replaces a single -1 with the missing size.
Shape infer_shape(Shape shape, std::int64_t n) {
    std::int64_t known = 1;      // product of the sizes that are given
    std::int64_t infer_at = -1;  // position of the -1, if there is one
    for (std::int64_t d = 0; d < static_cast<std::int64_t>(shape.size()); ++d) {
        if (shape[d] == -1 && infer_at == -1) {
            infer_at = d;
        } else if (shape[d] < 0) {  // any other negative size, or a second -1
            throw reshape_error(shape, n);
        } else {
            known *= shape[d];
        }
    }
    if (infer_at == -1) {
        if (known != n) throw reshape_error(shape, n);
    } else {
        // known == 0: any size would fit, so the -1 is ambiguous (PyTorch rejects it too).
        if (known == 0 || n % known != 0) throw reshape_error(shape, n);
        shape[infer_at] = n / known;
    }
    return shape;
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

const float* Tensor::data_ptr() const { return impl_->storage->data.data() + impl_->offset; }

std::vector<float> Tensor::to_vector() const {
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(numel()));
    gather_elements(*impl_, 0, impl_->offset, out);
    return out;
}

bool Tensor::is_same(const Tensor& other) const { return impl_ == other.impl_; }

// --- Autograd state (the engine itself is in autograd.cpp) --------------------------------

bool Tensor::requires_grad() const { return impl_->requires_grad; }

Tensor Tensor::set_requires_grad(bool value) {
    if (!is_leaf()) {
        throw std::logic_error("set_requires_grad: only leaves can change requires_grad; "
                               "the result of an operation gets it from its inputs");
    }
    impl_->requires_grad = value;
    return *this;
}

bool Tensor::is_leaf() const { return impl_->grad_fn == nullptr; }

std::optional<Tensor> Tensor::grad() const {
    if (!impl_->grad) return std::nullopt;
    return Tensor(impl_->grad);
}

void Tensor::zero_grad() { impl_->grad.reset(); }

std::shared_ptr<Node> Tensor::grad_fn() const { return impl_->grad_fn; }

void Tensor::set_grad_fn(std::shared_ptr<Node> node) {
    impl_->grad_fn = std::move(node);
    impl_->requires_grad = true;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << kPrintPrefix;
    print_elements(os, *t.impl_, 0, t.impl_->offset);
    return os << ')';
}

// --- Views ----------------------------------------------------------------------

Tensor Tensor::make_view(Shape shape, Strides strides, std::int64_t offset) const {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = impl_->storage;  // the pointer is copied, the data is shared
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->offset = offset;
    return Tensor(std::move(impl));
}

Tensor Tensor::transpose(std::int64_t dim0, std::int64_t dim1) const {
    const std::int64_t d0 = detail::wrap_dim(dim0, dim(), "transpose");
    const std::int64_t d1 = detail::wrap_dim(dim1, dim(), "transpose");
    Shape shape = impl_->shape;
    Strides strides = impl_->strides;
    std::swap(shape[d0], shape[d1]);
    std::swap(strides[d0], strides[d1]);
    return make_view(std::move(shape), std::move(strides), impl_->offset);
}

Tensor Tensor::slice(std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) const {
    const std::int64_t d = detail::wrap_dim(dim, this->dim(), "slice");  // the parameter hides the method dim()
    if (step <= 0) {
        throw std::invalid_argument("slice: step must be positive, got " + std::to_string(step));
    }
    const std::int64_t size = impl_->shape[d];
    // Python rules: negative bounds count from the end, then both are clamped to [0, size].
    if (start < 0) start += size;
    if (end < 0) end += size;
    start = std::clamp<std::int64_t>(start, 0, size);
    end = std::clamp<std::int64_t>(end, start, size);  // end before start gives an empty slice

    Shape shape = impl_->shape;
    Strides strides = impl_->strides;
    shape[d] = (end - start + step - 1) / step;  // ceil((end - start) / step)
    strides[d] *= step;                          // jump over step-1 elements each time
    return make_view(std::move(shape), std::move(strides), impl_->offset + start * impl_->strides[d]);
}

Tensor Tensor::reshape(const Shape& shape) const {
    const Shape new_shape = infer_shape(shape, numel());
    // Row-major strides fit the new shape only if the elements already lie in row-major order,
    // so a non-contiguous tensor is copied first.
    const Tensor src = contiguous();
    return src.make_view(new_shape, contiguous_strides(new_shape), src.offset());
}

bool Tensor::is_contiguous() const {
    std::int64_t expected = 1;  // stride that row-major order needs for dimension d
    for (std::int64_t d = dim() - 1; d >= 0; --d) {
        if (impl_->shape[d] == 1) continue;  // only index 0 exists, so this stride is never used
        if (impl_->strides[d] != expected) return false;
        expected *= impl_->shape[d];
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return from_data(to_vector(), impl_->shape);
}

Tensor Tensor::expand(const Shape& shape) const {
    check_shape(shape, "expand");
    const std::int64_t added = static_cast<std::int64_t>(shape.size()) - dim();  // new dims on the left
    if (added < 0) {
        throw std::invalid_argument("expand: cannot expand shape " + detail::shape_to_string(impl_->shape) +
                                    " to " + detail::shape_to_string(shape) + " with fewer dimensions");
    }
    // Stride 0 = "do not move in storage": every index along that dimension reads the same element.
    // New dims on the left and stretched size-1 dims get stride 0; matching dims keep their stride.
    Strides strides(shape.size(), 0);
    for (std::int64_t d = 0; d < dim(); ++d) {
        const std::int64_t size = impl_->shape[d];
        if (size == shape[added + d]) {
            strides[added + d] = impl_->strides[d];
        } else if (size != 1) {
            throw std::invalid_argument("expand: cannot expand shape " + detail::shape_to_string(impl_->shape) +
                                        " to " + detail::shape_to_string(shape) +
                                        ": only dimensions of size 1 can be stretched");
        }
    }
    return make_view(shape, std::move(strides), impl_->offset);
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

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    const std::size_t n = std::max(a.size(), b.size());
    Shape out(n);
    // i counts from the right (i = 0 is the last dimension); a missing dimension counts as size 1.
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t sa = i < a.size() ? a[a.size() - 1 - i] : 1;
        const std::int64_t sb = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (sa != sb && sa != 1 && sb != 1) {
            throw std::invalid_argument("broadcast_shapes: shapes " + detail::shape_to_string(a) + " and " +
                                        detail::shape_to_string(b) + " are not broadcastable (sizes " +
                                        std::to_string(sa) + " and " + std::to_string(sb) + " at dimension -" +
                                        std::to_string(i + 1) + ")");
        }
        out[n - 1 - i] = (sa == 1) ? sb : sa;
    }
    return out;
}

Tensor from_data(std::vector<float> data, const Shape& shape) {
    check_shape(shape, "from_data");
    const std::int64_t n = numel(shape);
    if (static_cast<std::int64_t>(data.size()) != n) {
        throw std::invalid_argument("from_data: shape " + detail::shape_to_string(shape) + " needs " +
                                    std::to_string(n) + " values, got " + std::to_string(data.size()));
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
