#pragma once

// Numerical gradient check for the tests (not part of the library).

#include "minitensor/autograd.hpp"
#include "minitensor/ops.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace mt_test {

// Function under test: takes the inputs, returns a tensor of any shape.
using Fn = std::function<mt::Tensor(const std::vector<mt::Tensor>&)>;

// Flat row-major position -> index, e.g. 4 in shape [2, 3] -> {1, 1}.
inline mt::Index unravel(std::int64_t flat, const mt::Shape& shape) {
    mt::Index index(shape.size());
    for (std::int64_t d = static_cast<std::int64_t>(shape.size()) - 1; d >= 0; --d) {
        index[d] = flat % shape[d];
        flat /= shape[d];
    }
    return index;
}

// Checks backward() against numerical derivatives (central differences):
//     d loss / d x  ~  (loss(x + eps) - loss(x - eps)) / (2 * eps)
// for every element x of every input. loss = sum(f(inputs) * w) with fixed random weights w,
// so each output element counts with a different weight (with a plain sum, errors in
// different elements could cancel out). Inputs must be leaves with requires_grad; their
// gradients are reset first. Passes if |backward - numerical| <= atol + rtol * |numerical|.
// float32 rounding limits the precision, hence the loose tolerances (PyTorch's gradcheck
// expects float64 for this reason).
inline ::testing::AssertionResult gradcheck(const Fn& f, std::vector<mt::Tensor> inputs, float eps = 1e-3f,
                                            float atol = 1e-2f, float rtol = 1e-2f) {
    const mt::Tensor weights = mt::randn(f(inputs).shape(), 1234);
    auto loss = [&] { return mt::sum(mt::mul(f(inputs), weights)); };

    // 1. Gradients from backward().
    for (mt::Tensor& t : inputs) t.zero_grad();
    loss().backward();

    // 2. Numerical derivative of every element, compared with step 1.
    mt::NoGradGuard no_grad;  // here we only need the values, not a graph
    for (std::size_t k = 0; k < inputs.size(); ++k) {
        mt::Tensor& input = inputs[k];
        const std::optional<mt::Tensor> grad = input.grad();
        if (!grad) return ::testing::AssertionFailure() << "input " << k << " got no gradient";

        for (std::int64_t i = 0; i < input.numel(); ++i) {
            const mt::Index index = unravel(i, input.shape());
            float& x = input.at(index);  // changed in place, restored below
            const float original = x;
            x = original + eps;
            const float plus = loss().at({});
            x = original - eps;
            const float minus = loss().at({});
            x = original;

            const float numerical = (plus - minus) / (2 * eps);
            const float analytic = grad->at(index);
            if (std::fabs(analytic - numerical) > atol + rtol * std::fabs(numerical)) {
                return ::testing::AssertionFailure() << "input " << k << ", element " << i << ": backward gives "
                                                     << analytic << ", numerical derivative is " << numerical;
            }
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace mt_test
