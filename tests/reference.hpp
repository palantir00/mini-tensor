#pragma once

// Reference data from PyTorch for the tests (not part of the library).
// The files in tests/data/ are written by scripts/gen_reference.py, which also describes
// the format. Names in a file: the inputs, "out", "grad_out" (none when out is a scalar)
// and "grad_<input>" for every input that requires grad.

#include "minitensor/tensor.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

namespace mt_test {

// All tensors of one case, by name.
using Reference = std::map<std::string, mt::Tensor>;

// Reads tests/data/<name>.txt. Throws std::runtime_error if the file is missing or malformed.
Reference load_reference(const std::string& name);

// Like torch.allclose: same shape and |actual - expected| <= atol + rtol * |expected|
// for every element. On failure, reports how many elements differ and the worst one.
::testing::AssertionResult allclose(const mt::Tensor& actual, const mt::Tensor& expected, float atol = 1e-5f,
                                    float rtol = 1e-5f);

}  // namespace mt_test
