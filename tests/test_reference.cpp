// Comparison with PyTorch. Every case has a twin with the same name in scripts/gen_reference.py;
// inputs, output and gradients come from tests/data/<case>.txt, so PyTorch is not needed here.

#include "minitensor/ops.hpp"

#include "reference.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using mt_test::Reference;
using Fn = std::function<mt::Tensor(const Reference&)>;

bool is_input(const std::string& name) { return name != "out" && name.rfind("grad_", 0) != 0; }

// Runs one case the way gen_reference.py did and compares every result with PyTorch:
//   1. inputs = all tensors in the file except out and grad_*; an input that has
//      grad_<name> in the file gets requires_grad
//   2. out = f(inputs), compared with "out"
//   3. out.backward(grad_out) if the file has grad_out, plain backward() for a scalar
//   4. the gradient of every such input compared with grad_<name>
void check_case(const std::string& name, const Fn& f) {
    const Reference ref = mt_test::load_reference(name);

    Reference inputs;
    for (const auto& [key, tensor] : ref) {
        if (!is_input(key)) continue;
        mt::Tensor input = tensor;
        if (ref.count("grad_" + key) > 0) input.set_requires_grad(true);
        inputs.emplace(key, input);
    }

    const mt::Tensor out = f(inputs);
    ASSERT_EQ(out.shape(), ref.at("out").shape()) << "out";  // backward would fail with a wrong shape
    EXPECT_TRUE(mt_test::allclose(out, ref.at("out"))) << "out";

    const auto grad_out = ref.find("grad_out");
    if (grad_out != ref.end()) {
        out.backward(grad_out->second);
    } else {
        out.backward();
    }

    for (const auto& [key, input] : inputs) {
        const auto expected = ref.find("grad_" + key);
        if (expected == ref.end()) continue;
        const std::optional<mt::Tensor> grad = input.grad();
        if (!grad) {
            ADD_FAILURE() << key << " got no gradient";
            continue;
        }
        EXPECT_TRUE(mt_test::allclose(*grad, expected->second)) << "grad_" << key;
    }
}

}  // namespace

// --- Element-wise with broadcasting -------------------------------------------------------

TEST(Reference, AddBroadcast) {
    check_case("add_broadcast", [](const Reference& in) { return in.at("a") + in.at("b"); });
}

TEST(Reference, SubBroadcast) {
    check_case("sub_broadcast", [](const Reference& in) { return in.at("a") - in.at("b"); });
}

TEST(Reference, MulBroadcast3d) {
    check_case("mul_broadcast_3d", [](const Reference& in) { return in.at("a") * in.at("b"); });
}

TEST(Reference, Relu) {
    check_case("relu", [](const Reference& in) { return mt::relu(in.at("x")); });
}

TEST(Reference, ReusedInput) {
    check_case("reused_input", [](const Reference& in) {
        const mt::Tensor& x = in.at("x");
        return x * x + x;
    });
}

// --- Matrix product -------------------------------------------------------------------------

TEST(Reference, Matmul) {
    check_case("matmul", [](const Reference& in) { return mt::matmul(in.at("a"), in.at("b")); });
}

TEST(Reference, MatmulTransposed) {
    check_case("matmul_transposed",
               [](const Reference& in) { return mt::matmul(in.at("a").transpose(0, 1), in.at("b")); });
}

// --- Reductions -----------------------------------------------------------------------------

TEST(Reference, SumDimKeepdim) {
    check_case("sum_dim_keepdim", [](const Reference& in) { return mt::sum(in.at("x"), 1, true); });
}

TEST(Reference, SumLastDim) {
    check_case("sum_last_dim", [](const Reference& in) { return mt::sum(in.at("x"), -1); });
}

TEST(Reference, SumTimesMean) {
    check_case("sum_times_mean", [](const Reference& in) { return mt::sum(in.at("a")) * mt::mean(in.at("b")); });
}

// --- Views ----------------------------------------------------------------------------------

TEST(Reference, ViewsChain) {
    // x[3, 4] -> transpose [4, 3] -> rows 1 and 3 [2, 3] -> [1, 6] (through a copy) -> [2, 6]
    check_case("views_chain", [](const Reference& in) {
        return in.at("x").transpose(0, 1).slice(0, 1, 4, 2).reshape({1, 6}).expand({2, 6});
    });
}

TEST(Reference, Slice3d) {
    check_case("slice_3d", [](const Reference& in) {
        const mt::Tensor& x = in.at("x");
        return x.slice(1, 1, 5, 2) * x.slice(1, 0, 2);
    });
}

// --- Everything together: one step of the XOR network from stage 6 ---------------------------

TEST(Reference, Mlp) {
    check_case("mlp", [](const Reference& in) {
        const mt::Tensor h = mt::relu(mt::matmul(in.at("x"), in.at("w1")) + in.at("b1"));
        const mt::Tensor d = mt::matmul(h, in.at("w2")) + in.at("b2") - in.at("y");
        return mt::mean(d * d);
    });
}

// --- The checks themselves --------------------------------------------------------------------

TEST(ReferenceHelpers, AllcloseFollowsTorchFormula) {
    const mt::Tensor expected = mt::from_data({1.0f, 100.0f}, {2});
    EXPECT_TRUE(mt_test::allclose(mt::from_data({1.000005f, 100.0009f}, {2}), expected));  // within atol + rtol*|b|
    EXPECT_FALSE(mt_test::allclose(mt::from_data({1.0001f, 100.0f}, {2}), expected));
    EXPECT_FALSE(mt_test::allclose(mt::from_data({1.0f, 100.0f, 0.0f}, {3}), expected));  // shape
    EXPECT_FALSE(mt_test::allclose(mt::from_data({std::nanf(""), 100.0f}, {2}), expected));
}

TEST(ReferenceHelpers, MissingFileThrows) {
    EXPECT_THROW(mt_test::load_reference("no_such_case"), std::runtime_error);
}
