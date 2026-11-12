#include "minitensor/autograd.hpp"
#include "minitensor/ops.hpp"

#include "gradcheck.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Values = std::vector<float>;

// Leaf that takes part in autograd.
mt::Tensor param(std::vector<float> values, const mt::Shape& shape) {
    return mt::from_data(std::move(values), shape).set_requires_grad(true);
}

// Gradient of t as plain numbers; fails the test if t has none.
Values grad_of(const mt::Tensor& t) {
    const std::optional<mt::Tensor> g = t.grad();
    if (!g) {
        ADD_FAILURE() << "tensor has no grad";
        return {};
    }
    return g->to_vector();
}

}  // namespace

// --- Recording the graph ----------------------------------------------------------------

TEST(Autograd, NewTensorIsLeafWithoutGrad) {
    mt::Tensor t = mt::ones({2});
    EXPECT_FALSE(t.requires_grad());
    EXPECT_TRUE(t.is_leaf());
    EXPECT_FALSE(t.grad().has_value());
    EXPECT_EQ(t.grad_fn(), nullptr);
}

TEST(Autograd, SetRequiresGradOnlyOnLeaves) {
    mt::Tensor x = mt::ones({2});
    mt::Tensor same = x.set_requires_grad(true);
    EXPECT_TRUE(same.is_same(x));  // returns the tensor itself, for chaining
    EXPECT_TRUE(x.requires_grad());
    mt::Tensor y = x * x;
    EXPECT_THROW(y.set_requires_grad(false), std::logic_error);
}

TEST(Autograd, OperationRecordsNodeWithItsInputs) {
    mt::Tensor a = param({1, 2}, {2});
    mt::Tensor b = mt::ones({2});  // does not require grad
    mt::Tensor c = a * b;
    EXPECT_TRUE(c.requires_grad());
    EXPECT_FALSE(c.is_leaf());
    ASSERT_NE(c.grad_fn(), nullptr);
    EXPECT_STREQ(c.grad_fn()->name(), "MulBackward");
    ASSERT_EQ(c.grad_fn()->inputs().size(), 2u);
    EXPECT_TRUE(c.grad_fn()->inputs()[0].is_same(a));
    EXPECT_TRUE(c.grad_fn()->inputs()[1].is_same(b));
}

TEST(Autograd, NoNodeWhenNoInputNeedsGrad) {
    mt::Tensor c = mt::ones({2}) + mt::ones({2});
    EXPECT_FALSE(c.requires_grad());
    EXPECT_TRUE(c.is_leaf());
}

TEST(Autograd, NoGradGuardStopsRecordingAndNests) {
    mt::Tensor x = param({1}, {});
    EXPECT_TRUE(mt::is_grad_enabled());
    {
        mt::NoGradGuard outer;
        {
            mt::NoGradGuard inner;
        }
        EXPECT_FALSE(mt::is_grad_enabled());  // the inner guard restored "off", not "on"
        EXPECT_FALSE((x * x).requires_grad());
    }
    EXPECT_TRUE(mt::is_grad_enabled());
    EXPECT_TRUE((x * x).requires_grad());
}

// --- backward(): the engine -------------------------------------------------------------

TEST(Backward, ExampleFromNotes) {
    // z = x*y + x at x = 2, y = 3:  dz/dx = y + 1 = 4,  dz/dy = x = 2
    mt::Tensor x = param({2}, {});
    mt::Tensor y = param({3}, {});
    mt::Tensor z = x * y + x;
    z.backward();
    EXPECT_EQ(grad_of(x), (Values{4}));
    EXPECT_EQ(grad_of(y), (Values{2}));
}

TEST(Backward, TensorUsedTwiceSumsContributions) {
    mt::Tensor x = param({1, 2, 3}, {3});
    mt::sum(x * x).backward();  // d/dx sum(x^2) = 2x
    EXPECT_EQ(grad_of(x), (Values{2, 4, 6}));
}

TEST(Backward, SharedIntermediateRunsOnceWithSummedGradient) {
    // t is used by two operations; its node must run once, after both, with 2t + 1 = 19.
    mt::Tensor x = param({3}, {});
    mt::Tensor t = x * x;      // 9
    mt::Tensor z = t * t + t;  // x^4 + x^2  ->  dz/dx = 4x^3 + 2x = 108 + 6
    z.backward();
    EXPECT_EQ(grad_of(x), (Values{114}));
}

TEST(Backward, AccumulatesAcrossCallsUntilZeroGrad) {
    mt::Tensor x = param({2}, {});
    mt::Tensor y = x * x;  // dy/dx = 2x = 4
    y.backward();
    mt::Tensor g = *x.grad();
    y.backward();  // gradients add up, as in PyTorch (which would also need retain_graph=True here)
    EXPECT_EQ(grad_of(x), (Values{8}));
    EXPECT_EQ(g.to_vector(), (Values{8}));  // added in place: the handle taken earlier sees it
    x.zero_grad();
    EXPECT_FALSE(x.grad().has_value());
}

TEST(Backward, ExplicitGradientForNonScalar) {
    mt::Tensor x = param({1, 2, 3}, {3});
    mt::Tensor y = x * mt::from_data({3}, {});
    y.backward(mt::from_data({1, 10, 100}, {3}));  // d loss / d y, given from outside
    EXPECT_EQ(grad_of(x), (Values{3, 30, 300}));
}

TEST(Backward, RejectsBadCalls) {
    mt::Tensor x = param({1, 2}, {2});
    mt::Tensor y = x * x;
    EXPECT_THROW(y.backward(), std::invalid_argument);               // 2 elements and no gradient
    EXPECT_THROW(y.backward(mt::ones({3})), std::invalid_argument);  // gradient of the wrong shape
    EXPECT_THROW(mt::ones({}).backward(), std::logic_error);         // nothing requires grad
}

TEST(Backward, OnLeafItself) {
    mt::Tensor x = param({5}, {});
    x.backward();
    EXPECT_EQ(grad_of(x), (Values{1}));
}

TEST(Backward, SkipsInputsWithoutGrad) {
    mt::Tensor a = param({1, 2}, {2});
    mt::Tensor b = mt::from_data({3, 4}, {2});
    mt::sum(a * b).backward();
    EXPECT_EQ(grad_of(a), (Values{3, 4}));
    EXPECT_FALSE(b.grad().has_value());
}

TEST(Backward, GradOwnsItsBuffer) {
    mt::Tensor x = param({1, 2}, {2});
    mt::Tensor y = param({3, 4}, {2});
    mt::sum(x + y).backward();  // both receive the very same gradient tensor...
    x.grad()->at({0}) = 42;     // ...but each grad is an own copy
    EXPECT_EQ(grad_of(y), (Values{1, 1}));

    mt::Tensor z = param({1, 2, 3}, {3});
    mt::sum(z).backward();                   // the formula returns an expand view (stride 0)...
    EXPECT_TRUE(z.grad()->is_contiguous());  // ...grad() keeps a contiguous copy
    EXPECT_EQ(z.grad()->strides(), (mt::Strides{1}));
}

// --- Gradient formulas, checked by hand -------------------------------------------------

TEST(OpsBackward, AddSumsOverBroadcastDims) {
    mt::Tensor a = param({1, 2, 3}, {3, 1});
    mt::Tensor b = param({10, 20, 30, 40}, {4});
    mt::sum(a + b).backward();  // [3, 1] + [4] -> [3, 4]
    EXPECT_EQ(a.grad()->shape(), (mt::Shape{3, 1}));
    EXPECT_EQ(grad_of(a), (Values{4, 4, 4}));  // each a[i] was used in 4 columns
    EXPECT_EQ(b.grad()->shape(), (mt::Shape{4}));
    EXPECT_EQ(grad_of(b), (Values{3, 3, 3, 3}));  // each b[j] was used in 3 rows
}

TEST(OpsBackward, SubNegatesSecondInput) {
    mt::Tensor a = param({1, 2}, {2});
    mt::Tensor b = param({3, 4}, {2});
    mt::sum(a - b).backward();
    EXPECT_EQ(grad_of(a), (Values{1, 1}));
    EXPECT_EQ(grad_of(b), (Values{-1, -1}));
}

TEST(OpsBackward, MulUsesTheOtherInput) {
    mt::Tensor a = param({1, 2, 3}, {3});
    mt::Tensor s = param({2}, {});  // scalar, broadcast to [3]
    mt::sum(a * s).backward();
    EXPECT_EQ(grad_of(a), (Values{2, 2, 2}));  // d/da = s
    EXPECT_EQ(grad_of(s), (Values{6}));        // d/ds = a1 + a2 + a3
}

TEST(OpsBackward, MatmulTransposesTheOtherInput) {
    mt::Tensor a = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::Tensor b = param({7, 8, 9, 10, 11, 12}, {3, 2});
    mt::sum(mt::matmul(a, b)).backward();  // dC = ones [2, 2]
    // dA = dC @ B^T: every row holds the row sums of B (15, 19, 23)
    EXPECT_EQ(grad_of(a), (Values{15, 19, 23, 15, 19, 23}));
    // dB = A^T @ dC: every column holds the column sums of A (5, 7, 9)
    EXPECT_EQ(grad_of(b), (Values{5, 5, 7, 7, 9, 9}));
}

TEST(OpsBackward, ReluPassesGradientWherePositive) {
    mt::Tensor x = param({-1, 0, 2, -3, 4}, {5});
    mt::sum(mt::relu(x)).backward();
    EXPECT_EQ(grad_of(x), (Values{0, 0, 1, 0, 1}));  // 0 at x = 0, as in PyTorch
}

TEST(OpsBackward, SumAndMean) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::sum(x).backward();
    EXPECT_EQ(grad_of(x), Values(6, 1.0f));
    x.zero_grad();
    mt::mean(x).backward();
    for (float g : grad_of(x)) EXPECT_FLOAT_EQ(g, 1.0f / 6);
}

TEST(OpsBackward, SumAlongDimCopiesGradientBack) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::Tensor w = mt::from_data({1, 10}, {2});
    mt::sum(mt::sum(x, 1) * w).backward();  // row i of x gets weight w[i]
    EXPECT_EQ(grad_of(x), (Values{1, 1, 1, 10, 10, 10}));

    x.zero_grad();
    mt::Tensor v = mt::from_data({1, 2, 3}, {1, 3});
    mt::sum(mt::sum(x, 0, true) * v).backward();  // keepdim: [1, 3]; column j gets weight v[j]
    EXPECT_EQ(grad_of(x), (Values{1, 2, 3, 1, 2, 3}));
}

// --- Views: their own nodes and gradient formulas ---------------------------------------

TEST(ViewsBackward, ViewOfTensorWithGradRecordsNode) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::Tensor t = x.transpose(0, 1);
    EXPECT_TRUE(t.requires_grad());
    EXPECT_FALSE(t.is_leaf());
    ASSERT_NE(t.grad_fn(), nullptr);
    EXPECT_STREQ(t.grad_fn()->name(), "TransposeBackward");
    EXPECT_TRUE(t.grad_fn()->inputs()[0].is_same(x));
    EXPECT_EQ(t.data_ptr(), x.data_ptr());  // still a view: same storage, nothing copied

    EXPECT_STREQ(x.slice(1, 0, 2).grad_fn()->name(), "SliceBackward");
    EXPECT_STREQ(x.reshape({3, 2}).grad_fn()->name(), "ReshapeBackward");
    EXPECT_STREQ(x.expand({4, 2, 3}).grad_fn()->name(), "ExpandBackward");
}

TEST(ViewsBackward, NoNodeWithoutGradOrUnderGuard) {
    mt::Tensor plain = mt::ones({2, 3});
    EXPECT_TRUE(plain.transpose(0, 1).is_leaf());
    EXPECT_FALSE(plain.slice(1, 0, 2).requires_grad());

    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::NoGradGuard no_grad;
    EXPECT_TRUE(x.transpose(0, 1).is_leaf());
    EXPECT_FALSE(x.reshape({6}).requires_grad());
}

TEST(ViewsBackward, TransposeSendsGradientBackTransposed) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::Tensor w = mt::from_data({10, 20, 30, 40, 50, 60}, {3, 2});
    mt::sum(x.transpose(0, 1) * w).backward();  // x[i][j] meets w[j][i]
    EXPECT_EQ(x.grad()->shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(grad_of(x), (Values{10, 30, 50, 20, 40, 60}));
}

TEST(ViewsBackward, SliceGivesZerosWhereNothingWasTaken) {
    mt::Tensor x = param({1, 2, 3, 4, 5}, {5});
    mt::Tensor w = mt::from_data({10, 20}, {2});
    mt::sum(x.slice(0, 1, 5, 2) * w).backward();  // takes x[1] and x[3]
    EXPECT_EQ(grad_of(x), (Values{0, 10, 0, 20, 0}));
}

TEST(ViewsBackward, SliceAlongLastDim) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::Tensor w = mt::from_data({1, 2, 3, 4}, {2, 2});
    mt::sum(x.slice(-1, 1, 3) * w).backward();  // columns 1 and 2
    EXPECT_EQ(grad_of(x), (Values{0, 1, 2, 0, 3, 4}));
}

TEST(ViewsBackward, OverlappingSlicesAddUp) {
    mt::Tensor x = param({1, 2, 3, 4, 5}, {5});
    mt::sum(x.slice(0, 0, 3) + x.slice(0, 2, 5)).backward();
    EXPECT_EQ(grad_of(x), (Values{1, 1, 2, 1, 1}));  // x[2] is in both slices
}

TEST(ViewsBackward, ReshapeSendsGradientBackInOriginalShape) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::Tensor w = mt::from_data({10, 20, 30, 40, 50, 60}, {3, 2});
    mt::sum(x.reshape({3, 2}) * w).backward();
    EXPECT_EQ(x.grad()->shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(grad_of(x), (Values{10, 20, 30, 40, 50, 60}));  // the same row-major order
}

TEST(ViewsBackward, ReshapeOfNonContiguousGoesThroughCopy) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    mt::Tensor y = x.transpose(0, 1).reshape({6});  // a copy: [x00, x10, x01, x11, x02, x12]
    ASSERT_NE(y.grad_fn(), nullptr);
    EXPECT_STREQ(y.grad_fn()->name(), "ReshapeBackward");
    const mt::Tensor& copy = y.grad_fn()->inputs()[0];
    ASSERT_NE(copy.grad_fn(), nullptr);
    EXPECT_STREQ(copy.grad_fn()->name(), "ContiguousBackward");
    EXPECT_STREQ(copy.grad_fn()->inputs()[0].grad_fn()->name(), "TransposeBackward");

    mt::sum(y * mt::from_data({1, 2, 3, 4, 5, 6}, {6})).backward();
    EXPECT_EQ(grad_of(x), (Values{1, 3, 5, 2, 4, 6}));
}

TEST(ViewsBackward, ContiguousNodeOnlyWhenItCopies) {
    mt::Tensor x = param({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_TRUE(x.contiguous().is_same(x));  // already row-major: the same tensor, still a leaf
    mt::Tensor c = x.transpose(0, 1).contiguous();
    ASSERT_NE(c.grad_fn(), nullptr);
    EXPECT_STREQ(c.grad_fn()->name(), "ContiguousBackward");
    mt::sum(c * mt::from_data({10, 20, 30, 40, 50, 60}, {3, 2})).backward();
    EXPECT_EQ(grad_of(x), (Values{10, 30, 50, 20, 40, 60}));  // same as transpose alone
}

TEST(ViewsBackward, ExpandSumsOverCopies) {
    mt::Tensor x = param({1, 2, 3}, {3, 1});
    mt::sum(x.expand({2, 3, 4})).backward();
    EXPECT_EQ(x.grad()->shape(), (mt::Shape{3, 1}));
    EXPECT_EQ(grad_of(x), (Values{8, 8, 8}));  // each element appears 2 * 4 times
}

// --- Graph ownership and memory ---------------------------------------------------------
// Ownership points backwards only: result -> its node -> the node's inputs -> their nodes ...
// A weak_ptr watches a node without owning it; expired() becomes true once the node is freed.

TEST(Graph, NodeIsFreedWithItsResult) {
    mt::Tensor x = param({1, 2}, {2});
    std::weak_ptr<mt::Node> node;
    std::weak_ptr<mt::Node> view_node;
    {
        mt::Tensor y = x * x;
        mt::Tensor t = x.reshape({2, 1});
        node = y.grad_fn();
        view_node = t.grad_fn();
        EXPECT_FALSE(node.expired());
        EXPECT_FALSE(view_node.expired());
    }
    EXPECT_TRUE(node.expired());  // y and t were the only owners of their nodes
    EXPECT_TRUE(view_node.expired());
}

TEST(Graph, ConsumerKeepsIntermediateAlive) {
    mt::Tensor x = param({1, 2}, {2});
    std::weak_ptr<mt::Node> inner;
    std::weak_ptr<mt::Node> outer;
    {
        mt::Tensor y = x * x;
        mt::Tensor z = y + x;
        inner = y.grad_fn();
        outer = z.grad_fn();
        y = mt::ones({1});              // rebinds the handle, like `y = ...` in Python
        EXPECT_FALSE(inner.expired());  // still owned: z -> AddBackward -> old y -> MulBackward
        mt::sum(z).backward();
        EXPECT_FALSE(inner.expired());  // backward does not free the graph (PyTorch does)
        EXPECT_EQ(grad_of(x), (Values{3, 5}));  // 2x + 1
    }
    EXPECT_TRUE(outer.expired());  // the whole chain goes with z
    EXPECT_TRUE(inner.expired());
}

TEST(Graph, GradHoldsNoGraph) {
    mt::Tensor x = param({-1, 2, 3}, {3});
    std::weak_ptr<mt::Node> node;
    {
        mt::Tensor loss = mt::sum(mt::relu(x) * x.slice(0, 0, 3));  // a view node in the graph too
        node = loss.grad_fn();
        loss.backward();
    }
    EXPECT_TRUE(node.expired());  // nothing made during backward keeps the graph alive
    ASSERT_TRUE(x.grad().has_value());
    EXPECT_TRUE(x.grad()->is_leaf());  // computed under NoGradGuard: plain numbers, no grad_fn
    EXPECT_FALSE(x.grad()->requires_grad());
}

// --- Gradcheck: backward() against numerical derivatives --------------------------------

TEST(Gradcheck, ElementwiseWithBroadcasting) {
    mt::Tensor a = mt::randn({2, 3}, 1).set_requires_grad(true);
    mt::Tensor b = mt::randn({3}, 2).set_requires_grad(true);
    EXPECT_TRUE(mt_test::gradcheck(
        [](const std::vector<mt::Tensor>& in) { return (in[0] + in[1]) * in[0] - in[1]; }, {a, b}));
}

TEST(Gradcheck, TwoLayerNetwork) {
    // relu(x @ w1 + b1) @ w2: the shape of the XOR network from stage 6
    mt::Tensor x = mt::randn({4, 2}, 3).set_requires_grad(true);
    mt::Tensor w1 = mt::randn({2, 8}, 4).set_requires_grad(true);
    mt::Tensor b1 = mt::randn({8}, 5).set_requires_grad(true);
    mt::Tensor w2 = mt::randn({8, 1}, 6).set_requires_grad(true);
    EXPECT_TRUE(mt_test::gradcheck(
        [](const std::vector<mt::Tensor>& in) {
            return mt::matmul(mt::relu(mt::matmul(in[0], in[1]) + in[2]), in[3]);
        },
        {x, w1, b1, w2}));
}

TEST(Gradcheck, Reductions) {
    mt::Tensor x = mt::randn({2, 3}, 7).set_requires_grad(true);
    EXPECT_TRUE(mt_test::gradcheck(
        [](const std::vector<mt::Tensor>& in) {
            const mt::Tensor& t = in[0];
            return mt::sum(t, 0, true) * t + mt::sum(t * t, 1, true) - mt::mean(t);
        },
        {x}));
}

TEST(Gradcheck, AllViews) {
    mt::Tensor x = mt::randn({3, 4}, 9).set_requires_grad(true);
    mt::Tensor w = mt::randn({3, 2}, 10).set_requires_grad(true);
    EXPECT_TRUE(mt_test::gradcheck(
        [](const std::vector<mt::Tensor>& in) {
            const mt::Tensor& x = in[0];                                   // [3, 4]
            const mt::Tensor cols = x.slice(1, 1, 4, 2);                   // [3, 2]: columns 1 and 3
            const mt::Tensor m = mt::matmul(cols.transpose(0, 1), in[1]);  // [2, 3] @ [3, 2] = [2, 2]
            const mt::Tensor rows = m.reshape({1, 4}).expand({3, 4});      // the same row 3 times
            return rows * x.transpose(0, 1).reshape({3, 4});               // copy, then regroup
        },
        {x, w}));
}

TEST(Gradcheck, SlicesOfThreeDimTensor) {
    // Slices along the middle dim (outer and inner both > 1); x[:, 1] is in both slices.
    mt::Tensor x = mt::randn({2, 5, 3}, 11).set_requires_grad(true);
    EXPECT_TRUE(mt_test::gradcheck(
        [](const std::vector<mt::Tensor>& in) { return in[0].slice(1, -4, 5, 2) * in[0].slice(1, 0, 2); },
        {x}));
}

namespace {

// A node with a deliberately wrong formula, to show that gradcheck notices.
// Also an example of a custom operation: compute the values, then attach your own node.
class WrongSquareBackward : public mt::Node {
public:
    using mt::Node::Node;
    const char* name() const override { return "WrongSquareBackward"; }
    std::vector<mt::Tensor> backward(const mt::Tensor& grad) const override {
        return {mt::mul(grad, inputs()[0])};  // wrong: d(x^2)/dx is 2x, not x
    }
};

mt::Tensor wrong_square(const mt::Tensor& x) {
    std::vector<float> values = x.to_vector();
    for (float& v : values) v *= v;
    mt::Tensor out = mt::from_data(std::move(values), x.shape());
    if (mt::is_grad_enabled() && x.requires_grad()) {
        out.set_grad_fn(std::make_shared<WrongSquareBackward>(std::vector<mt::Tensor>{x}));
    }
    return out;
}

}  // namespace

TEST(Gradcheck, CatchesWrongFormula) {
    mt::Tensor x = mt::randn({3}, 8).set_requires_grad(true);
    EXPECT_FALSE(mt_test::gradcheck([](const std::vector<mt::Tensor>& in) { return wrong_square(in[0]); }, {x}));
}

// --- In-place update: the optimizer step ------------------------------------------------

TEST(InPlace, SubChangesTheTensorItself) {
    mt::Tensor t = mt::from_data({1, 2, 3, 4}, {2, 2});
    const mt::Tensor alias = t;  // same tensor, so it sees the change
    t.sub_(mt::from_data({0.5f, 1, 1.5f, 2}, {2, 2}));
    EXPECT_EQ(alias.to_vector(), (Values{0.5f, 1, 1.5f, 2}));
}

TEST(InPlace, SubThroughViewChangesBase) {
    mt::Tensor base = mt::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    base.slice(1, 0, 3, 2).sub_(mt::ones({2, 2}));  // columns 0 and 2, strides [3, 2]
    EXPECT_EQ(base.to_vector(), (Values{0, 2, 2, 3, 5, 5}));
}

TEST(InPlace, SubReadsOtherInLogicalOrder) {
    mt::Tensor t = mt::zeros({2, 2});
    t.sub_(mt::from_data({1, 2, 3, 4}, {2, 2}).transpose(0, 1));  // [[1, 3], [2, 4]]
    EXPECT_EQ(t.to_vector(), (Values{-1, -3, -2, -4}));
}

TEST(InPlace, SubWithOverlappingViewUsesOldValues) {
    // t[1:4] -= t[0:3]: both views share the storage of t. Every element must use the old value
    // of its left neighbour, not one that was already changed: 2-1, 3-2, 4-3.
    mt::Tensor t = mt::from_data({1, 2, 3, 4}, {4});
    t.slice(0, 1, 4).sub_(t.slice(0, 0, 3));
    EXPECT_EQ(t.to_vector(), (Values{1, 1, 1, 1}));  // without copying other first: {1, 1, 2, 2}
}

TEST(InPlace, SubRejectsDifferentShape) {
    mt::Tensor t = mt::zeros({2, 3});
    EXPECT_THROW(t.sub_(mt::ones({3, 2})), std::invalid_argument);
    EXPECT_THROW(t.sub_(mt::ones({3})), std::invalid_argument);  // no broadcasting
}

TEST(InPlace, SubOnGradTensorOnlyUnderGuard) {
    mt::Tensor w = param({1, 2}, {2});
    EXPECT_THROW(w.sub_(mt::ones({2})), std::logic_error);
    EXPECT_EQ(w.to_vector(), (Values{1, 2}));  // unchanged
    {
        mt::NoGradGuard no_grad;
        w.sub_(mt::ones({2}));
    }
    EXPECT_EQ(w.to_vector(), (Values{0, 1}));
    EXPECT_TRUE(w.requires_grad());  // still a parameter, still a leaf
    EXPECT_TRUE(w.is_leaf());
}

TEST(InPlace, SgdStepMovesAgainstTheGradient) {
    // loss = sum(w * w), d loss / dw = 2w, so one step with lr 0.25 halves w.
    mt::Tensor w = param({2, -4}, {2});
    mt::sum(w * w).backward();
    {
        mt::NoGradGuard no_grad;
        w.sub_(mt::from_data({0.25f}, {}) * w.grad().value());
    }
    EXPECT_EQ(w.to_vector(), (Values{1, -2}));     // 2 - 0.25 * 4, -4 - 0.25 * (-8)
    EXPECT_FLOAT_EQ(mt::sum(w * w).at({}), 5.0f);  // the loss was 20
}
