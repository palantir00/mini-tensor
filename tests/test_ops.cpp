#include "minitensor/ops.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using Values = std::vector<float>;

// [[1, 2, 3],
//  [4, 5, 6]]   shape [2, 3]
mt::Tensor matrix_2x3() { return mt::from_data({1, 2, 3, 4, 5, 6}, {2, 3}); }

}  // namespace

// --- broadcast_shapes -------------------------------------------------------------

TEST(BroadcastShapes, FollowsNumpyRules) {
    EXPECT_EQ(mt::broadcast_shapes({2, 3}, {2, 3}), (mt::Shape{2, 3}));
    EXPECT_EQ(mt::broadcast_shapes({3, 1}, {4}), (mt::Shape{3, 4}));
    EXPECT_EQ(mt::broadcast_shapes({2, 1, 4}, {3, 1}), (mt::Shape{2, 3, 4}));
    EXPECT_EQ(mt::broadcast_shapes({}, {2, 3}), (mt::Shape{2, 3}));  // a scalar fits anything
    EXPECT_EQ(mt::broadcast_shapes({1}, {0}), (mt::Shape{0}));       // size 1 stretches to 0 too
}

TEST(BroadcastShapes, RejectsMismatch) {
    EXPECT_THROW(mt::broadcast_shapes({2, 3}, {2}), std::invalid_argument);  // 3 vs 2
    EXPECT_THROW(mt::broadcast_shapes({2, 3}, {3, 3}), std::invalid_argument);
}

// --- add / sub / mul --------------------------------------------------------------

TEST(Add, SameShape) {
    mt::Tensor c = mt::add(matrix_2x3(), mt::from_data({10, 20, 30, 40, 50, 60}, {2, 3}));
    EXPECT_EQ(c.shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(c.to_vector(), (Values{11, 22, 33, 44, 55, 66}));
}

TEST(Add, BroadcastsColumnAgainstRow) {
    mt::Tensor col = mt::from_data({1, 2, 3}, {3, 1});
    mt::Tensor row = mt::from_data({10, 20, 30, 40}, {4});
    mt::Tensor c = mt::add(col, row);  // [3, 1] + [4] -> [3, 4]
    EXPECT_EQ(c.shape(), (mt::Shape{3, 4}));
    EXPECT_EQ(c.to_vector(), (Values{11, 21, 31, 41,
                                     12, 22, 32, 42,
                                     13, 23, 33, 43}));
}

TEST(Add, ScalarBroadcastsToAnyShape) {
    mt::Tensor c = mt::add(mt::from_data({100}, {}), matrix_2x3());
    EXPECT_EQ(c.shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(c.to_vector(), (Values{101, 102, 103, 104, 105, 106}));
    EXPECT_EQ(mt::add(mt::from_data({1}, {}), mt::from_data({2}, {})).shape(), mt::Shape{});
}

TEST(Add, ReadsNonContiguousInput) {
    mt::Tensor t = matrix_2x3().transpose(0, 1);  // [[1, 4], [2, 5], [3, 6]]
    mt::Tensor c = mt::add(t, mt::from_data({10, 20}, {2}));
    EXPECT_EQ(c.shape(), (mt::Shape{3, 2}));
    EXPECT_EQ(c.to_vector(), (Values{11, 24, 12, 25, 13, 26}));
}

TEST(Add, ResultHasOwnStorage) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor c = mt::add(a, mt::zeros({2, 3}));
    EXPECT_TRUE(c.is_contiguous());
    c.at({0, 0}) = 42;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 1);
}

TEST(Add, RejectsIncompatibleShapes) {
    EXPECT_THROW(mt::add(matrix_2x3(), mt::ones({2})), std::invalid_argument);
}

TEST(SubMul, ElementwiseWithBroadcasting) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor col = mt::from_data({1, 10}, {2, 1});
    EXPECT_EQ(mt::sub(a, col).to_vector(), (Values{0, 1, 2, -6, -5, -4}));
    EXPECT_EQ(mt::mul(a, col).to_vector(), (Values{1, 2, 3, 40, 50, 60}));
}

TEST(Operators, CallElementwiseOps) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor two = mt::from_data({2}, {});
    EXPECT_EQ((a + two).to_vector(), (Values{3, 4, 5, 6, 7, 8}));
    EXPECT_EQ((a - two).to_vector(), (Values{-1, 0, 1, 2, 3, 4}));
    EXPECT_EQ((a * two).to_vector(), (Values{2, 4, 6, 8, 10, 12}));
    EXPECT_EQ((a * a - a).to_vector(), (Values{0, 2, 6, 12, 20, 30}));  // * before -, as usual in C++
}

// --- relu -------------------------------------------------------------------------

TEST(Relu, ZeroesNegatives) {
    mt::Tensor r = mt::relu(mt::from_data({-2, -0.5, 0, 0.5, 3, -1}, {2, 3}));
    EXPECT_EQ(r.shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(r.to_vector(), (Values{0, 0, 0, 0.5, 3, 0}));
}

TEST(Relu, ReadsNonContiguousInputAndKeepsNan) {
    mt::Tensor t = mt::from_data({-1, 2, 3, -4}, {2, 2}).transpose(0, 1);  // [[-1, 3], [2, -4]]
    EXPECT_EQ(mt::relu(t).to_vector(), (Values{0, 3, 2, 0}));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(std::isnan(mt::relu(mt::from_data({nan}, {})).at({})));
}

// --- matmul -----------------------------------------------------------------------

TEST(Matmul, TwoByThreeTimesThreeByTwo) {
    mt::Tensor b = mt::from_data({7, 8, 9, 10, 11, 12}, {3, 2});
    mt::Tensor c = mt::matmul(matrix_2x3(), b);
    // [1 2 3].[7 9 11] = 58   [1 2 3].[8 10 12] = 64
    // [4 5 6].[7 9 11] = 139  [4 5 6].[8 10 12] = 154
    EXPECT_EQ(c.shape(), (mt::Shape{2, 2}));
    EXPECT_EQ(c.to_vector(), (Values{58, 64, 139, 154}));
}

TEST(Matmul, ReadsTransposedInput) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor c = mt::matmul(a.transpose(0, 1), a);  // a^T a: [3, 2] @ [2, 3] -> [3, 3]
    EXPECT_EQ(c.shape(), (mt::Shape{3, 3}));
    EXPECT_EQ(c.to_vector(), (Values{17, 22, 27,
                                     22, 29, 36,
                                     27, 36, 45}));
    // Same numbers as with an explicit contiguous copy of a^T.
    EXPECT_EQ(c.to_vector(), mt::matmul(a.transpose(0, 1).contiguous(), a).to_vector());
}

TEST(Matmul, ReadsSlicedInputWithOffset) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor row1 = a.slice(0, 1, 2);                  // [[4, 5, 6]], offset 3
    mt::Tensor c = mt::matmul(row1, a.transpose(0, 1));  // [1, 3] @ [3, 2] -> [1, 2]
    EXPECT_EQ(c.shape(), (mt::Shape{1, 2}));
    EXPECT_EQ(c.to_vector(), (Values{32, 77}));  // [4 5 6].[1 2 3], [4 5 6].[4 5 6]
}

TEST(Matmul, EmptyInnerDimGivesZeros) {
    mt::Tensor c = mt::matmul(mt::ones({2, 0}), mt::ones({0, 3}));  // sums with no terms
    EXPECT_EQ(c.shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(c.to_vector(), (Values(6, 0.0f)));
}

TEST(Matmul, RejectsBadShapes) {
    mt::Tensor a = matrix_2x3();
    EXPECT_THROW(mt::matmul(a, a), std::invalid_argument);                    // inner sizes 3 and 2
    EXPECT_THROW(mt::matmul(a, mt::ones({3})), std::invalid_argument);        // 1-D
    EXPECT_THROW(mt::matmul(mt::ones({2, 2, 2}), a), std::invalid_argument);  // 3-D
}

// --- sum / mean -------------------------------------------------------------------

TEST(Sum, AllElementsGiveScalar) {
    mt::Tensor s = mt::sum(matrix_2x3());
    EXPECT_EQ(s.shape(), mt::Shape{});
    EXPECT_FLOAT_EQ(s.at({}), 21);
    EXPECT_FLOAT_EQ(mt::sum(mt::zeros({0})).at({}), 0);
}

TEST(Sum, AlongDim) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor s0 = mt::sum(a, 0);  // down each column
    EXPECT_EQ(s0.shape(), (mt::Shape{3}));
    EXPECT_EQ(s0.to_vector(), (Values{5, 7, 9}));

    mt::Tensor s1 = mt::sum(a, 1);  // along each row
    EXPECT_EQ(s1.shape(), (mt::Shape{2}));
    EXPECT_EQ(s1.to_vector(), (Values{6, 15}));
    EXPECT_EQ(mt::sum(a, -1).to_vector(), s1.to_vector());
}

TEST(Sum, KeepdimLeavesSizeOne) {
    mt::Tensor a = matrix_2x3();
    EXPECT_EQ(mt::sum(a, 0, true).shape(), (mt::Shape{1, 3}));

    mt::Tensor s1 = mt::sum(a, 1, true);
    EXPECT_EQ(s1.shape(), (mt::Shape{2, 1}));
    // [2, 1] broadcasts back against [2, 3]: every element minus its row sum.
    EXPECT_EQ(mt::sub(a, s1).to_vector(), (Values{-5, -4, -3, -11, -10, -9}));
}

TEST(Sum, MiddleDimOfThreeDims) {
    // [[[0, 1], [2, 3], [4, 5]],  [[6, 7], [8, 9], [10, 11]]]   shape [2, 3, 2]
    mt::Tensor t = mt::from_data({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {2, 3, 2});
    mt::Tensor s = mt::sum(t, 1);  // outer 2, size 3, inner 2
    EXPECT_EQ(s.shape(), (mt::Shape{2, 2}));
    EXPECT_EQ(s.to_vector(), (Values{6, 9, 24, 27}));
}

TEST(Sum, NonContiguousInput) {
    mt::Tensor t = matrix_2x3().transpose(0, 1);  // [[1, 4], [2, 5], [3, 6]]
    EXPECT_EQ(mt::sum(t, 0).to_vector(), (Values{6, 15}));
    EXPECT_EQ(mt::sum(t, 1).to_vector(), (Values{5, 7, 9}));
}

TEST(Sum, RejectsDimOutOfRange) {
    EXPECT_THROW(mt::sum(matrix_2x3(), 2), std::out_of_range);
    EXPECT_THROW(mt::sum(matrix_2x3(), -3), std::out_of_range);
}

TEST(Mean, AllElements) {
    mt::Tensor m = mt::mean(matrix_2x3());
    EXPECT_EQ(m.shape(), mt::Shape{});
    EXPECT_FLOAT_EQ(m.at({}), 3.5f);
    EXPECT_TRUE(std::isnan(mt::mean(mt::zeros({0})).at({})));  // 0 / 0
}
