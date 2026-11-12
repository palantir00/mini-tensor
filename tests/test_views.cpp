#include "minitensor/tensor.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Values = std::vector<float>;

// [[1, 2, 3],
//  [4, 5, 6]]   shape [2, 3], strides [3, 1]
mt::Tensor matrix_2x3() { return mt::from_data({1, 2, 3, 4, 5, 6}, {2, 3}); }

// 0, 1, ..., 9
mt::Tensor range_10() { return mt::from_data({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {10}); }

std::string to_string(const mt::Tensor& t) {
    std::ostringstream os;
    os << t;
    return os.str();
}

}  // namespace

// --- to_vector ------------------------------------------------------------------

TEST(ToVector, ReturnsRowMajorOrder) {
    EXPECT_EQ(matrix_2x3().to_vector(), (Values{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(mt::from_data({2.5}, {}).to_vector(), (Values{2.5}));
    EXPECT_TRUE(mt::zeros({0}).to_vector().empty());
}

// --- transpose ------------------------------------------------------------------

TEST(Transpose, SwapsShapeAndStrides) {
    mt::Tensor t = matrix_2x3().transpose(0, 1);
    EXPECT_EQ(t.shape(), (mt::Shape{3, 2}));
    EXPECT_EQ(t.strides(), (mt::Strides{1, 3}));
    EXPECT_EQ(t.offset(), 0);
    EXPECT_EQ(t.to_vector(), (Values{1, 4, 2, 5, 3, 6}));
    EXPECT_FLOAT_EQ(t.at({2, 1}), 6);
}

TEST(Transpose, ThreeDimensions) {
    mt::Tensor t = mt::zeros({2, 3, 4}).transpose(0, 2);
    EXPECT_EQ(t.shape(), (mt::Shape{4, 3, 2}));
    EXPECT_EQ(t.strides(), (mt::Strides{1, 4, 12}));
}

TEST(Transpose, AcceptsNegativeDims) {
    mt::Tensor t = matrix_2x3().transpose(-1, -2);
    EXPECT_EQ(t.shape(), (mt::Shape{3, 2}));
    EXPECT_EQ(t.strides(), (mt::Strides{1, 3}));
}

TEST(Transpose, RejectsDimOutOfRange) {
    mt::Tensor a = matrix_2x3();
    EXPECT_THROW(a.transpose(0, 2), std::out_of_range);
    EXPECT_THROW(a.transpose(-3, 0), std::out_of_range);
    EXPECT_THROW(mt::ones({}).transpose(0, 0), std::out_of_range);  // a scalar has no dims
}

TEST(Transpose, SharesStorage) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor t = a.transpose(0, 1);
    EXPECT_FALSE(t.is_same(a));  // a different tensor...
    t.at({0, 1}) = 42;           // ...over the same buffer
    EXPECT_FLOAT_EQ(a.at({1, 0}), 42);
}

// --- slice ----------------------------------------------------------------------

TEST(Slice, ColumnsWithStep) {
    mt::Tensor t = matrix_2x3().slice(1, 0, 3, 2);  // columns 0 and 2
    EXPECT_EQ(t.shape(), (mt::Shape{2, 2}));
    EXPECT_EQ(t.strides(), (mt::Strides{3, 2}));
    EXPECT_EQ(t.offset(), 0);
    EXPECT_EQ(t.to_vector(), (Values{1, 3, 4, 6}));
}

TEST(Slice, RowMovesOffset) {
    mt::Tensor t = matrix_2x3().slice(0, 1, 2);  // row 1
    EXPECT_EQ(t.shape(), (mt::Shape{1, 3}));
    EXPECT_EQ(t.strides(), (mt::Strides{3, 1}));
    EXPECT_EQ(t.offset(), 3);
    EXPECT_EQ(t.to_vector(), (Values{4, 5, 6}));
}

TEST(Slice, NegativeAndOutOfRangeBoundsLikePython) {
    mt::Tensor v = range_10();
    EXPECT_EQ(v.slice(0, -3, 10).to_vector(), (Values{7, 8, 9}));  // v[-3:]
    EXPECT_EQ(v.slice(0, 2, -2, 3).to_vector(), (Values{2, 5}));   // v[2:-2:3]
    EXPECT_EQ(v.slice(0, -100, 2).to_vector(), (Values{0, 1}));    // start clamped to 0
    EXPECT_EQ(v.slice(0, 8, 100).to_vector(), (Values{8, 9}));     // end clamped to 10
    EXPECT_EQ(v.slice(0, 5, 2).numel(), 0);                        // end before start: empty
    EXPECT_EQ(matrix_2x3().slice(-1, 1, 3).to_vector(), (Values{2, 3, 5, 6}));
}

TEST(Slice, OfSliceAddsOffsets) {
    mt::Tensor t = range_10().slice(0, 2, 10).slice(0, 1, 8, 2);  // [2..9], then every 2nd from 1
    EXPECT_EQ(t.offset(), 3);
    EXPECT_EQ(t.strides(), (mt::Strides{2}));
    EXPECT_EQ(t.to_vector(), (Values{3, 5, 7, 9}));
}

TEST(Slice, RejectsBadArguments) {
    mt::Tensor v = range_10();
    EXPECT_THROW(v.slice(0, 0, 5, 0), std::invalid_argument);
    EXPECT_THROW(v.slice(0, 5, 0, -1), std::invalid_argument);
    EXPECT_THROW(v.slice(1, 0, 5), std::out_of_range);
}

TEST(Slice, SharesStorage) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor row = a.slice(0, 1, 2);
    row.at({0, 2}) = 42;
    EXPECT_FLOAT_EQ(a.at({1, 2}), 42);
}

// --- is_contiguous / contiguous ---------------------------------------------------

TEST(IsContiguous, FollowsStrides) {
    mt::Tensor a = matrix_2x3();
    EXPECT_TRUE(a.is_contiguous());
    EXPECT_FALSE(a.transpose(0, 1).is_contiguous());
    EXPECT_TRUE(a.transpose(0, 1).transpose(0, 1).is_contiguous());
    EXPECT_FALSE(a.slice(1, 0, 3, 2).is_contiguous());
    EXPECT_TRUE(a.slice(0, 1, 2).is_contiguous());  // offset does not matter
    EXPECT_TRUE(mt::ones({}).is_contiguous());
}

TEST(IsContiguous, IgnoresStrideOfSizeOneDims) {
    mt::Tensor t = mt::from_data({1, 2, 3}, {1, 3}).transpose(0, 1);
    EXPECT_EQ(t.strides(), (mt::Strides{1, 3}));  // 3 is never used: dim 1 has only index 0
    EXPECT_TRUE(t.is_contiguous());
}

TEST(Contiguous, ReturnsSameTensorIfAlreadyContiguous) {
    mt::Tensor a = matrix_2x3();
    EXPECT_TRUE(a.contiguous().is_same(a));
}

TEST(Contiguous, CopiesNonContiguousTensor) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor t = a.transpose(0, 1);
    mt::Tensor c = t.contiguous();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.shape(), (mt::Shape{3, 2}));
    EXPECT_EQ(c.strides(), (mt::Strides{2, 1}));
    EXPECT_EQ(c.to_vector(), t.to_vector());
    c.at({0, 0}) = 42;  // own storage: a is not affected
    EXPECT_FLOAT_EQ(a.at({0, 0}), 1);
}

// --- reshape --------------------------------------------------------------------

TEST(Reshape, ContiguousTensorGivesView) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor r = a.reshape({3, 2});
    EXPECT_EQ(r.shape(), (mt::Shape{3, 2}));
    EXPECT_EQ(r.strides(), (mt::Strides{2, 1}));
    EXPECT_EQ(r.to_vector(), (Values{1, 2, 3, 4, 5, 6}));
    r.at({2, 1}) = 42;
    EXPECT_FLOAT_EQ(a.at({1, 2}), 42);
}

TEST(Reshape, KeepsOffsetOfContiguousView) {
    mt::Tensor r = matrix_2x3().slice(0, 1, 2).reshape({3});  // row 1 starts at offset 3
    EXPECT_EQ(r.offset(), 3);
    EXPECT_EQ(r.to_vector(), (Values{4, 5, 6}));
}

TEST(Reshape, NonContiguousTensorIsCopied) {
    mt::Tensor a = matrix_2x3();
    mt::Tensor r = a.transpose(0, 1).reshape({6});
    EXPECT_EQ(r.to_vector(), (Values{1, 4, 2, 5, 3, 6}));
    r.at({0}) = 42;  // a copy: a is not affected
    EXPECT_FLOAT_EQ(a.at({0, 0}), 1);
}

TEST(Reshape, InfersMinusOne) {
    EXPECT_EQ(matrix_2x3().reshape({-1}).shape(), (mt::Shape{6}));
    EXPECT_EQ(matrix_2x3().reshape({3, -1}).shape(), (mt::Shape{3, 2}));
    EXPECT_EQ(mt::zeros({2, 3, 4}).reshape({2, -1}).shape(), (mt::Shape{2, 12}));
    EXPECT_EQ(mt::zeros({0}).reshape({-1}).shape(), (mt::Shape{0}));
}

TEST(Reshape, RejectsInvalidShape) {
    mt::Tensor a = matrix_2x3();
    EXPECT_THROW(a.reshape({4}), std::invalid_argument);       // 4 != 6 elements
    EXPECT_THROW(a.reshape({-1, 4}), std::invalid_argument);   // 6 is not divisible by 4
    EXPECT_THROW(a.reshape({-1, -1}), std::invalid_argument);  // only one -1
    EXPECT_THROW(a.reshape({2, -3}), std::invalid_argument);
    EXPECT_THROW(mt::zeros({0}).reshape({0, -1}), std::invalid_argument);  // -1 could be anything
}

// --- expand ---------------------------------------------------------------------

TEST(Expand, SizeOneDimGetsStrideZero) {
    mt::Tensor col = mt::from_data({1, 2}, {2, 1});
    mt::Tensor e = col.expand({2, 3});
    EXPECT_EQ(e.shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(e.strides(), (mt::Strides{1, 0}));
    EXPECT_EQ(e.to_vector(), (Values{1, 1, 1, 2, 2, 2}));
}

TEST(Expand, AddsDimsOnTheLeft) {
    mt::Tensor row = mt::from_data({10, 20, 30}, {3});
    mt::Tensor e = row.expand({2, 3});
    EXPECT_EQ(e.strides(), (mt::Strides{0, 1}));
    EXPECT_EQ(e.to_vector(), (Values{10, 20, 30, 10, 20, 30}));

    mt::Tensor s = mt::from_data({7}, {}).expand({2, 2});  // a scalar fills any shape
    EXPECT_EQ(s.strides(), (mt::Strides{0, 0}));
    EXPECT_EQ(s.to_vector(), (Values{7, 7, 7, 7}));
}

TEST(Expand, KeepsStridesAndOffsetOfMatchingDims) {
    mt::Tensor t = matrix_2x3().transpose(0, 1).expand({4, 3, 2});  // strides [1, 3] -> [0, 1, 3]
    EXPECT_EQ(t.strides(), (mt::Strides{0, 1, 3}));

    mt::Tensor row1 = matrix_2x3().slice(0, 1, 2).expand({3, 3});  // [[4, 5, 6]] starts at offset 3
    EXPECT_EQ(row1.strides(), (mt::Strides{0, 1}));
    EXPECT_EQ(row1.offset(), 3);
    EXPECT_EQ(row1.to_vector(), (Values{4, 5, 6, 4, 5, 6, 4, 5, 6}));
}

TEST(Expand, IsNotContiguousAndSharesElements) {
    mt::Tensor row = mt::from_data({10, 20, 30}, {3});
    mt::Tensor e = row.expand({2, 3});
    EXPECT_FALSE(e.is_contiguous());
    e.at({1, 0}) = 42;  // both rows read the same storage element
    EXPECT_FLOAT_EQ(e.at({0, 0}), 42);
    EXPECT_FLOAT_EQ(row.at({0}), 42);

    mt::Tensor c = e.contiguous();  // the copy has 6 elements of its own
    EXPECT_EQ(c.strides(), (mt::Strides{3, 1}));
    EXPECT_EQ(c.to_vector(), (Values{42, 20, 30, 42, 20, 30}));
}

TEST(Expand, RejectsInvalidShape) {
    mt::Tensor a = matrix_2x3();
    EXPECT_THROW(a.expand({4, 3}), std::invalid_argument);  // size 2 is not 1
    EXPECT_THROW(a.expand({3}), std::invalid_argument);     // fewer dims
    EXPECT_THROW(a.expand({2, -1}), std::invalid_argument);
}

// --- Views and const, printing ----------------------------------------------------

TEST(Views, ConstHandleDoesNotMeanConstData) {
    const mt::Tensor a = matrix_2x3();
    mt::Tensor t = a.transpose(0, 1);  // allowed: transpose() is const
    t.at({0, 0}) = 42;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 42);
}

TEST(Views, PrintFollowsStrides) {
    EXPECT_EQ(to_string(matrix_2x3().transpose(0, 1)),
              "tensor([[1, 4],\n"
              "        [2, 5],\n"
              "        [3, 6]])");
}
