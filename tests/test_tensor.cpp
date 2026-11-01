#include "minitensor/tensor.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string to_string(const mt::Tensor& t) {
    std::ostringstream os;
    os << t;
    return os.str();
}

}  // namespace

// --- Shape helpers ------------------------------------------------------------

TEST(Numel, IsProductOfSizes) {
    EXPECT_EQ(mt::numel({2, 3, 4}), 24);
    EXPECT_EQ(mt::numel({}), 1);      // scalar
    EXPECT_EQ(mt::numel({3, 0}), 0);  // empty tensor
}

TEST(ContiguousStrides, AreRowMajor) {
    EXPECT_EQ(mt::contiguous_strides({2, 3, 4}), (mt::Strides{12, 4, 1}));
    EXPECT_EQ(mt::contiguous_strides({5}), (mt::Strides{1}));
    EXPECT_EQ(mt::contiguous_strides({}), (mt::Strides{}));
}

// --- Factories ----------------------------------------------------------------

TEST(FromData, StoresValuesRowMajor) {
    mt::Tensor t = mt::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_EQ(t.shape(), (mt::Shape{2, 3}));
    EXPECT_EQ(t.strides(), (mt::Strides{3, 1}));
    EXPECT_EQ(t.offset(), 0);
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6);
}

TEST(FromData, RejectsWrongNumberOfValues) {
    EXPECT_THROW(mt::from_data({1, 2, 3}, {2, 2}), std::invalid_argument);
}

TEST(Factories, RejectNegativeSize) {
    EXPECT_THROW(mt::zeros({2, -1}), std::invalid_argument);
    EXPECT_THROW(mt::randn({-3}, 0), std::invalid_argument);
}

TEST(Factories, ZerosAndOnes) {
    mt::Tensor z = mt::zeros({2, 3});
    mt::Tensor o = mt::ones({2, 3});
    for (std::int64_t i = 0; i < 2; ++i) {
        for (std::int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(z.at({i, j}), 0);
            EXPECT_FLOAT_EQ(o.at({i, j}), 1);
        }
    }
}

TEST(Factories, ScalarHasEmptyShape) {
    mt::Tensor s = mt::ones({});
    EXPECT_EQ(s.dim(), 0);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_FLOAT_EQ(s.at({}), 1);
}

TEST(Randn, SameSeedGivesSameValues) {
    mt::Tensor a = mt::randn({100}, 42);
    mt::Tensor b = mt::randn({100}, 42);
    mt::Tensor c = mt::randn({100}, 43);
    bool any_differs = false;
    for (std::int64_t i = 0; i < 100; ++i) {
        EXPECT_EQ(a.at({i}), b.at({i}));  // exact: same seed must give identical numbers
        if (a.at({i}) != c.at({i})) any_differs = true;
    }
    EXPECT_TRUE(any_differs);
}

TEST(Randn, LooksStandardNormal) {
    const std::int64_t n = 100000;
    mt::Tensor t = mt::randn({n}, 7);
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        const double x = t.at({i});
        sum += x;
        sum_sq += x * x;
    }
    const double mean = sum / n;
    const double var = sum_sq / n - mean * mean;
    EXPECT_NEAR(mean, 0.0, 0.02);  // standard error of the mean is about 0.003
    EXPECT_NEAR(var, 1.0, 0.02);
}

// --- Element access -----------------------------------------------------------

TEST(At, RejectsWrongNumberOfIndices) {
    mt::Tensor t = mt::zeros({2, 3});
    EXPECT_THROW(t.at({0}), std::invalid_argument);
    EXPECT_THROW(t.at({0, 0, 0}), std::invalid_argument);
}

TEST(At, RejectsOutOfRangeIndex) {
    mt::Tensor t = mt::zeros({2, 3});
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t.at({0, 3}), std::out_of_range);
    EXPECT_THROW(t.at({-1, 0}), std::out_of_range);
}

// at() must follow offset and strides, not assume row-major order.
TEST(At, FollowsOffsetAndStrides) {
    auto impl = std::make_shared<mt::TensorImpl>();
    impl->storage = std::make_shared<mt::Storage>(mt::Storage{{0, 1, 2, 3, 4, 5, 6, 7}});
    impl->shape = {2, 3};
    impl->strides = {1, 2};  // column-major
    impl->offset = 2;
    mt::Tensor t(impl);
    // element [i, j] = data[2 + i*1 + j*2]
    EXPECT_FLOAT_EQ(t.at({0, 0}), 2);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 3);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 6);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 7);
}

TEST(Tensor, RejectsStridesOfWrongLength) {
    auto impl = std::make_shared<mt::TensorImpl>();
    impl->storage = std::make_shared<mt::Storage>();
    impl->shape = {2, 3};
    impl->strides = {1};
    EXPECT_THROW(mt::Tensor{impl}, std::invalid_argument);
}

// --- Handle semantics ---------------------------------------------------------

TEST(Tensor, CopyIsTheSameTensor) {
    mt::Tensor a = mt::zeros({2, 2});
    mt::Tensor b = a;  // copies the handle, not the data
    b.at({0, 1}) = 42;
    EXPECT_FLOAT_EQ(a.at({0, 1}), 42);
    EXPECT_TRUE(a.is_same(b));
    EXPECT_FALSE(a.is_same(mt::zeros({2, 2})));
}

// --- Printing -----------------------------------------------------------------

TEST(Print, ScalarAndVector) {
    EXPECT_EQ(to_string(mt::from_data({2.5}, {})), "tensor(2.5)");
    EXPECT_EQ(to_string(mt::from_data({1, -2}, {2})), "tensor([1, -2])");
}

TEST(Print, MatrixLikePyTorch) {
    EXPECT_EQ(to_string(mt::from_data({1, 2, 3, 4, 5, 6}, {2, 3})),
              "tensor([[1, 2, 3],\n"
              "        [4, 5, 6]])");
}

TEST(Print, ThreeDimensions) {
    EXPECT_EQ(to_string(mt::from_data({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2})),
              "tensor([[[1, 2],\n"
              "         [3, 4]],\n"
              "\n"
              "        [[5, 6],\n"
              "         [7, 8]]])");
}
