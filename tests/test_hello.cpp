// Smoke tests for the build setup: GoogleTest is fetched, compiled and linked,
// and ctest discovers each TEST() separately.

#include <gtest/gtest.h>

#include <stdexcept>

TEST(Hello, GoogleTestWorks) {
    EXPECT_EQ(2 + 2, 4);
}

// The pattern used for error handling in the library: bad input throws,
// the test checks the exception type.
TEST(Hello, ExpectThrowWorks) {
    EXPECT_THROW(throw std::invalid_argument("bad shape"), std::invalid_argument);
}
