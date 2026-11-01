include(GoogleTest)

# One test executable for the whole project; ctest still sees every TEST() separately.
add_executable(mt_tests
  test_tensor.cpp
)
target_link_libraries(mt_tests PRIVATE minitensor GTest::gtest_main)
mt_set_warnings(mt_tests)

gtest_discover_tests(mt_tests)
