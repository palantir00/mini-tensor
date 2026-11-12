include(GoogleTest)

# One test executable for the whole project; ctest still sees every TEST() separately.
add_executable(mt_tests
  test_tensor.cpp
  test_views.cpp
  test_ops.cpp
  test_autograd.cpp
  test_reference.cpp
  reference.cpp
)
target_link_libraries(mt_tests PRIVATE minitensor GTest::gtest_main)
mt_set_warnings(mt_tests)

# Reference data from PyTorch (written by scripts/gen_reference.py). An absolute path,
# so the tests find it no matter which directory they are started from.
target_compile_definitions(mt_tests PRIVATE MT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

gtest_discover_tests(mt_tests)
