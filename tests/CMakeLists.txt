include(GoogleTest)

add_executable(test_hello test_hello.cpp)
target_link_libraries(test_hello PRIVATE GTest::gtest_main)
mt_set_warnings(test_hello)

# Registers every TEST() in the executable as a separate ctest test.
gtest_discover_tests(test_hello)
