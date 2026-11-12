# Programs that use the library the way a user would.
add_executable(xor_mlp xor_mlp.cpp)
target_link_libraries(xor_mlp PRIVATE minitensor)
mt_set_warnings(xor_mlp)

# The demo is also an end-to-end test: it exits with 1 if the network does not learn XOR.
add_test(NAME XorDemo COMMAND xor_mlp ${CMAKE_CURRENT_BINARY_DIR}/loss.csv)
