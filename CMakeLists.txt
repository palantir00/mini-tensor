cmake_minimum_required(VERSION 3.24)
project(minitensor VERSION 0.1.0 LANGUAGES CXX)

# --- Language standard ------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17 instead of -std=gnu++17

# Writes build/compile_commands.json, used by IntelliSense / clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Without a build type, Ninja compiles without -g (no debug info for gdb).
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Warnings for our own targets only; third-party code (GoogleTest) is not affected.
# MT_WERROR turns them into errors. CI sets it, so no warning can get into main; it stays OFF
# locally, so that a warning does not stop you while you experiment.
option(MT_WERROR "Treat warnings in mini-tensor's own targets as errors" OFF)
function(mt_set_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  if(MT_WERROR)
    target_compile_options(${target} PRIVATE -Werror)
  endif()
endfunction()

# --- Sanitizers (optional, in a separate build directory) -------------------
#   cmake -S . -B build-asan -G Ninja -DMT_SANITIZE=ON
# AddressSanitizer: out-of-bounds access, use after free; on Linux it includes LeakSanitizer
# (memory still allocated at exit that nothing points to, e.g. a shared_ptr cycle).
# UndefinedBehaviorSanitizer: signed overflow, bad shifts, misaligned pointers, ...
# -fno-sanitize-recover=all: stop at the first error. By default UBSan only prints a message
# and carries on, so the test would still pass. Set for every target (GoogleTest too), before
# any target is defined, so that all code and the final link use the same runtime.
option(MT_SANITIZE "Build with AddressSanitizer (+LeakSanitizer) and UndefinedBehaviorSanitizer" OFF)
if(MT_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- Library ------------------------------------------------------------------
add_library(minitensor
  src/tensor.cpp
  src/ops.cpp
  src/autograd.cpp
)
target_include_directories(minitensor PUBLIC include)
mt_set_warnings(minitensor)

# --- GoogleTest (pinned release, verified by hash) --------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
  URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
)
set(BUILD_GMOCK OFF)    # we only need gtest, not gmock
set(INSTALL_GTEST OFF)  # do not install gtest together with our project
FetchContent_MakeAvailable(googletest)

# --- Tests and examples -----------------------------------------------------
enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)  # examples/xor_mlp, also registered as a test
