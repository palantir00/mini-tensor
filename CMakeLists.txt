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
function(mt_set_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

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

# --- Tests ------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
