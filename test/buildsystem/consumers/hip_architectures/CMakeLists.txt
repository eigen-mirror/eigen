# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.17)
project(EigenHipArchitectures LANGUAGES CXX)
enable_testing()

set(EIGEN_TEST_HIP ON)
set(EIGEN_ADD_TEST_FILENAME_EXTENSION cu)
set(EIGEN_TEST_MAX_SIZE 320)
include("${EIGEN_SOURCE_DIR}/cmake/EigenTesting.cmake")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
add_custom_target(buildtests)
add_custom_target(buildtests_gpu)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" "int main() { return 0; }\n")

macro(hip_reset_flags)
endmacro()

# FindHIP consumes HIPCC_OPTIONS when it creates its custom compile commands;
# CMake's native HIP_ARCHITECTURES property cannot affect those commands.
function(hip_add_executable target source)
  cmake_parse_arguments(HIP "" "" "HIPCC_OPTIONS" ${ARGN})
  if(NOT "${HIP_HIPCC_OPTIONS}" STREQUAL "${expected_options}")
    message(FATAL_ERROR "${target}: expected HIPCC_OPTIONS [${expected_options}], got [${HIP_HIPCC_OPTIONS}]")
  endif()
  add_executable(${target} "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")
endfunction()

set(EIGEN_HIP_ARCHITECTURES gfx1151)
set(expected_options -std=c++14 --offload-arch=gfx1151)
ei_add_test_internal(probe single_architecture)

set(EIGEN_HIP_ARCHITECTURES gfx906 "gfx90a:xnack-")
set(expected_options -std=c++14 --offload-arch=gfx906 "--offload-arch=gfx90a:xnack-")
ei_add_test_internal(probe multiple_architectures)

set(EIGEN_HIP_ARCHITECTURES "")
set(expected_options -std=c++14)
ei_add_test_internal(probe empty_architectures)
