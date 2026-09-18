# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.21)
project(EigenHipArchitectures LANGUAGES CXX)
enable_testing()

set(EIGEN_TEST_HIP ON)
set(EIGEN_ADD_TEST_FILENAME_EXTENSION cu)
set(EIGEN_TEST_MAX_SIZE 320)
include("${EIGEN_SOURCE_DIR}/cmake/EigenTesting.cmake")
include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
add_custom_target(buildtests)
add_custom_target(buildtests_gpu)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cu" "int main() { return 0; }\n")

# Observe compiler detection and target setup without requiring ROCm or a GPU.
macro(enable_language language)
  if(NOT "${language}" STREQUAL "HIP" OR
     NOT "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "${expected_architectures}")
    message(FATAL_ERROR "Unexpected HIP language setup: ${language}, ${CMAKE_HIP_ARCHITECTURES}")
  endif()
endmacro()

function(check_architectures target eigen_architectures cmake_architectures expected_architectures)
  set(EIGEN_HIP_ARCHITECTURES "${eigen_architectures}")
  set(CMAKE_HIP_ARCHITECTURES "${cmake_architectures}")
  ei_gpu_testing_enable()
  ei_add_test_internal("${CMAKE_CURRENT_BINARY_DIR}/probe" ${target})
  get_target_property(actual_architectures ${target} HIP_ARCHITECTURES)
  get_source_file_property(language "${CMAKE_CURRENT_BINARY_DIR}/probe.cu" LANGUAGE)
  get_target_property(definitions ${target} COMPILE_DEFINITIONS)
  if(NOT "${actual_architectures}" STREQUAL "${expected_architectures}" OR
     NOT language STREQUAL "HIP" OR NOT "EIGEN_USE_HIP" IN_LIST definitions)
    message(FATAL_ERROR "${target}: unexpected HIP target setup: ${actual_architectures}, ${language}, ${definitions}")
  endif()
  # The wiring has been checked; generation uses the available host compiler.
  set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/probe.cu" PROPERTIES LANGUAGE CXX)
endfunction()

check_architectures(single_architecture gfx1151 "" gfx1151)
check_architectures(multiple_architectures "gfx906;gfx90a:xnack-" "" "gfx906;gfx90a:xnack-")
check_architectures(empty_architectures "" "" "")
check_architectures(cmake_override gfx906 gfx942 gfx942)
check_architectures(cmake_off gfx906 OFF OFF)
