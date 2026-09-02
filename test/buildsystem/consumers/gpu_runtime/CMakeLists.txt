# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(GpuRuntimeLinkage CXX)
include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")

# Inspect linkage without requiring either GPU compiler or an installed CUDA toolkit.
add_library(CUDA::cudart_static INTERFACE IMPORTED)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cu" "int main() { return 0; }\n")
set(EIGEN_GPU_TEST_MODE cuda-as-cxx)
set(EIGEN_TEST_CUDA_NVC ON)
ei_add_gpu_test_executable(nvc_test "${CMAKE_CURRENT_BINARY_DIR}/dummy.cu")
get_target_property(nvc_libraries nvc_test LINK_LIBRARIES)
if(nvc_libraries)
  message(FATAL_ERROR "NVHPC must supply its own CUDA runtime, got: ${nvc_libraries}")
endif()

set(EIGEN_TEST_CUDA_NVC OFF)
ei_add_gpu_test_executable(clang_test "${CMAKE_CURRENT_BINARY_DIR}/dummy.cu")
get_target_property(clang_libraries clang_test LINK_LIBRARIES)
if(NOT "CUDA::cudart_static" IN_LIST clang_libraries)
  message(FATAL_ERROR "The Windows Clang path still needs an explicit CUDA runtime")
endif()
