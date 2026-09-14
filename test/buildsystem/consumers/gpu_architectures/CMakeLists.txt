# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(GpuArchitectures NONE)
include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")

# Exercise route selection without requiring a toolkit, compiler, or GPU.
macro(find_package package)
  if(NOT "${package}" STREQUAL "CUDAToolkit")
    message(FATAL_ERROR "Unexpected package: ${package}")
  endif()
  set(CUDAToolkit_VERSION 12.8)
  set(CUDAToolkit_TARGET_DIR /cuda)
endmacro()

macro(enable_language language)
  if(NOT "${language}" STREQUAL "CUDA" OR NOT ROUTE STREQUAL "cuda-language")
    message(FATAL_ERROR "Unexpected language enablement: ${language}")
  endif()
endmacro()

set(EIGEN_TEST_CUDA ON)
if(ROUTE STREQUAL "nvc")
  set(EIGEN_TEST_CUDA_NVC ON)
elseif(ROUTE STREQUAL "windows-clang")
  set(EIGEN_TEST_CUDA_CLANG ON)
  set(WIN32 TRUE)
endif()
string(REPLACE "," ";" CMAKE_CUDA_ARCHITECTURES "${ARCHITECTURES}")
ei_gpu_testing_enable()
string(STRIP "${CMAKE_CXX_FLAGS}" actual_flags)
string(STRIP "${EXPECTED_FLAGS}" expected_flags)

if(ROUTE STREQUAL "cuda-language")
  if(NOT EIGEN_GPU_TEST_MODE STREQUAL "cuda-language")
    message(FATAL_ERROR "Expected the CUDA language route")
  endif()
elseif(NOT EIGEN_GPU_TEST_MODE STREQUAL "cuda-as-cxx" OR NOT actual_flags STREQUAL expected_flags)
  message(FATAL_ERROR "Unexpected C++ compiler flags: ${CMAKE_CXX_FLAGS}")
endif()
