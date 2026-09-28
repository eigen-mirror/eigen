# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(GpuArchitectures NONE)
include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")

# Exercise route selection without requiring a toolkit, compiler, or GPU.
macro(find_package package)
  get_filename_component(toolkit_root "${CUDAToolkit_ROOT}" REALPATH)
  get_filename_component(expected_toolkit_root "${EXPECTED_TOOLKIT_ROOT}" REALPATH)
  if(NOT "${package}" STREQUAL "CUDAToolkit" OR NOT toolkit_root STREQUAL expected_toolkit_root)
    message(FATAL_ERROR "Unexpected package: ${package}, CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
  endif()
  set(CUDAToolkit_VERSION 12.8)
  set(CUDAToolkit_TARGET_DIR /cuda)
endmacro()

# Compiler identification compiles with the custom flags.
macro(enable_language language)
  if(NOT "${language}" STREQUAL "CUDA" OR NOT ROUTE STREQUAL "cuda-language" OR
     NOT " ${CMAKE_CUDA_FLAGS} " MATCHES " -allow-unsupported-compiler ")
    message(FATAL_ERROR "Unexpected language enablement: ${language}, CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
  endif()
endmacro()

unset(ENV{CUDAToolkit_ROOT})
set(EIGEN_TEST_CUDA ON)
if(ROUTE STREQUAL "nvc")
  set(EIGEN_TEST_CUDA_NVC ON)
elseif(ROUTE STREQUAL "windows-clang")
  set(EIGEN_TEST_CUDA_CLANG ON)
  set(WIN32 TRUE)
else()
  set(EIGEN_CUDA_CXX_FLAGS -allow-unsupported-compiler)
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
