# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(GpuFlags NONE)
include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")

function(check_flags actual expected what)
  separate_arguments(actual NATIVE_COMMAND "${actual}")
  separate_arguments(expected NATIVE_COMMAND "${expected}")
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "${what}: expected [${expected}], got [${actual}]")
  endif()
endfunction()

macro(find_package package)
  set(CUDAToolkit_VERSION 12.8)
  set(CUDAToolkit_TARGET_DIR /cuda)
endmacro()

# Keep compiler discovery hardware-independent, but run CMake's real flag/cache initialization.
macro(enable_language language)
  check_flags("${CMAKE_CUDA_FLAGS}" "${EXPECTED_DETECTION} ${extra_flags}" "compiler detection flags")
  set(CMAKE_CUDA_COMPILER_ID EigenFlagTest)
  include(CMakeInitializeConfigs)
  include(CMakeCUDAInformation)
endmacro()

set(EIGEN_TEST_CUDA ON)
set(EIGEN_CUDA_COMPUTE_ARCH 75)
set(ENV{CUDAFLAGS} "${ENVIRONMENT_FLAGS}")
if(DEFINED NORMAL_FLAGS)
  set(CMAKE_CUDA_FLAGS "${NORMAL_FLAGS}")
endif()
set(extra_flags "${EIGEN_CUDA_CXX_FLAGS}")
if(EIGEN_TEST_CUDA_CLANG)
  string(PREPEND extra_flags "--cuda-path=/cuda ")
  set(WIN32 FALSE)
endif()

# Compiler and platform modules can add flags only after compiler identification.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/Modules")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/Modules/Compiler/EigenFlagTest-CUDA.cmake"
     "string(APPEND CMAKE_CUDA_FLAGS_INIT \" -DPLATFORM_FLAG=1\")\n")
ei_gpu_testing_enable()
check_flags("${CMAKE_CUDA_FLAGS}" "${EXPECTED_BUILD} ${extra_flags}" "build flags")
get_property(cache_flags CACHE CMAKE_CUDA_FLAGS PROPERTY VALUE)
check_flags("${cache_flags}" "${EXPECTED_CACHE}" "cached flags")
