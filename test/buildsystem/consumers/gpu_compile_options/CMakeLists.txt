# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(GpuCompileOptions CXX)
include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")
include("${EIGEN_SOURCE_DIR}/cmake/EigenTesting.cmake")

add_library(Eigen3::Eigen INTERFACE IMPORTED)
add_library(CUDA::cudart_static INTERFACE IMPORTED)
add_custom_target(buildtests)
add_custom_target(buildtests_gpu)
set(EIGEN_TEST_MAX_SIZE 320)

# Inspect the actual test registration without requiring a GPU compiler or toolkit.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cu" "int main() { return 0; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" "int main() { return 0; }\n")
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/dummy.cu" PROPERTIES LANGUAGE CXX)

function(check_options name mode compiler extension msvc expected_options expected_definitions)
  set(EIGEN_GPU_TEST_MODE "${mode}")
  set(CMAKE_CUDA_COMPILER_ID "${compiler}")
  set(EIGEN_ADD_TEST_FILENAME_EXTENSION "${extension}")
  set(MSVC "${msvc}")
  set(EIGEN_TEST_CUSTOM_CXX_FLAGS "/d2ReducedOptimizeHugeFunctions;/DEIGEN_STRONG_INLINE=inline;/Os;-DKEEP=1")
  ei_add_test_internal("${CMAKE_CURRENT_BINARY_DIR}/dummy" "${name}" "/W4 /DLOCAL_DEF=2")
  get_target_property(options "${name}" COMPILE_OPTIONS)
  get_target_property(definitions "${name}" COMPILE_DEFINITIONS)
  if(NOT options STREQUAL expected_options)
    message(FATAL_ERROR "${name}: expected options [${expected_options}], got [${options}]")
  endif()
  if(NOT definitions STREQUAL expected_definitions)
    message(FATAL_ERROR "${name}: expected definitions [${expected_definitions}], got [${definitions}]")
  endif()
endfunction()

set(msvc_options "/bigobj;/W4;/DLOCAL_DEF=2;/d2ReducedOptimizeHugeFunctions;/DEIGEN_STRONG_INLINE=inline;/Os;-DKEEP=1")
set(nvcc_options "-Xcompiler=/bigobj;-Xcompiler=/W4;-Xcompiler=/d2ReducedOptimizeHugeFunctions;-Xcompiler=/Os;-DKEEP=1")
set(definitions "EIGEN_TEST_MAX_SIZE=320")
check_options(nvcc cuda-language NVIDIA cu TRUE "${nvcc_options}"
              "${definitions};LOCAL_DEF=2;EIGEN_STRONG_INLINE=inline")
check_options(host cuda-language NVIDIA cpp TRUE "${msvc_options}" "${definitions}")
check_options(clang cuda-language Clang cu TRUE "${msvc_options}" "${definitions}")
check_options(cuda_as_cxx cuda-as-cxx NVIDIA cu TRUE "${msvc_options}" "${definitions}")
check_options(non_msvc cuda-language NVIDIA cu FALSE
              "/W4;/DLOCAL_DEF=2;/d2ReducedOptimizeHugeFunctions;/DEIGEN_STRONG_INLINE=inline;/Os;-DKEEP=1"
              "${definitions}")
