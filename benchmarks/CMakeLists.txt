# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# 3.18 for the CUDA language support the GPU subtree needs (CMAKE_CUDA_ARCHITECTURES); the CPU tree asks for
# nothing newer than 3.10 did.
cmake_minimum_required(VERSION 3.18)
project(EigenBenchmarks CXX)

option(EIGEN_BENCH_CPU "Build the CPU benchmarks" ON)
option(EIGEN_BENCH_CUDA "Build the CUDA benchmarks under GPU/ (needs a CUDA compiler and CMAKE_CUDA_ARCHITECTURES)" OFF)

find_package(benchmark REQUIRED)
find_package(BLAS QUIET)
find_package(Threads REQUIRED)

# Eigen is a header-only library; find it relative to this directory.
set(EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Helper: add a Google Benchmark target.
#   eigen_add_benchmark(name source [LIBRARIES lib1 lib2 ...] [DEFINITIONS def1 def2 ...])
function(eigen_add_benchmark name source)
  cmake_parse_arguments(BENCH "" "" "LIBRARIES;DEFINITIONS" ${ARGN})
  if(NOT IS_ABSOLUTE "${source}")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  endif()
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${EIGEN_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE benchmark::benchmark benchmark::benchmark_main)
  if(BENCH_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${BENCH_LIBRARIES})
  endif()
  target_compile_options(${name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>
  )
  target_compile_definitions(${name} PRIVATE NDEBUG)
  if(BENCH_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${BENCH_DEFINITIONS})
  endif()
endfunction()

if(EIGEN_BENCH_CPU)
  add_subdirectory(Core)
  add_subdirectory(Cholesky)
  add_subdirectory(LU)
  add_subdirectory(QR)
  add_subdirectory(SVD)
  add_subdirectory(Eigenvalues)
  add_subdirectory(Geometry)
  add_subdirectory(Sparse)
  add_subdirectory(FFT)
  add_subdirectory(Householder)
  add_subdirectory(Solvers)
  add_subdirectory(Tuning)
  add_subdirectory(BLAS)
  add_subdirectory(ThreadPool)
endif()

if(EIGEN_BENCH_CUDA)
  # Architectures default to the GPU present, which needs 3.24; older CMake has to be told.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_LESS 3.24)
      message(FATAL_ERROR "EIGEN_BENCH_CUDA needs CMAKE_CUDA_ARCHITECTURES (e.g. 89) with CMake < 3.24")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(GPU)
endif()
