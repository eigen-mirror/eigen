# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(LapackSymbols CXX)

# Stand-ins for eigen_lapack_static, eigen_blas_static and eigen_lapack_reference.
foreach(library lapack blas reference)
  add_library(fake_${library} STATIC symbols.cpp)
  string(TOUPPER "${library}" macro)
  target_compile_definitions(fake_${library} PRIVATE FAKE_${macro})
endforeach()
add_library(fake_reference_overlap STATIC symbols.cpp)
target_compile_definitions(fake_reference_overlap PRIVATE FAKE_REFERENCE FAKE_OVERLAP)
add_executable(fake_nm nm_output.cpp)

# Shared stand-ins where lapack/CMakeLists.txt checks the shared libraries' exports.
# Like eigen_lapack, they hide every symbol not marked for export.
set(shared_libraries "")
if(UNIX AND NOT APPLE AND NOT CYGWIN)
  add_library(fake_lapack_shared SHARED symbols.cpp)
  target_compile_definitions(fake_lapack_shared PRIVATE FAKE_LAPACK)
  add_library(fake_lapack_shared_hidden SHARED symbols.cpp)
  target_compile_definitions(fake_lapack_shared_hidden PRIVATE FAKE_LAPACK FAKE_HIDE_DPOTRF)
  add_library(fake_blas_shared SHARED symbols.cpp)
  target_compile_definitions(fake_blas_shared PRIVATE FAKE_BLAS)
  set_target_properties(fake_lapack_shared fake_lapack_shared_hidden fake_blas_shared PROPERTIES
                        CXX_VISIBILITY_PRESET hidden)
  set(shared_libraries
      "set(EIGEN_LAPACK_SHARED \"$<TARGET_FILE:fake_lapack_shared>\")\n"
      "set(EIGEN_LAPACK_SHARED_HIDDEN \"$<TARGET_FILE:fake_lapack_shared_hidden>\")\n"
      "set(EIGEN_BLAS_SHARED \"$<TARGET_FILE:fake_blas_shared>\")\n")
  string(REPLACE ";" "" shared_libraries "${shared_libraries}")
endif()

# The arguments lapack/CMakeLists.txt would pass to testing_symbols.cmake.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/libraries-$<CONFIG>.cmake" CONTENT
     "set(NM \"${CMAKE_NM}\")
set(FAKE_NM \"$<TARGET_FILE:fake_nm>\")
set(EIGEN_LAPACK \"$<TARGET_FILE:fake_lapack>\")
set(EIGEN_BLAS \"$<TARGET_FILE:fake_blas>\")
set(REFERENCE \"$<TARGET_FILE:fake_reference>\")
set(REFERENCE_OVERLAP \"$<TARGET_FILE:fake_reference_overlap>\")
${shared_libraries}")
