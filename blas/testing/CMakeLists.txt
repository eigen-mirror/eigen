# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
endif()

if(TARGET eigen_blas)
  set(eigen_blas_lib eigen_blas)
else()
  set(eigen_blas_lib eigen_blas_static)
endif()

if(EIGEN_BUILD_BLAS_ILP64)
  if(TARGET eigen_blas_ilp64)
    set(eigen_blas_ilp64_lib eigen_blas_ilp64)
  else()
    set(eigen_blas_ilp64_lib eigen_blas_ilp64_static)
  endif()
endif()

# Every test below carries the "blas" label, so that ctest -L blas runs exactly these.

# Netlib test driver <name>.f as target <name><suffix>, linked to <lib> and run in <dir>; extra arguments are
# Fortran compile options.
function(eigen_add_blas_driver name suffix lib dir)
  set(target "${name}${suffix}")
  add_executable(${target} ${name}.f)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name} RUNTIME_OUTPUT_DIRECTORY "${dir}")
  target_compile_options(${target} PRIVATE ${ARGN})
  # The GNU complex level-3 drivers miscompute their own error norms at -O3.
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU" AND name MATCHES "^[cz]blat3$")
    target_compile_options(${target} PRIVATE "-O0")
  endif()
  target_link_libraries(${target} ${lib} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO} ${EXTERNAL_LIBS})
  add_test(NAME ${target}
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/runblastest.sh" ${name} "${CMAKE_CURRENT_SOURCE_DIR}/${name}.dat"
    WORKING_DIRECTORY "${dir}")
  set_property(TEST ${target} PROPERTY LABELS blas)
  add_dependencies(buildtests ${target})
endfunction()

set(eigen_blas_drivers sblat1 sblat2 sblat3 dblat1 dblat2 dblat3 cblat1 cblat2 cblat3 zblat1 zblat2 zblat3)

if(CMAKE_Fortran_COMPILER)
  foreach(driver IN LISTS eigen_blas_drivers)
    eigen_add_blas_driver(${driver} "" ${eigen_blas_lib} "${CMAKE_CURRENT_BINARY_DIR}")
  endforeach()

  if(EIGEN_BUILD_BLAS_ILP64)
    include(CheckFortranCompilerFlag)
    set(EIGEN_FORTRAN_INT64_FLAG "")
    foreach(flag "-fdefault-integer-8" "-i8")
      string(REGEX REPLACE "[^A-Za-z0-9]" "_" flag_var "${flag}")
      check_fortran_compiler_flag("${flag}" EIGEN_FORTRAN_HAS_${flag_var})
      if(EIGEN_FORTRAN_HAS_${flag_var})
        set(EIGEN_FORTRAN_INT64_FLAG "${flag}")
        break()
      endif()
    endforeach()

    if(EIGEN_FORTRAN_INT64_FLAG)
      set(ilp64_test_dir "${CMAKE_CURRENT_BINARY_DIR}/ilp64")
      file(MAKE_DIRECTORY "${ilp64_test_dir}")
      foreach(driver IN LISTS eigen_blas_drivers)
        eigen_add_blas_driver(${driver} "_64" ${eigen_blas_ilp64_lib} "${ilp64_test_dir}" "${EIGEN_FORTRAN_INT64_FLAG}")
      endforeach()
    endif()
  endif()
endif()

# A C++ test <name>.cpp linked to <lib>, beside it so that Windows finds its DLL (no rpath there).
function(eigen_add_blas_cpp_test name lib)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE "${Eigen_SOURCE_DIR}" "${Eigen_SOURCE_DIR}/test")
  target_link_libraries(${name} ${lib} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:${lib}>")
  add_test(NAME ${name} COMMAND ${name})
  set_property(TEST ${name} PROPERTY LABELS blas)
  add_dependencies(buildtests ${name})
endfunction()

if(EIGEN_BUILD_BLAS_ILP64)
  eigen_add_blas_cpp_test(blas_ilp64 ${eigen_blas_ilp64_lib})
  target_compile_definitions(blas_ilp64 PRIVATE "EIGEN_USE_BLAS" "EIGEN_64BIT_BLAS")
endif()

eigen_add_blas_cpp_test(blas_large_stride ${eigen_blas_lib})

# Differential test against netlib's reference BLAS, built from the Reference-LAPACK release with its symbols
# left without the trailing underscore so that both libraries link into one program.
option(EIGEN_TEST_BLAS_REFERENCE "Test Eigen's BLAS against netlib reference BLAS (downloads Reference-LAPACK)" OFF)
if(EIGEN_TEST_BLAS_REFERENCE)
  if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "GNU|LLVMFlang|Flang")
    message(FATAL_ERROR "EIGEN_TEST_BLAS_REFERENCE needs a Fortran compiler that accepts -fno-underscoring.")
  endif()
  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  # SOURCE_SUBDIR names a directory without a CMakeLists.txt, so FetchContent_MakeAvailable only downloads the
  # release and does not add its own project to the build.
  FetchContent_Declare(eigen_reference_blas
    URL https://codeload.github.com/Reference-LAPACK/lapack/tar.gz/refs/tags/v3.12.1
    URL_HASH SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422
    TLS_VERIFY ON
    SOURCE_SUBDIR no-cmake-project)
  FetchContent_MakeAvailable(eigen_reference_blas)
  file(GLOB netlib_blas_sources "${eigen_reference_blas_SOURCE_DIR}/BLAS/SRC/*.f"
                                "${eigen_reference_blas_SOURCE_DIR}/BLAS/SRC/*.f90")
  add_library(eigen_netlib_blas STATIC ${netlib_blas_sources})
  target_compile_options(eigen_netlib_blas PRIVATE "-fno-underscoring")

  add_executable(blas_reference blas_reference.cpp)
  target_include_directories(blas_reference PRIVATE "${Eigen_SOURCE_DIR}" "${Eigen_SOURCE_DIR}/test")
  # The shared library where there is one: this test includes test/main.h, which undefines NDEBUG and redefines
  # eigen_assert, so its instances of Eigen's inline templates differ from the static library's, and in one program
  # the linker keeps one instance of each for both.
  target_link_libraries(blas_reference ${eigen_blas_lib} eigen_netlib_blas ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  set_target_properties(blas_reference PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:${eigen_blas_lib}>")
  add_test(NAME blas_reference COMMAND blas_reference)
  set_property(TEST blas_reference PROPERTY LABELS blas)
  add_dependencies(buildtests blas_reference)
endif()

# Every BLAS test above, for a job that builds only these and the LAPACK tests.
add_custom_target(buildtests_blas)
get_property(blas_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(REMOVE_ITEM blas_tests buildtests_blas)
add_dependencies(buildtests_blas ${blas_tests})
