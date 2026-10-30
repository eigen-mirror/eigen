# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(LapackSummary CXX)
enable_testing()
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Exercise LAPACK's real registration without a Fortran compiler or downloads.
include(CheckLanguage)
macro(check_language language)
  set(CMAKE_Fortran_COMPILER "unused-fortran")
endmacro()
macro(enable_language language)
endmacro()
set(EIGEN_BUILD_LAPACK ON)
set(EIGEN_BUILD_BLAS ON)
set(EIGEN_ENABLE_LAPACK_TESTS ON)
set(EIGEN_INSTALL OFF)
add_library(eigen_blas_static INTERFACE)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
add_custom_target(buildtests)
foreach(precision s d)
  add_executable(xlintst${precision} IMPORTED)
  set_target_properties(xlintst${precision} PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
endforeach()

set(fixture "${CMAKE_CURRENT_BINARY_DIR}/lapack")
set(reference "${CMAKE_CURRENT_BINARY_DIR}/lapack-reference")
file(MAKE_DIRECTORY "${reference}/SRC" "${reference}/TESTING" "${reference}/INSTALL")
foreach(precision s d)
  file(WRITE "${reference}/INSTALL/${precision}roundup_lwork.f" "")
endforeach()
set(FETCHCONTENT_SOURCE_DIR_EIGEN_LAPACK_REFERENCE "${reference}")
file(GLOB sources "${EIGEN_SOURCE_DIR}/lapack/*.cpp")
file(COPY ${sources} "${EIGEN_SOURCE_DIR}/lapack/CMakeLists.txt"
          "${EIGEN_SOURCE_DIR}/lapack/testing_summary.cmake" DESTINATION "${fixture}")
foreach(directory MATGEN LIN EIG)
  file(MAKE_DIRECTORY "${reference}/TESTING/${directory}")
  file(WRITE "${reference}/TESTING/${directory}/CMakeLists.txt" "")
endforeach()
file(WRITE "${reference}/TESTING/stest.in" "All tests passed ( 100 tests run)\n")
file(WRITE "${reference}/TESTING/dtest.in" "All tests passed ( 20 tests run)\n")
# LAPACK 3.4.1 TESTING/EIG/?CHKBL, ?CHKBK, ?CHKGL and ?CHKGK summary formats.
foreach(precision s d c z)
  add_executable(xeigtst${precision} IMPORTED)
  set_target_properties(xeigtst${precision} PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
  foreach(kind bal bak gbal gbak)
    if(kind MATCHES "^g")
      string(TOUPPER "${precision}g${kind}" routine)
    else()
      string(TOUPPER "${precision}ge${kind}" routine)
    endif()
    set(output " .. test output of ${routine} ..\n value of largest test error = 0.000D+00\n")
    if(kind STREQUAL "gbak")
      string(TOUPPER "${precision}" prefix)
      string(APPEND output
        " example number where ${prefix}GGBAL info is not 0 = 0\n"
        " example number where ${prefix}GGBAK(L) info is not 0 = 0\n"
        " example number where ${prefix}GGBAK(R) info is not 0 = 0\n")
    else()
      string(APPEND output " example number where info is not zero = 0\n")
      if(kind MATCHES "bal$")
        string(APPEND output " example number where ILO or IHI wrong = 0\n")
      endif()
    endif()
    string(APPEND output
      " example number having largest error = 0\n"
      " number of examples where info is not 0 = 0\n"
      " total number of examples tested = 13\n")
    file(WRITE "${reference}/TESTING/${precision}${kind}.in" "${output}")
  endforeach()
endforeach()
file(WRITE "${reference}/TESTING/runtest.cmake"
     "configure_file(\"\${INPUT}\" \"\${OUTPUT}\" COPYONLY)\n")

# Fail if configuration attempts to discover any Python interpreter.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")
foreach(module Python PythonInterp Python2 Python3)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/modules/Find${module}.cmake"
       "message(FATAL_ERROR \"LAPACK tests must not require Python\")\n")
endforeach()
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/modules")
add_subdirectory("${fixture}" lapack-build)
