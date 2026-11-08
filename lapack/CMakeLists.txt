# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

project(EigenLapack CXX)

if(EIGEN_BUILD_LAPACK AND EIGEN_BUILD_BLAS)

option(EIGEN_ENABLE_LAPACK_TESTS "Enable the Lapack unit tests" OFF)

if(EIGEN_ENABLE_LAPACK_TESTS)
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
  else()
    message(STATUS "A Fortran compiler was not found; LAPACK unit tests won't be enabled")
    set(EIGEN_ENABLE_LAPACK_TESTS OFF)
  endif()
endif()

add_custom_target(lapack)
include_directories(../blas)

set(EigenLapack_SRCS
  dsecnd_INT_CPU_TIME.cpp second_INT_CPU_TIME.cpp single.cpp double.cpp complex_single.cpp complex_double.cpp
  ilaver.cpp
)

# The reference LAPACK release that the routines follow: ILAVER reports it, and the tests build its sources.
set(EIGEN_LAPACK_REFERENCE_VERSION 3.12.1)
set(EIGEN_LAPACK_REFERENCE_SHA256 2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422)
string(REPLACE "." ";" eigen_lapack_version "${EIGEN_LAPACK_REFERENCE_VERSION}")
list(GET eigen_lapack_version 0 eigen_lapack_version_major)
list(GET eigen_lapack_version 1 eigen_lapack_version_minor)
list(GET eigen_lapack_version 2 eigen_lapack_version_patch)
set_property(SOURCE ilaver.cpp APPEND PROPERTY COMPILE_DEFINITIONS
  EIGEN_LAPACK_VERSION_MAJOR=${eigen_lapack_version_major}
  EIGEN_LAPACK_VERSION_MINOR=${eigen_lapack_version_minor}
  EIGEN_LAPACK_VERSION_PATCH=${eigen_lapack_version_patch})

set(EIGEN_LAPACK_DEF "eigen_lapack.def")

if(EIGEN_ENABLE_LAPACK_TESTS)

  include(FetchContent)
  FetchContent_Declare(eigen_lapack_reference
    URL https://codeload.github.com/Reference-LAPACK/lapack/tar.gz/refs/tags/v${EIGEN_LAPACK_REFERENCE_VERSION}
    URL_HASH SHA256=${EIGEN_LAPACK_REFERENCE_SHA256}
    TLS_VERIFY ON)
  FetchContent_GetProperties(eigen_lapack_reference)
  if(NOT eigen_lapack_reference_POPULATED)
    FetchContent_Populate(eigen_lapack_reference)
  endif()
  set(LAPACK_SOURCE_DIR "${eigen_lapack_reference_SOURCE_DIR}")
  set(LAPACK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

  # The test drivers link Eigen's routines and reference LAPACK's remaining ones.
  # A reference source is left out when eigen_lapack.def exports its routine;
  # LAPACK_Test_Symbols checks that no routine is defined twice.
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/${EIGEN_LAPACK_DEF}" EigenLapack_routines
       REGEX "^[ \t]*[a-z0-9_]+_[ \t]*$")
  list(TRANSFORM EigenLapack_routines REPLACE "^[ \t]*([a-z0-9_]+)_[ \t]*$" "\\1")
  file(GLOB ReferenceLapack_candidates "${LAPACK_SOURCE_DIR}/SRC/*.f"
       "${LAPACK_SOURCE_DIR}/SRC/*.F" "${LAPACK_SOURCE_DIR}/SRC/*.f90"
       "${LAPACK_SOURCE_DIR}/SRC/*.F90"
       "${LAPACK_SOURCE_DIR}/SRC/DEPRECATED/*.f")
  list(APPEND ReferenceLapack_candidates
    "${LAPACK_SOURCE_DIR}/INSTALL/sroundup_lwork.f"
    "${LAPACK_SOURCE_DIR}/INSTALL/droundup_lwork.f")
  set(ReferenceLapack_SRCS "")
  foreach(source IN LISTS ReferenceLapack_candidates)
    get_filename_component(routine "${source}" NAME_WE)
    string(TOLOWER "${routine}" routine)
    if(NOT routine IN_LIST EigenLapack_routines)
      list(APPEND ReferenceLapack_SRCS "${source}")
    endif()
  endforeach()
endif()

set(EIGEN_LAPACK_TARGETS "")

add_library(eigen_lapack_static STATIC ${EigenLapack_SRCS})
target_link_libraries(eigen_lapack_static  eigen_blas_static)
list(APPEND EIGEN_LAPACK_TARGETS eigen_lapack_static)

if (EIGEN_BUILD_SHARED_LIBS)
  add_library(eigen_lapack SHARED ${EigenLapack_SRCS} ${EIGEN_LAPACK_DEF})
  # Build LAPACK but link BLAS.
  target_compile_definitions(eigen_lapack PUBLIC "EIGEN_BLAS_LINK_DLL" "EIGEN_LAPACK_BUILD_DLL")
  target_link_libraries(eigen_lapack  eigen_blas)
  set_target_properties(eigen_lapack PROPERTIES CXX_VISIBILITY_PRESET hidden
    VERSION ${EIGEN_VERSION_NUMBER}
    SOVERSION ${EIGEN_MAJOR_VERSION})
  list(APPEND EIGEN_LAPACK_TARGETS eigen_lapack)
endif()

# The 64-bit integer (ILP64) library exports the same symbol names as the 32-bit one, with EIGEN_BLAS_INT
# (blas/blas.h) as the integer type, as eigen_blas_ilp64 does; it links against eigen_blas_ilp64.
# blas/ may shadow a cached EIGEN_BUILD_BLAS_ILP64=ON with OFF, so test for the target it would have built.
if(TARGET eigen_blas_ilp64_static)
  option(EIGEN_BUILD_LAPACK_ILP64 "Toggles the building of the 64-bit integer (ILP64) Eigen Lapack library" ON)
else()
  set(EIGEN_BUILD_LAPACK_ILP64 OFF)
endif()

if(EIGEN_BUILD_LAPACK_ILP64)
  add_library(eigen_lapack_ilp64_static STATIC ${EigenLapack_SRCS})
  target_compile_definitions(eigen_lapack_ilp64_static PUBLIC "EIGEN_64BIT_BLAS")
  target_link_libraries(eigen_lapack_ilp64_static eigen_blas_ilp64_static)
  list(APPEND EIGEN_LAPACK_TARGETS eigen_lapack_ilp64_static)

  if(EIGEN_BUILD_SHARED_LIBS AND TARGET eigen_blas_ilp64)
    add_library(eigen_lapack_ilp64 SHARED ${EigenLapack_SRCS} ${EIGEN_LAPACK_DEF})
    # EIGEN_LAPACK_BUILD_DLL stays private, as EIGEN_BLAS_BUILD_DLL does for eigen_blas_ilp64.
    target_compile_definitions(eigen_lapack_ilp64
      PUBLIC "EIGEN_BLAS_LINK_DLL" "EIGEN_64BIT_BLAS"
      PRIVATE "EIGEN_LAPACK_BUILD_DLL")
    target_link_libraries(eigen_lapack_ilp64 eigen_blas_ilp64)
    set_target_properties(eigen_lapack_ilp64 PROPERTIES CXX_VISIBILITY_PRESET hidden
      VERSION ${EIGEN_VERSION_NUMBER}
      SOVERSION ${EIGEN_MAJOR_VERSION})
    list(APPEND EIGEN_LAPACK_TARGETS eigen_lapack_ilp64)
  endif()
endif()

foreach(target IN LISTS EIGEN_LAPACK_TARGETS)
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${target} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  target_link_libraries(${target} Eigen3::Eigen)
  add_dependencies(lapack ${target})
  if(EIGEN_INSTALL)
    install(TARGETS ${target}
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
  endif()
endforeach()

# Checks of Eigen's routines that need no Fortran compiler.
if(EIGEN_BUILD_TESTING)
  function(eigen_add_lapack_cxx_test name source library)
    add_executable(${name} testing/${source}.cpp)
    target_include_directories(${name} PRIVATE "${Eigen_SOURCE_DIR}/test")
    target_link_libraries(${name} ${library})
    if(NOT EIGEN_LEAVE_TEST_IN_ALL_TARGET)
      set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
    add_test(NAME ${name} COMMAND ${name})
    add_dependencies(buildtests ${name})
  endfunction()
  eigen_add_lapack_cxx_test(lapack_routines lapack_routines eigen_lapack_static)
  eigen_add_lapack_cxx_test(lapack_wrappers lapack_wrappers eigen_lapack_static)
  if(EIGEN_BUILD_LAPACK_ILP64)
    # The same checks through 64-bit integers.
    eigen_add_lapack_cxx_test(lapack_wrappers_ilp64 lapack_wrappers eigen_lapack_ilp64_static)
  endif()
endif()


if(EIGEN_ENABLE_LAPACK_TESTS)

  # Reference LAPACK's own Fortran flags, such as -frecursive for gfortran and
  # a strict floating-point model for Intel Fortran.
  list(APPEND CMAKE_MODULE_PATH "${LAPACK_SOURCE_DIR}/CMAKE")
  include(CheckLAPACKCompilerFlags)
  CheckLAPACKCompilerFlags()

  # Test-only: never installed and not part of the eigen_lapack libraries.
  add_library(eigen_lapack_reference STATIC EXCLUDE_FROM_ALL ${ReferenceLapack_SRCS})
  set_target_properties(eigen_lapack_reference PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/reference_modules")
  target_link_libraries(eigen_lapack_reference PUBLIC eigen_lapack_static)

  set(BUILD_SINGLE      true)
  set(BUILD_DOUBLE      true)
  set(BUILD_COMPLEX     true)
  set(BUILD_COMPLEX16   true)

  if(MSVC_VERSION)
#  string(REPLACE "/STACK:10000000" "/STACK:900000000000000000"
#    CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  string(REGEX REPLACE "(.*)/STACK:(.*) (.*)" "\\1/STACK:900000000000000000 \\3"
    CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  endif()
  file(MAKE_DIRECTORY "${LAPACK_BINARY_DIR}/TESTING")
  set(LAPACK_LIBRARIES eigen_lapack_reference)
  set(BLAS_LIBRARIES eigen_blas_static)
  set(TMGLIB tmglib)
  set(LAPACK_VERSION ${EIGEN_LAPACK_REFERENCE_VERSION})
  set(LAPACK_MAJOR_VERSION ${eigen_lapack_version_major})
  # Upstream's test libraries are build-only dependencies.
  function(lapack_install_library target)
  endfunction()
  add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING/MATGEN" testing/MATGEN)
  add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING/LIN" testing/LIN)
  add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING/EIG" testing/EIG)

  # Builds every registered driver, and the Fortran-free lapack_* tests defined above.
  add_custom_target(buildtests_lapack)
  if(TARGET buildtests)
    add_dependencies(buildtests buildtests_lapack)
  endif()
  get_property(lapack_cxx_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  list(FILTER lapack_cxx_tests INCLUDE REGEX "^lapack_")
  if(lapack_cxx_tests)
    add_dependencies(buildtests_lapack ${lapack_cxx_tests})
  endif()
  set(EIGEN_LAPACK_TEST_NAMES "")
  set(EIGEN_LAPACK_TEST_OUTPUTS "")
  # A relative <input> names a file in reference LAPACK's TESTING directory.
  macro(add_lapack_test output input target)
    if(IS_ABSOLUTE "${input}")
      set(TEST_INPUT "${input}")
    else()
      set(TEST_INPUT "${LAPACK_SOURCE_DIR}/TESTING/${input}")
    endif()
    set(TEST_OUTPUT "${LAPACK_BINARY_DIR}/TESTING/${output}")
    get_filename_component(input_name "${input}" NAME)
    string(REPLACE "." "_" input_name "${input_name}")
    set(testName "${target}_${input_name}")
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Reference LAPACK runs ${input} with ${target}, which its TESTING directory does not define")
    endif()
    if(NOT EXISTS "${TEST_INPUT}")
      message(FATAL_ERROR "Reference LAPACK test input ${TEST_INPUT} does not exist")
    endif()
    add_dependencies(buildtests_lapack ${target})
    list(APPEND EIGEN_LAPACK_TEST_NAMES "LAPACK-${testName}")
    list(APPEND EIGEN_LAPACK_TEST_OUTPUTS "${TEST_OUTPUT}")
    add_test(NAME LAPACK-${testName}
      COMMAND "${CMAKE_COMMAND}"
      -DTEST=$<TARGET_FILE:${target}>
      -DINPUT=${TEST_INPUT}
      -DOUTPUT=${TEST_OUTPUT}
      -DINTDIR=${CMAKE_CFG_INTDIR}
      -P "${LAPACK_SOURCE_DIR}/TESTING/runtest.cmake")
  endmacro()

  # Register the same drivers and inputs as reference LAPACK's own
  # TESTING/CMakeLists.txt, whose precision guards all hold here.
  file(STRINGS "${LAPACK_SOURCE_DIR}/TESTING/CMakeLists.txt" reference_tests
       REGEX "^[ \t]*add_lapack_test\\(")
  if(NOT reference_tests)
    message(FATAL_ERROR "No reference LAPACK tests were registered")
  endif()
  foreach(reference_test IN LISTS reference_tests)
    if(NOT reference_test MATCHES
        "^[ \t]*add_lapack_test\\([ \t]*([^ \t)]+)[ \t]+([^ \t)]+)[ \t]+([^ \t)]+)[ \t]*\\)")
      message(FATAL_ERROR "Unrecognized reference LAPACK test registration: ${reference_test}")
    endif()
    add_lapack_test("${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" "${CMAKE_MATCH_3}")
  endforeach()
  # Reference LAPACK's inputs stop at N = 50, below some of Eigen's blocking
  # thresholds; these repeat the paths that reach Eigen's routines at up to
  # the drivers' NMAX = 132.
  foreach(precision s d c z)
    add_lapack_test(${precision}test_large.out
                    "${CMAKE_CURRENT_SOURCE_DIR}/testing/${precision}test_large.in" xlintst${precision})
  endforeach()

  set(EIGEN_LAPACK_TEST_ALLOWED_FAILURES "${CMAKE_CURRENT_SOURCE_DIR}/testing_allowed_failures.txt"
      CACHE FILEPATH "Per-output failure counts that LAPACK_Test_Summary accepts; empty accepts none")
  string(REPLACE ";" "\n" lapack_outputs "${EIGEN_LAPACK_TEST_OUTPUTS}")
  file(WRITE "${LAPACK_BINARY_DIR}/lapack_test_outputs.txt" "${lapack_outputs}\n")
  add_test(
    NAME LAPACK_Test_Summary
    WORKING_DIRECTORY ${LAPACK_BINARY_DIR}
    COMMAND "${CMAKE_COMMAND}"
            "-DOUTPUT_MANIFEST=${LAPACK_BINARY_DIR}/lapack_test_outputs.txt"
            "-DREPORT_FILE=${LAPACK_BINARY_DIR}/TESTING/testing_results.txt"
            "-DFAILURES_FILE=${LAPACK_BINARY_DIR}/TESTING/testing_failures.txt"
            "-DALLOWED_FAILURES=${EIGEN_LAPACK_TEST_ALLOWED_FAILURES}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/testing_summary.cmake"
  )
  set_tests_properties(${EIGEN_LAPACK_TEST_NAMES} PROPERTIES FIXTURES_SETUP lapack_outputs)
  set_tests_properties(LAPACK_Test_Summary PROPERTIES FIXTURES_REQUIRED lapack_outputs)

  if(CMAKE_NM)
    set(symbols_args "")
    # ELF only: Cygwin is UNIX but builds PE DLLs, which nm -D cannot read.
    if(TARGET eigen_lapack AND UNIX AND NOT APPLE AND NOT CYGWIN)
      set(symbols_args "-DEIGEN_LAPACK_SHARED=$<TARGET_FILE:eigen_lapack>"
                       "-DEIGEN_BLAS_SHARED=$<TARGET_FILE:eigen_blas>")
      add_dependencies(buildtests_lapack eigen_lapack)
    endif()
    add_test(
      NAME LAPACK_Test_Symbols
      COMMAND "${CMAKE_COMMAND}"
              "-DNM=${CMAKE_NM}"
              "-DDEF_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${EIGEN_LAPACK_DEF}"
              "-DEIGEN_LAPACK=$<TARGET_FILE:eigen_lapack_static>"
              "-DEIGEN_BLAS=$<TARGET_FILE:eigen_blas_static>"
              "-DREFERENCE=$<TARGET_FILE:eigen_lapack_reference>"
              ${symbols_args}
              -P "${CMAKE_CURRENT_SOURCE_DIR}/testing_symbols.cmake"
    )
  endif()

endif()

elseif(EIGEN_BUILD_LAPACK AND NOT EIGEN_BUILD_BLAS)
 message(FATAL_ERROR "EIGEN_BUILD_LAPACK requires EIGEN_BUILD_BLAS")
endif() #EIGEN_BUILD_LAPACK
