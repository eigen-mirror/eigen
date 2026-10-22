# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

macro(ei_add_property prop value)
  get_property(previous GLOBAL PROPERTY ${prop})
  if ((NOT previous) OR (previous STREQUAL ""))
    set_property(GLOBAL PROPERTY ${prop} "${value}")
  else()
    set_property(GLOBAL PROPERTY ${prop} "${previous} ${value}")
  endif()
endmacro()

#internal. See documentation of ei_add_test for details.
macro(ei_add_test_internal testname testname_with_suffix)
  set(targetname ${testname_with_suffix})

  if(EIGEN_ADD_TEST_FILENAME_EXTENSION)
    set(filename ${testname}.${EIGEN_ADD_TEST_FILENAME_EXTENSION})
  else()
    set(filename ${testname}.cpp)
  endif()

  # Add the current target to the list of subtest targets
  get_property(EIGEN_SUBTESTS_LIST GLOBAL PROPERTY EIGEN_SUBTESTS_LIST)
  set(EIGEN_SUBTESTS_LIST "${EIGEN_SUBTESTS_LIST}${targetname}\n")
  set_property(GLOBAL PROPERTY EIGEN_SUBTESTS_LIST "${EIGEN_SUBTESTS_LIST}")

  set(is_gpu_test OFF)
  if(EIGEN_ADD_TEST_FILENAME_EXTENSION STREQUAL cu)
    set(is_gpu_test ON)
    ei_add_gpu_test_executable(${targetname} ${filename})
  else()
    add_executable(${targetname} ${filename})
  endif()

  add_dependencies(buildtests ${targetname})

  if (is_gpu_test)
    add_dependencies(buildtests_gpu ${targetname})
  endif()

  if(EIGEN_NO_ASSERTION_CHECKING)
    target_compile_definitions(${targetname} PRIVATE EIGEN_NO_ASSERTION_CHECKING=1)
  else()
    if(EIGEN_DEBUG_ASSERTS)
      target_compile_definitions(${targetname} PRIVATE EIGEN_DEBUG_ASSERTS=1)
    endif()
  endif()

  target_compile_definitions(${targetname} PRIVATE EIGEN_TEST_MAX_SIZE=${EIGEN_TEST_MAX_SIZE})

  set(test_compile_options "")
  if(MSVC)
    list(APPEND test_compile_options "-bigobj")
  endif()

  # let the user pass flags.
  if(${ARGC} GREATER 2)
    separate_arguments(compile_options NATIVE_COMMAND "${ARGV2}")
    list(APPEND test_compile_options ${compile_options})
  endif()
  list(APPEND test_compile_options ${EIGEN_TEST_CUSTOM_CXX_FLAGS})

  if(is_gpu_test AND MSVC AND EIGEN_GPU_TEST_MODE STREQUAL "cuda-language"
     AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # These options are host (MSVC) flags in either spelling: nvcc reads a bare
    # /option as an input file and rejects MSVC's -options, so both go through
    # -Xcompiler. Definitions must reach both compilation passes. nvcc's own
    # options belong in CMAKE_CUDA_FLAGS.
    foreach(option IN LISTS test_compile_options)
      if(option MATCHES "^[/-]D(.+)")
        target_compile_definitions(${targetname} PRIVATE "${CMAKE_MATCH_1}")
      elseif(option MATCHES "^[/-]")
        string(REGEX REPLACE "^/" "-" option "${option}")
        target_compile_options(${targetname} PRIVATE "-Xcompiler=${option}")
      else()
        target_compile_options(${targetname} PRIVATE "${option}")
      endif()
    endforeach()
  else()
    target_compile_options(${targetname} PRIVATE ${test_compile_options})
  endif()

  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${targetname} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  if(EXTERNAL_LIBS)
    target_link_libraries(${targetname} ${EXTERNAL_LIBS})
  endif()
  if(EIGEN_TEST_CUSTOM_LINKER_FLAGS)
    target_link_libraries(${targetname} ${EIGEN_TEST_CUSTOM_LINKER_FLAGS})
  endif()
  target_link_libraries(${targetname} Eigen3::Eigen)

  if(${ARGC} GREATER 3)
    set(libs_to_link ${ARGV3})
    # it could be that some cmake module provides a bad library string " "  (just spaces),
    # and that severely breaks target_link_libraries ("can't link to -l-lstdc++" errors).
    # so we check for strings containing only spaces.
    string(STRIP "${libs_to_link}" libs_to_link_stripped)
    string(LENGTH "${libs_to_link_stripped}" libs_to_link_stripped_length)
    if(${libs_to_link_stripped_length} GREATER 0)
      # notice: no double quotes around ${libs_to_link} here. It may be a list.
      target_link_libraries(${targetname} ${libs_to_link})
    endif()
  endif()

  add_test(NAME ${testname_with_suffix} COMMAND "${targetname}")

  # Specify target and test labels according to EIGEN_CURRENT_SUBPROJECT
  get_property(current_subproject GLOBAL PROPERTY EIGEN_CURRENT_SUBPROJECT)
  if ((current_subproject) AND (NOT (current_subproject STREQUAL "")))
    set_property(TARGET ${targetname} PROPERTY LABELS "Build${current_subproject}")
    add_dependencies("Build${current_subproject}" ${targetname})
    set_property(TEST ${testname_with_suffix} PROPERTY LABELS "${current_subproject}")
  endif()
  if (is_gpu_test)
    # Add gpu tag for testing only GPU tests.
    set_property(TEST ${testname_with_suffix} APPEND PROPERTY LABELS "gpu")
    set_property(TEST ${testname_with_suffix} PROPERTY SKIP_RETURN_CODE 77)
  endif()

  if(EIGEN_SYCL)
    # Force include of the SYCL file at the end to avoid errors.
    set_property(TARGET ${targetname} PROPERTY COMPUTECPP_INCLUDE_AFTER 1)
    # Link against pthread and add sycl to target
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${targetname} Threads::Threads)
    add_sycl_to_target(TARGET ${targetname} SOURCES ${filename})
  endif(EIGEN_SYCL)

  # Match on the parent test name (e.g. "packetmath" → all packetmath_N) or
  # the specific subtest name (e.g. "packetmath_2"); see EigenConfigureTesting.cmake
  # for why this lives here instead of in a post-hoc registration.
  if ("${testname_with_suffix}" IN_LIST ei_smoke_test_list OR "${testname}" IN_LIST ei_smoke_test_list)
    set_property(TEST ${testname_with_suffix} APPEND PROPERTY LABELS "smoketest")
    add_dependencies(buildsmoketests ${targetname})
  endif()
endmacro(ei_add_test_internal)
# Macro to add a test
#
# the unique mandatory parameter testname must correspond to a file
# <testname>.cpp which follows this pattern:
#
# #include "main.h"
# void test_<testname>() { ... }
#
# Depending on the contents of that file, this macro can have 2 behaviors,
# see below.
#
# The optional 2nd parameter is libraries to link to.
#
# A. Default behavior
#
# this macro adds an executable <testname> as well as a ctest test
# named <testname> too.
#
# On platforms with bash simply run:
#   "ctest -V" or "ctest -V -R <testname>"
# On other platform use ctest as usual
#
# B. Multi-part behavior
#
# If the source file matches the regexp
#    CALL_SUBTEST_[0-9]+|EIGEN_TEST_PART_[0-9]+
# then it is interpreted as a multi-part test. The behavior then depends on the
# CMake option EIGEN_SPLIT_LARGE_TESTS, which is ON by default.
#
# If EIGEN_SPLIT_LARGE_TESTS is OFF, the behavior is the same as in A (the multi-part
# aspect is ignored).
#
# If EIGEN_SPLIT_LARGE_TESTS is ON, the test is split into multiple executables
#   test_<testname>_<N>
# where N runs from 1 to the greatest occurrence found in the source file. Each of these
# executables is built passing -DEIGEN_TEST_PART_N. This allows to split large tests
# into smaller executables.
#
# Moreover, targets <testname> are still generated, they
# have the effect of building all the parts of the test.
#
# A test listed in EigenTestPartGroups.cmake compiles each listed range of
# parts "<first>-<last>" as one executable <testname>_<first>, built with every
# member's EIGEN_TEST_PART_<N> defined; unlisted parts stay separate.  Only
# parts that differ by CALL_SUBTEST_<N> alone may be grouped, so a source with
# explicit EIGEN_TEST_PART_<N> markers, or a part named individually in the
# smoke-test list, is a configure error.
#
# Again, ctest -R allows to run all matching tests.
macro(ei_add_test testname)
  get_property(EIGEN_TESTS_LIST GLOBAL PROPERTY EIGEN_TESTS_LIST)
  set(EIGEN_TESTS_LIST "${EIGEN_TESTS_LIST}${testname}\n")
  set_property(GLOBAL PROPERTY EIGEN_TESTS_LIST "${EIGEN_TESTS_LIST}")

  if(EIGEN_ADD_TEST_FILENAME_EXTENSION)
    set(filename ${testname}.${EIGEN_ADD_TEST_FILENAME_EXTENSION})
  else()
    set(filename ${testname}.cpp)
  endif()

  file(READ "${filename}" test_source)
  string(REGEX MATCHALL "CALL_SUBTEST_[0-9]+|EIGEN_TEST_PART_[0-9]+|EIGEN_SUFFIXES(;[0-9]+)+"
         occurrences "${test_source}")
  string(REGEX REPLACE "CALL_SUBTEST_|EIGEN_TEST_PART_|EIGEN_SUFFIXES" "" suffixes "${occurrences}")
  list(REMOVE_DUPLICATES suffixes)
  set(explicit_suffixes "")
  if( (NOT EIGEN_SPLIT_LARGE_TESTS) AND suffixes)
    # Check whether we have EIGEN_TEST_PART_* statements, in which case we likely must enforce splitting.
    # For instance, indexed_view activate a different c++ version for each part.
    string(REGEX MATCHALL "EIGEN_TEST_PART_[0-9]+" occurrences "${test_source}")
    string(REGEX REPLACE "EIGEN_TEST_PART_" "" explicit_suffixes "${occurrences}")
    list(REMOVE_DUPLICATES explicit_suffixes)
  endif()
  if( (EIGEN_SPLIT_LARGE_TESTS AND suffixes) OR explicit_suffixes)
    add_custom_target(${testname})
    set(ei_grouped_suffixes "")
    if(EIGEN_SPLIT_LARGE_TESTS AND DEFINED ei_test_part_groups_${testname})
      if(test_source MATCHES "EIGEN_TEST_PART_[0-9]+")
        message(FATAL_ERROR "${testname}: parts with EIGEN_TEST_PART_<N> markers cannot be grouped")
      endif()
      foreach(ei_part_range IN LISTS ei_test_part_groups_${testname})
        if(NOT ei_part_range MATCHES "^([0-9]+)-([0-9]+)$")
          message(FATAL_ERROR "${testname}: part group \"${ei_part_range}\" is not of the form <first>-<last>")
        endif()
        set(ei_part_first ${CMAKE_MATCH_1})
        set(ei_part_last ${CMAKE_MATCH_2})
        set(ei_part_members "")
        foreach(suffix ${suffixes})
          if(suffix GREATER_EQUAL ei_part_first AND suffix LESS_EQUAL ei_part_last)
            if("${testname}_${suffix}" IN_LIST ei_smoke_test_list)
              message(FATAL_ERROR "${testname}_${suffix} is a smoke test and cannot be grouped")
            endif()
            list(APPEND ei_part_members ${suffix})
          endif()
        endforeach()
        if(ei_part_members)
          list(SORT ei_part_members COMPARE NATURAL)
          list(GET ei_part_members 0 ei_part_name)
          ei_add_test_internal(${testname} ${testname}_${ei_part_name} "${ARGV1}" "${ARGV2}")
          add_dependencies(${testname} ${testname}_${ei_part_name})
          foreach(suffix ${ei_part_members})
            target_compile_definitions(${testname}_${ei_part_name} PRIVATE -DEIGEN_TEST_PART_${suffix}=1)
          endforeach()
          list(APPEND ei_grouped_suffixes ${ei_part_members})
        endif()
      endforeach()
    endif()
    foreach(suffix ${suffixes})
      if(suffix IN_LIST ei_grouped_suffixes)
        continue()
      endif()
      ei_add_test_internal(${testname} ${testname}_${suffix} "${ARGV1}" "${ARGV2}")
      add_dependencies(${testname} ${testname}_${suffix})
      target_compile_definitions(${testname}_${suffix} PRIVATE -DEIGEN_TEST_PART_${suffix}=1)
    endforeach()
  else()
    ei_add_test_internal(${testname} ${testname} "${ARGV1}" "${ARGV2}")
    target_compile_definitions(${testname} PRIVATE -DEIGEN_TEST_PART_ALL=1)
  endif()
endmacro()

set(EIGEN_FAILTEST_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/EigenFailtest.cmake")

# Adds a compile-failure test pair from ${testname}.cpp: ${testname}_ok must
# compile, and ${testname}_ko, built with EIGEN_SHOULD_FAIL_TO_BUILD, must not.
# Neither test builds anything: both inspect the executables that the
# buildfailtests fixture produced, so ei_add_failtest_fixture() must follow the
# last ei_add_failtest() call.
macro(ei_add_failtest testname)

  set(test_target_ok ${testname}_ok)
  set(test_target_ko ${testname}_ko)

  add_executable(${test_target_ok} ${testname}.cpp)
  add_executable(${test_target_ko} ${testname}.cpp)
  target_link_libraries(${test_target_ok} PRIVATE Eigen3::Eigen)
  target_link_libraries(${test_target_ko} PRIVATE Eigen3::Eigen)
  set_target_properties(${test_target_ok} ${test_target_ko} PROPERTIES
                        EXCLUDE_FROM_ALL TRUE
                        EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  target_compile_definitions(${test_target_ko} PRIVATE EIGEN_SHOULD_FAIL_TO_BUILD)
  set_property(GLOBAL APPEND PROPERTY EIGEN_FAILTEST_TARGETS ${test_target_ok} ${test_target_ko})

  set(test_check_args
      -DEIGEN_FAILTEST_ACTION=check
      -DEIGEN_FAILTEST_BINARY_DIR=${CMAKE_BINARY_DIR}
      -DEIGEN_FAILTEST_CONFIG=$<CONFIG>
      -DEIGEN_FAILTEST_LOCK=${CMAKE_CURRENT_BINARY_DIR}/buildfailtests.lock
      -DEIGEN_FAILTEST_OK=${test_target_ok}
      -DEIGEN_FAILTEST_OK_FILE=$<TARGET_FILE:${test_target_ok}>)
  add_test(NAME ${test_target_ok}
           COMMAND ${CMAKE_COMMAND} ${test_check_args} -P ${EIGEN_FAILTEST_SCRIPT})
  add_test(NAME ${test_target_ko}
           COMMAND ${CMAKE_COMMAND} ${test_check_args}
                   -DEIGEN_FAILTEST_KO=${test_target_ko}
                   -DEIGEN_FAILTEST_KO_FILE=$<TARGET_FILE:${test_target_ko}>
                   -P ${EIGEN_FAILTEST_SCRIPT})
  # Disable emulator if cross-compiling.
  if (CMAKE_CROSSCOMPILING)
    set_property(TEST ${test_target_ok} PROPERTY CROSSCOMPILING_EMULATOR "")
    set_property(TEST ${test_target_ko} PROPERTY CROSSCOMPILING_EMULATOR "")
  endif()

  set_tests_properties(${test_target_ok} ${test_target_ko} PROPERTIES
                       FIXTURES_REQUIRED eigen_failtest LABELS failtest)
endmacro()

# Registers the buildfailtests test, the fixture that every ei_add_failtest()
# test requires.  It builds all failtest targets in one keep-going invocation,
# so the compiles run in parallel under a single build system instead of one
# build per test, which would have to be serialized: concurrent builds over one
# binary directory collide.
function(ei_add_failtest_fixture)
  get_property(targets GLOBAL PROPERTY EIGEN_FAILTEST_TARGETS)
  if(NOT targets)
    return()
  endif()

  add_custom_target(buildfailtests)
  add_dependencies(buildfailtests ${targets})

  set(files "")
  foreach(target IN LISTS targets)
    string(APPEND files "  \"$<TARGET_FILE:${target}>\"\n")
  endforeach()
  set(list_file "${CMAKE_CURRENT_BINARY_DIR}/buildfailtests-$<CONFIG>.cmake")
  file(GENERATE OUTPUT "${list_file}"
       CONTENT "set(EIGEN_FAILTEST_TARGETS ${targets})\nset(EIGEN_FAILTEST_FILES\n${files})\n")

  add_test(NAME buildfailtests
           COMMAND ${CMAKE_COMMAND}
                   -DEIGEN_FAILTEST_ACTION=build
                   -DEIGEN_FAILTEST_BINARY_DIR=${CMAKE_BINARY_DIR}
                   -DEIGEN_FAILTEST_CONFIG=$<CONFIG>
                   -DEIGEN_FAILTEST_GENERATOR=${CMAKE_GENERATOR}
                   -DEIGEN_FAILTEST_LIST=${list_file}
                   -P ${EIGEN_FAILTEST_SCRIPT})
  if (CMAKE_CROSSCOMPILING)
    set_property(TEST buildfailtests PROPERTY CROSSCOMPILING_EMULATOR "")
  endif()
  set_tests_properties(buildfailtests PROPERTIES FIXTURES_SETUP eigen_failtest LABELS failtest)
endfunction()

# print a summary of the different options
macro(ei_testing_print_summary)
  message(STATUS "************************************************************")
  message(STATUS "***    Eigen's unit tests configuration summary          ***")
  message(STATUS "************************************************************")
  message(STATUS "")
  message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
  message(STATUS "Build site:        ${SITE}")
  message(STATUS "Build string:      ${BUILDNAME}")
  get_property(EIGEN_TESTING_SUMMARY GLOBAL PROPERTY EIGEN_TESTING_SUMMARY)
  get_property(EIGEN_TESTED_BACKENDS GLOBAL PROPERTY EIGEN_TESTED_BACKENDS)
  get_property(EIGEN_MISSING_BACKENDS GLOBAL PROPERTY EIGEN_MISSING_BACKENDS)
  message(STATUS "Enabled backends:  ${EIGEN_TESTED_BACKENDS}")
  message(STATUS "Disabled backends: ${EIGEN_MISSING_BACKENDS}")

  if(EIGEN_DEFAULT_TO_ROW_MAJOR)
    message(STATUS "Default order:     Row-major")
  else()
    message(STATUS "Default order:     Column-major")
  endif()

  if(EIGEN_TEST_NO_EXPLICIT_ALIGNMENT)
    message(STATUS "Explicit alignment (hence vectorization) disabled")
  elseif(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
    message(STATUS "Explicit vectorization disabled (alignment kept enabled)")
  else()

  message(STATUS "Maximal matrix/vector size: ${EIGEN_TEST_MAX_SIZE}")

    if(EIGEN_TEST_SSE2)
      message(STATUS "SSE2:              ON")
    else()
      message(STATUS "SSE2:              Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSE3)
      message(STATUS "SSE3:              ON")
    else()
      message(STATUS "SSE3:              Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSSE3)
      message(STATUS "SSSE3:             ON")
    else()
      message(STATUS "SSSE3:             Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSE4_1)
      message(STATUS "SSE4.1:            ON")
    else()
      message(STATUS "SSE4.1:            Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSE4_2)
      message(STATUS "SSE4.2:            ON")
    else()
      message(STATUS "SSE4.2:            Using architecture defaults")
    endif()

    if(EIGEN_TEST_AVX)
      message(STATUS "AVX:               ON")
    else()
      message(STATUS "AVX:               Using architecture defaults")
    endif()

    if(EIGEN_TEST_AVX2)
      message(STATUS "AVX2:              ON")
    else()
      message(STATUS "AVX2:              Using architecture defaults")
    endif()

    if(EIGEN_TEST_FMA)
      message(STATUS "FMA:               ON")
    else()
      message(STATUS "FMA:               Using architecture defaults")
    endif()

    if(EIGEN_TEST_AVX512)
      message(STATUS "AVX512:            ON")
    else()
      message(STATUS "AVX512:            Using architecture defaults")
    endif()

    if(EIGEN_TEST_AVX512DQ)
      message(STATUS "AVX512DQ:          ON")
    else()
      message(STATUS "AVX512DQ:          Using architecture defaults")
    endif()

    if(EIGEN_TEST_ALTIVEC)
      message(STATUS "Altivec:           ON")
    else()
      message(STATUS "Altivec:           Using architecture defaults")
    endif()

    if(EIGEN_TEST_VSX)
      message(STATUS "VSX:               ON")
    else()
      message(STATUS "VSX:               Using architecture defaults")
    endif()

    if(EIGEN_TEST_MSA)
      message(STATUS "MIPS MSA:          ON")
    else()
      message(STATUS "MIPS MSA:          Using architecture defaults")
    endif()

    if(EIGEN_TEST_NEON)
      message(STATUS "ARM NEON:          ON")
    else()
      message(STATUS "ARM NEON:          Using architecture defaults")
    endif()

    if(EIGEN_TEST_NEON64)
      message(STATUS "ARMv8 NEON:        ON")
    else()
      message(STATUS "ARMv8 NEON:        Using architecture defaults")
    endif()

    if(EIGEN_TEST_ZVECTOR)
      message(STATUS "S390X ZVECTOR:     ON")
    else()
      message(STATUS "S390X ZVECTOR:     Using architecture defaults")
    endif()

    if(EIGEN_TEST_LSX)
        message(STATUS "LSX:               ON")
    else()
        message(STATUS "LSX:               Using architecture defaults")
    endif()

    if(EIGEN_TEST_SYCL)
      if(EIGEN_SYCL_TRISYCL)
        message(STATUS "SYCL:              ON (using triSYCL)")
      elseif(EIGEN_SYCL_ComputeCpp)
        message(STATUS "SYCL:              ON (using computeCPP)")
      elseif(EIGEN_SYCL_DPCPP)
        message(STATUS "SYCL:              ON (using DPCPP)")
      endif()
    else()
      message(STATUS "SYCL:              OFF")
    endif()
    if(EIGEN_TEST_CUDA)
      get_property(EIGEN_CUDA_ARCH_SUMMARY GLOBAL PROPERTY EIGEN_CUDA_COMPUTE_ARCH_RESOLVED)
      if(EIGEN_CUDA_ARCH_SUMMARY)
        string(REPLACE ";" ", sm_" EIGEN_CUDA_ARCH_SUMMARY ", sm_${EIGEN_CUDA_ARCH_SUMMARY}")
      endif()
      if(EIGEN_TEST_CUDA_NVC)
        set(EIGEN_CUDA_DRIVER "nvc++")
      elseif(EIGEN_TEST_CUDA_CLANG)
        set(EIGEN_CUDA_DRIVER "clang")
      else()
        set(EIGEN_CUDA_DRIVER "nvcc")
      endif()
      message(STATUS "CUDA:              ON (using ${EIGEN_CUDA_DRIVER}${EIGEN_CUDA_ARCH_SUMMARY})")
    else()
      message(STATUS "CUDA:              OFF")
    endif()
    if(EIGEN_TEST_HIP)
      message(STATUS "HIP:               ON (using hipcc)")
    else()
      message(STATUS "HIP:               OFF")
    endif()

  endif() # vectorization / alignment options

  message(STATUS "\n${EIGEN_TESTING_SUMMARY}")

  message(STATUS "************************************************************")
endmacro()

macro(ei_init_testing)
  define_property(GLOBAL PROPERTY EIGEN_CURRENT_SUBPROJECT BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_TESTED_BACKENDS BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_MISSING_BACKENDS BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_TESTING_SUMMARY BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_TESTS_LIST BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_SUBTESTS_LIST BRIEF_DOCS " " FULL_DOCS " ")

  set_property(GLOBAL PROPERTY EIGEN_TESTED_BACKENDS "")
  set_property(GLOBAL PROPERTY EIGEN_MISSING_BACKENDS "")
  set_property(GLOBAL PROPERTY EIGEN_TESTING_SUMMARY "")
  set_property(GLOBAL PROPERTY EIGEN_TESTS_LIST "")
  set_property(GLOBAL PROPERTY EIGEN_SUBTESTS_LIST "")

  define_property(GLOBAL PROPERTY EIGEN_FAILTEST_FAILURE_COUNT BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_FAILTEST_COUNT BRIEF_DOCS " " FULL_DOCS " ")

  set_property(GLOBAL PROPERTY EIGEN_FAILTEST_FAILURE_COUNT "0")
  set_property(GLOBAL PROPERTY EIGEN_FAILTEST_COUNT "0")
endmacro()

macro(ei_set_sitename)
  # if the sitename is not yet set, try to set it
  if(NOT ${SITE} OR ${SITE} STREQUAL "")
    set(eigen_computername $ENV{COMPUTERNAME})
    set(eigen_hostname $ENV{HOSTNAME})
    if(eigen_hostname)
      set(SITE ${eigen_hostname})
    elseif(eigen_computername)
      set(SITE ${eigen_computername})
    endif()
  endif()
  # in case it is already set, enforce lower case
  if(SITE)
    string(TOLOWER ${SITE} SITE)
  endif()
endmacro()

function(ei_get_compilerver VAR)
  set(compiler "${CMAKE_CXX_COMPILER_ID}")
  if(compiler STREQUAL "")
    set(compiler "unknown")
  endif()
  string(TOLOWER "${compiler}" compiler)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND
     NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(APPEND compiler "-msvc")
  endif()

  set(version "${CMAKE_CXX_COMPILER_VERSION}")
  if(version STREQUAL "")
    set(version "unknown")
  endif()
  set(${VAR} "${compiler}-${version}" PARENT_SCOPE)
endfunction()

function(ei_get_cxxflags VAR)
  set(flags "")
  # Report requested test options, not an ISA inferred from pointer width or host.
  foreach(feature IN ITEMS NEON NEON64 ZVECTOR Z14 VSX ALTIVEC MSA LSX SVE SME
                           SSE2 SSE3 SSSE3 SSE4_1 SSE4_2 AVX AVX2 AVX512 AVX512DQ AVX512FP16 FMA F16C)
    if(EIGEN_TEST_${feature})
      if(feature STREQUAL "SVE" AND NOT "${EIGEN_TEST_SVE_VL}" STREQUAL "")
        list(APPEND flags "SVE${EIGEN_TEST_SVE_VL}")
      else()
        list(APPEND flags "${feature}")
      endif()
    endif()
  endforeach()
  if(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
    list(APPEND flags NOVEC)
  endif()
  if(NOT flags)
    list(APPEND flags default)
  endif()

  if(EIGEN_TEST_OPENMP)
    list(APPEND flags OMP)
  endif()
  if(EIGEN_DEFAULT_TO_ROW_MAJOR)
    list(APPEND flags ROWMAJ)
  endif()
  list(JOIN flags "-" flags)
  set(${VAR} "${flags}" PARENT_SCOPE)
endfunction()

function(ei_get_target_architecture VAR)
  # Apple universal builds and Visual Studio targets may differ from the host.
  if(APPLE AND NOT "${CMAKE_OSX_ARCHITECTURES}" STREQUAL "")
    set(architectures "${CMAKE_OSX_ARCHITECTURES}")
  elseif(NOT "${CMAKE_CXX_COMPILER_TARGET}" STREQUAL "")
    # Keep the full target: QCC selectors are not necessarily target triples.
    set(architectures "${CMAKE_CXX_COMPILER_TARGET}")
  elseif(NOT "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "")
    set(architectures "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
  elseif(NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "")
    string(REGEX REPLACE ",.*$" "" architectures "${CMAKE_GENERATOR_PLATFORM}")
    if(architectures STREQUAL "")
      set(architectures "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
  else()
    set(architectures "${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  set(target_architectures "")
  foreach(architecture IN LISTS architectures)
    string(TOLOWER "${architecture}" architecture)
    if(architecture MATCHES "^(amd64|x64|x86_64)$")
      set(architecture x86_64)
    elseif(architecture MATCHES "^(win32|x86|i[3-6]86)$")
      set(architecture x86)
    elseif(architecture MATCHES "^(arm64|aarch64)$")
      set(architecture aarch64)
    endif()
    # Match the GNU-only -m32 handling in the top-level test configuration.
    if(EIGEN_TEST_32BIT AND CMAKE_COMPILER_IS_GNUCXX AND architecture STREQUAL "x86_64")
      set(architecture x86)
    endif()
    list(APPEND target_architectures "${architecture}")
  endforeach()
  if(NOT target_architectures)
    list(APPEND target_architectures unknown)
  endif()
  list(REMOVE_DUPLICATES target_architectures)
  list(SORT target_architectures)
  list(JOIN target_architectures "+" architecture)
  set(${VAR} "${architecture}" PARENT_SCOPE)
endfunction()

macro(ei_set_build_string)
  ei_get_compilerver(LOCAL_COMPILER_VERSION)
  ei_get_cxxflags(LOCAL_COMPILER_FLAGS)
  ei_get_target_architecture(LOCAL_TARGET_ARCHITECTURE)

  set(TMP_BUILD_STRING "${CMAKE_SYSTEM}-${LOCAL_TARGET_ARCHITECTURE}-${LOCAL_COMPILER_VERSION}")
  if(NOT "${LOCAL_COMPILER_FLAGS}" STREQUAL "")
    string(APPEND TMP_BUILD_STRING "-${LOCAL_COMPILER_FLAGS}")
  endif()

  if(EIGEN_TEST_EXTERNAL_BLAS)
    string(APPEND TMP_BUILD_STRING "-external_blas")
  endif()

  if(EIGEN_TEST_32BIT AND CMAKE_COMPILER_IS_GNUCXX)
    string(APPEND TMP_BUILD_STRING "-32bit")
  elseif("${CMAKE_SIZEOF_VOID_P}" MATCHES "^[1-9][0-9]*$")
    math(EXPR LOCAL_POINTER_BITS "8 * ${CMAKE_SIZEOF_VOID_P}")
    string(APPEND TMP_BUILD_STRING "-${LOCAL_POINTER_BITS}bit")
  else()
    string(APPEND TMP_BUILD_STRING "-unknown-bits")
  endif()

  if(EIGEN_BUILD_STRING_SUFFIX)
    string(APPEND TMP_BUILD_STRING "-${EIGEN_BUILD_STRING_SUFFIX}")
  endif()
  string(TOLOWER "${TMP_BUILD_STRING}" BUILDNAME)
endmacro()



# Split all tests listed in EIGEN_TESTS_LIST into num_splits many targets
# named buildtestspartN with N = { 0, ..., num_splits-1}.
#
# The intention behind the existence of this macro is the size of Eigen's
# testsuite. Together with the relatively big compile-times building all tests
# can take a substantial amount of time depending on the available hardware.
#
# The last buildtestspartN target will build possible remaining tests.
#
# An example:
#
#   EIGEN_TESTS_LIST= [ test1, test2, test3, test4, test5, test6, test7 ]
#
# A call to ei_split_testsuite(3) creates the following targets with dependencies
#
#   Target                      Dependencies
#   ------                      ------------
#   buildtestspart0             test1, test2
#   buildtestspart1             test3, test4
#   buildtestspart2             test5, test6, test7
#
macro(ei_split_testsuite num_splits)
  get_property(EIGEN_TESTS_LIST GLOBAL PROPERTY EIGEN_TESTS_LIST)

  # Translate EIGEN_TESTS_LIST into a CMake list
  string(REGEX REPLACE "\n" " " EIGEN_TESTS_LIST "${EIGEN_TESTS_LIST}")
  set(EIGEN_TESTS_LIST "${EIGEN_TESTS_LIST}")
  separate_arguments(EIGEN_TESTS_LIST)

  set(eigen_test_count "0")
  foreach(t IN ITEMS ${EIGEN_TESTS_LIST})
    math(EXPR eigen_test_count "${eigen_test_count}+1")
  endforeach()

  # Get number of tests per target
  math(EXPR num_tests_per_target "${eigen_test_count}/${num_splits} - ${eigen_test_count}/${num_splits} % 1")

  set(test_idx "0")
  math(EXPR target_bound "${num_splits}-1")
  foreach(part RANGE "0" "${target_bound}")
    # Create target
    set(current_target "buildtestspart${part}")
    add_custom_target("${current_target}")
    math(EXPR upper_bound "${test_idx} + ${num_tests_per_target} - 1")
    foreach(test_idx RANGE "${test_idx}" "${upper_bound}")
      list(GET EIGEN_TESTS_LIST "${test_idx}" curr_test)
      add_dependencies("${current_target}" "${curr_test}")
    endforeach()
    math(EXPR test_idx "${test_idx} + ${num_tests_per_target}")
  endforeach()

  # Handle the possibly remaining tests
  math(EXPR test_idx "${num_splits} * ${num_tests_per_target}")
  math(EXPR target_bound "${eigen_test_count} - 1")
  foreach(test_idx RANGE "${test_idx}" "${target_bound}")
    list(GET EIGEN_TESTS_LIST "${test_idx}" curr_test)
    add_dependencies("${current_target}" "${curr_test}")
  endforeach()
endmacro(ei_split_testsuite num_splits)
