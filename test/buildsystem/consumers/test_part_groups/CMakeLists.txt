# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Registers a five-part test through ei_add_test with parts 1-3 grouped and
# checks the resulting targets.  PART_GROUPS_CASE selects a variant that must
# fail to configure instead: "smoke" names a grouped part in the smoke-test
# list, "explicit" groups a source with an EIGEN_TEST_PART_<N> marker.

cmake_minimum_required(VERSION 3.18)
project(TestPartGroups CXX)
include("${EIGEN_SOURCE_DIR}/cmake/EigenTesting.cmake")

add_library(Eigen3::Eigen INTERFACE IMPORTED)
add_custom_target(buildtests)
add_custom_target(buildsmoketests)
set(EIGEN_TEST_MAX_SIZE 320)
set(EIGEN_SPLIT_LARGE_TESTS ON)
set(ei_smoke_test_list grouped_5)
set(ei_test_part_groups_grouped "1-3")

set(body "CALL_SUBTEST_1 CALL_SUBTEST_2 CALL_SUBTEST_3 CALL_SUBTEST_4 CALL_SUBTEST_5\n")
if(PART_GROUPS_CASE STREQUAL "smoke")
  list(APPEND ei_smoke_test_list grouped_2)
elseif(PART_GROUPS_CASE STREQUAL "explicit")
  string(APPEND body "#ifdef EIGEN_TEST_PART_2\n#endif\n")
endif()

# ei_add_test reads <testname>.cpp relative to the current source directory, so
# the test and its registration live in a generated subdirectory.
set(gen "${CMAKE_CURRENT_BINARY_DIR}/gen")
file(WRITE "${gen}/grouped.cpp" "${body}int main() { return 0; }\n")
file(WRITE "${gen}/CMakeLists.txt" "ei_add_test(grouped)\n")
add_subdirectory("${gen}" gen)

function(expect_part_defines target expected)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "missing target ${target}")
  endif()
  get_target_property(definitions ${target} COMPILE_DEFINITIONS)
  list(FILTER definitions INCLUDE REGEX "^EIGEN_TEST_PART_")
  if(NOT definitions STREQUAL expected)
    message(FATAL_ERROR "${target}: expected [${expected}], got [${definitions}]")
  endif()
endfunction()

expect_part_defines(grouped_1 "EIGEN_TEST_PART_1=1;EIGEN_TEST_PART_2=1;EIGEN_TEST_PART_3=1")
expect_part_defines(grouped_4 "EIGEN_TEST_PART_4=1")
expect_part_defines(grouped_5 "EIGEN_TEST_PART_5=1")
foreach(absorbed grouped_2 grouped_3)
  if(TARGET ${absorbed})
    message(FATAL_ERROR "${absorbed} should be compiled as part of grouped_1")
  endif()
endforeach()
get_target_property(parent_deps grouped MANUALLY_ADDED_DEPENDENCIES)
if(NOT parent_deps STREQUAL "grouped_1;grouped_4;grouped_5")
  message(FATAL_ERROR "grouped: expected dependencies [grouped_1;grouped_4;grouped_5], got [${parent_deps}]")
endif()
get_target_property(smoke_deps buildsmoketests MANUALLY_ADDED_DEPENDENCIES)
if(NOT smoke_deps STREQUAL "grouped_5")
  message(FATAL_ERROR "buildsmoketests: expected [grouped_5], got [${smoke_deps}]")
endif()
