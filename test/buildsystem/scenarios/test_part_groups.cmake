# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# ei_add_test compiles the part ranges listed for a test as one executable, and
# refuses a grouping that would change what a part compiles or drop a smoke test.

bs_configure("grouped parts" "${BS_CONSUMER_DIR}/test_part_groups" "${WORK_DIR}/grouped"
             "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}")

bs_configure_expect_failure("grouped smoke part" "${BS_CONSUMER_DIR}/test_part_groups"
                            "${WORK_DIR}/smoke" output
                            "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}" "-DPART_GROUPS_CASE=smoke")
if(NOT output MATCHES "grouped_2 is a smoke test and cannot be grouped")
  bs_fail("configure failed for some reason other than the smoke-test check\n----\n${output}\n----")
endif()

bs_configure_expect_failure("grouped explicit parts" "${BS_CONSUMER_DIR}/test_part_groups"
                            "${WORK_DIR}/explicit" output
                            "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}" "-DPART_GROUPS_CASE=explicit")
if(NOT output MATCHES "parts with EIGEN_TEST_PART_<N> markers cannot be grouped")
  bs_fail("configure failed for some reason other than the marker check\n----\n${output}\n----")
endif()
