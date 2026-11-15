#!cmake -P
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# The tags below belong to the generated file, not to this one.
# REUSE-IgnoreStart
file(WRITE split_test_helper.h
  "// SPDX-FileCopyrightText: The Eigen Authors\n"
  "// SPDX-License-Identifier: MPL-2.0\n\n"
  "#ifndef EIGEN_TEST_SPLIT_TEST_HELPER_H\n"
  "#define EIGEN_TEST_SPLIT_TEST_HELPER_H\n\n"
)
# REUSE-IgnoreEnd
foreach(i RANGE 1 999)
  file(APPEND split_test_helper.h
    "#if defined(EIGEN_TEST_PART_${i}) || defined(EIGEN_TEST_PART_ALL)\n"
    "#define CALL_SUBTEST_${i}(FUNC) CALL_SUBTEST(FUNC)\n"
    "#else\n"
    "#define CALL_SUBTEST_${i}(FUNC)\n"
    "#endif\n\n"
  )
endforeach()
file(APPEND split_test_helper.h "#endif  // EIGEN_TEST_SPLIT_TEST_HELPER_H\n")
