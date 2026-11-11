// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include "stl_container_test_helpers.h"

template <typename T>
using TestContainer = std::vector<T, Eigen::aligned_allocator<T>>;

// the code below triggered an invalid warning with gcc >= 7
// eigen/Eigen/src/Core/util/Memory.h:189:12: warning: argument 1 value '18446744073709551612' exceeds maximum object
// size 9223372036854775807 This has been reported to gcc there: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=87544
void std_vector_gcc_warning() {
  using T = Eigen::Vector3f;
  std::vector<T, Eigen::aligned_allocator<T> > v;
  v.push_back(T(1.0f, 2.0f, 3.0f));
}

EIGEN_DECLARE_TEST(stdvector) {
  // some non vectorizable fixed sizes
  CALL_SUBTEST_1(check_stl_container_matrix<TestContainer>(Vector2f()));
  CALL_SUBTEST_1(check_stl_container_matrix<TestContainer>(Matrix3f()));
  CALL_SUBTEST_2(check_stl_container_matrix<TestContainer>(Matrix3d()));

  // some vectorizable fixed sizes
  CALL_SUBTEST_1(check_stl_container_matrix<TestContainer>(Matrix2f()));
  CALL_SUBTEST_1(check_stl_container_matrix<TestContainer>(Vector4f()));
  CALL_SUBTEST_1(check_stl_container_matrix<TestContainer>(Matrix4f()));
  CALL_SUBTEST_2(check_stl_container_matrix<TestContainer>(Matrix4d()));

  // some dynamic sizes
  CALL_SUBTEST_3(check_stl_container_matrix<TestContainer>(MatrixXd(1, 1)));
  CALL_SUBTEST_3(check_stl_container_matrix<TestContainer>(VectorXd(20)));
  CALL_SUBTEST_3(check_stl_container_matrix<TestContainer>(RowVectorXf(20)));
  CALL_SUBTEST_3(check_stl_container_matrix<TestContainer>(MatrixXcf(10, 10)));

  // some Transform
  CALL_SUBTEST_4(check_stl_container_transform<TestContainer>(Projective2f()));
  CALL_SUBTEST_4(check_stl_container_transform<TestContainer>(Projective3f()));
  CALL_SUBTEST_4(check_stl_container_transform<TestContainer>(Projective3d()));
  // CALL_SUBTEST(check_stl_container_transform<TestContainer>(Projective4d()));

  // some Quaternion
  CALL_SUBTEST_5(check_stl_container_quaternion<TestContainer>(Quaternionf()));
  CALL_SUBTEST_5(check_stl_container_quaternion<TestContainer>(Quaterniond()));
}
