// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/StdDeque>
#include <Eigen/Geometry>
#include "stl_container_test_helpers.h"

template <typename T>
using TestContainer = std::deque<T, Eigen::aligned_allocator<T>>;

EIGEN_DECLARE_TEST(stddeque) {
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
  CALL_SUBTEST_4(check_stl_container_transform<TestContainer>(Affine2f()));
  CALL_SUBTEST_4(check_stl_container_transform<TestContainer>(Affine3f()));
  CALL_SUBTEST_4(check_stl_container_transform<TestContainer>(Affine3d()));

  // some Quaternion
  CALL_SUBTEST_5(check_stl_container_quaternion<TestContainer>(Quaternionf()));
  CALL_SUBTEST_5(check_stl_container_quaternion<TestContainer>(Quaterniond()));
}
