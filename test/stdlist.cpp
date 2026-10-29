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
#include <Eigen/StdList>
#include <Eigen/Geometry>
#include "stl_container_test_helpers.h"

template <typename MatrixType>
void check_stdlist_matrix(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();
  MatrixType x = MatrixType::Random(rows, cols), y = MatrixType::Random(rows, cols);
  std::list<MatrixType, Eigen::aligned_allocator<MatrixType> > v(10, MatrixType::Zero(rows, cols)), w(20, y);
  check_stl_container_front(v, w, x, y, MatrixType::Zero(rows, cols));
}

template <typename TransformType>
void check_stdlist_transform(const TransformType&) {
  using MatrixType = typename TransformType::MatrixType;
  TransformType x(MatrixType::Random()), y(MatrixType::Random()), ti = TransformType::Identity();
  std::list<TransformType, Eigen::aligned_allocator<TransformType> > v(10, ti), w(20, y);
  check_stl_container_front(v, w, x, y, ti);
}

template <typename QuaternionType>
void check_stdlist_quaternion(const QuaternionType&) {
  using Coefficients = typename QuaternionType::Coefficients;
  QuaternionType x(Coefficients::Random()), y(Coefficients::Random()), qi = QuaternionType::Identity();
  std::list<QuaternionType, Eigen::aligned_allocator<QuaternionType> > v(10, qi), w(20, y);
  check_stl_container_front(v, w, x, y, qi);
}

EIGEN_DECLARE_TEST(stdlist) {
  // some non vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdlist_matrix(Vector2f()));
  CALL_SUBTEST_1(check_stdlist_matrix(Matrix3f()));
  CALL_SUBTEST_2(check_stdlist_matrix(Matrix3d()));

  // some vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdlist_matrix(Matrix2f()));
  CALL_SUBTEST_1(check_stdlist_matrix(Vector4f()));
  CALL_SUBTEST_1(check_stdlist_matrix(Matrix4f()));
  CALL_SUBTEST_2(check_stdlist_matrix(Matrix4d()));

  // some dynamic sizes
  CALL_SUBTEST_3(check_stdlist_matrix(MatrixXd(1, 1)));
  CALL_SUBTEST_3(check_stdlist_matrix(VectorXd(20)));
  CALL_SUBTEST_3(check_stdlist_matrix(RowVectorXf(20)));
  CALL_SUBTEST_3(check_stdlist_matrix(MatrixXcf(10, 10)));

  // some Transform
  CALL_SUBTEST_4(check_stdlist_transform(Affine2f()));
  CALL_SUBTEST_4(check_stdlist_transform(Affine3f()));
  CALL_SUBTEST_4(check_stdlist_transform(Affine3d()));

  // some Quaternion
  CALL_SUBTEST_5(check_stdlist_quaternion(Quaternionf()));
  CALL_SUBTEST_5(check_stdlist_quaternion(Quaterniond()));
}
