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

template <typename MatrixType>
void check_stdvector_matrix(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();
  MatrixType x = MatrixType::Random(rows, cols), y = MatrixType::Random(rows, cols);
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType> > v(10, MatrixType::Zero(rows, cols)), w(20, y);
  check_stl_container_indexed(v, w, x, y);
}

template <typename TransformType>
void check_stdvector_transform(const TransformType&) {
  using MatrixType = typename TransformType::MatrixType;
  TransformType x(MatrixType::Random()), y(MatrixType::Random());
  std::vector<TransformType, Eigen::aligned_allocator<TransformType> > v(10), w(20, y);
  check_stl_container_indexed(v, w, x, y);
}

template <typename QuaternionType>
void check_stdvector_quaternion(const QuaternionType&) {
  using Coefficients = typename QuaternionType::Coefficients;
  QuaternionType x(Coefficients::Random()), y(Coefficients::Random()), qi = QuaternionType::Identity();
  std::vector<QuaternionType, Eigen::aligned_allocator<QuaternionType> > v(10, qi), w(20, y);
  check_stl_container_indexed(v, w, x, y);
}

// the code below triggered an invalid warning with gcc >= 7
// eigen/Eigen/src/Core/util/Memory.h:189:12: warning: argument 1 value '18446744073709551612' exceeds maximum object
// size 9223372036854775807 This has been reported to gcc there: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=87544
void std_vector_gcc_warning() {
  typedef Eigen::Vector3f T;
  std::vector<T, Eigen::aligned_allocator<T> > v;
  v.push_back(T(1.0f, 2.0f, 3.0f));
}

EIGEN_DECLARE_TEST(stdvector) {
  // some non vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdvector_matrix(Vector2f()));
  CALL_SUBTEST_1(check_stdvector_matrix(Matrix3f()));
  CALL_SUBTEST_2(check_stdvector_matrix(Matrix3d()));

  // some vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdvector_matrix(Matrix2f()));
  CALL_SUBTEST_1(check_stdvector_matrix(Vector4f()));
  CALL_SUBTEST_1(check_stdvector_matrix(Matrix4f()));
  CALL_SUBTEST_2(check_stdvector_matrix(Matrix4d()));

  // some dynamic sizes
  CALL_SUBTEST_3(check_stdvector_matrix(MatrixXd(1, 1)));
  CALL_SUBTEST_3(check_stdvector_matrix(VectorXd(20)));
  CALL_SUBTEST_3(check_stdvector_matrix(RowVectorXf(20)));
  CALL_SUBTEST_3(check_stdvector_matrix(MatrixXcf(10, 10)));

  // some Transform
  CALL_SUBTEST_4(check_stdvector_transform(Projective2f()));
  CALL_SUBTEST_4(check_stdvector_transform(Projective3f()));
  CALL_SUBTEST_4(check_stdvector_transform(Projective3d()));
  // CALL_SUBTEST(heck_stdvector_transform(Projective4d()));

  // some Quaternion
  CALL_SUBTEST_5(check_stdvector_quaternion(Quaternionf()));
  CALL_SUBTEST_5(check_stdvector_quaternion(Quaterniond()));
}
