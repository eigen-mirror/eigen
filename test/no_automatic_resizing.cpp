// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Test resizing restrictions and assignment to empty matrices and arrays.

#define EIGEN_NO_AUTOMATIC_RESIZING
#include "main.h"
#include <Eigen/Core>

template <typename Scalar>
void testNoAutomaticResizing() {
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
  using Array = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ArrayVector = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

  const Index rows = internal::random<Index>(1, 50);
  const Index cols = internal::random<Index>(1, 50);

  // Assignment of Zero expression to default-constructed matrix.
  {
    Matrix M;
    M = Matrix::Zero(rows, cols);
    VERIFY_IS_EQUAL(M.rows(), rows);
    VERIFY_IS_EQUAL(M.cols(), cols);
    VERIFY_IS_EQUAL(M.norm(), Scalar(0));
  }

  // Assignment of Zero expression to default-constructed array.
  {
    Array A;
    A = Array::Zero(rows, cols);
    VERIFY_IS_EQUAL(A.rows(), rows);
    VERIFY_IS_EQUAL(A.cols(), cols);
  }

  // Assignment of Ones expression to default-constructed matrix.
  {
    Matrix M;
    M = Matrix::Ones(rows, cols);
    VERIFY_IS_EQUAL(M.rows(), rows);
    VERIFY_IS_EQUAL(M.cols(), cols);
  }

  // Assignment of Random expression to default-constructed matrix.
  {
    Matrix M;
    M = Matrix::Random(rows, cols);
    VERIFY_IS_EQUAL(M.rows(), rows);
    VERIFY_IS_EQUAL(M.cols(), cols);
  }

  // Assignment from another matrix to default-constructed matrix.
  {
    Matrix src = Matrix::Random(rows, cols);
    Matrix dst;
    dst = src;
    VERIFY_IS_EQUAL(dst.rows(), rows);
    VERIFY_IS_EQUAL(dst.cols(), cols);
    VERIFY_IS_APPROX(dst, src);
  }

  // Vector assignment to default-constructed vector.
  {
    Vector v;
    v = Vector::Zero(rows);
    VERIFY_IS_EQUAL(v.size(), rows);
  }

  // RowVector assignment to default-constructed row vector.
  {
    RowVector v;
    v = RowVector::Zero(cols);
    VERIFY_IS_EQUAL(v.size(), cols);
  }

  // Array vector assignment to default-constructed array vector.
  {
    ArrayVector v;
    v = ArrayVector::Zero(rows);
    VERIFY_IS_EQUAL(v.size(), rows);
  }

  // Column access after Zero initialization (reproducer for reported bug).
  {
    Array A;
    A = Array::Zero(rows, cols);
    for (Index j = 0; j < cols; ++j) {
      auto c = A.col(j);
      VERIFY_IS_EQUAL(c.rows(), rows);
    }
  }
}

void testProductNoAutomaticResizing() {
  const MatrixXd lhs = MatrixXd::Ones(3, 2);
  const MatrixXd rhs = MatrixXd::Constant(2, 4, 2.0);
  const MatrixXd expected = MatrixXd::Constant(3, 4, 4.0);

  MatrixXd dst;
  dst.noalias() = lhs * rhs;
  VERIFY_IS_APPROX(dst, expected);
  dst.noalias() = 2.0 * (lhs * rhs);
  VERIFY_IS_APPROX(dst, 2.0 * expected);
  dst.noalias() = lhs * rhs;
  VERIFY_IS_APPROX(dst, expected);

  dst.resize(2, 6);  // Same coefficient count does not imply compatible dimensions.
  dst.setOnes();
  VERIFY_RAISES_ASSERT(dst.noalias() = lhs * rhs);
  VERIFY_RAISES_ASSERT(dst.noalias() = 2.0 * (lhs * rhs));
  VERIFY_IS_EQUAL(dst.rows(), 2);
  VERIFY_IS_EQUAL(dst.cols(), 6);
  VERIFY_IS_EQUAL(dst.sum(), 12.0);

  RowVectorXd row;
  row.noalias() = lhs * Vector2d::Ones();
  VERIFY_IS_APPROX(row, RowVector3d::Constant(2.0));
  row.resize(2);
  VERIFY_RAISES_ASSERT(row.noalias() = lhs * Vector2d::Ones());
}

EIGEN_DECLARE_TEST(no_automatic_resizing) {
  CALL_SUBTEST_1(testNoAutomaticResizing<float>());
  CALL_SUBTEST_2(testNoAutomaticResizing<double>());
  CALL_SUBTEST_2(testProductNoAutomaticResizing());
  CALL_SUBTEST_3(testNoAutomaticResizing<std::complex<double>>());
}
