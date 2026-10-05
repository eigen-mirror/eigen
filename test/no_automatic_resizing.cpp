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
#include <Eigen/SparseCore>

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

// Isolate this check from the additional shape assertions in assignment evaluators.
template <typename PlainObject>
struct ResizeToMatchProbe : PlainObject {
  using PlainObject::_resize_to_match;
};

template <typename Vector>
void testResizeToMatchVector() {
  ResizeToMatchProbe<Vector> dst;
  MatrixXd source = MatrixXd::Ones(2, 3);
  VERIFY_RAISES_ASSERT(dst._resize_to_match(source));
  dst.resize(6);
  dst.setConstant(-1);
  VERIFY_RAISES_ASSERT(dst._resize_to_match(source));
  VERIFY_IS_EQUAL(dst.sum(), -6);

  // A non-empty vector accepts a same-length source of either runtime orientation and keeps its shape.
  source.resize(6, 1);
  dst._resize_to_match(source);
  source.resize(1, 6);
  dst._resize_to_match(source);
  VERIFY_IS_EQUAL(dst.rows(), Vector::RowsAtCompileTime == 1 ? 1 : 6);
  VERIFY_IS_EQUAL(dst.cols(), Vector::RowsAtCompileTime == 1 ? 6 : 1);
  source.resize(1, 5);
  VERIFY_RAISES_ASSERT(dst._resize_to_match(source));

  ResizeToMatchProbe<Vector> empty;
  empty._resize_to_match(source);
  VERIFY_IS_EQUAL(empty.size(), 5);
  VERIFY_IS_EQUAL(empty.rows(), Vector::RowsAtCompileTime == 1 ? 1 : 5);
  VERIFY_IS_EQUAL(empty.cols(), Vector::RowsAtCompileTime == 1 ? 5 : 1);
}

template <int Order>
void testVectorOrientationNoAutomaticResizing() {
  using Matrix = Eigen::Matrix<double, Dynamic, Dynamic, Order>;
  using Array = Eigen::Array<double, Dynamic, Dynamic, Order>;
  const Matrix column = Matrix::Constant(3, 1, 4.0);
  const Matrix row = Matrix::Constant(1, 3, 5.0);
  RowVectorXd r = RowVectorXd::Constant(3, -1.0);
  VectorXd c = VectorXd::Constant(3, -1.0);
  VERIFY_RAISES_ASSERT(r = column);
  VERIFY_RAISES_ASSERT(c = row);
  VERIFY_IS_EQUAL(r.sum(), -3.0);
  VERIFY_IS_EQUAL(c.sum(), -3.0);

  Eigen::Array<double, 1, Dynamic> ar = Eigen::Array<double, 1, Dynamic>::Constant(3, -1.0);
  ArrayXd ac = ArrayXd::Constant(3, -1.0);
  const Array arrayColumn = Array::Constant(3, 1, 4.0);
  const Array arrayRow = Array::Constant(1, 3, 5.0);
  VERIFY_RAISES_ASSERT(ar = arrayColumn);
  VERIFY_RAISES_ASSERT(ac = arrayRow);

  const Matrix identity = Matrix::Identity(3, 3);
  VERIFY_RAISES_ASSERT(r.noalias() = identity * column);
  VERIFY_RAISES_ASSERT(c.noalias() = row * identity);
  VERIFY_RAISES_ASSERT(r.noalias() = 2.0 * (identity * column));
  VERIFY_RAISES_ASSERT(c.noalias() = 2.0 * (row * identity));

  // Compile-time vector orientation conversion remains supported.
  c << 1, 2, 3;
  r = c;
  VERIFY_IS_EQUAL(r, RowVector3d(1, 2, 3));
  c = RowVector3d(4, 5, 6);
  VERIFY_IS_EQUAL(c, Vector3d(4, 5, 6));
  ac << 1, 2, 3;
  ar = ac;
  VERIFY_IS_EQUAL(ar.matrix(), RowVector3d(1, 2, 3));
  ac = ar;
  VERIFY_IS_EQUAL(ac.matrix(), Vector3d(1, 2, 3));
  r = row;
  c = column;
  VERIFY_IS_EQUAL(r.sum(), 15.0);
  VERIFY_IS_EQUAL(c.sum(), 12.0);
}

template <int Order>
void testSparseNoAutomaticResizing() {
  SparseMatrix<double, Order> row(1, 3), column(3, 1);
  row.insert(0, 2) = 7.0;
  column.insert(2, 0) = 8.0;
  VectorXd c = VectorXd::Constant(3, -1.0);
  RowVectorXd r = RowVectorXd::Constant(3, -1.0);
  VERIFY_RAISES_ASSERT(c = row);
  VERIFY_RAISES_ASSERT(r = column);
  c = column;
  r = row;
  VERIFY_IS_EQUAL(c, Vector3d(0, 0, 8));
  VERIFY_IS_EQUAL(r, RowVector3d(0, 0, 7));
}

template <typename = void>
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
  CALL_SUBTEST_2(testProductNoAutomaticResizing<>());
  CALL_SUBTEST_2(testVectorOrientationNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_2(testVectorOrientationNoAutomaticResizing<RowMajor>());
  CALL_SUBTEST_2(testResizeToMatchVector<VectorXd>());
  CALL_SUBTEST_2(testResizeToMatchVector<RowVectorXd>());
  CALL_SUBTEST_2(testResizeToMatchVector<ArrayXd>());
  CALL_SUBTEST_2((testResizeToMatchVector<Eigen::Array<double, 1, Dynamic>>()));
  CALL_SUBTEST_3(testNoAutomaticResizing<std::complex<double>>());
  CALL_SUBTEST_4(testSparseNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_4(testSparseNoAutomaticResizing<RowMajor>());
}
