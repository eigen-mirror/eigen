// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Must be defined before including any Eigen headers.
#define EIGEN_NO_AUTOMATIC_RESIZING

#include "main.h"

// Helper to create a random matrix respecting compile-time fixed dimensions.
template <typename MatrixType>
MatrixType random_matrix() {
  enum { RowsAtCompileTime = MatrixType::RowsAtCompileTime, ColsAtCompileTime = MatrixType::ColsAtCompileTime };
  Index rows = (RowsAtCompileTime == Dynamic) ? internal::random<Index>(1, 10) : Index(RowsAtCompileTime);
  Index cols = (ColsAtCompileTime == Dynamic) ? internal::random<Index>(1, 10) : Index(ColsAtCompileTime);
  return MatrixType::Random(rows, cols);
}

template <typename MatrixType>
void noresize_assign_to_empty() {
  MatrixType src = random_matrix<MatrixType>();

  // Assigning to a default-constructed (empty) destination should work.
  MatrixType dst;
  dst = src;
  VERIFY_IS_EQUAL(dst.rows(), src.rows());
  VERIFY_IS_EQUAL(dst.cols(), src.cols());
  VERIFY_IS_APPROX(dst, src);
}

template <typename MatrixType>
void noresize_assign_expression_to_empty() {
  MatrixType a = random_matrix<MatrixType>();
  MatrixType b(a.rows(), a.cols());
  b.setRandom();

  // Assigning an expression to an empty destination should work.
  MatrixType dst;
  dst = a + b;
  VERIFY_IS_EQUAL(dst.rows(), a.rows());
  VERIFY_IS_EQUAL(dst.cols(), a.cols());
  VERIFY_IS_APPROX(dst, a + b);
}

template <typename MatrixType>
void noresize_construct_from_expression() {
  MatrixType a = random_matrix<MatrixType>();

  // Construction from an expression should work.
  MatrixType dst = a * 2;
  VERIFY_IS_EQUAL(dst.rows(), a.rows());
  VERIFY_IS_EQUAL(dst.cols(), a.cols());
  VERIFY_IS_APPROX(dst, a * 2);
}

template <typename MatrixType>
void noresize_col_access() {
  MatrixType src = random_matrix<MatrixType>();

  // Assigning to empty, then accessing columns should work.
  MatrixType dst;
  dst = src;
  for (Index j = 0; j < src.cols(); ++j) {
    VERIFY_IS_APPROX(dst.col(j), src.col(j));
  }
}

template <typename MatrixType>
void noresize_size_mismatch() {
  enum { RowsAtCompileTime = MatrixType::RowsAtCompileTime, ColsAtCompileTime = MatrixType::ColsAtCompileTime };
  Index rows = (RowsAtCompileTime == Dynamic) ? internal::random<Index>(2, 10) : Index(RowsAtCompileTime);
  Index cols = (ColsAtCompileTime == Dynamic) ? internal::random<Index>(2, 10) : Index(ColsAtCompileTime);
  MatrixType src = MatrixType::Random(rows, cols);
  // Create a destination with at least one mismatched dynamic dimension.
  Index dst_rows = (RowsAtCompileTime == Dynamic) ? rows + 1 : rows;
  Index dst_cols = (ColsAtCompileTime == Dynamic) ? cols + 1 : cols;
  MatrixType dst = MatrixType::Random(dst_rows, dst_cols);

  // Assigning to a non-empty destination with different size should assert.
  VERIFY_RAISES_ASSERT(dst = src);
}

EIGEN_DECLARE_TEST(noresize) {
  CALL_SUBTEST_1(noresize_assign_to_empty<MatrixXf>());
  CALL_SUBTEST_1(noresize_assign_to_empty<MatrixXd>());
  CALL_SUBTEST_1(noresize_assign_to_empty<MatrixXcf>());
  CALL_SUBTEST_1(noresize_assign_to_empty<MatrixXcd>());
  CALL_SUBTEST_2(noresize_assign_to_empty<ArrayXXd>());
  CALL_SUBTEST_2(noresize_assign_to_empty<ArrayXXcd>());
  CALL_SUBTEST_3(noresize_assign_to_empty<VectorXf>());
  CALL_SUBTEST_3(noresize_assign_to_empty<RowVectorXd>());

  CALL_SUBTEST_4(noresize_assign_expression_to_empty<MatrixXd>());
  CALL_SUBTEST_4(noresize_assign_expression_to_empty<ArrayXXd>());

  CALL_SUBTEST_5(noresize_construct_from_expression<MatrixXd>());
  CALL_SUBTEST_5(noresize_construct_from_expression<ArrayXXd>());

  CALL_SUBTEST_6(noresize_col_access<MatrixXd>());
  CALL_SUBTEST_6(noresize_col_access<MatrixXf>());

  CALL_SUBTEST_7(noresize_size_mismatch<MatrixXd>());
  CALL_SUBTEST_7(noresize_size_mismatch<MatrixXf>());
  CALL_SUBTEST_7(noresize_size_mismatch<VectorXd>());
}
