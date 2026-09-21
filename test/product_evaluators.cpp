// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

template <typename ProductType, typename Mat>
void check_scaled_product(const ProductType& product, const Mat& expected) {
  Mat actual = product;
  VERIFY_IS_EQUAL(actual, expected);
  actual = product + Mat::Zero(expected.rows(), expected.cols());
  VERIFY_IS_EQUAL(actual, expected);
  actual.setZero();
  actual.noalias() += product;
  VERIFY_IS_EQUAL(actual, expected);
  actual.noalias() -= product;
  VERIFY_IS_EQUAL(actual, Mat::Zero(expected.rows(), expected.cols()));
  Mat triangle = Mat::Constant(expected.rows(), expected.cols(), typename Mat::Scalar(7));
  Mat triangleExpected = triangle;
  triangle.template triangularView<Upper>() = product;
  triangleExpected.template triangularView<Upper>() = expected;
  VERIFY_IS_EQUAL(triangle, triangleExpected);
  triangle.template triangularView<Lower>() = product;
  triangleExpected.template triangularView<Lower>() = expected;
  VERIFY_IS_EQUAL(triangle, triangleExpected);
}

template <typename Scalar, int Mode, int Order>
void scaled_unit_triangular_product() {
  using Mat = Matrix<Scalar, 3, 3, Order>;
  Mat matrix;
  matrix << 9, 2, 3, 4, 9, 6, 7, 8, 9;
  const Mat rhs = Mat::Constant(2);
  const Matrix<Scalar, 3, 1> diagonal = Matrix<Scalar, 3, 1>::Constant(2);
  Mat expected = Mat::Zero(), expectedDiagonal = Mat::Zero();
  for (Index j = 0; j < 3; ++j) {
    for (Index i = 0; i < 3; ++i) {
      for (Index k = 0; k < 3; ++k) {
        const bool stored = Mode == UnitLower ? i > k : i < k;
        const Scalar value = i == k ? Scalar(1) : stored ? matrix(i, k) : Scalar(0);
        expected(i, j) += Scalar(3) * value * rhs(k, j);
      }
      const bool stored = Mode == UnitLower ? i > j : i < j;
      expectedDiagonal(i, j) = Scalar(3) * (i == j ? Scalar(1) : stored ? matrix(i, j) : Scalar(0)) * diagonal(j);
    }
  }
  check_scaled_product(Scalar(3) * (matrix.template triangularView<Mode>() * rhs), expected);
  check_scaled_product(Scalar(3) * (matrix.template triangularView<Mode>() * diagonal.asDiagonal()), expectedDiagonal);
  Mat aliased = matrix;
  aliased = Scalar(3) * (aliased.template triangularView<Mode>() * rhs);
  VERIFY_IS_EQUAL(aliased, expected);
}

template <typename Scalar, int Order>
void scaled_permutation_product() {
  using Mat = Matrix<Scalar, 3, 3, Order>;
  Mat matrix;
  matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  PermutationMatrix<3> permutation;
  permutation.indices() << 2, 0, 1;
  Mat expected;
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) expected(permutation.indices()(i), j) = Scalar(3) * matrix(i, j);
  check_scaled_product(Scalar(3) * (permutation * matrix), expected);
  Mat aliased = matrix;
  aliased = Scalar(3) * (permutation * aliased);
  VERIFY_IS_EQUAL(aliased, expected);
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) expected(i, j) = Scalar(3) * matrix(permutation.indices()(i), j);
  check_scaled_product(Scalar(3) * (permutation.inverse() * matrix), expected);
}

template <int Order>
void scaled_structured_product() {
  using Scalar = std::complex<double>;
  using Mat = Matrix<Scalar, 3, 3, Order>;
  Mat matrix;
  matrix << Scalar(1), Scalar(2, 1), Scalar(3, 2), Scalar(4, 3), Scalar(5), Scalar(6, 4), Scalar(7, 5), Scalar(8, 6),
      Scalar(9);
  const Scalar alpha(2, 3);
  const Mat rhs = Mat::Constant(Scalar(2, 1));
  const Matrix<Scalar, 3, 1> diagonal = Matrix<Scalar, 3, 1>::Constant(Scalar(3, 2));
  Mat expected;
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) expected(i, j) = alpha * (diagonal(i) * matrix(i, j));
  check_scaled_product(alpha * (diagonal.asDiagonal() * matrix), expected);
  const DiagonalMatrix<Scalar, 3> ownedDiagonal(diagonal);
  check_scaled_product(alpha * (ownedDiagonal * matrix), expected);

  expected.setZero();
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i)
      for (Index k = 0; k <= i; ++k) expected(i, j) += alpha * (matrix(i, k) * rhs(k, j));
  check_scaled_product(alpha * (matrix.template triangularView<Lower>() * rhs), expected);

  expected.setZero();
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i)
      for (Index k = 0; k < 3; ++k) {
        const Scalar value = i >= k ? matrix(i, k) : numext::conj(matrix(k, i));
        expected(i, j) += alpha * (value * rhs(k, j));
      }
  check_scaled_product(alpha * (matrix.template selfadjointView<Lower>() * rhs), expected);
}

template <int Mode, int Order>
void scaled_selfadjoint_diagonal_product() {
  using Scalar = std::complex<double>;
  using Mat = Matrix<Scalar, 3, 3, Order>;
  Mat matrix;
  matrix << Scalar(1), Scalar(2, 1), Scalar(3, 2), Scalar(4, 3), Scalar(5), Scalar(6, 4), Scalar(7, 5), Scalar(8, 6),
      Scalar(9);
  Matrix<Scalar, 3, 1> diagonal;
  diagonal << Scalar(1, -1), Scalar(2, 3), Scalar(-4, 2);
  const Scalar alpha(2, 3);
  Mat expected, expectedLeft;
  for (Index j = 0; j < 3; ++j) {
    for (Index i = 0; i < 3; ++i) {
      const bool stored = Mode == Lower ? i >= j : i <= j;
      const Scalar value = stored ? matrix(i, j) : numext::conj(matrix(j, i));
      expected(i, j) = alpha * (value * diagonal(j));
      expectedLeft(i, j) = alpha * (diagonal(i) * value);
    }
  }
  check_scaled_product(alpha * (matrix.template selfadjointView<Mode>() * diagonal.asDiagonal()), expected);
  const DiagonalMatrix<Scalar, 3> ownedDiagonal(diagonal);
  check_scaled_product(alpha * (matrix.template selfadjointView<Mode>() * ownedDiagonal), expected);
  check_scaled_product(alpha * (diagonal.asDiagonal() * matrix.template selfadjointView<Mode>()), expectedLeft);
  Mat aliased = matrix;
  aliased = alpha * (aliased.template selfadjointView<Mode>() * diagonal.asDiagonal()) + Mat::Zero();
  VERIFY_IS_EQUAL(aliased, expected);
}

EIGEN_DECLARE_TEST(product_evaluators) {
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    CALL_SUBTEST_1((scaled_unit_triangular_product<double, UnitLower, ColMajor>()));
    CALL_SUBTEST_1((scaled_unit_triangular_product<double, UnitLower, RowMajor>()));
    CALL_SUBTEST_1((scaled_unit_triangular_product<double, UnitUpper, ColMajor>()));
    CALL_SUBTEST_1((scaled_unit_triangular_product<double, UnitUpper, RowMajor>()));
    CALL_SUBTEST_1((scaled_unit_triangular_product<std::complex<double>, UnitLower, ColMajor>()));
    CALL_SUBTEST_1((scaled_unit_triangular_product<std::complex<double>, UnitUpper, RowMajor>()));
    CALL_SUBTEST_2((scaled_permutation_product<double, ColMajor>()));
    CALL_SUBTEST_2((scaled_permutation_product<double, RowMajor>()));
    CALL_SUBTEST_2((scaled_permutation_product<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_2((scaled_permutation_product<std::complex<double>, RowMajor>()));
    CALL_SUBTEST_3((scaled_structured_product<ColMajor>()));
    CALL_SUBTEST_3((scaled_structured_product<RowMajor>()));
    CALL_SUBTEST_3((scaled_selfadjoint_diagonal_product<Lower, ColMajor>()));
    CALL_SUBTEST_3((scaled_selfadjoint_diagonal_product<Lower, RowMajor>()));
    CALL_SUBTEST_3((scaled_selfadjoint_diagonal_product<Upper, ColMajor>()));
    CALL_SUBTEST_3((scaled_selfadjoint_diagonal_product<Upper, RowMajor>()));
  }
}
