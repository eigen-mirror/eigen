// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

template <typename LhsScalar, typename RhsScalar, int Order>
void outer_product_scalar_types() {
  using Scalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  using Lhs = Matrix<LhsScalar, Dynamic, 1>;
  using Rhs = Matrix<RhsScalar, 1, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  STATIC_CHECK((internal::product_type<Lhs, Rhs>::value == OuterProduct));
  for (Index rows : {2, 3, 16, 17}) {
    for (Index cols : {2, 3, 16, 17}) {
      const Lhs lhs = Lhs::Constant(rows, LhsScalar(2));
      const Rhs rhs = Rhs::Constant(cols, RhsScalar(3));
      Mat storage = Mat::Constant(2 * rows, 2 * cols, Scalar(7));
      Mat expected = storage;
      Map<Mat, 0, Stride<Dynamic, 2>> dst(storage.data(), rows, cols, Stride<Dynamic, 2>(2 * storage.outerStride(), 2));
      for (int operation = 0; operation < 3; ++operation) {
        storage.setConstant(Scalar(7));
        expected = storage;
        if (operation == 0) dst.noalias() = lhs * rhs;
        if (operation == 1) dst.noalias() += lhs * rhs;
        if (operation == 2) dst.noalias() -= lhs * rhs;
        for (Index j = 0; j < cols; ++j) {
          for (Index i = 0; i < rows; ++i) {
            const Scalar value = lhs(i) * rhs(j);
            expected(2 * i, 2 * j) = operation == 0 ? value : operation == 1 ? Scalar(7) + value : Scalar(7) - value;
          }
        }
        VERIFY_IS_EQUAL(storage, expected);
      }
    }
  }
}

template <typename Real, int Order>
void outer_product_mixed_special_values() {
  using Scalar = std::complex<Real>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Real infinity = NumTraits<Real>::infinity();
  const Real nan = NumTraits<Real>::quiet_NaN();
  const Scalar values[] = {Scalar(infinity, 2),       Scalar(2, infinity),      Scalar(-infinity, -2),
                           Scalar(-2, -infinity),     Scalar(nan, 2),           Scalar(2, nan),
                           Scalar(Real(0), -Real(0)), Scalar(-Real(0), Real(0))};
  Matrix<Scalar, Dynamic, 1> lhs;
  Matrix<Real, 1, Dynamic> rhs;
  Mat actual, reversed;
  for (Index n : {2, 3, 16, 17}) {
    for (const Scalar& value : values) {
      for (Real factor : {Real(2), Real(-2)}) {
        lhs.setConstant(n, value);
        rhs.setConstant(n, factor);
        for (int operation = 0; operation < 3; ++operation) {
          actual.setConstant(n, n, Scalar(1, 1));
          reversed = actual;
          if (operation == 0) {
            actual.noalias() = lhs * rhs;
            reversed.noalias() = rhs.transpose() * lhs.transpose();
          }
          if (operation == 1) {
            actual.noalias() += lhs * rhs;
            reversed.noalias() += rhs.transpose() * lhs.transpose();
          }
          if (operation == 2) {
            actual.noalias() -= lhs * rhs;
            reversed.noalias() -= rhs.transpose() * lhs.transpose();
          }
          const Scalar product(value.real() * factor, value.imag() * factor);
          const Scalar expected = operation == 0   ? product
                                  : operation == 1 ? Scalar(1, 1) + product
                                                   : Scalar(1, 1) - product;
          for (Index j = 0; j < n; ++j) {
            for (Index i = 0; i < n; ++i) {
              for (int component = 0; component < 2; ++component) {
                const Real reference = component == 0 ? expected.real() : expected.imag();
                for (Real result : {component == 0 ? actual(i, j).real() : actual(i, j).imag(),
                                    component == 0 ? reversed(i, j).real() : reversed(i, j).imag()}) {
                  if ((numext::isnan)(reference)) {
                    VERIFY((numext::isnan)(result));
                  } else {
                    VERIFY_IS_EQUAL(result, reference);
                    if (reference == Real(0)) VERIFY_IS_EQUAL((std::signbit)(result), (std::signbit)(reference));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

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
  check_scaled_product((alpha * matrix.template selfadjointView<Mode>()) * diagonal.asDiagonal(), expected);
  check_scaled_product(diagonal.asDiagonal() * (alpha * matrix.template selfadjointView<Mode>()), expectedLeft);
  check_scaled_product((alpha * matrix.template selfadjointView<Mode>()) * ownedDiagonal, expected);
  check_scaled_product(ownedDiagonal * (alpha * matrix.template selfadjointView<Mode>()), expectedLeft);
  const Mat conjugated = matrix.conjugate();
  check_scaled_product((alpha * conjugated.conjugate().template selfadjointView<Mode>()) * diagonal.asDiagonal(),
                       expected);
  check_scaled_product(diagonal.asDiagonal() * (alpha * conjugated.conjugate().template selfadjointView<Mode>()),
                       expectedLeft);
  const Matrix<Scalar, 3, 1> actualDiagonal =
      ((alpha * matrix.template selfadjointView<Mode>()) * diagonal.asDiagonal()).diagonal();
  VERIFY_IS_EQUAL(actualDiagonal, expected.diagonal());
  aliased = matrix;
  aliased = (alpha * aliased.template selfadjointView<Mode>()) * diagonal.asDiagonal() + Mat::Zero();
  VERIFY_IS_EQUAL(aliased, expected);
  aliased = matrix;
  aliased = diagonal.asDiagonal() * (alpha * aliased.template selfadjointView<Mode>()) + Mat::Zero();
  VERIFY_IS_EQUAL(aliased, expectedLeft);
}

EIGEN_DECLARE_TEST(product_evaluators) {
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    CALL_SUBTEST_4((outer_product_scalar_types<double, double, ColMajor>()));
    CALL_SUBTEST_4((outer_product_scalar_types<double, double, RowMajor>()));
    CALL_SUBTEST_4((outer_product_scalar_types<std::complex<double>, double, ColMajor>()));
    CALL_SUBTEST_4((outer_product_scalar_types<std::complex<double>, double, RowMajor>()));
    CALL_SUBTEST_4((outer_product_scalar_types<double, std::complex<double>, ColMajor>()));
    CALL_SUBTEST_4((outer_product_scalar_types<double, std::complex<double>, RowMajor>()));
    CALL_SUBTEST_4((outer_product_mixed_special_values<float, ColMajor>()));
    CALL_SUBTEST_4((outer_product_mixed_special_values<float, RowMajor>()));
    CALL_SUBTEST_4((outer_product_mixed_special_values<double, ColMajor>()));
    CALL_SUBTEST_4((outer_product_mixed_special_values<double, RowMajor>()));

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
