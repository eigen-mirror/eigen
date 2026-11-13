// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING
#include "main.h"
#include <Eigen/SparseCore>

template <typename Matrix, typename Expression>
void check_indexed_update(const Matrix& input, const Matrix& expected, const Expression& expression, int allocations) {
  Matrix dst = Matrix::Constant(expected.rows(), expected.cols(), typename Matrix::Scalar(3));
  const Matrix initial = dst;
  VERIFY_EVALUATION_COUNT((dst.noalias() += expression), allocations);
  VERIFY_IS_APPROX(dst, (initial + expected).eval());
  VERIFY_EVALUATION_COUNT((dst.noalias() -= expression), allocations);
  VERIFY_IS_APPROX(dst, initial);
  Matrix padded = Matrix::Zero(expected.rows() + 2, expected.cols() + 2);
  auto block = padded.block(1, 1, expected.rows(), expected.cols());
  VERIFY_EVALUATION_COUNT((block.noalias() += expression), allocations);
  VERIFY_IS_APPROX(block, expected);
  VERIFY_EVALUATION_COUNT((block.noalias() -= expression), allocations);
  VERIFY(padded.isZero(0));
  dst.noalias() = expression;
  VERIFY_IS_APPROX(dst, expected);
  EIGEN_UNUSED_VARIABLE(input);
}

template <typename Scalar, int Order>
void permutation_updates() {
  using Matrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, Order>;
  for (Index n : {Index(0), Index(1), Index(7), Index(24)}) {
    PermutationMatrix<Dynamic> p(n);
    p.setIdentity();
    for (Index i = 0; i < n; ++i) p.applyTranspositionOnTheRight(i, internal::random<Index>(i, n - 1));
    const Matrix a = Matrix::Random(n, n + 3), b = Matrix::Random(n + 3, n);
    Matrix left(n, n + 3), inverseLeft(n, n + 3), right(n + 3, n), inverseRight(n + 3, n);
    for (Index i = 0; i < n; ++i)
      for (Index j = 0; j < n + 3; ++j) {
        left(p.indices()(i), j) = a(i, j);
        inverseLeft(i, j) = a(p.indices()(i), j);
        right(j, i) = b(j, p.indices()(i));
        inverseRight(j, p.indices()(i)) = b(j, i);
      }
    const auto check = [&](const Matrix& input, const Matrix& expected, const auto& makeProduct) {
      check_indexed_update(input, expected, makeProduct(input), 0);
      Matrix alias = input;
      alias += makeProduct(alias);
      VERIFY_IS_APPROX(alias, (input + expected).eval());
      alias = input;
      alias -= makeProduct(alias);
      VERIFY_IS_APPROX(alias, (input - expected).eval());
    };
    check(a, left, [&](const auto& x) { return p * x; });
    check(a, inverseLeft, [&](const auto& x) { return p.inverse() * x; });
    check(b, right, [&](const auto& x) { return x * p; });
    check(b, inverseRight, [&](const auto& x) { return x * p.inverse(); });
    check_indexed_update(a, left, p * a.block(0, 0, n, n + 3), 0);
    check_indexed_update(a, Matrix(inverseRight.transpose()), p * b.transpose(), 0);
    check_indexed_update(b, Matrix(inverseLeft.transpose()), a.transpose() * p, 0);
    check_indexed_update(a, Matrix(left + left), p * (a + a), 0);
    const Map<const Matrix, Unaligned, Stride<Dynamic, Dynamic>> mapped(
        a.data(), a.rows(), a.cols(), Stride<Dynamic, Dynamic>(a.outerStride(), a.innerStride()));
    check_indexed_update(a, left, p * mapped, 0);
    SparseMatrix<Scalar, Order> sparse = a.sparseView();
    Matrix dst = Matrix::Zero(n, n + 3);
    dst += p * sparse;
    VERIFY_IS_APPROX(dst, left);
  }
}

EIGEN_DECLARE_TEST(indexed_product_updates) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((permutation_updates<double, ColMajor>()));
    CALL_SUBTEST_1((permutation_updates<double, RowMajor>()));
    CALL_SUBTEST_2((permutation_updates<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_2((permutation_updates<std::complex<double>, RowMajor>()));
  }
}
