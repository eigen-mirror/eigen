// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// A product served by the coefficient-based evaluator must be evaluated in the storage order its expression reports:
// Reverse, Redux, PartialRedux, Block, Replicate and FindCoeff index its packets and coefficients by that order, or
// size their loops by it. Each consumer is checked against the materialized product.

#include "main.h"

// Integer entries stay small so that the products cannot overflow.
template <typename MatrixType>
MatrixType random_operand() {
  using Scalar = typename MatrixType::Scalar;
  return MatrixType::NullaryExpr([] {
    return NumTraits<Scalar>::IsInteger ? internal::random<Scalar>(Scalar(-8), Scalar(8)) : internal::random<Scalar>();
  });
}

template <typename Prod>
void check_min_max(const Prod&, const Matrix<typename Prod::Scalar, Dynamic, Dynamic>&, std::true_type) {}

template <typename Prod>
void check_min_max(const Prod& prod, const Matrix<typename Prod::Scalar, Dynamic, Dynamic>& ref, std::false_type) {
  VERIFY_IS_APPROX(prod.minCoeff(), ref.minCoeff());
  Index i = -1, j = -1;
  const typename Prod::Scalar m = prod.maxCoeff(&i, &j);
  VERIFY(i >= 0 && i < ref.rows() && j >= 0 && j < ref.cols());
  VERIFY_IS_APPROX(m, ref(i, j));
  VERIFY_IS_APPROX(m, ref.maxCoeff());
}

template <typename Prod>
void check_consumers(const Prod& prod) {
  using Scalar = typename Prod::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Plain = Matrix<Scalar, Dynamic, Dynamic>;
  const Plain& lhs = prod.lhs();
  const Plain& rhs = prod.rhs();
  const Plain ref = lhs * rhs;

  VERIFY_IS_EQUAL(bool(Prod::IsRowMajor), bool(int(internal::evaluator<Prod>::Flags) & RowMajorBit));
  VERIFY_IS_APPROX(Plain(prod), ref);
  VERIFY_IS_APPROX(Plain(prod.reverse()), Plain(ref.reverse()));
  VERIFY_IS_APPROX(Plain(prod.colwise().reverse()), Plain(ref.colwise().reverse()));
  VERIFY_IS_APPROX(Plain(prod.rowwise().reverse()), Plain(ref.rowwise().reverse()));
  VERIFY_IS_APPROX(Plain(prod.colwise().sum()), Plain(ref.colwise().sum()));
  VERIFY_IS_APPROX(Plain(prod.rowwise().sum()), Plain(ref.rowwise().sum()));
  double magnitude = 0;
  for (Index j = 0; j < rhs.cols(); ++j)
    for (Index i = 0; i < lhs.rows(); ++i)
      for (Index k = 0; k < lhs.cols(); ++k)
        magnitude += static_cast<double>(numext::abs(lhs(i, k))) * static_cast<double>(numext::abs(rhs(k, j)));
  // Two k-term products and mn-term reductions; the factor 4 also covers complex arithmetic.
  const double sum_bound =
      4 * (double(lhs.cols()) + double(ref.size())) * double(NumTraits<RealScalar>::epsilon()) * magnitude;
  VERIFY((numext::isfinite)(sum_bound));
  VERIFY(static_cast<double>(numext::abs(prod.sum() - ref.sum())) <= sum_bound);
  VERIFY_IS_APPROX(prod.squaredNorm(), ref.squaredNorm());
  VERIFY_IS_APPROX(prod.cwiseAbs().sum(), ref.cwiseAbs().sum());
  check_min_max(prod, ref, internal::bool_constant<NumTraits<Scalar>::IsComplex>());
  const Matrix<Scalar, Dynamic, Dynamic, ColMajor> col_major = prod.block(0, 0, ref.rows(), ref.cols());
  const Matrix<Scalar, Dynamic, Dynamic, RowMajor> row_major = prod.block(0, 0, ref.rows(), ref.cols());
  VERIFY_IS_APPROX(Plain(col_major), ref);
  VERIFY_IS_APPROX(Plain(row_major), ref);
  VERIFY_IS_APPROX(Plain(prod.replicate(2, 3)), Plain(ref.replicate(2, 3)));
}

template <typename Scalar, int Rows, int Cols, int Order>
using MatrixOf = typename internal::make_proper_matrix_type<Scalar, Rows, Cols, Order>::type;

template <typename Scalar, int M, int K, int N>
void lazy_products() {
  const MatrixOf<Scalar, M, K, RowMajor> lr = random_operand<MatrixOf<Scalar, M, K, RowMajor>>();
  const MatrixOf<Scalar, M, K, ColMajor> lc = random_operand<MatrixOf<Scalar, M, K, ColMajor>>();
  const MatrixOf<Scalar, K, N, RowMajor> rr = random_operand<MatrixOf<Scalar, K, N, RowMajor>>();
  const MatrixOf<Scalar, K, N, ColMajor> rc = random_operand<MatrixOf<Scalar, K, N, ColMajor>>();
  check_consumers(lr.lazyProduct(rr));
  check_consumers(lc.lazyProduct(rr));
  check_consumers(lr.lazyProduct(rc));
  check_consumers(lc.lazyProduct(rc));
  // A nested product has no preferred storage order.
  const Matrix<Scalar, K, K> identity = Matrix<Scalar, K, K>::Identity();
  check_consumers((lc * identity).lazyProduct(rr));
  check_consumers(lr.lazyProduct(identity * rc));
  // Expression operands.
  const MatrixOf<Scalar, K, M, ColMajor> lct = random_operand<MatrixOf<Scalar, K, M, ColMajor>>();
  const Matrix<Scalar, M + 2, K + 2, RowMajor> big = random_operand<Matrix<Scalar, M + 2, K + 2, RowMajor>>();
  check_consumers(lct.transpose().lazyProduct(rr));
  check_consumers((Scalar(2) * lr).lazyProduct(rr));
  check_consumers(lr.conjugate().lazyProduct(rr));
  check_consumers(big.template block<M, K>(1, 1).lazyProduct(rr));

  const Matrix<Scalar, Dynamic, Dynamic, RowMajor> dlr = lr, drr = rr;
  const Matrix<Scalar, Dynamic, Dynamic, ColMajor> dlc = lc, drc = rc;
  check_consumers(dlr.lazyProduct(drr));
  check_consumers(dlc.lazyProduct(drr));
  check_consumers(dlr.lazyProduct(drc));
}

// Default products with a unit inner dimension are served by the coefficient-based evaluator without being
// materialized.
template <typename Scalar>
void outer_products() {
  const Matrix<Scalar, 8, 8, RowMajor> a = random_operand<Matrix<Scalar, 8, 8, RowMajor>>();
  const Matrix<Scalar, 8, 8, ColMajor> b = random_operand<Matrix<Scalar, 8, 8, ColMajor>>();
  const Matrix<Scalar, 1, 7> v = random_operand<Matrix<Scalar, 1, 7>>();
  check_consumers(a.col(2) * v);
  check_consumers(b.col(2) * v);
  check_consumers(a.col(2) * a.row(3).template head<7>());
  // Depth 1 from the rhs alone: the lhs has a runtime number of columns.
  const Matrix<Scalar, Dynamic, Dynamic, RowMajor> d = a.col(5);
  check_consumers(d * v);
}

EIGEN_DECLARE_TEST(product_storage_order) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((lazy_products<float, 3, 3, 5>()));
    CALL_SUBTEST_1((lazy_products<float, 8, 4, 5>()));
    CALL_SUBTEST_1((lazy_products<float, 16, 2, 17>()));
    CALL_SUBTEST_1(outer_products<float>());
    CALL_SUBTEST_2((lazy_products<double, 6, 2, 7>()));
    CALL_SUBTEST_2((lazy_products<double, 4, 1, 9>()));
    CALL_SUBTEST_2((lazy_products<double, 1, 3, 8>()));
    CALL_SUBTEST_2(outer_products<double>());
    CALL_SUBTEST_3((lazy_products<std::complex<float>, 5, 3, 6>()));
    CALL_SUBTEST_3(outer_products<std::complex<float>>());
    CALL_SUBTEST_4((lazy_products<std::complex<double>, 4, 2, 5>()));
    CALL_SUBTEST_5((lazy_products<int, 8, 3, 8>()));
    CALL_SUBTEST_5(outer_products<int>());
  }
}
