// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING
#define TEST_IGNORE_STACK_ALLOCATED_TEMPORARY
#include "main.h"
#include <Eigen/Core>
#include "NonCommutativeScalar.h"

template <typename Lhs, typename Rhs>
void check_scaled_unit_product(const Lhs& lhs, const Rhs& rhs, const typename Lhs::Scalar& alpha) {
  using Scalar = typename Lhs::Scalar;
  using Real = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, (int(Lhs::Flags) & RowMajorBit) ? RowMajor : ColMajor>;
  const MatrixType left(lhs);
  MatrixType expected = MatrixType::Zero(lhs.rows(), rhs.cols());
  Real budget = 0;
  for (Index j = 0; j < rhs.cols(); ++j)
    for (Index i = 0; i < lhs.rows(); ++i) {
      Real sumAbs = 0;
      for (Index k = 0; k < lhs.cols(); ++k) {
        expected(i, j) += left(i, k) * rhs.coeff(k, j);
        sumAbs += numext::abs(left(i, k)) * numext::abs(rhs.coeff(k, j));
      }
      expected(i, j) = alpha * expected(i, j);
      budget = (std::max)(budget, numext::abs(alpha) * sumAbs);
    }
  // gamma_depth times the independent sum of absolute terms, with headroom for complex arithmetic and updates.
  const Real bound = Real(64 * (lhs.cols() + 1)) * NumTraits<Real>::epsilon() * (Real(1) + budget);
  const auto verify = [&expected, bound](const auto& actual, const auto& reference) {
    if (expected.size()) VERIFY_LE((actual - reference).cwiseAbs().template maxCoeff<PropagateNaN>(), bound);
  };
  using Destination = std::conditional_t<Rhs::ColsAtCompileTime == 1, Matrix<Scalar, Dynamic, 1>, MatrixType>;
  Destination dst(lhs.rows(), rhs.cols());
  const auto expression = alpha * (lhs * rhs);
#ifndef EIGEN_USE_BLAS
  VERIFY_EVALUATION_COUNT((dst.noalias() = expression), 0);
#else
  dst.noalias() = expression;  // BLAS TRMM needs its own operand workspaces.
#endif
  verify(dst, expected);
  dst.setOnes();
#ifndef EIGEN_USE_BLAS
  VERIFY_EVALUATION_COUNT((dst.noalias() += expression), 0);
#else
  dst.noalias() += expression;
#endif
  verify(dst, (MatrixType::Ones(dst.rows(), dst.cols()) + expected).eval());
#ifndef EIGEN_USE_BLAS
  VERIFY_EVALUATION_COUNT((dst.noalias() -= expression), 0);
#else
  dst.noalias() -= expression;
#endif
  verify(dst, MatrixType::Ones(dst.rows(), dst.cols()));

  MatrixType ordinary = expression;
  verify(ordinary, expected);
  MatrixType nested = MatrixType::Ones(dst.rows(), dst.cols()) + expression;
  verify(nested, (MatrixType::Ones(dst.rows(), dst.cols()) + expected).eval());
  MatrixType storage = MatrixType::Zero(dst.rows() + 2, dst.cols() + 2);
  storage.block(1, 1, dst.rows(), dst.cols()).noalias() = expression;
  verify(storage.block(1, 1, dst.rows(), dst.cols()), expected);
  if (dst.size()) {
    Matrix<Scalar, Dynamic, 1> buffer(2 * dst.size());
    const Index outer = 2 * (MatrixType::IsRowMajor ? dst.cols() : dst.rows());
    Map<MatrixType, 0, Stride<Dynamic, Dynamic>> strided(buffer.data(), dst.rows(), dst.cols(),
                                                         Stride<Dynamic, Dynamic>(outer, 2));
    strided.noalias() = expression;
    verify(strided, expected);
    VectorXi rowIndices(dst.rows()), colIndices(dst.cols());
    for (Index i = 0; i < dst.rows(); ++i) rowIndices(i) = int((i + 1) % dst.rows());
    for (Index j = 0; j < dst.cols(); ++j) colIndices(j) = int((j + 1) % dst.cols());
    auto indexed = storage(rowIndices, colIndices);
    indexed.noalias() = expression;
    verify(indexed, expected);
  }
}

template <typename Scalar, int Order>
void scaled_unit_products() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Scalar alpha = Scalar(3.14159265358979323846) + Scalar(internal::random<Scalar>());
  for (Index n : {Index(0), Index(1), Index(4), Index(24), Index(64)}) {
    MatrixType a = MatrixType::Random(n, n), b = MatrixType::Random(n, n + 2);
    a.diagonal().setConstant(Scalar(NumTraits<typename NumTraits<Scalar>::Real>::quiet_NaN()));
    check_scaled_unit_product(a.template triangularView<UnitLower>(), b, alpha);
    check_scaled_unit_product(a.template triangularView<UnitUpper>(), b, -alpha);
    if (n) {
      Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
      check_scaled_unit_product(a.template triangularView<UnitLower>(), v, alpha);
      check_scaled_unit_product(a.template triangularView<UnitUpper>(), v.asDiagonal(), alpha);
      MatrixType alias = b;
      const MatrixType expected = alpha * (a.template triangularView<UnitLower>() * b).eval();
      alias = alpha * (a.template triangularView<UnitLower>() * alias);
      VERIFY_IS_APPROX(alias, expected);
      alias = a;
      alias = alpha * (alias.template triangularView<UnitLower>() * b);
      VERIFY_IS_APPROX(alias, expected);
    }
  }
  Matrix<Scalar, 4, 4, Order> fixed = Matrix<Scalar, 4, 4, Order>::Random();
  check_scaled_unit_product(fixed.template triangularView<UnitUpper>(), fixed, alpha);
  MatrixType a = MatrixType::Random(26, 24), b = MatrixType::Random(24, 27);
  check_scaled_unit_product(a.template triangularView<UnitLower>(), b, alpha);
  check_scaled_unit_product(a.template triangularView<UnitUpper>(), b, alpha);
  auto adjoint = a.adjoint();
  MatrixType c = MatrixType::Random(26, 7);
  check_scaled_unit_product(adjoint.template triangularView<UnitUpper>(), c, alpha);
}

#if defined(EIGEN_TEST_PART_5) || defined(EIGEN_TEST_PART_ALL)
template <typename Lhs, typename Rhs>
void check_scaled_unit_zero(const Lhs& lhs, const Rhs& rhs, const typename Lhs::Scalar& alpha) {
  using Scalar = typename Lhs::Scalar;
  using Destination =
      std::conditional_t<Rhs::ColsAtCompileTime == 1, Matrix<Scalar, Dynamic, 1>, Matrix<Scalar, Dynamic, Dynamic>>;
  const auto expression = alpha * (lhs * rhs);
  Destination dst;
  dst.noalias() = expression;
  VERIFY_IS_EQUAL(dst.rows(), lhs.rows());
  VERIFY_IS_EQUAL(dst.cols(), rhs.cols());
  VERIFY((dst.array() == Scalar(0)).all());
  dst.setConstant(Scalar(7));
  dst.noalias() += expression;
  VERIFY((dst.array() == Scalar(7)).all());
  dst.noalias() -= expression;
  VERIFY((dst.array() == Scalar(7)).all());
}

template <typename Scalar, int LhsOrder, int RhsOrder>
void scaled_unit_zero_scales() {
  using Real = typename NumTraits<Scalar>::Real;
  using Lhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<Scalar, Dynamic, Dynamic, RhsOrder>;
  for (Real value : {NumTraits<Real>::infinity(), NumTraits<Real>::quiet_NaN()}) {
    for (Index n : {Index(0), Index(1), Index(4), Index(24)}) {
      const Lhs a = Lhs::Constant(n, n, Scalar(value));
      const Rhs b = Rhs::Constant(n, n + 2, Scalar(value));
      const Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Constant(n, Scalar(value));
      for (Scalar alpha : {Scalar(0), Scalar(-Real(0))}) {
        check_scaled_unit_zero(a.template triangularView<UnitLower>(), b, alpha);
        check_scaled_unit_zero(a.template triangularView<UnitUpper>(), b, alpha);
        check_scaled_unit_zero(a.template triangularView<UnitLower>(), v, alpha);
        check_scaled_unit_zero(a.template triangularView<UnitUpper>(), v, alpha);
        check_scaled_unit_zero(a.template triangularView<UnitLower>(), v.asDiagonal(), alpha);
        check_scaled_unit_zero(a.template triangularView<UnitUpper>(), v.asDiagonal(), alpha);
      }
    }
  }
  const Lhs empty(3, 0);
  const Rhs rhs(0, 4);
  check_scaled_unit_zero(empty.template triangularView<UnitUpper>(), rhs, Scalar(0));
}

void scaled_unit_fallbacks() {
  scaled_unit_zero_scales<double, ColMajor, ColMajor>();
  scaled_unit_zero_scales<double, ColMajor, RowMajor>();
  scaled_unit_zero_scales<double, RowMajor, ColMajor>();
  scaled_unit_zero_scales<double, RowMajor, RowMajor>();
  scaled_unit_zero_scales<std::complex<double>, ColMajor, ColMajor>();
  scaled_unit_zero_scales<std::complex<double>, ColMajor, RowMajor>();
  scaled_unit_zero_scales<std::complex<double>, RowMajor, ColMajor>();
  scaled_unit_zero_scales<std::complex<double>, RowMajor, RowMajor>();

  const MatrixXcd complexA = MatrixXcd::Random(24, 24), complexB = MatrixXcd::Random(24, 7);
  MatrixXcd mixed(24, 7);
  mixed.noalias() = 2.0 * (complexA.triangularView<UnitLower>() * complexB);
  VERIFY_IS_APPROX(mixed, (2.0 * (complexA.triangularView<UnitLower>() * complexB).eval()).eval());
  const MatrixXcd scaled = (std::complex<double>(2, 1) * (complexA.triangularView<UnitLower>() * complexB)).eval();
  ArrayXXcd array = std::complex<double>(2, 1) * (complexA.triangularView<UnitLower>() * complexB);
  VERIFY_IS_APPROX(array.matrix(), scaled);
  const MatrixXcd zero = MatrixXcd::Zero(64, 64);
  const MatrixXcd zeroProduct = zero.triangularView<UnitLower>() * zero;
  for (const std::complex<double> scale :
       {std::complex<double>(-2, 0), std::complex<double>(0, -2), std::complex<double>(-2, -1),
        std::complex<double>(2, 1), std::complex<double>(-0.0, -2), std::complex<double>(2, -0.0)}) {
    const MatrixXcd expected = scale * zeroProduct;
    MatrixXcd actual(64, 64);
    actual.noalias() = scale * (zero.triangularView<UnitLower>() * zero);
    for (Index i = 0; i < actual.size(); ++i) {
      VERIFY_IS_EQUAL(actual(i), expected(i));
    }
  }
  MatrixXd finite = MatrixXd::Random(24, 24), other = MatrixXd::Random(24, 7), dst(24, 7);
  dst.noalias() = 2.0 * (finite.triangularView<UnitLower>() * (3.0 * other));
  VERIFY_IS_APPROX(dst, (2.0 * (finite.triangularView<UnitLower>() * (3.0 * other)).eval()).eval());

  using Q = noncommutative_scalar::Quaternion;
  using QMatrix = Matrix<Q, Dynamic, Dynamic>;
  const Q alpha(0, 0, 0, 1);
  const QMatrix qa = QMatrix::Constant(3, 3, Q(0, 1, 0, 0)), qb = QMatrix::Constant(3, 3, Q(0, 0, 1, 0));
  const QMatrix qp = qa.triangularView<UnitUpper>() * qb;
  QMatrix qdst(3, 3);
  qdst.noalias() = alpha * (qa.triangularView<UnitUpper>() * qb);
  VERIFY(qdst == (alpha * qp).eval());
}
#endif

#if defined(EIGEN_TEST_PART_6) || defined(EIGEN_TEST_PART_ALL)
void scaled_unit_eligibility() {
  MatrixXd a(4, 4);
  auto unit = a.triangularView<UnitLower>();
  auto lower = a.triangularView<Lower>();
  auto scaled = 2.0 * a;
  auto scaledTriangle = scaled.triangularView<UnitLower>();
  STATIC_CHECK((internal::product_can_scale_unit_triangle<decltype(unit), MatrixXd>::value));
  STATIC_CHECK((internal::product_can_scale_unit_triangle<decltype(unit), DiagonalMatrix<double, Dynamic>>::value));
  STATIC_CHECK((!internal::product_can_scale_unit_triangle<decltype(lower), MatrixXd>::value));
  STATIC_CHECK((!internal::product_can_scale_unit_triangle<MatrixXd, decltype(unit)>::value));
  STATIC_CHECK((!internal::product_can_scale_unit_triangle<decltype(unit), MatrixXf>::value));
  STATIC_CHECK((!internal::product_can_scale_unit_triangle<decltype(unit), decltype(scaled)>::value));
  STATIC_CHECK((!internal::product_can_scale_unit_triangle<decltype(scaledTriangle), MatrixXd>::value));
  using QMatrix = Matrix<noncommutative_scalar::Quaternion, Dynamic, Dynamic>;
  STATIC_CHECK((!internal::product_can_scale_unit_triangle<TriangularView<QMatrix, UnitLower>, QMatrix>::value));
}
#endif

EIGEN_DECLARE_TEST(scaled_unit_triangular_product) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((scaled_unit_products<float, ColMajor>()));
    CALL_SUBTEST_1((scaled_unit_products<float, RowMajor>()));
    CALL_SUBTEST_2((scaled_unit_products<double, ColMajor>()));
    CALL_SUBTEST_2((scaled_unit_products<double, RowMajor>()));
    CALL_SUBTEST_3((scaled_unit_products<std::complex<float>, ColMajor>()));
    CALL_SUBTEST_3((scaled_unit_products<std::complex<float>, RowMajor>()));
    CALL_SUBTEST_4((scaled_unit_products<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_4((scaled_unit_products<std::complex<double>, RowMajor>()));
    CALL_SUBTEST_5(scaled_unit_fallbacks());
    CALL_SUBTEST_6(scaled_unit_eligibility());
  }
}
