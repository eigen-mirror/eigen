// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

template <typename Expression>
void verify_zero_product(const Expression& expression) {
  using Scalar = typename Expression::Scalar;
  typename Expression::PlainObject dst(expression.rows(), expression.cols());
  dst.setConstant(Scalar(7));
  dst.noalias() = expression;
  VERIFY((dst.array() == Scalar(0)).all());
  dst.setConstant(Scalar(7));
  dst.noalias() += expression;
  VERIFY((dst.array() == Scalar(7)).all());
  dst.noalias() -= expression;
  VERIFY((dst.array() == Scalar(7)).all());
}

template <typename Scalar, int Order, int Mode>
void triangular_zero_alpha(Index n, const Scalar& nonfinite) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  const MatrixType a = MatrixType::Constant(n, n, nonfinite);
  const MatrixType b = MatrixType::Constant(n, n + 2, nonfinite);
  const VectorType v = VectorType::Constant(n, nonfinite);
  const Scalar zero(0);
  verify_zero_product(zero * (a.template triangularView<Mode>() * b));
  verify_zero_product(zero * (b.transpose() * a.template triangularView<Mode>()));
  verify_zero_product(zero * (a.template triangularView<Mode>() * v));
  verify_zero_product(zero * (v.transpose() * a.template triangularView<Mode>()));
  verify_zero_product(a.template triangularView<Mode>() * (zero * b));
  verify_zero_product((zero * b.transpose()) * a.template triangularView<Mode>());
  verify_zero_product(a.template triangularView<Mode>() * (zero * v));
  EIGEN_IF_CONSTEXPR ((Mode & UnitDiag) == 0) {
    verify_zero_product((zero * a).template triangularView<Mode>() * b);
    verify_zero_product(b.transpose() * (zero * a).template triangularView<Mode>());
    verify_zero_product((zero * a).template triangularView<Mode>() * v);
  } else {
    const MatrixType finite = MatrixType::Ones(n, n + 2);
    MatrixType dst = (zero * a).template triangularView<Mode>() * finite;
    VERIFY_IS_EQUAL(dst, finite);
    dst = finite.transpose() * (zero * a).template triangularView<Mode>();
    VERIFY_IS_EQUAL(dst, finite.transpose());
    const VectorType ones = VectorType::Ones(n);
    VectorType result = (zero * a).template triangularView<Mode>() * ones;
    VERIFY_IS_EQUAL(result, ones);
    Matrix<Scalar, 1, Dynamic> row = ones.transpose() * (zero * a).template triangularView<Mode>();
    VERIFY_IS_EQUAL(row, ones.transpose());

    const Array<Scalar, Dynamic, Dynamic, Order> arrayLeft = (zero * a).template triangularView<Mode>() * finite;
    VERIFY_IS_EQUAL(arrayLeft.matrix(), finite);
    const Array<Scalar, Dynamic, Dynamic, Order> arrayRight =
        finite.transpose() * (zero * a).template triangularView<Mode>();
    VERIFY_IS_EQUAL(arrayRight.matrix(), finite.transpose());
    const Array<Scalar, Dynamic, 1> arrayVector = (zero * a).template triangularView<Mode>() * ones;
    VERIFY_IS_EQUAL(arrayVector.matrix(), ones);
    const Array<Scalar, 1, Dynamic, RowMajor> arrayRow = ones.transpose() * (zero * a).template triangularView<Mode>();
    VERIFY_IS_EQUAL(arrayRow.matrix(), ones.transpose());
  }
}

template <typename Scalar, int Order, int UpLo>
void selfadjoint_zero_alpha(Index n, const Scalar& nonfinite) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  const MatrixType a = MatrixType::Constant(n, n, nonfinite);
  const VectorType v = VectorType::Constant(n, nonfinite);
  const Scalar zero(0);
  verify_zero_product(zero * (a.template selfadjointView<UpLo>() * a));
  verify_zero_product(zero * (a * a.template selfadjointView<UpLo>()));
  verify_zero_product(zero * (a.template selfadjointView<UpLo>() * v));
  verify_zero_product(zero * (v.transpose() * a.template selfadjointView<UpLo>()));
  verify_zero_product(a.template selfadjointView<UpLo>() * (zero * a));

  MatrixType dst = MatrixType::Constant(n, n, Scalar(7));
  dst.template selfadjointView<UpLo>().rankUpdate(v, zero);
  VERIFY((dst.array() == Scalar(7)).all());
  dst.template selfadjointView<UpLo>().rankUpdate(a, zero);
  VERIFY((dst.array() == Scalar(7)).all());
  dst.template selfadjointView<UpLo>().rankUpdate(v, v, zero);
  VERIFY((dst.array() == Scalar(7)).all());
  dst.template selfadjointView<UpLo>().rankUpdate(zero * v);
  VERIFY((dst.array() == Scalar(7)).all());

  dst.template triangularView<UpLo>() = (zero * a) * a.adjoint();
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) VERIFY_IS_EQUAL(dst(i, j), ((UpLo == Lower ? i >= j : i <= j) ? zero : Scalar(7)));
  dst.setConstant(Scalar(7));
  dst.template triangularView<UpLo>() = (zero * v) * v.adjoint();
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) VERIFY_IS_EQUAL(dst(i, j), ((UpLo == Lower ? i >= j : i <= j) ? zero : Scalar(7)));

  // Exercise the non-unit destination-stride fallbacks as well as the packed kernels.
  MatrixType storage = MatrixType::Constant(2 * n, 2 * n, Scalar(7));
  Map<MatrixType, Unaligned, Stride<Dynamic, Dynamic>> strided(storage.data(), n, n,
                                                               Stride<Dynamic, Dynamic>(2 * storage.outerStride(), 2));
  strided.template selfadjointView<UpLo>().rankUpdate(v, zero);
  strided.template selfadjointView<UpLo>().rankUpdate(v, v, zero);
  VERIFY((storage.array() == Scalar(7)).all());
  if (n != 0) {
    using RealScalar = typename NumTraits<Scalar>::Real;
    const RealScalar smallReal = (std::numeric_limits<RealScalar>::min)();
    const Scalar small(smallReal);
    const VectorType large = VectorType::Constant(n, Scalar(RealScalar(1) / smallReal));
    storage.setZero();
    strided.template selfadjointView<UpLo>().rankUpdate(small * large);
    VERIFY_IS_EQUAL(strided(0, 0), Scalar(1));
    strided.template selfadjointView<UpLo>().rankUpdate(small * large, small * large);
    VERIFY_IS_EQUAL(strided(0, 0), Scalar(3));
    strided.template triangularView<UpLo>() = (small * large) * (small * large).adjoint();
    VERIFY_IS_EQUAL(strided(0, 0), Scalar(1));
    const Scalar denormal((std::numeric_limits<RealScalar>::denorm_min)());
    const auto nested = Scalar(0.25) * (large * denormal);
    const Scalar value = Scalar(0.25) * (large(0) * denormal);
    storage.setZero();
    strided.template selfadjointView<UpLo>().rankUpdate(nested);
    VERIFY_IS_EQUAL(strided(0, 0), value * numext::conj(value));
  }
}

struct zero_alpha_counted_coeff {
  Index* reads;
  double operator()(Index, Index) const {
    ++*reads;
    return NumTraits<double>::quiet_NaN();
  }
};

struct zero_alpha_no_equality {};
struct value_only_scalar {
  double value, derivative;
  value_only_scalar(double v, double d = 0) : value(v), derivative(d) {}
  bool operator==(const value_only_scalar& other) const { return value == other.value; }
};

void zero_alpha_skips_evaluation() {
  VERIFY(!internal::is_identically_zero(zero_alpha_no_equality{}));
  VERIFY(!internal::is_identically_zero(value_only_scalar{0, 1}));
  for (Index n : {4, 24}) {
    const MatrixXd a = MatrixXd::Constant(n, n, NumTraits<double>::quiet_NaN());
    MatrixXd result = 1e-200 * (a.triangularView<Lower>() * (1e-200 * a));
    VERIFY((numext::isnan)(result(0, 0)));
    result = 1e-200 * (a.selfadjointView<Lower>() * (1e-200 * a));
    VERIFY((numext::isnan)(result(0, 0)));
    VectorXd v = 1e-200 * (a.selfadjointView<Lower>() * (1e-200 * a.col(0)));
    VERIFY((numext::isnan)(v(0)));
    const MatrixXd id = MatrixXd::Identity(n, n);
    result = id.triangularView<Lower>() * (0.0 * MatrixXd::Constant(n, n, NumTraits<double>::quiet_NaN()));
    VERIFY((numext::isnan)(result(0, 0)));
    result.setZero();
    result.selfadjointView<Lower>().rankUpdate(1e-170 * a.col(0), 1e-170 * a.col(0));
    VERIFY((numext::isnan)(result(0, 0)));
    result.setZero();
    result.selfadjointView<Lower>().rankUpdate(1e-170 * a.col(0));
    VERIFY((numext::isnan)(result(0, 0)));
  }
  Index reads = 0;
  const auto source = MatrixXd::NullaryExpr(24, 24, zero_alpha_counted_coeff{&reads});
  MatrixXd dst = MatrixXd::Ones(24, 24);
  dst.noalias() += 0.0 * (source.triangularView<Lower>() * source);
  VERIFY_IS_EQUAL(reads, 0);
  dst.noalias() += 0.0 * (source.selfadjointView<Lower>() * source);
  VERIFY_IS_EQUAL(reads, 0);
  dst.selfadjointView<Lower>().rankUpdate(source, 0.0);
  VERIFY_IS_EQUAL(reads, 0);

  const Matrix4d fixed = Matrix4d::Constant(NumTraits<double>::quiet_NaN());
  verify_zero_product(0.0 * (fixed.triangularView<Lower>() * fixed));
  verify_zero_product(0.0 * (fixed * fixed.triangularView<UnitUpper>()));
  verify_zero_product(0.0 * (fixed.selfadjointView<Lower>() * fixed));

  const double small = (std::numeric_limits<double>::min)();
  const Matrix4d large = Matrix4d::Constant(1.0 / small);
  const Matrix4d tiny = (small * (large * small)).triangularView<Lower>() * Matrix4d::Ones();
  VERIFY_IS_EQUAL(tiny(0, 0), small);
}

template <typename Scalar, int Order>
void structured_zero_alpha() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  for (Index n : {Index(0), Index(1), Index(4), Index(24)}) {
    for (const Scalar nonfinite :
         {Scalar(NumTraits<RealScalar>::infinity()), Scalar(NumTraits<RealScalar>::quiet_NaN())}) {
      triangular_zero_alpha<Scalar, Order, Lower>(n, nonfinite);
      triangular_zero_alpha<Scalar, Order, Upper>(n, nonfinite);
      triangular_zero_alpha<Scalar, Order, UnitLower>(n, nonfinite);
      triangular_zero_alpha<Scalar, Order, UnitUpper>(n, nonfinite);
      selfadjoint_zero_alpha<Scalar, Order, Lower>(n, nonfinite);
      selfadjoint_zero_alpha<Scalar, Order, Upper>(n, nonfinite);
    }
  }
}

EIGEN_DECLARE_TEST(structured_product_zero_alpha) {
  CALL_SUBTEST_1((structured_zero_alpha<float, ColMajor>()));
  CALL_SUBTEST_2((structured_zero_alpha<float, RowMajor>()));
  CALL_SUBTEST_3((structured_zero_alpha<double, ColMajor>()));
  CALL_SUBTEST_4((structured_zero_alpha<double, RowMajor>()));
  CALL_SUBTEST_5((structured_zero_alpha<std::complex<float>, ColMajor>()));
  CALL_SUBTEST_6((structured_zero_alpha<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_7((structured_zero_alpha<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_8((structured_zero_alpha<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_1(zero_alpha_skips_evaluation());
}
