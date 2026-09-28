// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "packetmath_test_shared.h"
#include <Eigen/Geometry>
#include <Eigen/SparseCore>

template <typename Real>
void comparison_helpers() {
  const Real inf = NumTraits<Real>::infinity();
  const Real nan = NumTraits<Real>::quiet_NaN();
  VERIFY(test_isCwiseApprox(inf, inf, true));
  VERIFY(!test_isCwiseApprox(inf, -inf, false));
  VERIFY(!test_isApprox(inf, -inf));
  VERIFY(test_isApprox(inf, inf));
  VERIFY(test_isApprox(nan, nan));
  VERIFY(!test_isMuchSmallerThan(inf, inf));
  VERIFY(!test_isMuchSmallerThan(nan, nan));
  VERIFY(!test_isApproxOrLessThan(inf, -inf));
  VERIFY(!test_isApproxOrLessThan(nan, nan));
  VERIFY(test_isApproxOrLessThan(inf, inf));
  VERIFY(test_isCwiseApprox(nan, nan, true));
  VERIFY(!test_isCwiseApprox(nan, Real(0), false));
  VERIFY(!test_isCwiseApprox(inf, NumTraits<Real>::highest(), false));
  VERIFY(test_isCwiseApprox(Real(0), Real(-0.0), true));
  VERIFY(!test_isCwiseApprox(Real(1), Real(2), false));
  const Real eps = NumTraits<Real>::epsilon();
  const Real near = Real(1) + Real(8) * eps;
  VERIFY(test_isCwiseApprox(Real(1), near, false, Real(8) * eps));
  VERIFY(!test_isCwiseApprox(Real(1), near, false, Real(7) * eps));
  VERIFY(test_isCwiseApprox(Real(1), Real(1) + test_precision<Real>() / Real(2), false));

  using Complex = std::complex<Real>;
  VERIFY(test_isApprox(Complex(1), Real(1)));
  VERIFY(test_isApprox(Real(1), Complex(1)));
  VERIFY(test_isMuchSmallerThan(Complex(0), Real(1)));
  VERIFY(test_isMuchSmallerThan(Real(0), Complex(1)));
  using ArrayType = Array<Complex, 2, 2>;
  ArrayType x, y;
  x << Complex(inf), Complex(nan), Complex(1), Complex(0);
  y = x;
  VERIFY(test_isCwiseApprox(x, y, true));
  VERIFY(test_isCwiseApprox(x, y, false));
  y(1, 0) = Complex(1, test_precision<Real>() / Real(2));
  VERIFY(test_isCwiseApprox(x, y, false));
  VERIFY(!test_isCwiseApprox(x, y, true));
  y(0, 0) = Complex(-inf);
  VERIFY(!test_isCwiseApprox(x, y, false));
  VERIFY(!test_isCwiseApprox(x, y.row(0), false));
  VERIFY(!test_isApprox(Complex(inf), Complex(inf)));
  VERIFY(!test_isApprox(Complex(nan), Complex(nan)));

  for (Real scale :
       {Real(1), NumTraits<Real>::highest() / Real(32), (numext::numeric_limits<Real>::min)() * Real(32)}) {
    x.setConstant(Complex(scale));
    y = Real(2) * x;
    VERIFY(!test_isCwiseApprox(x, y, false));
    VERIFY(!test_isApprox(Complex(scale), Complex(Real(2) * scale)));
    VERIFY(!test_isMuchSmallerThan(Complex(scale), Complex(Real(2) * scale)));
    VERIFY(test_isMuchSmallerThan(x * Real(0), y));
    VERIFY(test_isMuchSmallerThan(x * Real(0), scale));
    VERIFY_IS_EQUAL(test_relative_error(x, y), Real(1));
    VERIFY_IS_EQUAL(test_relative_error(x.real(), y), Real(1));
    VERIFY_IS_EQUAL(test_relative_error(scale, Real(2) * scale), Real(1));
  }

  x.setZero();
  y.setZero();
  VERIFY_IS_EQUAL(test_relative_error(x, y), Real(0));
  VERIFY_IS_EQUAL(test_relative_error(Real(0), Real(0)), Real(0));
  y.setOnes();
  VERIFY((numext::isinf)(test_relative_error(x, y)));
  const SparseMatrix<Complex> sx = x.matrix().sparseView();
  const SparseMatrix<Complex> sy = y.matrix().sparseView();
  VERIFY(test_isCwiseApprox(sx, sx, true));
  VERIFY(!test_isCwiseApprox(sx, sy, false));

  using Vector = Matrix<Real, 2, 1>;
  ParametrizedLine<Real, 2> a(Vector::Ones(), Vector::UnitX());
  ParametrizedLine<Real, 2> b(Vector::Ones(), -Vector::UnitX());
  VERIFY_IS_EQUAL(test_relative_error(a, b), Real(2));
}

void integer_diagnostics() {
  const int low = NumTraits<int>::lowest();
  const int high = NumTraits<int>::highest();
  VERIFY(test_relative_error(low, high) >= 2.0);
  VERIFY(test_relative_error(Matrix2i::Constant(low), Matrix2i::Constant(high)) >= 2.0);
  VERIFY_IS_EQUAL(test_relative_error(false, false), 0.0f);
  VERIFY((numext::isinf)(test_relative_error(false, true)));
  VERIFY(test_isCwiseApprox(false, false, true));
  VERIFY(!test_isCwiseApprox(false, true, false));
  VERIFY_IS_EQUAL(test_relative_error(half(1), 2.0f), half(1));
  VERIFY_IS_EQUAL(test_relative_error(bfloat16(1), 2.0f), bfloat16(1));
}

template <typename Real>
void scaled_comparison_helpers() {
  using MatrixType = Matrix<Real, 2, 2>;
  const MatrixType zero = MatrixType::Zero();
  for (Index i = 0; i < zero.size(); ++i) {
    MatrixType value = zero;
    value(i) = NumTraits<Real>::quiet_NaN();
    VERIFY(!verifyIsApproxScaled(value, zero, Real(1)));
    VERIFY(!verifyIsApproxScaled(value.array(), zero.array(), Real(1)));
  }
  VERIFY(!verifyIsApproxScaled(zero, zero, NumTraits<Real>::infinity()));
  VERIFY(!verifyIsApproxScaled(zero, zero, NumTraits<Real>::quiet_NaN()));
  const Matrix<Real, Dynamic, Dynamic> empty(0, 3);
  VERIFY(verifyIsApproxScaled(empty, empty, Real(1)));
}

void packet_comparison_helpers() {
  const double inf = NumTraits<double>::infinity();
  const double opposite = -inf;
  const double one = 1, near = 1 + 1e-8;
  VERIFY(test::areApprox(&inf, &inf, 1));
  VERIFY(!test::areApprox(&inf, &opposite, 1));
  VERIFY(!test::areApprox(&inf, &opposite, 1, NumTraits<double>::epsilon()));
  // Packet checks retain dummy_precision(), which is tighter than the legacy test_precision().
  VERIFY(!test::areApprox(&one, &near, 1));
}

EIGEN_DECLARE_TEST(numerical_test_helpers) {
  comparison_helpers<float>();
  comparison_helpers<double>();
  comparison_helpers<long double>();
  integer_diagnostics();
  scaled_comparison_helpers<float>();
  scaled_comparison_helpers<double>();
  packet_comparison_helpers();
}
