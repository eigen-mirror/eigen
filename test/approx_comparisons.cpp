// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "fp_control.h"
#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
#include "AnnoyingScalar.h"
#include "CustomComplex.h"

template <typename Scalar, int Options>
void approx_comparisons_floating() {
  using Real = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Options>;
  const Real precision = Real(0.125);
  MatrixType x(3, 5), y(3, 5);
  for (const Real scale :
       {Real(1), NumTraits<Real>::highest() / Real(32), (numext::numeric_limits<Real>::min)() * Real(32)}) {
    x.setConstant(Scalar(scale));
    y = x * Real(2);
    internal::set_is_malloc_allowed(false);
    VERIFY(x.isApprox(x, precision));
    VERIFY(!x.isApprox(y, precision));
    VERIFY(!y.isApprox(x, precision));
    VERIFY(!x.array().isApprox(y.array(), precision));
    VERIFY(!x.transpose().isApprox(y.transpose(), precision));
    VERIFY(!x.isMuchSmallerThan(y, precision));
    VERIFY(x.isMuchSmallerThan(y, Real(0.5)));
    VERIFY(!x.isMuchSmallerThan(scale, precision));
    VERIFY(x.isMuchSmallerThan(Real(4) * scale, Real(1)));
    VERIFY(x.isApprox(y, Real(1)));
    VERIFY(x.isApprox(y, Real(-1)));
    VERIFY(x.isApprox(x, Real(0)));
    VERIFY(!x.isApprox(y, Real(0)));
    y = x * (Real(1) + precision / Real(2));
    VERIFY(x.isApprox(y, precision));
    VERIFY(x.array().isApprox(y.array(), precision));
    VERIFY((x.template block<2, 3>(0, 1).isApprox(y.template block<2, 3>(0, 1), precision)));
    internal::set_is_malloc_allowed(true);
    // The scaled path evaluates a lazy operand, which allocates for dynamic sizes.
    VERIFY(!x.isApprox(MatrixType::Zero(3, 5), precision));
    Matrix<Scalar, Dynamic, Dynamic, Options == ColMajor ? RowMajor : ColMajor> other = y;
    VERIFY(x.isApprox(other, precision));
  }

  // Norms and differences can exceed the scalar range even though every coefficient is finite.
  x.setConstant(Scalar(NumTraits<Real>::highest()));
  y = -x;
  VERIFY(x.isApprox(x, Real(0)));
  VERIFY(!x.isApprox(y, Real(1)));
  VERIFY(x.isApprox(y, Real(2)));
  VERIFY(!x.isMuchSmallerThan(x, precision));
  // The squared difference, but not the bound, can overflow the accumulator here.
  const Real root = numext::ldexp(Real(1), (NumTraits<Real>::max_exponent() - 1) / 2);
  x.setZero();
  x(0, 0) = Scalar(root);
  y = -x;
  VERIFY(!x.isApprox(y, Real(1.9)));
  VERIFY(x.isApprox(y, Real(2)));

  // Squaring either the difference or the tolerance would underflow.
  const Real tiny = (numext::numeric_limits<Real>::min)() * Real(16);
  x.setZero();
  y.setZero();
  x(0, 0) = y(0, 0) = Scalar(1);
  x(2, 4) = Scalar(tiny);
  VERIFY(!x.isApprox(y, Real(0)));
  VERIFY(!x.isApprox(y, tiny / Real(2)));
  VERIFY(x.isApprox(y, tiny * Real(2)));
  VERIFY(x.isApprox(y, NumTraits<Real>::highest()));
  x.setZero();
  VERIFY(x.isMuchSmallerThan(y, Real(0)));
  VERIFY(x.isMuchSmallerThan(Real(1), Real(0)));

  for (Index rows : {Index(0), Index(3)}) {
    x.resize(rows, 0);
    y.resize(rows, 0);
    VERIFY(x.isApprox(y));
    VERIFY(x.isMuchSmallerThan(y));
    VERIFY(x.isMuchSmallerThan(Real(0)));
  }
}

// The scaled path views plain objects, maps and their blocks in place, whatever their strides, and evaluates other
// operands, on the stack when their size is bounded.
template <typename Scalar>
void approx_comparisons_strided_operands() {
  using Real = typename NumTraits<Scalar>::Real;
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RowMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using StridedMap = Map<const ColMatrix, 0, Stride<Dynamic, Dynamic>>;
  const Real precision = Real(0.125);
  // Squares of these coefficients underflow, which selects the scaled path.
  const Real scale = (numext::numeric_limits<Real>::min)() * Real(32);
  const ColMatrix x = ColMatrix::Random(6, 8) * scale;
  const ColMatrix y = x * (Real(1) + precision / Real(2));
  const ColMatrix z = x * Real(2);
  const RowMatrix xr = x, yr = y, zr = z;
  const Stride<Dynamic, Dynamic> stride(12, 2);
  const ColMatrix ys = StridedMap(y.data(), 3, 4, stride);
  internal::set_is_malloc_allowed(false);
  VERIFY(xr.isApprox(y, precision));
  VERIFY(!xr.isApprox(z, precision));
  VERIFY(x.row(2).isApprox(yr.row(2), precision));
  VERIFY(!x.row(2).isApprox(zr.row(2), precision));
  VERIFY(xr.col(3).isApprox(y.col(3), precision));
  VERIFY(!xr.col(3).isApprox(z.col(3), precision));
  VERIFY(StridedMap(x.data(), 3, 4, stride).isApprox(ys, precision));
  VERIFY(!StridedMap(x.data(), 3, 4, stride).isApprox(StridedMap(z.data(), 3, 4, stride), precision));
  VERIFY((Real(2) * x.template topLeftCorner<3, 3>()).isApprox(Real(2) * y.template topLeftCorner<3, 3>(), precision));
  VERIFY(!(Real(2) * x.template topLeftCorner<3, 3>()).isApprox(x.template topLeftCorner<3, 3>(), precision));
  internal::set_is_malloc_allowed(true);
  // These have direct access, but their strides do not address every coefficient.
  VERIFY(x(2, seq(0, 7, 2)).isApprox(yr(2, seq(0, 7, 2)), precision));
  VERIFY(xr(seq(0, 5, 2), 3).isApprox(y(seq(0, 5, 2), 3), precision));
  VERIFY(x.row(2).realView().isApprox(yr.row(2).realView(), precision));
}

template <typename Real>
void approx_comparisons_rounding_boundary() {
  using Vector = Matrix<Real, 1, 1>;
  const Real eps = NumTraits<Real>::epsilon();
  // One nonzero component and power-of-two scales make the relative difference exactly 8*eps.
  for (int exponent : {0, NumTraits<Real>::min_exponent() + 4, NumTraits<Real>::max_exponent() - 8}) {
    const Real scale = numext::ldexp(Real(1), exponent);
    const Vector x = Vector::Constant(scale);
    const Vector y = Vector::Constant(scale * (Real(1) + Real(8) * eps));
    VERIFY(!x.isApprox(y, Real(7) * eps));
    VERIFY(x.isApprox(y, Real(8) * eps));
    VERIFY(x.isApprox(y, Real(9) * eps));
  }
}

template <typename Real>
void approx_comparisons_flush_to_zero() {
  ScopedFlushToZero flushToZero;
  using Vector = Matrix<Real, 2, 1>;
  // The difference of these normal operands is subnormal.
  const Real normalMin = (numext::numeric_limits<Real>::min)();
  const Vector x(Real(1.5) * normalMin, Real(0)), y(normalMin, Real(0));
  VERIFY(!x.isApprox(y, Real(0.25)));
  VERIFY(!x.isApprox(y, Real(0)));
  VERIFY(x.isApprox(y, Real(0.5)));
  VERIFY_IS_EQUAL(test_relative_error(x, y), Real(0.5));
  approx_comparisons_rounding_boundary<Real>();
}

// The squared differences, 2^-118 and just below 2^-126, accumulate in float, where FTZ flushes the second.
void approx_comparisons_flush_to_zero_bfloat16() {
  ScopedFlushToZero flushToZero;
  using Vector = Matrix<bfloat16, 2, 1>;
  // Opaque, so the compiler cannot square it with IEEE subnormals at compile time.
  volatile float tail = numext::ldexp(1.9921875f, -64);
  const Vector x(bfloat16(numext::ldexp(1.0f, -52)), bfloat16(0.0f));
  const Vector y(bfloat16(numext::ldexp(1.0f + 1.0f / 128, -52)), bfloat16(float(tail)));
  VERIFY(!x.isApprox(y, bfloat16(1.0f / 128)));
}

template <typename Real>
void approx_comparisons_special_values() {
  using Vector = Matrix<Real, 2, 1>;
  const Real inf = NumTraits<Real>::infinity();
  const Real nan = NumTraits<Real>::quiet_NaN();
  Vector x = Vector::Ones(), y = Vector::Ones();
  for (Index i = 0; i < 2; ++i) {
    y = x;
    y(i) = nan;
    VERIFY(!x.isApprox(y));
    VERIFY(!y.isApprox(x));
    VERIFY(!y.isApprox(y));
    VERIFY(!x.isMuchSmallerThan(y));
    VERIFY(!y.isMuchSmallerThan(x));
    y(i) = inf;
    VERIFY(!x.isApprox(y));
    VERIFY(!y.isApprox(y));
    VERIFY(!y.isMuchSmallerThan(y));
    VERIFY(x.isMuchSmallerThan(y));
  }
  VERIFY(!x.isApprox(x, nan));
  VERIFY(!x.isMuchSmallerThan(x, nan));
  VERIFY(!x.isMuchSmallerThan(nan));
  VERIFY(!x.isMuchSmallerThan(inf, Real(0)));

  if (subnormalDivisionIsExact<Real>()) {
    const Real tiny = (numext::numeric_limits<Real>::denorm_min)();
    x.setConstant(tiny);
    y = Real(2) * x;
    VERIFY(!x.isApprox(y));
    VERIFY(x.isApprox(x, Real(0)));
    VERIFY(!x.isApprox(Vector::Zero()));
    VERIFY(!x.isMuchSmallerThan(y));
    VERIFY(!x.isMuchSmallerThan(tiny));
  }
}

template <typename Scalar>
void approx_comparisons_exact() {
  using MatrixType = Matrix<Scalar, 2, 2>;
  const MatrixType zero = MatrixType::Zero();
  const MatrixType one = MatrixType::Ones();
  VERIFY(zero.isApprox(zero));
  VERIFY(one.isApprox(one));
  VERIFY(!zero.isApprox(one));
  VERIFY(!one.isApprox(zero));
  VERIFY(zero.isMuchSmallerThan(one));
  VERIFY(zero.isMuchSmallerThan(Scalar(0)));
  VERIFY(!one.isMuchSmallerThan(one));
  VERIFY(!one.isMuchSmallerThan(Scalar(1)));
}

void approx_comparisons_expressions() {
  Matrix2d x;
  x << 1, 2, 3, 4;
  VERIFY((x * x).isApprox((x * x).eval()));
  VERIFY((x + x).isApprox(2 * x));
  VERIFY((x * x * 0).isMuchSmallerThan(x * x));
  VERIFY((x * x * 0).isMuchSmallerThan(1.0));
  VERIFY(!Matrix2f::Ones().isMuchSmallerThan(Matrix2d::Constant(1e200), 0.0f));
  VERIFY(Matrix2f::Ones().isMuchSmallerThan(Matrix2d::Constant(1e200)));
  VERIFY(!Matrix2d::Ones().isMuchSmallerThan(Matrix2f::Constant(1e-30f)));
  VERIFY(Matrix2d::Constant(1e-300).isMuchSmallerThan(Matrix2f::Constant(1e10f), 1e-300));
  VERIFY(!Matrix2f::Constant(1e-23f).isMuchSmallerThan(Matrix2d::Constant(1e-20), 1e-5f));
  VERIFY(Matrix2f::Constant(1e20f).isMuchSmallerThan(Matrix2d::Constant(1e30), 1e-5f));
  const Matrix2d tiny = Matrix2d::Constant(1e-200);
  const Matrix2cd complexTiny = tiny.cast<std::complex<double>>();
  VERIFY(tiny.isApprox(complexTiny));
  VERIFY(complexTiny.isApprox(tiny));
  STATIC_CHECK(!internal::use_scaled_comparison<AnnoyingScalar>::value);
  approx_comparisons_exact<AnnoyingScalar>();
  const Matrix<AnnoyingScalar, 2, 2> custom = Matrix<AnnoyingScalar, 2, 2>::Ones();
  VERIFY(test_isCwiseApprox(custom, custom, false));
  VERIFY(test_relative_error(custom, custom) == AnnoyingScalar(0));
  // A custom complex scalar over a built-in real has no dedicated coefficient overload.
  using CustomComplexScalar = CustomComplex<double>;
  STATIC_CHECK(!internal::use_scaled_comparison<CustomComplexScalar>::value);
  const Matrix<CustomComplexScalar, 2, 2> customComplex =
      Matrix<CustomComplexScalar, 2, 2>::Constant(CustomComplexScalar(1, 2));
  VERIFY(test_isCwiseApprox(customComplex, customComplex, true));
  VERIFY(test_isCwiseApprox(customComplex, customComplex, false));
  VERIFY(!test_isCwiseApprox(customComplex, (customComplex * CustomComplexScalar(2)).eval(), false));
  VERIFY_IS_EQUAL(test_relative_error(customComplex, customComplex), 0.0);
}

template <typename Real>
void approx_comparisons_complex_components() {
  using Complex = std::complex<Real>;
  using Vector = Matrix<Complex, 2, 1>;
  const Real big = NumTraits<Real>::highest();
  const Vector x = Vector::Constant(Complex(big, big));
  const Vector y = -x;
  VERIFY(x.isApprox(x, Real(0)));
  VERIFY(!x.isApprox(y, Real(1)));
  VERIFY(x.isApprox(y, Real(2)));
  VERIFY(!x.isMuchSmallerThan(y, Real(0.5)));
  VERIFY(!x.isMuchSmallerThan(big, Real(1)));
}

// nested_eval keeps these cheap expressions lazy, so they reach the scaled path without direct access.
template <typename Real>
void approx_comparisons_lazy_operands() {
  using Complex = std::complex<Real>;
  using Vector3 = Matrix<Real, 3, 1>;
  using Vector4 = Matrix<Real, 4, 1>;
  using Array4 = Array<Real, 4, 1>;
  const Real precision = Real(0.125);
  const Array4 base = Array4::LinSpaced(Real(1), Real(2));
  for (int exponent : {0, NumTraits<Real>::min_exponent() + 4, NumTraits<Real>::max_exponent() - 8}) {
    const Real scale = numext::ldexp(Real(1), exponent);
    const Vector3 x = Vector3::Constant(scale);
    Vector4 y;
    y << x, Real(1);
    VERIFY(x.homogeneous().isApprox(y, precision));
    VERIFY(y.isApprox(x.homogeneous(), precision));
    VERIFY_IS_EQUAL(test_relative_error(x.homogeneous(), y), Real(0));
    VERIFY(!x.homogeneous().isMuchSmallerThan(y, precision));
    y *= Real(2);
    VERIFY(!x.homogeneous().isApprox(y, precision));
    VERIFY(!y.isApprox(x.homogeneous(), precision));
    VERIFY(test_relative_error(x.homogeneous(), y) > precision);

    const Array4 a = base * scale;
    VERIFY((a + a).isApprox(a * Real(2), precision));
    VERIFY(!(a + a).isApprox(a, precision));
    VERIFY_IS_EQUAL(test_relative_error(a + a, a * Real(2)), Real(0));
  }
  // A unary functor whose result is complex but whose argument is real.
  const Complex power(Real(2), Real(1));
  VERIFY(base.pow(power).isApprox(base.template cast<Complex>().pow(power), precision));
  VERIFY(test_relative_error(base.pow(power), base.template cast<Complex>().pow(power)) <= precision);
}

template <typename Real>
void approx_comparisons_scale_invariance() {
  using Vector = Matrix<Real, 17, 1>;
  const Real precision = Real(0.125);
  const long double precision2 = static_cast<long double>(precision) * static_cast<long double>(precision);
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    const Vector a = Vector::Random();
    for (int close = 0; close < 2; ++close) {
      Vector b = Vector::Random();
      if (close) b = a + precision * Real(0.25) * b;
      long double difference = 0, a2 = 0, b2 = 0;
      for (Index i = 0; i < a.size(); ++i) {
        const long double av = static_cast<long double>(a(i));
        const long double bv = static_cast<long double>(b(i));
        difference += (av - bv) * (av - bv);
        a2 += av * av;
        b2 += bv * bv;
      }
      const bool expected = difference <= precision2 * (std::min)(a2, b2);
      for (int exponent : {0, NumTraits<Real>::min_exponent() + 4, NumTraits<Real>::max_exponent() - 8}) {
        const Real scale = numext::ldexp(Real(1), exponent);
        const Vector x = a * scale, y = b * scale;
        VERIFY_IS_EQUAL(x.isApprox(y, precision), expected);
      }
    }
  }
}

EIGEN_DECLARE_TEST(approx_comparisons) {
  CALL_SUBTEST_1((approx_comparisons_floating<float, ColMajor>()));
  CALL_SUBTEST_1((approx_comparisons_floating<double, RowMajor>()));
  CALL_SUBTEST_1(approx_comparisons_strided_operands<float>());
  CALL_SUBTEST_1(approx_comparisons_strided_operands<double>());
  CALL_SUBTEST_1(approx_comparisons_special_values<float>());
  CALL_SUBTEST_1(approx_comparisons_special_values<double>());
  CALL_SUBTEST_1(approx_comparisons_rounding_boundary<float>());
  CALL_SUBTEST_1(approx_comparisons_rounding_boundary<double>());
  CALL_SUBTEST_1(approx_comparisons_scale_invariance<float>());
  CALL_SUBTEST_1(approx_comparisons_scale_invariance<double>());
  CALL_SUBTEST_1(approx_comparisons_flush_to_zero<float>());
  CALL_SUBTEST_1(approx_comparisons_flush_to_zero<double>());
  CALL_SUBTEST_2((approx_comparisons_floating<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_2((approx_comparisons_floating<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_2(approx_comparisons_strided_operands<std::complex<float>>());
  CALL_SUBTEST_2(approx_comparisons_complex_components<float>());
  CALL_SUBTEST_2(approx_comparisons_complex_components<double>());
  CALL_SUBTEST_2(approx_comparisons_lazy_operands<float>());
  CALL_SUBTEST_2(approx_comparisons_lazy_operands<double>());
  CALL_SUBTEST_3((approx_comparisons_floating<half, ColMajor>()));
  CALL_SUBTEST_3((approx_comparisons_floating<bfloat16, RowMajor>()));
  CALL_SUBTEST_3((approx_comparisons_floating<long double, ColMajor>()));
  CALL_SUBTEST_3(approx_comparisons_rounding_boundary<half>());
  CALL_SUBTEST_3(approx_comparisons_rounding_boundary<bfloat16>());
  CALL_SUBTEST_3(approx_comparisons_flush_to_zero<bfloat16>());
  CALL_SUBTEST_3(approx_comparisons_flush_to_zero_bfloat16());
  CALL_SUBTEST_4(approx_comparisons_exact<int>());
  CALL_SUBTEST_4(approx_comparisons_exact<unsigned int>());
  CALL_SUBTEST_4(approx_comparisons_exact<bool>());
  CALL_SUBTEST_4(approx_comparisons_expressions());
}
