// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/QR>
#include <iomanip>
#include <vector>

void check(bool b, bool ref) {
  std::cout << b;
  if (b == ref)
    std::cout << " OK  ";
  else
    std::cout << " BAD ";
}

template <typename T>
void check_inf_nan(bool dryrun) {
  Matrix<T, Dynamic, 1> m(10);
  m.setRandom();
  m(3) = std::numeric_limits<T>::quiet_NaN();

  if (dryrun) {
    std::cout << "std::isfinite(" << m(3) << ") = ";
    check((std::isfinite)(m(3)), false);
    std::cout << "  ; numext::isfinite = ";
    check((numext::isfinite)(m(3)), false);
    std::cout << "\n";
    std::cout << "std::isinf(" << m(3) << ")    = ";
    check((std::isinf)(m(3)), false);
    std::cout << "  ; numext::isinf    = ";
    check((numext::isinf)(m(3)), false);
    std::cout << "\n";
    std::cout << "std::isnan(" << m(3) << ")    = ";
    check((std::isnan)(m(3)), true);
    std::cout << "  ; numext::isnan    = ";
    check((numext::isnan)(m(3)), true);
    std::cout << "\n";
    std::cout << "allFinite: ";
    check(m.allFinite(), 0);
    std::cout << "\n";
    std::cout << "hasNaN:    ";
    check(m.hasNaN(), 1);
    std::cout << "\n";
    std::cout << "\n";
  } else {
    if ((std::isfinite)(m(3))) {
      g_test_level = 1;
      VERIFY(!(numext::isfinite)(m(3)));
      g_test_level = 0;
    }
    if ((std::isinf)(m(3))) {
      g_test_level = 1;
      VERIFY(!(numext::isinf)(m(3)));
      g_test_level = 0;
    }
    if (!(std::isnan)(m(3))) {
      g_test_level = 1;
      VERIFY((numext::isnan)(m(3)));
      g_test_level = 0;
    }
    if ((std::isfinite)(m(3))) {
      g_test_level = 1;
      VERIFY(!m.allFinite());
      g_test_level = 0;
    }
    if (!(std::isnan)(m(3))) {
      g_test_level = 1;
      VERIFY(m.hasNaN());
      g_test_level = 0;
    }
  }
  T hidden_zero = (std::numeric_limits<T>::min)() * (std::numeric_limits<T>::min)();
  m(4) /= hidden_zero;
  if (dryrun) {
    std::cout << "std::isfinite(" << m(4) << ") = ";
    check((std::isfinite)(m(4)), false);
    std::cout << "  ; numext::isfinite = ";
    check((numext::isfinite)(m(4)), false);
    std::cout << "\n";
    std::cout << "std::isinf(" << m(4) << ")    = ";
    check((std::isinf)(m(4)), true);
    std::cout << "  ; numext::isinf    = ";
    check((numext::isinf)(m(4)), true);
    std::cout << "\n";
    std::cout << "std::isnan(" << m(4) << ")    = ";
    check((std::isnan)(m(4)), false);
    std::cout << "  ; numext::isnan    = ";
    check((numext::isnan)(m(4)), false);
    std::cout << "\n";
    std::cout << "allFinite: ";
    check(m.allFinite(), 0);
    std::cout << "\n";
    std::cout << "hasNaN:    ";
    check(m.hasNaN(), 1);
    std::cout << "\n";
    std::cout << "\n";
  } else {
    if ((std::isfinite)(m(3))) {
      g_test_level = 1;
      VERIFY(!(numext::isfinite)(m(4)));
      g_test_level = 0;
    }
    if (!(std::isinf)(m(3))) {
      g_test_level = 1;
      VERIFY((numext::isinf)(m(4)));
      g_test_level = 0;
    }
    if ((std::isnan)(m(3))) {
      g_test_level = 1;
      VERIFY(!(numext::isnan)(m(4)));
      g_test_level = 0;
    }
    if ((std::isfinite)(m(3))) {
      g_test_level = 1;
      VERIFY(!m.allFinite());
      g_test_level = 0;
    }
    if (!(std::isnan)(m(3))) {
      g_test_level = 1;
      VERIFY(m.hasNaN());
      g_test_level = 0;
    }
  }
  m(3) = 0;
  if (dryrun) {
    std::cout << "std::isfinite(" << m(3) << ") = ";
    check((std::isfinite)(m(3)), true);
    std::cout << "  ; numext::isfinite = ";
    check((numext::isfinite)(m(3)), true);
    std::cout << "\n";
    std::cout << "std::isinf(" << m(3) << ")    = ";
    check((std::isinf)(m(3)), false);
    std::cout << "  ; numext::isinf    = ";
    check((numext::isinf)(m(3)), false);
    std::cout << "\n";
    std::cout << "std::isnan(" << m(3) << ")    = ";
    check((std::isnan)(m(3)), false);
    std::cout << "  ; numext::isnan    = ";
    check((numext::isnan)(m(3)), false);
    std::cout << "\n";
    std::cout << "allFinite: ";
    check(m.allFinite(), 0);
    std::cout << "\n";
    std::cout << "hasNaN:    ";
    check(m.hasNaN(), 0);
    std::cout << "\n";
    std::cout << "\n\n";
  } else {
    if (!(std::isfinite)(m(3))) {
      g_test_level = 1;
      VERIFY((numext::isfinite)(m(3)));
      g_test_level = 0;
    }
    if ((std::isinf)(m(3))) {
      g_test_level = 1;
      VERIFY(!(numext::isinf)(m(3)));
      g_test_level = 0;
    }
    if ((std::isnan)(m(3))) {
      g_test_level = 1;
      VERIFY(!(numext::isnan)(m(3)));
      g_test_level = 0;
    }
    if ((std::isfinite)(m(3))) {
      g_test_level = 1;
      VERIFY(!m.allFinite());
      g_test_level = 0;
    }
    if ((std::isnan)(m(3))) {
      g_test_level = 1;
      VERIFY(!m.hasNaN());
      g_test_level = 0;
    }
  }
}

template <typename RealScalar>
void check_complex_rowmajor_adjoint_product() {
  using Scalar = std::complex<RealScalar>;
  using RowMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  RowMatrix mat(2, 2);
  mat << Scalar(1, 2), Scalar(3, -4), Scalar(-5, 6), Scalar(7, 8);

  RowMatrix expected(2, 2);
  expected << Scalar(66, 0), Scalar(8, -92), Scalar(8, 92), Scalar(138, 0);

  const RowMatrix row_major_result = mat.adjoint() * mat;
  const ColMatrix col_major_result = mat.adjoint() * mat;

  VERIFY_IS_APPROX(mat.adjoint() * mat, expected);
  VERIFY_IS_APPROX(row_major_result, expected);
  VERIFY_IS_APPROX(col_major_result, expected);
}

template <typename RealScalar>
void check_complex_packet_arithmetic() {
  using Scalar = std::complex<RealScalar>;
  using Vector2 = Matrix<Scalar, 2, 1>;

  Vector2 values;
  values << Scalar(RealScalar(0.53645928880954319), RealScalar(-0.60489662966980218)),
      Scalar(RealScalar(0.25774142970757641), RealScalar(0.10793998506041591));
  Scalar divisor(RealScalar(1.6611441458336193), RealScalar(-0.21123424512127231));
  Scalar factor(RealScalar(1.2499121678643004), RealScalar(0.36146968008699221));

  Vector2 quotient = values / divisor;
  Vector2 expected_quotient;
  expected_quotient << values.coeff(0) / divisor, values.coeff(1) / divisor;
  VERIFY_IS_APPROX(quotient, expected_quotient);

  Vector2 inverse = values.array().inverse();
  Vector2 expected_inverse;
  expected_inverse << Scalar(RealScalar(1)) / values.coeff(0), Scalar(RealScalar(1)) / values.coeff(1);
  VERIFY_IS_APPROX(inverse, expected_inverse);

  Vector2 product = values * factor;
  Vector2 expected_product;
  expected_product << values.coeff(0) * factor, values.coeff(1) * factor;
  VERIFY_IS_APPROX(product, expected_product);

  Vector2 conjugate_product = values.conjugate().cwiseProduct(Vector2::Constant(factor));
  Vector2 expected_conjugate_product;
  expected_conjugate_product << numext::conj(values.coeff(0)) * factor, numext::conj(values.coeff(1)) * factor;
  VERIFY_IS_APPROX(conjugate_product, expected_conjugate_product);

  // numext::divide guards against intermediate overflow (c^2 + d^2) near max representable bounds.
  // The divisor stays below max_val / 4: above that its reciprocal is subnormal, and GCC's
  // -ffast-math vector division (rcpps plus a Newton step) flushes it to zero.
  const RealScalar max_val = (std::numeric_limits<RealScalar>::max)();
  const RealScalar big_val = max_val / RealScalar(16);
  const Scalar big_num(RealScalar(0.9) * big_val, RealScalar(0));
  const Scalar big_denom(big_val, RealScalar(0));
  const Scalar safe_div_result = numext::divide(big_num, big_denom);
  VERIFY((numext::isfinite)(safe_div_result.real()));
  VERIFY_IS_APPROX(safe_div_result, Scalar(RealScalar(0.9), RealScalar(0)));
  VERIFY_IS_APPROX(numext::divide(RealScalar(0.9) * max_val, max_val), RealScalar(0.9));

  // Verify vectorized expression division (vec / denom) on numbers that would cause
  // intermediate overflow (c^2 + d^2 > max) in naive complex division.
  const RealScalar intermediate_scale = numext::sqrt(max_val) * RealScalar(2);
  const Scalar intermediate_num(RealScalar(0.9) * intermediate_scale, RealScalar(0));
  const Scalar intermediate_denom(intermediate_scale, RealScalar(0));
  Matrix<Scalar, Dynamic, 1> vec(8);
  vec.fill(intermediate_num);
  Matrix<Scalar, Dynamic, 1> vec_div = vec / intermediate_denom;
  for (Index i = 0; i < vec.size(); ++i) {
    VERIFY((numext::isfinite)(vec_div(i).real()));
    VERIFY_IS_APPROX(vec_div(i), Scalar(RealScalar(0.9), RealScalar(0)));
  }
}

template <typename RealScalar>
void check_complex_packet_math_functions() {
  using Scalar = std::complex<RealScalar>;
  using Vector4 = Matrix<Scalar, 4, 1>;

  Vector4 values;
  values << Scalar(RealScalar(0.53645928880954319), RealScalar(-0.60489662966980218)),
      Scalar(RealScalar(0.25774142970757641), RealScalar(0.10793998506041591)),
      Scalar(RealScalar(-0.83239073966000054), RealScalar(0.026801457199547407)),
      Scalar(RealScalar(1.6611441458336193), RealScalar(-0.21123424512127231));

  Vector4 sqrt_result = values.array().sqrt();
  Vector4 log_result = values.array().log();
  Vector4 exp_result = values.array().exp();

  Vector4 expected_sqrt, expected_log, expected_exp;
  for (Index i = 0; i < values.size(); ++i) {
    expected_sqrt[i] = std::sqrt(values[i]);
    expected_log[i] = std::log(values[i]);
    expected_exp[i] = std::exp(values[i]);
  }

  VERIFY_IS_APPROX(sqrt_result, expected_sqrt);
  VERIFY_IS_APPROX(log_result, expected_log);
  VERIFY_IS_APPROX(exp_result, expected_exp);
}

template <typename RealScalar>
void check_complex_householder_qr() {
  using Scalar = std::complex<RealScalar>;
  using Matrix32 = Matrix<Scalar, 3, 2>;
  using Matrix22 = Matrix<Scalar, 2, 2>;

  Matrix32 mat;
  mat << Scalar(RealScalar(0.59688070592806186), RealScalar(-0.21123424512127231)),
      Scalar(RealScalar(0.83239073966000054), RealScalar(0.026801457199547407)),
      Scalar(RealScalar(0.53645928880954319), RealScalar(-0.60489662966980218)),
      Scalar(RealScalar(0.21393694076781777), RealScalar(0.43459380694554106)),
      Scalar(RealScalar(0.25774142970757641), RealScalar(0.10793998506041591)),
      Scalar(RealScalar(0.60835279200704262), RealScalar(-0.51422689790671194));

  HouseholderQR<Matrix32> qr(mat);
  Matrix32 q = qr.householderQ() * Matrix32::Identity();
  Matrix22 r = qr.matrixQR().template topRows<2>().template triangularView<Upper>();
  VERIFY_IS_APPROX(mat, q * r);
}

// The complex self-adjoint 1-norm takes sqrt(re^2 + im^2) in packets and falls back to the scalar
// abs when a component is too large to square. The decision must not depend on an infinity or
// NaN surviving fast-math: an approximate packet sqrt can turn the overflow into a NaN that a
// maximum then discards. Large entries both on the diagonal and inside a packet of a column.
template <typename RealScalar>
void check_complex_selfadjoint_l1norm() {
  using Scalar = std::complex<RealScalar>;
  RealScalar big = numext::sqrt(NumTraits<RealScalar>::highest()) * RealScalar(1e3);
  Matrix<Scalar, 8, 8> m = Matrix<Scalar, 8, 8>::Identity() * big;
  VERIFY_IS_APPROX(m.template selfadjointView<Lower>().l1Norm(), big);
  VERIFY_IS_APPROX(m.template selfadjointView<Upper>().l1Norm(), big);
  m.setIdentity();
  m(2, 0) = m(0, 2) = Scalar(big, big);
  RealScalar expected = numext::abs(Scalar(big, big)) + RealScalar(1);
  VERIFY_IS_APPROX(m.template selfadjointView<Lower>().l1Norm(), expected);
  VERIFY_IS_APPROX(m.template selfadjointView<Upper>().l1Norm(), expected);
}

template <typename RealScalar>
void check_complex_fastmath() {
  check_complex_rowmajor_adjoint_product<RealScalar>();
  check_complex_packet_arithmetic<RealScalar>();
  check_complex_packet_math_functions<RealScalar>();
  check_complex_householder_qr<RealScalar>();
  check_complex_selfadjoint_l1norm<RealScalar>();
}

// The packet implementations of these functions manipulate signs of non-zero values through a
// -0.0 bitmask. Under fast-math flags compilers consider -0.0 and +0.0 interchangeable and may
// substitute one such constant for the other (GCC's value numbering does this on RISC-V), which
// silently zeroes the mask unless it is constructed from integer bits (see psignmask and
// https://gitlab.com/libeigen/eigen/-/merge_requests/2698). Sign handling of *non-zero* inputs
// and outputs is not relaxed by fast-math, so these checks must hold.
template <typename Scalar>
void check_sign_dependent_functions() {
  using ArrayType = Array<Scalar, Dynamic, 1>;
  const Index n = 64;

  ArrayType ya(n), xa(n);
  for (Index i = 0; i < n; ++i) {
    Scalar s = Scalar(1) + Scalar(i) / Scalar(n);
    // All four quadrants.
    ya[i] = (i & 1) ? s : -s;
    xa[i] = (i & 2) ? Scalar(2) * s : Scalar(-2) * s;
  }
  // Cover the |x| == |y| special path of patan2 in all four quadrants.
  ya[0] = Scalar(1), xa[0] = Scalar(1);
  ya[1] = Scalar(1), xa[1] = Scalar(-1);
  ya[2] = Scalar(-1), xa[2] = Scalar(1);
  ya[3] = Scalar(-1), xa[3] = Scalar(-1);

  const ArrayType atan2_result = ya.atan2(xa);
  for (Index i = 0; i < n; ++i) {
    VERIFY_IS_APPROX(atan2_result[i], std::atan2(ya[i], xa[i]));
  }

  // Mixed-sign inputs, no exact zeros, |w| up to ~3 to hit the small- and large-|x| branches.
  const ArrayType w = ArrayType::LinSpaced(n, Scalar(-3), Scalar(3)) + Scalar(0.017);
  const ArrayType w_unit = w / Scalar(3.2);  // in (-1, 1) for atanh
  const ArrayType atan_result = w.atan();
  const ArrayType sinh_result = w.sinh();
  const ArrayType tanh_result = w.tanh();
  const ArrayType asinh_result = w.asinh();
  const ArrayType atanh_result = w_unit.atanh();
  const ArrayType cbrt_result = w.cbrt();
  for (Index i = 0; i < n; ++i) {
    VERIFY_IS_APPROX(atan_result[i], std::atan(w[i]));
    VERIFY_IS_APPROX(sinh_result[i], std::sinh(w[i]));
    VERIFY_IS_APPROX(tanh_result[i], std::tanh(w[i]));
    VERIFY_IS_APPROX(asinh_result[i], std::asinh(w[i]));
    VERIFY_IS_APPROX(atanh_result[i], std::atanh(w_unit[i]));
    VERIFY_IS_APPROX(cbrt_result[i], std::cbrt(w[i]));
  }
}

// Packet exp, expm1, log1p, asinh and acosh under -ffast-math, against exact references. A wrong value, not isfinite or
// isnan (which fast-math folds), shows an overflow or a cancellation. The inputs and the C library references go
// through volatiles: fast-math folds calls on constants and rewrites acosh and the like, even inside std::acosh.
using LongDoubleFunction = long double (*)(long double);

// Whether v is finite, from its bits: fast-math folds isfinite and compares as if a NaN or an infinity never occurred.
template <typename Scalar>
bool has_finite_bits(Scalar v) {
  using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  constexpr int kMantissaBits = NumTraits<Scalar>::digits() - 1;
  constexpr int kExponentBits = int(8 * sizeof(Scalar)) - 1 - kMantissaBits;
  const Bits exponent_mask = static_cast<Bits>(((Bits(1) << kExponentBits) - 1) << kMantissaBits);
  return (numext::bit_cast<Bits>(v) & exponent_mask) != exponent_mask;
}

template <typename Scalar, typename Func>
bool check_fastmath_accuracy(const char* name, const std::vector<Scalar>& values, Func func,
                             LongDoubleFunction reference) {
  LongDoubleFunction volatile ref = reference;
  using ArrayType = Array<Scalar, Dynamic, 1>;
  // Long enough for several packets of any size and a remainder.
  const Index n = 3 * 64 + 5;
  // The kernels are within 2 ulp, and the limit leaves room for the reciprocal approximation that fast-math divides
  // by.
  const long double tolerance = 4 * static_cast<long double>(NumTraits<Scalar>::epsilon());
  ArrayType x(n);
  for (Index i = 0; i < n; ++i) {
    const volatile Scalar* v = &values[static_cast<size_t>(i) % values.size()];
    x[i] = *v;
  }
  const ArrayType y = func(x);
  bool passed = true;
  for (Index i = 0; i < n; ++i) {
    const long double expected = ref(static_cast<long double>(x[i]));
    const long double error = numext::abs(static_cast<long double>(y[i]) - expected);
    // An infinite reference would make the comparison below true for every finite y.
    if (!has_finite_bits(static_cast<Scalar>(expected)) || !has_finite_bits(y[i]) ||
        !(error <= tolerance * numext::abs(expected))) {
      passed = false;
      if (static_cast<size_t>(i) < values.size()) {
        std::cerr << std::setprecision(std::numeric_limits<Scalar>::max_digits10) << name << "(" << x[i]
                  << ") = " << y[i] << ", expected " << static_cast<Scalar>(expected) << "\n";
      }
    }
  }
  return passed;
}

template <typename Scalar>
void check_exp_family_accuracy() {
  using ArrayType = Array<Scalar, Dynamic, 1>;
  const bool is_float = std::is_same<Scalar, float>::value;
  // Small and moderate arguments, where expm1, log1p and asinh rely on the low bits of fl(1 + x).
  const Scalar tiny = is_float ? Scalar(1e-7) : Scalar(1e-10);
  const std::vector<Scalar> small_values = {tiny,        -tiny,       Scalar(1e-3), Scalar(-1e-3),
                                            Scalar(0.1), Scalar(0.7), Scalar(3)};
  // The largest finite results of exp and expm1 need a scale 2^m with m at the top of the exponent range.
  const Scalar exp_edge = is_float ? Scalar(88.38) : Scalar(709.1);
  const std::vector<Scalar> large_values = {
      exp_edge,    is_float ? Scalar(88.5) : Scalar(709.5), is_float ? Scalar(88.72) : Scalar(709.78), Scalar(10),
      Scalar(-10), is_float ? Scalar(-80) : Scalar(-700)};
  // Arguments up to the largest finite value, where asinh and acosh switch to log(2 x) without squaring x.
  const std::vector<Scalar> huge_values = {is_float ? Scalar(1e10) : Scalar(1e150),
                                           is_float ? Scalar(1e30) : Scalar(1e300),
                                           is_float ? Scalar(1e38) : Scalar(1e307), NumTraits<Scalar>::highest()};

  auto exp_fn = [](const ArrayType& x) -> ArrayType { return x.exp(); };
  LongDoubleFunction exp_ref = ::expl;
  auto expm1_fn = [](const ArrayType& x) -> ArrayType { return x.expm1(); };
  LongDoubleFunction expm1_ref = ::expm1l;
  auto log1p_fn = [](const ArrayType& x) -> ArrayType { return x.log1p(); };
  LongDoubleFunction log1p_ref = ::log1pl;
  auto asinh_fn = [](const ArrayType& x) -> ArrayType { return x.asinh(); };
  LongDoubleFunction asinh_ref = ::asinhl;
  auto acosh_fn = [](const ArrayType& x) -> ArrayType { return x.acosh(); };
  LongDoubleFunction acosh_ref = ::acoshl;
  auto sinh_fn = [](const ArrayType& x) -> ArrayType { return x.sinh(); };
  LongDoubleFunction sinh_ref = ::sinhl;
  auto cosh_fn = [](const ArrayType& x) -> ArrayType { return x.cosh(); };
  LongDoubleFunction cosh_ref = ::coshl;

  // Every call runs, so that one failure report lists all the functions that are wrong.
  bool passed = true;
  passed &= check_fastmath_accuracy("exp", small_values, exp_fn, exp_ref);
  passed &= check_fastmath_accuracy("exp", large_values, exp_fn, exp_ref);
  passed &= check_fastmath_accuracy("expm1", small_values, expm1_fn, expm1_ref);
  passed &= check_fastmath_accuracy("expm1", large_values, expm1_fn, expm1_ref);
  passed &= check_fastmath_accuracy("log1p", small_values, log1p_fn, log1p_ref);
  passed &= check_fastmath_accuracy("log1p", huge_values, log1p_fn, log1p_ref);
  passed &= check_fastmath_accuracy("asinh", small_values, asinh_fn, asinh_ref);
  passed &= check_fastmath_accuracy("asinh", huge_values, asinh_fn, asinh_ref);
  std::vector<Scalar> negative_huge_values;
  for (Scalar v : huge_values) negative_huge_values.push_back(-v);
  passed &= check_fastmath_accuracy("asinh", negative_huge_values, asinh_fn, asinh_ref);

  // acosh is defined for x >= 1: x = 1 + tiny, 1 + 1e-3, ... and the huge values.
  std::vector<Scalar> acosh_values = huge_values;
  for (Scalar v : small_values) {
    if (v > Scalar(0)) acosh_values.push_back(Scalar(1) + v);
  }
  passed &= check_fastmath_accuracy("acosh", acosh_values, acosh_fn, acosh_ref);

  // sinh and cosh scale exp(r) by 2^(n - 1) from the same reduction as expm1, and are finite up to ln(2 highest).
  std::vector<Scalar> hyperbolic_values = large_values;
  // sinhl and coshl overflow at 710.3 where long double has the range of double (MSVC, Apple arm64).
  if (std::numeric_limits<long double>::max_exponent > std::numeric_limits<Scalar>::max_exponent) {
    hyperbolic_values.push_back(is_float ? Scalar(89.2) : Scalar(710.3));
    hyperbolic_values.push_back(is_float ? Scalar(-89.2) : Scalar(-710.3));
  }
  hyperbolic_values.push_back(Scalar(-1.5));
  passed &= check_fastmath_accuracy("sinh", hyperbolic_values, sinh_fn, sinh_ref);
  passed &= check_fastmath_accuracy("cosh", hyperbolic_values, cosh_fn, cosh_ref);
  VERIFY(passed);
}

// For x = 1000, the packet logistic computes exp(x) = inf, which it must recognize and map to 1, though fast-math
// deletes a comparison with an infinity literal.
long double logistic_reference(long double x) { return 1 / (1 + ::expl(-x)); }

template <typename Scalar>
void check_logistic_overflow() {
  using ArrayType = Array<Scalar, Dynamic, 1>;
  const std::vector<Scalar> values = {Scalar(1000), Scalar(-5), Scalar(100), Scalar(-50), Scalar(0.5)};
  VERIFY(check_fastmath_accuracy(
      "logistic", values, [](const ArrayType& x) -> ArrayType { return x.logistic(); }, logistic_reference));
}

// Finite arguments of sqrt and rsqrt, whose Newton steps compare the argument with an infinity.
long double rsqrt_reference(long double x) { return 1 / ::sqrtl(x); }

// bfloat16 divides in float, where -ffast-math may compute 1 / (1 + e) by an approximation that flushes it to 0 once
// 1 + e exceeds 2^126: logistic(88) has e = exp(88) = 1.6e38 and is 1.
void check_logistic_bfloat16() {
  const float values[] = {88, 89, 100, 30, -30, 0, 50, -50};
  const int n = 37;
  Array<bfloat16, Dynamic, 1> x(n);
  for (int i = 0; i < n; ++i) {
    const volatile float* v = &values[i % 8];
    x[i] = bfloat16(*v);
  }
  const Array<bfloat16, Dynamic, 1> y = x.logistic();
  LongDoubleFunction volatile reference = logistic_reference;
  for (int i = 0; i < n; ++i) {
    const long double expected = reference(static_cast<long double>(static_cast<float>(x[i])));
    const long double tolerance = 2 * static_cast<long double>(NumTraits<bfloat16>::epsilon());
    VERIFY(has_finite_bits(static_cast<float>(y[i])));
    VERIFY(numext::abs(static_cast<long double>(static_cast<float>(y[i])) - expected) <= tolerance * expected);
  }
}

template <typename Scalar>
void check_sqrt_rsqrt_finite() {
  using ArrayType = Array<Scalar, Dynamic, 1>;
  const std::vector<Scalar> values = {Scalar(4), Scalar(9), Scalar(2), Scalar(1e-3), Scalar(7.5), Scalar(1e20)};
  bool passed = true;
  passed &= check_fastmath_accuracy(
      "sqrt", values, [](const ArrayType& x) -> ArrayType { return x.sqrt(); }, ::sqrtl);
  passed &= check_fastmath_accuracy(
      "rsqrt", values, [](const ArrayType& x) -> ArrayType { return x.rsqrt(); }, rsqrt_reference);
  VERIFY(passed);
}

// exp(x + iy) with exp(x) above the largest finite value and cos(y) or sin(y) small enough that the component is
// finite: the packet code splits exp(x) = 2^n exp(r) for those, and finds them by the overflow of exp(x), which
// fast-math must not hide.
template <typename Scalar>
void check_complex_exp_overflow() {
  using Complex = std::complex<Scalar>;
  const bool is_float = std::is_same<Scalar, float>::value;
  // Real parts at x_overflow of pexp_complex, the bound of the split, which lies 4 epsilon below ln(highest), and next
  // to it; ln(highest), where exp(x) is the largest finite value; and larger ones. Up to x_overflow exp(x) is computed
  // by the packet exp.
  using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  const Scalar x_overflow = Scalar(NumTraits<Scalar>::max_exponent()) * Scalar(EIGEN_LN2) *
                            (Scalar(1) - Scalar(4) * NumTraits<Scalar>::epsilon());
  const Bits x_overflow_bits = numext::bit_cast<Bits>(x_overflow);
  const std::vector<Scalar> x_values = {
      is_float ? Scalar(89) : Scalar(710),
      is_float ? Scalar(92) : Scalar(720),
      is_float ? Scalar(95) : Scalar(735),
      numext::bit_cast<Scalar>(Bits(x_overflow_bits - 1)),
      x_overflow,
      numext::bit_cast<Scalar>(Bits(x_overflow_bits + 1)),
      static_cast<Scalar>(::logl(static_cast<long double>(NumTraits<Scalar>::highest())))};
  const Scalar y_values[] = {Scalar(1.5707963267948966), Scalar(3.141592653589793), Scalar(-1.5707963267948966),
                             Scalar(0.7)};
  const Index n = 4 * static_cast<Index>(x_values.size());
  Array<Complex, Dynamic, 1> z(n);
  for (Index i = 0; i < n; ++i) {
    const volatile Scalar* x = &x_values[static_cast<size_t>(i / 4)];
    const volatile Scalar* y = &y_values[i % 4];
    const Scalar re = *x, im = *y;  // std::complex takes const Scalar&, which cannot bind to a volatile lvalue.
    z[i] = Complex(re, im);
  }
  const Array<Complex, Dynamic, 1> w = z.exp();
  // The reference is exp(x) cos(y) = 2^n m with m = exp(r) cos(y), x = n ln2 + r, |r| <= ln2 / 2, in double: exp(x)
  // itself overflows, and long double is no wider than double with MSVC and on Apple arm64. The constants are those of
  // fdlibm's exp: ln2_hi has 32 trailing zero bits, so n ln2_hi is exact, and the first difference goes through a
  // volatile because fast-math fuses n ln2_hi + n ln2_lo into the rounded n (ln2_hi + ln2_lo).
  const volatile double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10;
  const double highest = static_cast<double>(NumTraits<Scalar>::highest());
  const double tolerance = 8 * static_cast<double>(NumTraits<Scalar>::epsilon());
  for (Index i = 0; i < n; ++i) {
    const double x = static_cast<double>(z[i].real()), y = static_cast<double>(z[i].imag());
    const int exponent = static_cast<int>(std::floor(x / EIGEN_LN2 + 0.5));
    const volatile double reduced_hi = x - exponent * ln2_hi;
    const double r = reduced_hi - exponent * ln2_lo;
    const double m = std::exp(r);
    const double mantissa[2] = {m * std::cos(y), m * std::sin(y)};
    const Scalar actual[2] = {w[i].real(), w[i].imag()};
    for (int k = 0; k < 2; ++k) {
      // A component above the largest finite value is infinite, which fast-math cannot be asked about. Compared before
      // scaling, so that no infinity occurs.
      if (numext::abs(mantissa[k]) <= std::ldexp(highest, -exponent)) {
        const double expected = std::ldexp(mantissa[k], exponent);
        VERIFY(has_finite_bits(actual[k]));
        VERIFY(numext::abs(static_cast<double>(actual[k]) - expected) <= tolerance * numext::abs(expected));
      }
    }
  }
}

EIGEN_DECLARE_TEST(fastmath) {
  std::cout << "*** float *** \n\n";
  check_inf_nan<float>(true);
  std::cout << "*** double ***\n\n";
  check_inf_nan<double>(true);
  std::cout << "*** long double *** \n\n";
  check_inf_nan<long double>(true);

  check_inf_nan<float>(false);
  check_inf_nan<double>(false);
  check_inf_nan<long double>(false);

  CALL_SUBTEST_1(check_complex_fastmath<float>());
  CALL_SUBTEST_2(check_complex_fastmath<double>());
  CALL_SUBTEST_3(check_sign_dependent_functions<float>());
  CALL_SUBTEST_4(check_sign_dependent_functions<double>());
  CALL_SUBTEST_5(check_exp_family_accuracy<float>());
  CALL_SUBTEST_6(check_exp_family_accuracy<double>());
  CALL_SUBTEST_7(check_sqrt_rsqrt_finite<float>());
  CALL_SUBTEST_8(check_sqrt_rsqrt_finite<double>());
  CALL_SUBTEST_9(check_logistic_overflow<float>());
  CALL_SUBTEST_10(check_logistic_overflow<double>());
  CALL_SUBTEST_11(check_logistic_bfloat16());
  CALL_SUBTEST_12(check_complex_exp_overflow<float>());
  CALL_SUBTEST_13(check_complex_exp_overflow<double>());
}
