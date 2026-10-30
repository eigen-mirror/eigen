// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christoph Hertzberg <chtz@informatik.uni-bremen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "fp_control.h"
#ifndef small
#define small char  // Windows rpcndr.h defines this before users include AutoDiff.
#define EIGEN_TEST_DEFINED_SMALL
#endif
#include <contrib/Eigen/AutoDiff>
#ifdef EIGEN_TEST_DEFINED_SMALL
#undef small
#undef EIGEN_TEST_DEFINED_SMALL
#endif

/*
 * In this file scalar derivations are tested for correctness.
 * TODO add more tests!
 */

template <typename Scalar>
void check_atan2() {
  typedef Matrix<Scalar, 1, 1> Deriv1;
  typedef AutoDiffScalar<Deriv1> AD;

  AD x(internal::random<Scalar>(-3.0, 3.0), Deriv1::UnitX());

  using std::exp;
  Scalar r = exp(internal::random<Scalar>(-10, 10));

  AD s = sin(x), c = cos(x);
  AD res = atan2(r * s, r * c);

  VERIFY_IS_APPROX(res.value(), x.value());
  VERIFY_IS_APPROX(res.derivatives(), x.derivatives());

  res = atan2(r * s + 0, r * c + 0);
  VERIFY_IS_APPROX(res.value(), x.value());
  VERIFY_IS_APPROX(res.derivatives(), x.derivatives());
}

template <typename Scalar>
void check_hyperbolic_functions() {
  using std::cosh;
  using std::sinh;
  using std::tanh;
  typedef Matrix<Scalar, 1, 1> Deriv1;
  typedef AutoDiffScalar<Deriv1> AD;
  Deriv1 p = Deriv1::Random();
  AD val(p.x(), Deriv1::UnitX());

  Scalar cosh_px = std::cosh(p.x());
  AD res1 = tanh(val);
  VERIFY_IS_APPROX(res1.value(), std::tanh(p.x()));
  VERIFY_IS_APPROX(res1.derivatives().x(), Scalar(1.0) / (cosh_px * cosh_px));

  AD res2 = sinh(val);
  VERIFY_IS_APPROX(res2.value(), std::sinh(p.x()));
  VERIFY_IS_APPROX(res2.derivatives().x(), cosh_px);

  AD res3 = cosh(val);
  VERIFY_IS_APPROX(res3.value(), cosh_px);
  VERIFY_IS_APPROX(res3.derivatives().x(), std::sinh(p.x()));

  // Check constant values.
  const Scalar sample_point = Scalar(1) / Scalar(3);
  val = AD(sample_point, Deriv1::UnitX());
  res1 = tanh(val);
  VERIFY_IS_APPROX(res1.derivatives().x(), Scalar(0.896629559604914));

  res2 = sinh(val);
  VERIFY_IS_APPROX(res2.derivatives().x(), Scalar(1.056071867829939));

  res3 = cosh(val);
  VERIFY_IS_APPROX(res3.derivatives().x(), Scalar(0.339540557256150));
}

template <typename Scalar>
void check_limits_specialization() {
  typedef Eigen::Matrix<Scalar, 1, 1> Deriv;
  typedef Eigen::AutoDiffScalar<Deriv> AD;

  typedef std::numeric_limits<AD> A;
  typedef std::numeric_limits<Scalar> B;

  // workaround "unused typedef" warning:
  VERIFY(!bool(std::is_same<B, A>::value));

  VERIFY(bool(std::is_base_of<B, A>::value));
}

// Regression for issue #1870: min(ADS, ADS) and max(ADS, ADS) must agree on
// which argument wins on a tie.  Previously min returned the second argument
// while max returned the first, so the two could yield different derivatives
// for inputs with equal values but different derivative vectors.
template <typename Scalar>
void check_min_max_equal_values() {
  typedef Eigen::Matrix<Scalar, 2, 1> Deriv2;
  typedef Eigen::AutoDiffScalar<Deriv2> AD;

  AD a(Scalar(1), Deriv2(Scalar(1), Scalar(0)));
  AD b(Scalar(1), Deriv2(Scalar(0), Scalar(1)));

  // Parenthesized form avoids the min/max macros installed by test/main.h
  // and lets ADL pick up the AutoDiffScalar overloads.
  VERIFY_IS_EQUAL((min)(a, b).derivatives(), a.derivatives());
  VERIFY_IS_EQUAL((max)(a, b).derivatives(), a.derivatives());
  VERIFY_IS_EQUAL((min)(b, a).derivatives(), b.derivatives());
  VERIFY_IS_EQUAL((max)(b, a).derivatives(), b.derivatives());
}

template <typename Derivatives>
void check_atan() {
  using Scalar = typename Derivatives::Scalar;
  using AD = AutoDiffScalar<Derivatives>;
  using std::atan;
  const Scalar tolerance = Scalar(8) * NumTraits<Scalar>::epsilon();
  Derivatives seed(2);
  seed << Scalar(2), Scalar(-3);
  const Scalar values[] = {Scalar(-3),
                           Scalar(-1),
                           Scalar(-0.5),
                           Scalar(-0.0),
                           Scalar(0),
                           Scalar(0.5),
                           Scalar(1),
                           Scalar(3),
                           std::nextafter(Scalar(1), Scalar(0)),
                           std::nextafter(Scalar(1), Scalar(2)),
                           internal::random<Scalar>(Scalar(-8), Scalar(8))};
  for (Scalar value : values) {
    AD x(value, seed);
    const AD result = atan(x);
    const AD dispatched = numext::atan(x);
    VERIFY_IS_EQUAL(result.value(), atan(value));
    VERIFY_IS_EQUAL(bool((numext::signbit)(result.value())), bool((numext::signbit)(atan(value))));
    for (Index i = 0; i < seed.size(); ++i) {
      const long double v = static_cast<long double>(value);
      const Scalar expected = static_cast<Scalar>(static_cast<long double>(seed(i)) / (1 + v * v));
      VERIFY(numext::abs(result.derivatives()(i) - expected) <= tolerance * numext::abs(expected));
    }
    VERIFY_IS_EQUAL(dispatched.value(), result.value());
    VERIFY_IS_EQUAL(dispatched.derivatives(), result.derivatives());
    const AutoDiffScalar<Derivatives&> referenced(value, seed);
    const AD from_reference = atan(referenced);
    VERIFY_IS_EQUAL(from_reference.derivatives(), result.derivatives());
    const AD composed = atan(Scalar(2) * x - x);
    VERIFY_IS_EQUAL(composed.value(), result.value());
    VERIFY_IS_EQUAL(composed.derivatives(), result.derivatives());
    const auto deferred = atan(x);
    const Derivatives derivatives = deferred.derivatives();
    VERIFY_IS_EQUAL(derivatives, result.derivatives());
  }

  // The derivative stays normal even though x*x overflows and its reciprocal underflows.
  const Scalar largest = (NumTraits<Scalar>::highest)();
  for (Scalar sign : {Scalar(-1), Scalar(1)}) {
    seed << largest, -largest;
    const AD result = atan(AD(sign * (largest / Scalar(4)), seed));
    const Scalar expected = Scalar(16) / largest;
    VERIFY((numext::isfinite)(result.derivatives()(0)));
    VERIFY(numext::abs(result.derivatives()(0) / expected - Scalar(1)) <= tolerance);
    VERIFY(numext::abs(result.derivatives()(1) / expected + Scalar(1)) <= tolerance);
  }

  seed << Scalar(2), Scalar(-3);
  for (Scalar sign : {Scalar(-1), Scalar(1)}) {
    const AD result = atan(AD(sign * NumTraits<Scalar>::infinity(), seed));
    VERIFY_IS_EQUAL(result.value(), atan(sign * NumTraits<Scalar>::infinity()));
    VERIFY(result.derivatives().isZero());
  }
  const AD nan_result = atan(AD(NumTraits<Scalar>::quiet_NaN(), seed));
  VERIFY((numext::isnan)(nan_result.value()));
  VERIFY((numext::isnan)(nan_result.derivatives()(0)));
  VERIFY((numext::isnan)(nan_result.derivatives()(1)));

  const AD constant = atan(AD(Scalar(0.5)));
  VERIFY(constant.derivatives().isZero());
}

template <typename Scalar>
void check_atan_second_derivative() {
  using FirstDerivative = AutoDiffScalar<Matrix<Scalar, 1, 1>>;
  using SecondDerivative = AutoDiffScalar<Matrix<FirstDerivative, 1, 1>>;
  for (Scalar value : {Scalar(0.5), Scalar(2)}) {
    SecondDerivative x;
    x.value() = FirstDerivative(value);
    x.value().derivatives()(0) = Scalar(1);
    x.derivatives()(0) = FirstDerivative(Scalar(1));
    const SecondDerivative result = atan(x);
    const Scalar denominator = Scalar(1) + value * value;
    const Scalar expected = -Scalar(2) * value / (denominator * denominator);
    const Scalar tolerance = Scalar(16) * NumTraits<Scalar>::epsilon();
    VERIFY(numext::abs(result.value().derivatives()(0) - Scalar(1) / denominator) <= tolerance);
    VERIFY(numext::abs(result.derivatives()(0).value() - Scalar(1) / denominator) <= tolerance);
    VERIFY(numext::abs(result.derivatives()(0).derivatives()(0) - expected) <= tolerance);
  }
}

template <typename Scalar>
void check_issue234_operations() {
  using Derivatives = Matrix<Scalar, 2, 1>;
  using AD = AutoDiffScalar<Derivatives>;
  const AD x(Scalar(3), Derivatives::UnitX());
  const AD y = Scalar(1) - Scalar(2) * x;
  const AD opposite = -x;
  VERIFY(x < Scalar(4));
  VERIFY(x < AD(Scalar(4)));
  VERIFY_IS_EQUAL(y.value(), Scalar(-5));
  VERIFY_IS_EQUAL(y.derivatives(), Derivatives(Scalar(-2), Scalar(0)));
  VERIFY_IS_EQUAL(opposite.value(), Scalar(-3));
  VERIFY_IS_EQUAL(opposite.derivatives(), -x.derivatives());
  Matrix<AD, 2, 1> v;
  v << x, AD(Scalar(4), Derivatives::UnitY());
  const AD length = v.norm();
  const Scalar tolerance = Scalar(8) * NumTraits<Scalar>::epsilon();
  VERIFY(numext::abs(length.value() - Scalar(5)) <= tolerance * Scalar(5));
  VERIFY(numext::abs(length.derivatives()(0) - Scalar(3) / Scalar(5)) <= tolerance);
  VERIFY(numext::abs(length.derivatives()(1) - Scalar(4) / Scalar(5)) <= tolerance);
}

template <typename Derivatives, typename Function, typename NumextFunction, typename Reference,
          typename FirstDerivative, typename SecondDerivative>
void check_unary_math(Function function, NumextFunction numext_function, Reference reference,
                      FirstDerivative first_derivative, SecondDerivative second_derivative,
                      std::initializer_list<typename Derivatives::Scalar> values) {
  using Scalar = typename Derivatives::Scalar;
  using AD = AutoDiffScalar<Derivatives>;
  // Up to two square roots and divisions, compared with long-double analytic derivatives.
  const long double tolerance = 16.L * static_cast<long double>(NumTraits<Scalar>::epsilon());
  Derivatives seed(2);
  seed << Scalar(1), Scalar(-2);
  for (Scalar value : values) {
    const AD x(value, seed);
    const auto deferred = function(x);
    const AD result = deferred;
    VERIFY_IS_EQUAL(result.value(), reference(value));
    const AD dispatched = numext_function(x);
    VERIFY_IS_EQUAL(dispatched.value(), result.value());
    for (Index i = 0; i < 2; ++i) {
      const long double expected =
          static_cast<long double>(seed(i)) * first_derivative(static_cast<long double>(value));
      const long double error = std::abs(static_cast<long double>(result.derivatives()(i)) - expected);
      VERIFY(error <= tolerance * std::abs(expected));
      VERIFY_IS_EQUAL(dispatched.derivatives()(i), result.derivatives()(i));
    }
    const AD expression = function(x + x);
    const AD doubled = function(AD(Scalar(2) * value, Scalar(2) * seed));
    VERIFY_IS_EQUAL(expression.value(), doubled.value());
    VERIFY_IS_EQUAL(expression.derivatives(), doubled.derivatives());
    AutoDiffScalar<Derivatives&> reference_input(value, seed);
    const AD referenced = function(reference_input);
    VERIFY_IS_EQUAL(referenced.derivatives(), result.derivatives());
    const AD constant = function(AD(value));
    VERIFY(constant.derivatives().isZero());

    using Inner = AutoDiffScalar<Matrix<Scalar, 1, 1>>;
    using Outer = AutoDiffScalar<Matrix<Inner, 1, 1>>;
    Outer nested;
    nested.value() = Inner(value);
    nested.value().derivatives()(0) = Scalar(1);
    nested.derivatives()(0) = Inner(Scalar(1));
    const Outer twice = function(nested);
    const long double expected_second = second_derivative(static_cast<long double>(value));
    VERIFY(std::abs(static_cast<long double>(twice.derivatives()(0).derivatives()(0)) - expected_second) <=
           tolerance * std::abs(expected_second));
  }
  const AD nan_result = function(AD(NumTraits<Scalar>::quiet_NaN(), seed));
  VERIFY((numext::isnan)(nan_result.value()));
  VERIFY((numext::isnan)(nan_result.derivatives()(0)));
}

template <typename Scalar, typename Function, typename Reference>
void check_unary_special_values(Function function, Reference reference) {
  using AD = AutoDiffScalar<Matrix<Scalar, 1, 1>>;
  const Scalar infinity = NumTraits<Scalar>::infinity();
  for (Scalar value :
       {Scalar(0), -Scalar(0), Scalar(-1), Scalar(1), -infinity, infinity, numext::nextafter(Scalar(-1), -infinity),
        numext::nextafter(Scalar(-1), Scalar(0)), numext::nextafter(Scalar(1), Scalar(0)),
        numext::nextafter(Scalar(1), infinity), std::numeric_limits<Scalar>::denorm_min()}) {
    const AD result = function(AD(value, Matrix<Scalar, 1, 1>::Ones()));
    const Scalar expected = reference(value);
    if ((numext::isnan)(expected)) {
      VERIFY((numext::isnan)(result.value()));
    } else {
      VERIFY_IS_EQUAL(result.value(), expected);
      VERIFY_IS_EQUAL(bool((numext::signbit)(result.value())), bool((numext::signbit)(expected)));
    }
  }
}

template <typename Scalar, typename Function, typename FirstDerivative>
void check_unary_large_derivative(Function function, FirstDerivative first_derivative,
                                  std::initializer_list<Scalar> values,
                                  Scalar seed = NumTraits<Scalar>::highest() / Scalar(4)) {
  using AD = AutoDiffScalar<Matrix<Scalar, 1, 1>>;
  for (Scalar value : values) {
    const AD result = function(AD(value, Matrix<Scalar, 1, 1>::Constant(seed)));
    const long double expected = static_cast<long double>(seed) * first_derivative(static_cast<long double>(value));
    // The scaled seed keeps the reference normal even when the unscaled derivative is tiny.
    VERIFY(std::abs(static_cast<long double>(result.derivatives()(0)) - expected) <=
           16.L * static_cast<long double>(NumTraits<Scalar>::epsilon()) * std::abs(expected));
  }
}

template <typename Derivatives>
void check_subnormal_math_derivatives() {
  using Scalar = typename Derivatives::Scalar;
  using AD = AutoDiffScalar<Derivatives>;
  if (!(subnormalInputProbe<Scalar>() > Scalar(0)) || !(underflowProbe<Scalar>() > Scalar(0))) return;
  const Scalar denorm = std::numeric_limits<Scalar>::denorm_min();
  const Scalar normal = (std::numeric_limits<Scalar>::min)();
  const Scalar boundary = numext::nextafter(Scalar(1), Scalar(0));
  const Scalar inner = numext::nextafter(boundary, Scalar(0));
  const long double epsilon = static_cast<long double>(NumTraits<Scalar>::epsilon());
  const auto check = [&](const auto& function, const auto& reference, std::initializer_list<Scalar> values) {
    for (Scalar value : values) {
      for (Scalar seed : {denorm, Scalar(2) * denorm, normal / Scalar(2), normal}) {
        Derivatives derivatives(2);
        derivatives << seed, -seed;
        const AD input(value, derivatives);
        const auto deferred = function(input);
        const AD result = deferred;
        const AutoDiffScalar<Derivatives&> referenced(value, derivatives);
        const AD from_reference = function(referenced);
        VERIFY_IS_EQUAL(result.derivatives(), from_reference.derivatives());
        for (Index i = 0; i < derivatives.size(); ++i) {
          const long double expected =
              reference(static_cast<long double>(derivatives(i)), static_cast<long double>(value));
          if ((std::isinf)(expected)) {
            VERIFY((numext::isinf)(result.derivatives()(i)));
            VERIFY_IS_EQUAL((std::signbit)(result.derivatives()(i)), (std::signbit)(expected));
          } else {
            // Allow ordinary rounding plus one subnormal quantum when the final result is tiny.
            const long double bound = 16 * epsilon * std::abs(expected) + static_cast<long double>(denorm);
            VERIFY(std::abs(static_cast<long double>(result.derivatives()(i)) - expected) <= bound);
          }
        }
      }
    }
  };
  check([](const auto& x) { return log10(x); },
        [](long double seed, long double x) { return (seed / x) / std::log(10.L); },
        {denorm, Scalar(2) * denorm, normal / Scalar(2), numext::nextafter(normal, Scalar(0)), normal,
         numext::nextafter(normal, Scalar(1)), Scalar(0.125), Scalar(0.5), Scalar(1)});
  check([](const auto& x) { return atanh(x); },
        [](long double seed, long double x) { return seed / ((1 - x) * (1 + x)); },
        {-Scalar(1), -boundary, -inner, Scalar(-0.5), Scalar(0), Scalar(0.5), inner, boundary, Scalar(1)});
}

template <typename Derivatives>
void check_additional_math() {
  using Scalar = typename Derivatives::Scalar;
#define CHECK_AUTODIFF_MATH(NAME, FIRST, SECOND, ...)                                            \
  check_unary_math<Derivatives>(                                                                 \
      [](const auto& x) {                                                                        \
        using std::NAME;                                                                         \
        return NAME(x);                                                                          \
      },                                                                                         \
      [](const auto& x) { return numext::NAME(x); }, [](Scalar x) { return std::NAME(x); },      \
      [](long double x) { return FIRST; }, [](long double x) { return SECOND; }, {__VA_ARGS__}); \
  check_unary_special_values<Scalar>(                                                            \
      [](const auto& x) {                                                                        \
        using std::NAME;                                                                         \
        return NAME(x);                                                                          \
      },                                                                                         \
      [](Scalar x) { return std::NAME(x); })
  CHECK_AUTODIFF_MATH(log1p, 1 / (1 + x), -1 / ((1 + x) * (1 + x)), Scalar(-0.25), Scalar(0), Scalar(1e-8), Scalar(0.5),
                      Scalar(3));
  CHECK_AUTODIFF_MATH(expm1, std::exp(x), std::exp(x), Scalar(-10), Scalar(-0.5), Scalar(0), Scalar(1e-8), Scalar(3));
  CHECK_AUTODIFF_MATH(log2, 1 / (x * std::log(2.L)), -1 / (x * x * std::log(2.L)), Scalar(0.125), Scalar(0.5),
                      Scalar(1), Scalar(8));
  CHECK_AUTODIFF_MATH(log10, 1 / (x * std::log(10.L)), -1 / (x * x * std::log(10.L)), Scalar(0.125), Scalar(0.5),
                      Scalar(1), Scalar(10));
  CHECK_AUTODIFF_MATH(cbrt, 1 / (3 * std::cbrt(x) * std::cbrt(x)), -2 / (9 * x * std::cbrt(x) * std::cbrt(x)),
                      Scalar(-8), Scalar(-0.125), Scalar(0.125), Scalar(8));
  CHECK_AUTODIFF_MATH(asinh, 1 / std::sqrt(1 + x * x), -x / std::pow(1 + x * x, 1.5L), Scalar(-8), Scalar(-0.5),
                      Scalar(0), Scalar(0.5), Scalar(8));
  CHECK_AUTODIFF_MATH(acosh, 1 / std::sqrt(x * x - 1), -x / std::pow(x * x - 1, 1.5L),
                      numext::nextafter(Scalar(1), Scalar(2)), Scalar(2), Scalar(8));
  CHECK_AUTODIFF_MATH(atanh, 1 / (1 - x * x), 2 * x / ((1 - x * x) * (1 - x * x)), Scalar(-0.25), Scalar(0),
                      Scalar(0.25));
#undef CHECK_AUTODIFF_MATH
  const Scalar large = NumTraits<Scalar>::highest() / Scalar(4);
  const Scalar large_seed = NumTraits<Scalar>::highest() * Scalar(0.6);
  check_unary_large_derivative<Scalar>([](const auto& x) { return atanh(x); },
                                       [](long double x) { return 1 / (1 - x * x); }, {Scalar(-0.5), Scalar(0.5)},
                                       large_seed);
  check_unary_large_derivative<Scalar>([](const auto& x) { return log10(x); },
                                       [](long double x) { return 1 / (x * std::log(10.L)); }, {Scalar(0.5)},
                                       large_seed);
  check_unary_large_derivative<Scalar>([](const auto& x) { return log1p(x); },
                                       [](long double x) { return 1 / (1 + x); }, {large});
  check_unary_large_derivative<Scalar>([](const auto& x) { return log2(x); },
                                       [](long double x) { return 1 / (x * std::log(2.L)); }, {large});
  check_unary_large_derivative<Scalar>([](const auto& x) { return log10(x); },
                                       [](long double x) { return 1 / (x * std::log(10.L)); }, {large});
  check_unary_large_derivative<Scalar>([](const auto& x) { return cbrt(x); },
                                       [](long double x) { return 1 / (3 * std::cbrt(x) * std::cbrt(x)); },
                                       {-large, large});
  check_unary_large_derivative<Scalar>([](const auto& x) { return asinh(x); },
                                       [](long double x) { return 1 / std::hypot(1.L, x); }, {-large, large});
  check_unary_large_derivative<Scalar>([](const auto& x) { return acosh(x); },
                                       [](long double x) { return 1 / (std::sqrt(x - 1) * std::sqrt(x + 1)); },
                                       {large});
  check_subnormal_math_derivatives<Derivatives>();
}

EIGEN_DECLARE_TEST(autodiff_scalar) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(check_atan2<float>());
    CALL_SUBTEST_2(check_atan2<double>());
    CALL_SUBTEST_3(check_hyperbolic_functions<float>());
    CALL_SUBTEST_4(check_hyperbolic_functions<double>());
    CALL_SUBTEST_5(check_limits_specialization<double>());
    CALL_SUBTEST_6(check_min_max_equal_values<float>());
    CALL_SUBTEST_7(check_min_max_equal_values<double>());
    CALL_SUBTEST_8(check_atan<Vector2f>());
    CALL_SUBTEST_8(check_atan<VectorXf>());
    CALL_SUBTEST_8(check_issue234_operations<float>());
    CALL_SUBTEST_8(check_atan_second_derivative<float>());
    CALL_SUBTEST_9(check_atan<Vector2d>());
    CALL_SUBTEST_9(check_atan<VectorXd>());
    CALL_SUBTEST_9(check_issue234_operations<double>());
    CALL_SUBTEST_9(check_atan_second_derivative<double>());
    CALL_SUBTEST_10(check_additional_math<Vector2f>());
    CALL_SUBTEST_10(check_additional_math<VectorXf>());
    CALL_SUBTEST_11(check_additional_math<Vector2d>());
    CALL_SUBTEST_11(check_additional_math<VectorXd>());
  }
}
