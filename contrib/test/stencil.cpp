// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <complex>

#include "main.h"
#include "fp_control.h"
#include <contrib/Eigen/NumericalDiff>

using Eigen::Array;
using Eigen::Index;
using Eigen::makeStencil;
using Eigen::Stencil;

// Independent oracle: for order-Derivative weights at x0 = 0 over points p_0..p_{Size-1},
// sum_i weights[i] * p_i^k == (k == Derivative ? Derivative! : 0) for every k = 0..Size-1. This
// holds for any polynomial reproduced exactly by an (Size-1)-degree interpolant, independently of
// how Stencil's recursion is implemented.
template <int Derivative, typename Scalar, int Size>
void verify_polynomial_exactness(const Array<Scalar, Size, 1> &points) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  const Stencil<Derivative, Scalar, Size> s(points);
  const Array<Scalar, Size, 1> w = s.weights();
  const Array<Scalar, Size, 1> p = s.points();

  for (Index i = 0; i < Size; ++i) {
    VERIFY_IS_EQUAL(p[i], points[i]);
    VERIFY((numext::isfinite)(w[i]));
  }

  Scalar fact(1);
  for (Index i = 2; i <= Derivative; ++i) fact *= Scalar(i);

  Scalar term[Size];
  for (Index i = 0; i < Size; ++i) term[i] = Scalar(1);

  for (Index k = 0; k < Size; ++k) {
    Scalar acc(0);
    RealScalar magSum(0);
    for (Index i = 0; i < Size; ++i) {
      const Scalar contribution = w[i] * term[i];
      acc += contribution;
      magSum += numext::abs(contribution);
    }
    const Scalar expected = (k == Derivative) ? fact : Scalar(0);
    const RealScalar tol =
        RealScalar(50) * RealScalar(Size) * (magSum + RealScalar(1)) * Eigen::NumTraits<RealScalar>::epsilon();
    VERIFY((numext::isfinite)(acc));
    VERIFY((numext::isfinite)(tol));
    VERIFY(numext::abs(acc - expected) <= tol);
    for (Index i = 0; i < Size; ++i) term[i] *= p[i];
  }
}

void test_textbook_central() {
  const Array<double, 3, 1> points{{-1.0, 0.0, 1.0}};
  const double tol = 10.0 * Eigen::NumTraits<double>::epsilon();

  const Stencil<1, double, 3> d1(points);
  const Array<double, 3, 1> w1 = d1.weights();
  VERIFY(numext::abs(w1[0] - (-0.5)) <= tol);
  VERIFY(numext::abs(w1[1] - 0.0) <= tol);
  VERIFY(numext::abs(w1[2] - 0.5) <= tol);

  const Stencil<2, double, 3> d2(points);
  const Array<double, 3, 1> w2 = d2.weights();
  VERIFY(numext::abs(w2[0] - 1.0) <= tol);
  VERIFY(numext::abs(w2[1] - (-2.0)) <= tol);
  VERIFY(numext::abs(w2[2] - 1.0) <= tol);
}

void test_textbook_forward() {
  const Array<double, 3, 1> points{{0.0, 1.0, 2.0}};
  const double tol = 10.0 * Eigen::NumTraits<double>::epsilon();

  const Stencil<1, double, 3> d1 = makeStencil<1>(points);
  const Array<double, 3, 1> w1 = d1.weights();
  VERIFY(numext::abs(w1[0] - (-1.5)) <= tol);
  VERIFY(numext::abs(w1[1] - 2.0) <= tol);
  VERIFY(numext::abs(w1[2] - (-0.5)) <= tol);
}

// Derivative == 0 (interpolation), and the boundary case Derivative == Size - 1, fall out of the
// same sweep since verify_polynomial_exactness checks every k = 0..Size-1 already.
void test_polynomial_exactness_uniform() {
  const Array<double, 5, 1> points{{-2.0, -1.0, 0.0, 1.0, 2.0}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<2>(points);
  verify_polynomial_exactness<3>(points);
  verify_polynomial_exactness<4>(points);
}

// Widely varying spacing: exercises the "arbitrary spacing" case without relying on any internal
// reordering to keep the recursion well conditioned.
void test_polynomial_exactness_nonuniform() {
  const Array<double, 5, 1> points{{-1000.0, -1.0, 0.0, 1.0, 1000.0}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<2>(points);
  verify_polynomial_exactness<4>(points);
}

// Points that do not include the evaluation point 0.
void test_polynomial_exactness_no_zero() {
  const Array<double, 4, 1> points{{-3.0, -1.0, 2.0, 5.0}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<3>(points);
}

void test_polynomial_exactness_float() {
  const Array<float, 4, 1> points{{-2.0f, -1.0f, 1.0f, 2.0f}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<3>(points);
}

template <int Derivative, typename Scalar, int Size>
void verify_power_of_two_scaling(const Array<Scalar, Size, 1>& points) {
  const Stencil<Derivative, Scalar, Size> reference(points);
  // Power-of-two scaling is exact while the recurrence stays in the normal range.
  for (const int exponent : {-27, -20, -40, 0, 20, 27, 40}) {
    const Scalar step = std::ldexp(Scalar(1), exponent);
    const Array<Scalar, Size, 1> scaledPoints = points * step;
    const Stencil<Derivative, Scalar, Size> scaled(scaledPoints);
    for (Index i = 0; i < Size; ++i) {
      const Scalar expected = std::ldexp(reference.weight(i), -Derivative * exponent);
      VERIFY((numext::isfinite)(expected));
      VERIFY((numext::isfinite)(scaled.weight(i)));
      VERIFY_IS_EQUAL(scaled.weight(i), expected);
    }
  }
}

template <typename Scalar>
void test_scaled_points() {
  const Array<Scalar, 9, 1> points{{-4, -3, -2, -1, 0, 1, 2, 3, 4}};
  const Array<Scalar, 7, 1> nonuniform{{3, -1, 0, 4, -4, 1, -2}};
  verify_power_of_two_scaling<0>(points);
  verify_power_of_two_scaling<1>(points);
  verify_power_of_two_scaling<2>(points);
  verify_power_of_two_scaling<0>(nonuniform);
  verify_power_of_two_scaling<1>(nonuniform);
  verify_power_of_two_scaling<2>(nonuniform);
}

void test_scaled_points_flush_modes() {
  Eigen::forEachFlushToZeroMode([](Eigen::FlushToZeroMode) { test_scaled_points<float>(); });
}

void test_polynomial_exactness_complex() {
  using Scalar = std::complex<double>;
  const Array<Scalar, 4, 1> points{{Scalar(-1.0, 1.0), Scalar(0.0, 0.0), Scalar(1.0, 1.0), Scalar(2.0, -1.0)}};
  verify_polynomial_exactness<0>(points);
  verify_polynomial_exactness<1>(points);
  verify_polynomial_exactness<3>(points);
}

// No internal reordering: weights()[i] must keep tracking points()[i] as given, so permuting the
// input permutes the output identically.
void test_order_preservation() {
  const Array<double, 5, 1> points{{-1000.0, -1.0, 0.0, 1.0, 1000.0}};
  const Array<double, 5, 1> reversed{{1000.0, 1.0, 0.0, -1.0, -1000.0}};

  const Stencil<2, double, 5> s(points);
  const Stencil<2, double, 5> sReversed(reversed);
  const Array<double, 5, 1> w = s.weights();
  const Array<double, 5, 1> wReversed = sReversed.weights();

  for (Index i = 0; i < 5; ++i) {
    const double tol = 100.0 * Eigen::NumTraits<double>::epsilon() * (numext::abs(w[i]) + 1.0);
    VERIFY(numext::abs(w[i] - wReversed[4 - i]) <= tol);
  }
}

// Repeated points make a `points[n] - points[nu]` denominator in the recursion vanish.
void test_repeated_points_assert() {
  const Array<double, 3, 1> points{{0.0, 0.0, 1.0}};
  VERIFY_RAISES_ASSERT((Stencil<1, double, 3>(points)));
}

// The C-array constructor and weight()/point() bypass Eigen::Array and Map respectively, so unlike
// the DenseBase constructor and weights()/points(), these are real constant expressions on every
// supported C++14 compiler.
constexpr double centralPoints[] = {-1.0, 0.0, 1.0};
constexpr Stencil<1, double, 3> centralDiff1 = makeStencil<1>(centralPoints);
static_assert(centralDiff1.weight(0) == -0.5, "");
static_assert(centralDiff1.weight(1) == 0.0, "");
static_assert(centralDiff1.weight(2) == 0.5, "");
static_assert(centralDiff1.point(0) == -1.0, "");
static_assert(centralDiff1.point(1) == 0.0, "");
static_assert(centralDiff1.point(2) == 1.0, "");

constexpr float smallStep = 1.0f / 134217728.0f;  // 2^-27
constexpr float unitPoints[] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
constexpr float smallPoints[] = {-4 * smallStep, -3 * smallStep, -2 * smallStep, -smallStep,   0,
                                 smallStep,      2 * smallStep,  3 * smallStep,  4 * smallStep};
constexpr Stencil<2, float, 9> unitDiff2(unitPoints);
constexpr Stencil<2, float, 9> smallDiff2(smallPoints);
static_assert(smallDiff2.weight(6) == unitDiff2.weight(6) / (smallStep * smallStep), "");
static_assert(smallDiff2.weight(7) == unitDiff2.weight(7) / (smallStep * smallStep), "");
static_assert(smallDiff2.weight(8) == unitDiff2.weight(8) / (smallStep * smallStep), "");

void test_compile_time_smoke() {
  const Array<double, 3, 1> w = centralDiff1.weights();
  VERIFY(numext::abs(w[0] - (-0.5)) <= Eigen::NumTraits<double>::epsilon());
  VERIFY(numext::abs(w[1] - 0.0) <= Eigen::NumTraits<double>::epsilon());
  VERIFY(numext::abs(w[2] - 0.5) <= Eigen::NumTraits<double>::epsilon());
}

void test_constant_member_linkage() {
  using StencilType = Stencil<1, double, 3>;
  // Volatile pointers retain the odr-use even in optimized test builds.
  const int* volatile pointCount = &StencilType::PointCount;
  const int* volatile derivativeOrder = &StencilType::DerivativeOrder;
  VERIFY_IS_EQUAL(*pointCount, 3);
  VERIFY_IS_EQUAL(*derivativeOrder, 1);
}

EIGEN_DECLARE_TEST(stencil) {
  CALL_SUBTEST(test_textbook_central());
  CALL_SUBTEST(test_textbook_forward());
  CALL_SUBTEST(test_polynomial_exactness_uniform());
  CALL_SUBTEST(test_polynomial_exactness_nonuniform());
  CALL_SUBTEST(test_polynomial_exactness_no_zero());
  CALL_SUBTEST(test_polynomial_exactness_float());
  CALL_SUBTEST(test_scaled_points<float>());
  CALL_SUBTEST(test_scaled_points<double>());
  CALL_SUBTEST(test_scaled_points_flush_modes());
  CALL_SUBTEST(test_polynomial_exactness_complex());
  CALL_SUBTEST(test_order_preservation());
  CALL_SUBTEST(test_repeated_points_assert());
  CALL_SUBTEST(test_compile_time_smoke());
  CALL_SUBTEST(test_constant_member_linkage());
}
