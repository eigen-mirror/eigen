// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MATHFUNCTIONSIMPL_H
#define EIGEN_MATHFUNCTIONSIMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal Fast reciprocal using Newton-Raphson's method.

 Preconditions:
   1. The starting guess provided in approx_a_recip must have at least half
      the leading mantissa bits in the correct result, such that a single
      Newton-Raphson step is sufficient to get within 1-2 ulps of the correct
      result.
   2. If a is zero, approx_a_recip must be infinite with the same sign as a.
   3. If a is infinite, approx_a_recip must be zero with the same sign as a.

   If the preconditions are satisfied, which they are for the _*_rcp_ps
   instructions on x86, the result has a maximum relative error of 2 ulps,
   and correctly handles reciprocals of zero, infinity, and NaN.
*/
template <typename Packet, int Steps>
struct generic_reciprocal_newton_step {
  static_assert(Steps > 0, "Steps must be at least 1.");
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& a, const Packet& approx_a_recip) {
    using Scalar = typename unpacket_traits<Packet>::type;
    const Packet one = pset1<Packet>(Scalar(1));
    // Refine the approximation using one Newton-Raphson step:
    //   x_{i} = x_{i-1} * (2 - a * x_{i-1})
    const Packet x = generic_reciprocal_newton_step<Packet, Steps - 1>::run(a, approx_a_recip);
    const Packet tmp = pnmadd(a, x, one);
    // If tmp is NaN, it means that a is either +/-0 or +/-Inf.
    // In this case return the approximation directly.
    const Packet is_not_nan = pcmp_eq(tmp, tmp);
    // Use two FMAs instead of FMA+FMUL to improve precision.
    return pselect(is_not_nan, pmadd(x, tmp, x), x);
  }
};

template <typename Packet>
struct generic_reciprocal_newton_step<Packet, 0> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& /*unused*/, const Packet& approx_rsqrt) {
    return approx_rsqrt;
  }
};

/** \internal Fast reciprocal sqrt using Newton-Raphson's method.

 Preconditions:
   1. The starting guess provided in approx_a_recip must have at least half
      the leading mantissa bits in the correct result, such that a single
      Newton-Raphson step is sufficient to get within 1-2 ulps of the correct
      result.
   2. If a is zero, approx_a_recip must be infinite with the same sign as a.
   3. If a is infinite, approx_a_recip must be zero with the same sign as a.

   If the preconditions are satisfied, which they are for the _*_rcp_ps
   instructions on x86, the result has a maximum relative error of 2 ulps,
   and correctly handles zero, infinity, and NaN. Positive denormals are
   treated as zero.
*/
template <typename Packet, int Steps>
struct generic_rsqrt_newton_step {
  static_assert(Steps > 0, "Steps must be at least 1.");
  using Scalar = typename unpacket_traits<Packet>::type;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& a, const Packet& approx_rsqrt) {
    const Scalar kMinusHalf = Scalar(-1) / Scalar(2);
    const Packet cst_minus_half = pset1<Packet>(kMinusHalf);
    const Packet cst_minus_one = pset1<Packet>(Scalar(-1));

    Packet inv_sqrt = approx_rsqrt;
    for (int step = 0; step < Steps; ++step) {
      // Refine the approximation using one Newton-Raphson step:
      // h_n = (x * inv_sqrt) * inv_sqrt - 1 (so that h_n is nearly 0).
      // inv_sqrt = inv_sqrt - 0.5 * inv_sqrt * h_n
      Packet r2 = pmul(a, inv_sqrt);
      Packet half_r = pmul(inv_sqrt, cst_minus_half);
      Packet h_n = pmadd(r2, inv_sqrt, cst_minus_one);
      inv_sqrt = pmadd(half_r, h_n, inv_sqrt);
    }

    // If inv_sqrt is NaN, then either:
    // 1) the input is NaN
    // 2) zero and infinity were multiplied
    // In either of these cases, return approx_rsqrt
    return pselect(pisnan(inv_sqrt), approx_rsqrt, inv_sqrt);
  }
};

template <typename Packet>
struct generic_rsqrt_newton_step<Packet, 0> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& /*unused*/, const Packet& approx_rsqrt) {
    return approx_rsqrt;
  }
};

/** \internal Fast sqrt using Newton-Raphson's method.

 Preconditions:
   1. The starting guess for the reciprocal sqrt provided in approx_rsqrt must
      have at least half the leading mantissa bits in the correct result, such
      that a single Newton-Raphson step is sufficient to get within 1-2 ulps of
      the correct result.
   2. If a is zero, approx_rsqrt must be infinite.
   3. If a is infinite, approx_rsqrt must be zero.

   If the preconditions are satisfied, which they are for the _*_rsqrt_ps
   instructions on x86, the result has a maximum relative error of 2 ulps,
   and correctly handles zero and infinity, and NaN. Positive denormal inputs
   are treated as zero.
*/
template <typename Packet, int Steps = 1>
struct generic_sqrt_newton_step {
  static_assert(Steps > 0, "Steps must be at least 1.");

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& a, const Packet& approx_rsqrt) {
    using Scalar = typename unpacket_traits<Packet>::type;
    const Packet one_point_five = pset1<Packet>(Scalar(1.5));
    const Packet minus_half = pset1<Packet>(Scalar(-0.5));
    // If a is inf or zero, return a directly.
    const Packet inf_mask = pcmp_eq(a, pset1<Packet>(NumTraits<Scalar>::infinity()));
    const Packet return_a = por(pcmp_eq(a, pzero(a)), inf_mask);
    // Do a single step of Newton's iteration for reciprocal square root:
    //   x_{n+1} = x_n * (1.5 + (-0.5 * x_n) * (a * x_n))).
    // The Newton's step is computed this way to avoid over/under-flows.
    Packet rsqrt = pmul(approx_rsqrt, pmadd(pmul(minus_half, approx_rsqrt), pmul(a, approx_rsqrt), one_point_five));
    for (int step = 1; step < Steps; ++step) {
      rsqrt = pmul(rsqrt, pmadd(pmul(minus_half, rsqrt), pmul(a, rsqrt), one_point_five));
    }

    // Return sqrt(x) = x * rsqrt(x) for non-zero finite positive arguments.
    // Return a itself for 0 or +inf, NaN for negative arguments.
    return pselect(return_a, a, pmul(a, rsqrt));
  }
};

template <typename RealScalar>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE RealScalar positive_real_hypot(const RealScalar& x,
                                                                               const RealScalar& y) {
  // IEEE IEC 60559 special cases.
  if ((numext::isinf)(x) || (numext::isinf)(y)) return NumTraits<RealScalar>::infinity();
  if ((numext::isnan)(x) || (numext::isnan)(y)) return NumTraits<RealScalar>::quiet_NaN();

  EIGEN_USING_STD(sqrt);
  RealScalar p = numext::maxi(x, y);
  if (numext::is_exactly_zero(p)) return RealScalar(0);
  RealScalar qp = numext::mini(y, x) / p;
  return p * sqrt(RealScalar(1) + qp * qp);
}

template <typename Scalar>
struct hypot_impl {
  using RealScalar = typename NumTraits<Scalar>::Real;
  static EIGEN_DEVICE_FUNC inline RealScalar run(const Scalar& x, const Scalar& y) {
    return positive_real_hypot<RealScalar>(numext::abs(x), numext::abs(y));
  }
};

template <typename ComplexT, bool Reciprocal>
EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE constexpr ComplexT complex_sqrt_extreme(const ComplexT& z) {
  using T = typename NumTraits<ComplexT>::Real;
  const T x = numext::real(z);
  const T y = numext::imag(z);
  const T zero = T(0);
  const T inf = NumTraits<T>::infinity();
  EIGEN_IF_CONSTEXPR (Reciprocal) {
    if ((numext::isinf)(x) || (numext::isinf)(y)) return ComplexT(zero, numext::copysign(zero, -y));
  } else {
    if ((numext::isinf)(y)) return ComplexT(inf, y);
    if ((numext::isinf)(x)) {
      const T other = (numext::isnan)(y) ? y : zero;
      return x > zero ? ComplexT(inf, numext::copysign(other, y))
                      : ComplexT(numext::abs(other), numext::copysign(inf, y));
    }
  }
  if ((numext::isnan)(x) || (numext::isnan)(y)) {
    return ComplexT(NumTraits<T>::quiet_NaN(), NumTraits<T>::quiet_NaN());
  }
  if (numext::is_exactly_zero(x) && numext::is_exactly_zero(y)) {
    return Reciprocal ? ComplexT(inf, NumTraits<T>::quiet_NaN()) : ComplexT(zero, y);
  }

  T ax = numext::abs(x);
  T ay = numext::abs(y);
  const T p = numext::maxi(ax, ay);
  const T r = numext::mini(ax, ay) / p;
  const T h = numext::sqrt(T(1) + r * r);
  // Evaluate at z/p, restoring sqrt(p) only after taking the square root.
  ax /= p;
  ay /= p;
  const T scale = numext::sqrt(p);
  const T sum = ax + h;
  const T w = numext::sqrt(T(0.5) * sum);
  T major = w * scale;
  T minor = zero;
  EIGEN_IF_CONSTEXPR (Reciprocal) {
    major = (w / h) / scale;
    minor = (ay / sum) * major;
  } else {
    // Use the original y: normalizing z/p can underflow its smaller component.
    minor = numext::abs(y) / (T(2) * major);
  }
  if (numext::is_exactly_zero(x)) minor = major;
  const T imag_sign = Reciprocal ? -y : y;
  return x < zero ? ComplexT(minor, numext::copysign(major, imag_sign))
                  : ComplexT(major, numext::copysign(minor, imag_sign));
}

template <typename ComplexT, bool Reciprocal>
EIGEN_DEVICE_FUNC constexpr ComplexT complex_sqrt_impl(const ComplexT& z) {
  using T = typename NumTraits<ComplexT>::Real;
  const T x = numext::real(z);
  const T y = numext::imag(z);
  const T ax = numext::abs(x);
  const T ay = numext::abs(y);
  const bool real_larger = ax > ay;
  const T p = real_larger ? ax : ay;
  const T q = real_larger ? ay : ax;
  // These bounds keep (|x| + |z|)/2 normal and finite. Using the same comparison
  // for p and q preserves a NaN in either component.
  if (EIGEN_PREDICT_FALSE(!(p > T(2) * (numext::numeric_limits<T>::min)() && p <= NumTraits<T>::highest() / T(4)))) {
    return complex_sqrt_extreme<ComplexT, Reciprocal>(z);
  }
  const T r = q / p;
  const T abs_z = p * numext::sqrt(T(1) + r * r);
  const T sum = ax + abs_z;
  const T w = numext::sqrt(T(0.5) * sum);
  T major = w;
  T minor = T(0);
  EIGEN_IF_CONSTEXPR (Reciprocal) {
    major = w / abs_z;
    // |y|/(2*w*|z|) = (|y|/(|x| + |z|)) * (w/|z|), without a cubic-scale denominator.
    minor = (ay / sum) * major;
  } else {
    minor = ay / (T(2) * w);
  }
  if (numext::is_exactly_zero(x)) minor = major;
  const T imag_sign = Reciprocal ? -y : y;
  return x < T(0) ? ComplexT(minor, numext::copysign(major, imag_sign))
                  : ComplexT(major, numext::copysign(minor, imag_sign));
}

// Principal square root, with the branch cut selected by the sign of the imaginary part.
template <typename ComplexT>
EIGEN_DEVICE_FUNC constexpr ComplexT complex_sqrt(const ComplexT& z) {
  return complex_sqrt_impl<ComplexT, false>(z);
}

template <typename ComplexT>
EIGEN_DEVICE_FUNC constexpr ComplexT complex_rsqrt(const ComplexT& z) {
  return complex_sqrt_impl<ComplexT, true>(z);
}

template <typename ComplexT>
EIGEN_DEVICE_FUNC constexpr ComplexT complex_log(const ComplexT& z) {
  // Computes complex log.
  using T = typename NumTraits<ComplexT>::Real;
  T a = numext::abs(z);
  EIGEN_USING_STD(atan2);
  T b = atan2(z.imag(), z.real());
  return ComplexT(numext::log(a), b);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATHFUNCTIONSIMPL_H
