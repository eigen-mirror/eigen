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

   If the preconditions are satisfied, the result has a maximum relative error
   of 2 ulps, and correctly handles reciprocals of zero, infinity, and NaN. The
   estimates of rcpps, good to 1.5 * 2^-12, fall just short of the first
   precondition for float: one step is within 2 ulps where the multiply-adds are
   fused and within 3.1 ulps where they are not. Lanes where the
   approximation flushed a denormal input or result to zero take a division.
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
    // Use two FMAs instead of FMA+FMUL to improve precision.
    const Packet refined = pmadd(x, tmp, x);
    // The step yields NaN when a is +/-0, +/-Inf or NaN, and NaN or 0 where the approximation flushed a denormal a or
    // 1 / a to zero (x = +/-inf, or x = 0 for |a| > 1 / min). Divide in those lanes.
    const Packet redo = pcmp_lt_or_nan(pabs(refined), pset1<Packet>((std::numeric_limits<Scalar>::denorm_min)()));
    return predux_any(redo) ? pselect(redo, pdiv(one, a), refined) : refined;
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

   If the preconditions are satisfied, the result has a maximum relative error
   of 2 ulps, and correctly handles zero, infinity, and NaN. Positive denormals
   are treated as zero. The estimates of rsqrtps, good to 1.5 * 2^-12, fall just
   short of the first precondition for float: one step leaves an error of up to
   3.4 ulps with Intel's estimate (2.1 ulps with AMD's), and 1 in 4 results off
   by one ulp or more.
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

   If the preconditions are satisfied, the result has a maximum relative error
   of 3 ulps, and correctly handles zero and infinity, and NaN. Positive denormal
   inputs are treated as zero. The last step multiplies the refined reciprocal
   square root by a, which rounds three times after the residual is formed; where
   a fused multiply-add is available, a last step on the square root itself,
   y + (r / 2) (a - y^2) with y = a r, is within 1 ulp in one operation fewer
   (see psqrt<Packet16f>).
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

// sqrt(x^2 + y^2) for x, y >= 0. The generic version divides, so that no intermediate over- or underflows; it is
// within 2 ulps. float and double have their own below.
template <typename RealScalar, typename EnableIf = void>
struct positive_real_hypot_impl {
  EIGEN_DEVICE_FUNC static constexpr EIGEN_STRONG_INLINE RealScalar run(const RealScalar& x, const RealScalar& y) {
    // IEEE IEC 60559 special cases.
    if ((numext::isinf)(x) || (numext::isinf)(y)) return NumTraits<RealScalar>::infinity();
    if ((numext::isnan)(x) || (numext::isnan)(y)) return NumTraits<RealScalar>::quiet_NaN();

    RealScalar p = numext::maxi(x, y);
    if (numext::is_exactly_zero(p)) return RealScalar(0);
    RealScalar qp = numext::mini(y, x) / p;
    return p * numext::sqrt<RealScalar>(RealScalar(1) + qp * qp);
  }
};

// Host code only: GPU and SYCL device code keeps the generic version. Double precision is slow or missing there,
// and neither of the two below has been run on a device.
#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(EIGEN_SYCL_DEVICE_ONLY)
// float in double precision, where neither square rounds, over- or underflows: the sum of squares is rounded once
// before the square root, and the result is the correctly rounded one except for a few inputs in a billion.
template <>
struct positive_real_hypot_impl<
    float,
    std::enable_if_t<std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559 &&
                     (std::numeric_limits<double>::digits >= 2 * std::numeric_limits<float>::digits + 2) &&
                     (std::numeric_limits<double>::max_exponent > 2 * std::numeric_limits<float>::max_exponent + 1)>> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float run(const float& x, const float& y) {
    // An infinite operand gives +inf even if the other one is NaN (IEC 60559). The operands are tested themselves:
    // GCC 15 folds the same test away once it knows the result to be NaN (it takes both operands of a NaN sum to be
    // NaN).
    if (EIGEN_PREDICT_FALSE((numext::isinf)(x) || (numext::isinf)(y))) return NumTraits<float>::infinity();
    const double dx = static_cast<double>(x);
    const double dy = static_cast<double>(y);
    const double xx = dx * dx;
    return static_cast<float>(numext::sqrt(xx + dy * dy));
  }
};

// double, as the square root of the sum of squares: no division, and one branch for operands whose squares neither
// overflow nor lose bits to underflow. Where fma is an instruction the larger square is fused into the sum, so that
// only the smaller square and the sum round before the square root, and the result is within 1.1 ulps; without it the
// three roundings leave it within 1.2 ulps. Other operands, subnormal ones included, are scaled by a power of two.
template <>
struct positive_real_hypot_impl<double, std::enable_if_t<std::numeric_limits<double>::is_iec559>> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double run(const double& x, const double& y) {
    // The larger and the smaller operand, in the form compilers turn into max and min instructions: a branch on x > y
    // would be mispredicted half of the time for random operands.
    const double p = x > y ? x : y;
    const double q = y < x ? y : x;
    // 2^-483 <= x + y <= 2^510 puts the larger operand in [2^-484, 2^510], where p^2 + q^2 is below 2^1022 and p^2 at
    // least 2^-968, and excludes NaN and infinite operands. Testing the sum instead of p keeps the selection above
    // free of branches with GCC.
    const double sum = x + y;
    if (EIGEN_PREDICT_TRUE(sum >= 4.004166190366202e-146 && sum <= 3.3519519824856493e+153)) {
      return numext::sqrt(sum_of_squares(p, q, has_fast_fma<double>()));
    }
    return run_extreme(x, y);
  }

 private:
  // Out of line, and without a second look at the sum: GCC (13 and 15 at least, with its default -ftrapping-math)
  // threads the failed range test above into "sum < bound" for a NaN sum when the two share a function, which sent
  // hypot(inf, NaN) down the scaled path.
  EIGEN_DEVICE_FUNC static EIGEN_DONT_INLINE double run_extreme(double x, double y) {
    // IEEE IEC 60559 special cases.
    if ((numext::isinf)(x) || (numext::isinf)(y)) return NumTraits<double>::infinity();
    if ((numext::isnan)(x) || (numext::isnan)(y)) return NumTraits<double>::quiet_NaN();
    const double p = numext::maxi(x, y);
    const double q = numext::mini(x, y);
    if (numext::is_exactly_zero(p)) return 0.0;
    // Scaled by 2^-514 above the range tested in run and by 2^600 below it, which brings subnormal operands into the
    // normal range.
    const bool large = p > 1.0;
    const double scale = large ? 1.8645851828000517e-155 : 4.149515568880993e+180;
    const double unscale = large ? 5.363123171977039e+154 : 2.409919865102884e-181;
    return unscale * numext::sqrt(sum_of_squares(p * scale, q * scale, has_fast_fma<double>()));
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double sum_of_squares(const double& p, const double& q, std::true_type) {
    const double qq = q * q;
    return numext::fma(p, p, qq);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double sum_of_squares(const double& p, const double& q,
                                                                     std::false_type) {
    const double pp = p * p;
    const double qq = q * q;
    return pp + qq;
  }
};
#endif

template <typename RealScalar>
EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE RealScalar positive_real_hypot(const RealScalar& x,
                                                                               const RealScalar& y) {
  return positive_real_hypot_impl<RealScalar>::run(x, y);
}

template <typename Scalar>
struct hypot_impl {
  using RealScalar = typename NumTraits<Scalar>::Real;
  static EIGEN_DEVICE_FUNC inline RealScalar run(const Scalar& x, const Scalar& y) {
    return positive_real_hypot<RealScalar>(numext::abs(x), numext::abs(y));
  }
};

template <typename ComplexT, bool Reciprocal>
EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE ComplexT complex_sqrt_extreme(const ComplexT& z) {
  using T = typename NumTraits<ComplexT>::Real;
  const T x = numext::real(z);
  const T y = numext::imag(z);
  const T zero = T(0);
  const T inf = NumTraits<T>::infinity();
  EIGEN_IF_CONSTEXPR (Reciprocal) {
    if ((numext::isinf)(x) || (numext::isinf)(y)) return ComplexT(zero, numext::copysign<T>(zero, -y));
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
  T b = numext::atan2(z.imag(), z.real());
  // log|z| is finite where |z| itself overflows or rounds to a subnormal: scale z exactly by a power of two first.
  if (EIGEN_PREDICT_FALSE((numext::isinf)(a) && (numext::isfinite)(z.real()) && (numext::isfinite)(z.imag()))) {
    const T half = T(0.5);
    return ComplexT(numext::log(numext::abs(ComplexT(z.real() * half, z.imag() * half))) + T(EIGEN_LN2), b);
  }
  if (EIGEN_PREDICT_FALSE(a < (numext::numeric_limits<T>::min)() && !numext::is_exactly_zero(a))) {
    const T eps = NumTraits<T>::epsilon();
    return ComplexT(numext::log(numext::abs(ComplexT(z.real() / eps, z.imag() / eps))) + numext::log(eps), b);
  }
  return ComplexT(numext::log(a), b);
}

// log2|z| + i arg(z) log2(e), scaled as in complex_log. The real part comes from log2, which is exact where |z| is a
// power of two; log2(e) log|z| can be an ulp off there.
template <typename ComplexT>
EIGEN_DEVICE_FUNC ComplexT complex_log2(const ComplexT& z) {
  using T = typename NumTraits<ComplexT>::Real;
  const T a = numext::abs(z);
  const T b = T(EIGEN_LOG2E) * numext::atan2(z.imag(), z.real());
  if (EIGEN_PREDICT_FALSE((numext::isinf)(a) && (numext::isfinite)(z.real()) && (numext::isfinite)(z.imag()))) {
    const T half = T(0.5);
    return ComplexT(numext::log2(numext::abs(ComplexT(z.real() * half, z.imag() * half))) + T(1), b);
  }
  if (EIGEN_PREDICT_FALSE(a < (numext::numeric_limits<T>::min)() && !numext::is_exactly_zero(a))) {
    const T eps = NumTraits<T>::epsilon();
    return ComplexT(numext::log2(numext::abs(ComplexT(z.real() / eps, z.imag() / eps))) + numext::log2(eps), b);
  }
  return ComplexT(numext::log2(a), b);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATHFUNCTIONSIMPL_H
