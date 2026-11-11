// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018-2025 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_ARCH_GENERIC_PACKET_MATH_DOUBLE_WORD_H
#define EIGEN_ARCH_GENERIC_PACKET_MATH_DOUBLE_WORD_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// This function splits x into the nearest integer n and fractional part r,
// such that x = n + r holds exactly.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void absolute_split(const Packet& x, Packet& n, Packet& r) {
  n = pround(x);
  r = psub(x, n);
}

// Knuth two-sum: x + y = s_hi + s_lo exactly, with no magnitude ordering required.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twosum(const Packet& x, const Packet& y, Packet& s_hi, Packet& s_lo) {
  s_hi = padd(x, y);
  pbarrier(s_hi);
  Packet y_part = psub(s_hi, x);
  pbarrier(y_part);
  Packet x_part = psub(s_hi, y_part);
  pbarrier(x_part);
  Packet x_err = psub(x, x_part);
  pbarrier(x_err);
  Packet y_err = psub(y, y_part);
  pbarrier(y_err);
  s_lo = padd(x_err, y_err);
}

// This function computes the sum {s_hi, s_lo}, such that x + y = s_hi + s_lo
// holds exactly, and s_hi = fl(x+y), if |x| >= |y|.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void fast_twosum(const Packet& x, const Packet& y, Packet& s_hi, Packet& s_lo) {
  s_hi = padd(x, y);
  const Packet t = psub(s_hi, x);
  s_lo = psub(y, t);
}

#ifdef EIGEN_VECTORIZE_FMA
// Given x, y, and xy = fl(x*y), return the residual such that x*y = xy + residual exactly.
template <typename Packet, std::enable_if_t<!is_scalar<Packet>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet twoprod_low(const Packet& x, const Packet& y, const Packet& xy) {
  return pmsub(x, y, xy);
}

template <typename Scalar, std::enable_if_t<is_scalar<Scalar>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar twoprod_low(const Scalar& x, const Scalar& y, const Scalar& xy) {
  // Error-free products require FMA even when EIGEN_SCALAR_MADD_USE_FMA disables fusion in scalar madd.
  return twoprod_low_fma(x, y, xy);
}

// This function implements the extended precision product of
// a pair of floating point numbers. Given {x, y}, it computes the pair
// {p_hi, p_lo} such that x * y = p_hi + p_lo holds exactly and
// p_hi = fl(x * y). PreventReassociation preserves the separately rounded non-FMA operations.
template <typename Packet, bool PreventReassociation = false>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twoprod(const Packet& x, const Packet& y, Packet& p_hi, Packet& p_lo) {
  p_hi = pmul(x, y);
  p_lo = twoprod_low(x, y, p_hi);
}

#else

// Dekker's product requires separately rounded operations. GCC's C++ default -ffp-contract=fast may fuse
// fl(x*y) into its consumers, so use fma for scalar types where it is an instruction (has_fast_fma).
// This does not prevent contraction in the remaining scalar or packet fallback paths.
template <typename Scalar, std::enable_if_t<has_fast_fma<Scalar>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar twoprod_low(const Scalar& x, const Scalar& y, const Scalar& xy) {
  return twoprod_low_fma(x, y, xy);
}

template <typename Scalar, bool PreventReassociation = false, std::enable_if_t<has_fast_fma<Scalar>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twoprod(const Scalar& x, const Scalar& y, Scalar& p_hi, Scalar& p_lo) {
  p_hi = x * y;
  p_lo = twoprod_low(x, y, p_hi);
}

// This function implements the Veltkamp splitting. Given a floating point
// number x it returns the pair {x_hi, x_lo} such that x_hi + x_lo = x holds
// exactly and that half of the significand of x fits in x_hi.
// This is Algorithm 3 from Jean-Michel Muller, "Elementary Functions",
// 3rd edition, Birkh\"auser, 2016.
template <typename Packet, bool PreventReassociation = false>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void veltkamp_splitting(const Packet& x, Packet& x_hi, Packet& x_lo) {
  using Scalar = typename unpacket_traits<Packet>::type;
  constexpr int shift = (NumTraits<Scalar>::digits() + 1) / 2;
  const Scalar shift_scale = Scalar(uint64_t(1) << shift);  // Scalar constructor not necessarily constexpr.
  Packet gamma = pmul(pset1<Packet>(shift_scale + Scalar(1)), x);
  EIGEN_IF_CONSTEXPR (PreventReassociation) pbarrier(gamma);
  Packet rho = psub(x, gamma);
  EIGEN_IF_CONSTEXPR (PreventReassociation) pbarrier(rho);
  x_hi = padd(rho, gamma);
  EIGEN_IF_CONSTEXPR (PreventReassociation) pbarrier(x_hi);
  x_lo = psub(x, x_hi);
}

// A version of twoprod that takes x, y, and fl(x*y) as input and returns the p_lo such that
// x * y = xy + p_lo holds exactly. PreventReassociation preserves the separately rounded operations.
template <typename Packet, bool PreventReassociation = false, std::enable_if_t<!has_fast_fma<Packet>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet twoprod_low(const Packet& x, const Packet& y, const Packet& xy) {
  Packet x_hi, x_lo, y_hi, y_lo;
  veltkamp_splitting<Packet, PreventReassociation>(x, x_hi, x_lo);
  veltkamp_splitting<Packet, PreventReassociation>(y, y_hi, y_lo);

  Packet p_lo = pmsub(x_hi, y_hi, xy);
  EIGEN_IF_CONSTEXPR (PreventReassociation) pbarrier(p_lo);
  p_lo = pmadd(x_hi, y_lo, p_lo);
  EIGEN_IF_CONSTEXPR (PreventReassociation) pbarrier(p_lo);
  p_lo = pmadd(x_lo, y_hi, p_lo);
  EIGEN_IF_CONSTEXPR (PreventReassociation) pbarrier(p_lo);
  p_lo = pmadd(x_lo, y_lo, p_lo);
  return p_lo;
}

// This function implements Dekker's algorithm for products x * y.
// Given floating point numbers {x, y}, it computes {p_hi, p_lo} such that
// x * y = p_hi + p_lo exactly and p_hi = fl(x * y).
template <typename Packet, bool PreventReassociation = false, std::enable_if_t<!has_fast_fma<Packet>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twoprod(const Packet& x, const Packet& y, Packet& p_hi, Packet& p_lo) {
  Packet hi = pmul(x, y);
  EIGEN_IF_CONSTEXPR (PreventReassociation) pbarrier(hi);
  const Packet lo = twoprod_low<Packet, PreventReassociation>(x, y, hi);
  p_hi = hi;
  p_lo = lo;
}

#endif  // EIGEN_VECTORIZE_FMA

// Whether twoprod_full_range takes Dekker's product for Scalar rather than a software FMA: a binary IEEE format with
// highest eps^6 >= min, so that an operand rescaled by eps^2 or 1 / eps^2 lands where the product is exact. libstdc++
// reports is_iec559 for IBM double-double too, whose operations are not correctly rounded; emin = 1 - emax excludes it.
// Other scalars keep the FMA, and so do GPU and SYCL devices, which fuse natively and have no optimization barrier.
#if defined(EIGEN_VECTORIZE_FMA) || defined(EIGEN_GPUCC) || defined(EIGEN_USE_SYCL)
template <typename Scalar>
struct twoprod_full_range_uses_dekker : std::false_type {};
#else
template <typename Scalar>
struct twoprod_full_range_uses_dekker
    : bool_constant<!has_fast_fma<Scalar>::value && std::numeric_limits<Scalar>::is_iec559 &&
                    std::numeric_limits<Scalar>::radix == 2 &&
                    std::numeric_limits<Scalar>::min_exponent == 3 - std::numeric_limits<Scalar>::max_exponent &&
                    std::numeric_limits<Scalar>::max_exponent - std::numeric_limits<Scalar>::min_exponent >=
                        6 * std::numeric_limits<Scalar>::digits> {};

// Veltkamp's splitting of x overflows for |x| > highest / 2^ceil(digits / 2), and Dekker's partial products lose bits
// to underflow once ulp(x) ulp(y) < denorm_min. With 1/2 <= |y| <= 2, neither happens for min / eps^2 <= |x| <=
// highest eps^2. Outside that interval x and p_hi are rescaled by a power of two, which keeps p_hi = fl(x y) while
// p_hi is normal, and p_lo then takes the single rounding of fma(x, y, -p_hi).
template <typename Scalar, std::enable_if_t<twoprod_full_range_uses_dekker<Scalar>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twoprod_full_range_impl(const Scalar& x, const Scalar& y, Scalar& p_hi,
                                                                   Scalar& p_lo) {
  const Scalar eps2 = NumTraits<Scalar>::epsilon() * NumTraits<Scalar>::epsilon();
  const Scalar abs_x = numext::abs(x);
  Scalar hi = x * y;
  pbarrier(hi);
  p_hi = hi;
  if (abs_x > NumTraits<Scalar>::highest() * eps2) {
    p_lo = twoprod_low<Scalar, true>(Scalar(x * eps2), y, Scalar(hi * eps2)) / eps2;
  } else if (abs_x < (std::numeric_limits<Scalar>::min)() / eps2) {
    p_lo = twoprod_low<Scalar, true>(Scalar(x / eps2), y, Scalar(hi / eps2)) * eps2;
  } else {
    p_lo = twoprod_low<Scalar, true>(x, y, hi);
  }
}
#endif

template <typename Scalar, std::enable_if_t<!twoprod_full_range_uses_dekker<Scalar>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twoprod_full_range_impl(const Scalar& x, const Scalar& y, Scalar& p_hi,
                                                                   Scalar& p_lo) {
  p_hi = x * y;
  p_lo = twoprod_low_fma(x, y, p_hi);
}

// Declared in MathFunctions.h, for the scalar functions there.
template <typename Scalar>
EIGEN_DEVICE_FUNC void twoprod_full_range(const Scalar& x, const Scalar& y, Scalar& p_hi, Scalar& p_lo) {
  twoprod_full_range_impl(x, y, p_hi, p_lo);
}

// This function implements the addition of two double word numbers
// represented by {x_hi, x_lo} and {y_hi, y_lo}, with operands of either sign
// and magnitude: Knuth's two-sum of the high parts is error-free, the low
// parts are added to its error term, and the result is renormalized.
// This is Algorithm 5 (SloppyDWPlusDW) from Joldes, Muller, & Popescu (2017),
// "Tight and rigorous error bounds for basic building blocks of double-word
// arithmetic", https://hal.science/hal-01351529: the error is a few u^2
// relative to |x| + |y|, not to the sum, which may cancel.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twosum(const Packet& x_hi, const Packet& x_lo, const Packet& y_hi,
                                                  const Packet& y_lo, Packet& s_hi, Packet& s_lo) {
  Packet s = padd(x_hi, y_hi);
  Packet y_part = psub(s, x_hi);
  Packet err = padd(psub(x_hi, psub(s, y_part)), psub(y_hi, y_part));
  fast_twosum(s, padd(err, padd(x_lo, y_lo)), s_hi, s_lo);
}

// The difference {x_hi, x_lo} - {y_hi, y_lo} of two double word numbers,
// the same algorithm as twosum with the sign of y folded into the operations.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twodiff(const Packet& x_hi, const Packet& x_lo, const Packet& y_hi,
                                                   const Packet& y_lo, Packet& s_hi, Packet& s_lo) {
  Packet s = psub(x_hi, y_hi);
  Packet y_part = psub(x_hi, s);
  Packet err = psub(psub(x_hi, padd(s, y_part)), psub(y_hi, y_part));
  fast_twosum(s, padd(err, psub(x_lo, y_lo)), s_hi, s_lo);
}

// This is a version of twosum for double word numbers,
// which assumes that |x_hi| >= |y_hi|.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void fast_twosum(const Packet& x_hi, const Packet& x_lo, const Packet& y_hi,
                                                       const Packet& y_lo, Packet& s_hi, Packet& s_lo) {
  Packet r_hi, r_lo;
  fast_twosum(x_hi, y_hi, r_hi, r_lo);
  const Packet s = padd(padd(y_lo, r_lo), x_lo);
  fast_twosum(r_hi, s, s_hi, s_lo);
}

// This is a version of twosum for adding a floating point number x to
// double word number {y_hi, y_lo} number, with the assumption
// that |x| >= |y_hi|.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void fast_twosum(const Packet& x, const Packet& y_hi, const Packet& y_lo,
                                                       Packet& s_hi, Packet& s_lo) {
  Packet r_hi, r_lo;
  fast_twosum(x, y_hi, r_hi, r_lo);
  const Packet s = padd(y_lo, r_lo);
  fast_twosum(r_hi, s, s_hi, s_lo);
}

// This function implements the multiplication of a double word
// number represented by {x_hi, x_lo} by a floating point number y.
// It returns the result as a pair {p_hi, p_lo} such that
// (x_hi + x_lo) * y = p_hi + p_lo holds with a relative error
// of less than 2*2^{-2p}, where p is the number of significand bits
// in the floating point type.
// This is Algorithm 7 from Jean-Michel Muller, "Elementary Functions",
// 3rd edition, Birkh\"auser, 2016.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twoprod(const Packet& x_hi, const Packet& x_lo, const Packet& y,
                                                   Packet& p_hi, Packet& p_lo) {
  Packet c_hi, c_lo1;
  twoprod(x_hi, y, c_hi, c_lo1);
  const Packet c_lo2 = pmul(x_lo, y);
  Packet t_hi, t_lo1;
  fast_twosum(c_hi, c_lo2, t_hi, t_lo1);
  const Packet t_lo2 = padd(t_lo1, c_lo1);
  fast_twosum(t_hi, t_lo2, p_hi, p_lo);
}

// This function implements the multiplication of two double word
// numbers represented by {x_hi, x_lo} and {y_hi, y_lo}.
// For normalized inputs (x_hi = RN(x_hi + x_lo), likewise for y), the result
// {p_hi, p_lo} has relative error < 5*u^2 for p-bit significands, u = 2^{-p}, p >= 8.
// This assumes round-to-nearest, no overflow/underflow, an exact two-product
// residual, and separately rounded operations outside that residual.
// Each DW*FP product has relative error <= (3/2 + 4*u)*u^2: Joldes, Muller,
// and Popescu (2017), "Tight and rigorous error bounds for basic building
// blocks of double-word arithmetic", Theorem 4.1, https://hal.science/hal-01351529.
// The final sum adds absolute error <= u^2*((3+3*u+u^2)*abs(p_hi_hi)
// + (4+3*u+u^2)*abs(p_lo_hi)); using abs(y_lo) <= u*abs(y_hi) gives the bound above.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void twoprod(const Packet& x_hi, const Packet& x_lo, const Packet& y_hi,
                                                   const Packet& y_lo, Packet& p_hi, Packet& p_lo) {
  Packet p_hi_hi, p_hi_lo;
  twoprod(x_hi, x_lo, y_hi, p_hi_hi, p_hi_lo);
  Packet p_lo_hi, p_lo_lo;
  twoprod(x_hi, x_lo, y_lo, p_lo_hi, p_lo_lo);
  fast_twosum(p_hi_hi, p_hi_lo, p_lo_hi, p_lo_lo, p_hi, p_lo);
}

// This function implements the multiplication of two double word numbers
// {x_hi, x_lo} and {y_hi, y_lo} at a lower cost than twoprod above: with
// {c_hi, c_lo} = x_hi*y_hi exactly, it renormalizes c_hi + (c_lo + x_hi*y_lo +
// x_lo*y_hi), dropping x_lo*y_lo. Scale x_hi and y_hi into [1, 2), so that
// |x_lo|, |y_lo| <= u = 2^-p for p-bit significands.
// Where pmadd is fused (is_fused_madd, and float and double scalars under
// EIGEN_VECTORIZE_FMA), two pmadds accumulate the cross terms onto c_lo. If
// x_hi*y_hi < 2, each rounds a value below 4*u, by at most 2*u^2, and the
// dropped term is at most u^2, so the relative error is below 5*u^2 to first
// order; if x_hi*y_hi >= 2 it is below 3.5*u^2. Measured maxima: 4.87*u^2 (float)
// and 4.80*u^2 (double), 4.93*u^2 and 4.98*u^2 for squares.
// Elsewhere the cross products are rounded and summed before c_lo is added:
// Algorithm 10 (DWTimesDW1) of Joldes, Muller, & Popescu (2017),
// https://hal.science/hal-01351529, measured at most 4.9997*u^2 (float) and
// 4.94*u^2 (double), 4.00*u^2 for squares. Chaining unfused pmadds instead would
// round each cross product as well, for up to 7*u^2 (measured 6.93*u^2).
// Measured against an exact reference over 5*10^7 normalized operand pairs per
// case plus a hill-climb search; round-to-nearest, no overflow or underflow.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void fast_twoprod(const Packet& x_hi, const Packet& x_lo, const Packet& y_hi,
                                                        const Packet& y_lo, Packet& p_hi, Packet& p_lo) {
  Packet c_hi, c_lo;
  twoprod(x_hi, y_hi, c_hi, c_lo);
#ifdef EIGEN_VECTORIZE_FMA
  // Scalar float and double pmadd are numext::fma here, so scalars round like the packets.
  constexpr bool kScalarFused = std::is_same<Packet, float>::value || std::is_same<Packet, double>::value;
#else
  constexpr bool kScalarFused = false;
#endif
  EIGEN_IF_CONSTEXPR (is_fused_madd<Packet>::value || kScalarFused) {
    c_lo = pmadd(x_hi, y_lo, pmadd(x_lo, y_hi, c_lo));
  } else {
    c_lo = padd(c_lo, padd(pmul(x_hi, y_lo), pmul(x_lo, y_hi)));
  }
  fast_twosum(c_hi, c_lo, p_hi, p_lo);
}

// This function implements the division of double word {x_hi, x_lo}
// by float y. This is Algorithm 15 from "Tight and rigorous error bounds
// for basic building blocks of double-word arithmetic", Joldes, Muller, & Popescu,
// 2017. https://hal.archives-ouvertes.fr/hal-01351529
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void doubleword_div_fp(const Packet& x_hi, const Packet& x_lo, const Packet& y,
                                                             Packet& z_hi, Packet& z_lo) {
  const Packet t_hi = pdiv(x_hi, y);
  Packet pi_hi, pi_lo;
  twoprod(t_hi, y, pi_hi, pi_lo);
  const Packet delta_hi = psub(x_hi, pi_hi);
  const Packet delta_t = psub(delta_hi, pi_lo);
  const Packet delta = padd(delta_t, x_lo);
  const Packet t_lo = pdiv(delta, y);
  fast_twosum(t_hi, t_lo, z_hi, z_lo);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_ARCH_GENERIC_PACKET_MATH_DOUBLE_WORD_H
