// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018-2025 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_ARCH_GENERIC_PACKET_MATH_POW_H
#define EIGEN_ARCH_GENERIC_PACKET_MATH_POW_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

//----------------------------------------------------------------------
// Cubic Root Functions
//----------------------------------------------------------------------

// This function implements a single step of Halley's iteration for
// computing x = y^(1/3):
//   x_{k+1} = x_k - (x_k^3 - y) x_k / (2x_k^3 + y)
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet cbrt_halley_iteration_step(const Packet& x_k,
                                                                                      const Packet& y) {
  using Scalar = typename unpacket_traits<Packet>::type;
  Packet x_k_cb = pmul(x_k, pmul(x_k, x_k));
  Packet denom = pmadd(pset1<Packet>(Scalar(2)), x_k_cb, y);
  Packet num = psub(x_k_cb, y);
  Packet r = pdiv(num, denom);
  return pnmadd(x_k, r, x_k);
}

// Decompose the input such that x^(1/3) = y^(1/3) * 2^e_div3, and y is in the
// interval [0.125,1].
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet cbrt_decompose(const Packet& x, Packet& e_div3) {
  using Scalar = typename unpacket_traits<Packet>::type;
  // Extract the significand s in the range [0.5,1) and exponent e, such that
  // x = 2^e * s.
  Packet e, s;
  s = pfrexp(x, e);

  // Split the exponent into a part divisible by 3 and the remainder.
  // e = 3*e_div3 + e_mod3.
  constexpr Scalar kOneThird = Scalar(1) / 3;
  e_div3 = pceil(pmul(e, pset1<Packet>(kOneThird)));
  Packet e_mod3 = pnmadd(pset1<Packet>(Scalar(3)), e_div3, e);

  // Replace s by y = (s * 2^e_mod3).
  return pldexp_fast(s, e_mod3);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet cbrt_special_cases_and_sign(const Packet& x,
                                                                                       const Packet& abs_root) {
  // Set sign.
  const Packet sign_mask = psignmask<Packet>();
  const Packet x_sign = pand(sign_mask, x);
  Packet root = por(x_sign, abs_root);

  // Pass non-finite and zero values of x straight through.
  const Packet is_not_finite = por(pisinf(x), pisnan(x));
  const Packet is_zero = pcmp_eq(pzero(x), x);
  const Packet use_x = por(is_not_finite, is_zero);
  return pselect(use_x, x, root);
}

// Generic implementation of cbrt(x) for float.
//
// The algorithm computes the cubic root of the input by first
// decomposing it into an exponent and significand
//   x = s * 2^e.
//
// We can then write the cube root as
//
//   x^(1/3) = 2^(e/3) * s^(1/3)
//           = 2^((3*e_div3 + e_mod3)/3) * s^(1/3)
//           = 2^(e_div3) * 2^(e_mod3/3) * s^(1/3)
//           = 2^(e_div3) * (s * 2^e_mod3)^(1/3)
//
// where e_div3 = ceil(e/3) and e_mod3 = e - 3*e_div3.
//
// The cube root of the second term y = (s * 2^e_mod3)^(1/3) is coarsely
// approximated using a cubic polynomial and subsequently refined using a
// single step of Halley's iteration, and finally the two terms are combined
// using pldexp_fast.
//
// Note: Many alternatives exist for implementing cbrt. See, for example,
// the excellent discussion in Kahan's note:
//   https://csclub.uwaterloo.ca/~pbarfuss/qbrt.pdf
// This particular implementation was found to be very fast and accurate
// among several alternatives tried, but is probably not "optimal" on all
// platforms.
//
// This is accurate to 2 ULP.
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pcbrt_float(const Packet& x) {
  using Scalar = typename unpacket_traits<Packet>::type;
  static_assert(std::is_same<Scalar, float>::value, "Scalar type must be float");

  // Decompose the input such that x^(1/3) = y^(1/3) * 2^e_div3, and y is in the
  // interval [0.125,1].
  Packet e_div3;
  const Packet y = cbrt_decompose(pabs(x), e_div3);

  // Compute initial approximation accurate to 5.22e-3.
  // The polynomial was computed using Rminimax.
  constexpr float alpha[] = {5.9220016002655029296875e-01f, -1.3859539031982421875e+00f, 1.4581282138824462890625e+00f,
                             3.408401906490325927734375e-01f};
  Packet r = ppolevl<Packet, 3>::run(y, alpha);

  // Take one step of Halley's iteration.
  r = cbrt_halley_iteration_step(r, y);

  // Finally multiply by 2^(e_div3)
  r = pldexp_fast(r, e_div3);

  return cbrt_special_cases_and_sign(x, r);
}

// Generic implementation of cbrt(x) for double.
//
// The algorithm is identical to the one for float except that a different initial
// approximation is used for y^(1/3) and two Halley iteration steps are performed.
//
// This is accurate to 1 ULP.
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet pcbrt_double(const Packet& x) {
  using Scalar = typename unpacket_traits<Packet>::type;
  static_assert(std::is_same<Scalar, double>::value, "Scalar type must be double");

  // Decompose the input such that x^(1/3) = y^(1/3) * 2^e_div3, and y is in the
  // interval [0.125,1].
  Packet e_div3;
  const Packet y = cbrt_decompose(pabs(x), e_div3);

  // Compute initial approximation accurate to 0.016.
  // The polynomial was computed using Rminimax.
  constexpr double alpha[] = {-4.69470621553356115551736138513660989701747894287109375e-01,
                              1.072314636518546304699839311069808900356292724609375e+00,
                              3.81249427609571867048288140722434036433696746826171875e-01};
  Packet r = ppolevl<Packet, 2>::run(y, alpha);

  // Take two steps of Halley's iteration.
  r = cbrt_halley_iteration_step(r, y);
  r = cbrt_halley_iteration_step(r, y);

  // Finally multiply by 2^(e_div3).
  r = pldexp_fast(r, e_div3);
  return cbrt_special_cases_and_sign(x, r);
}

//----------------------------------------------------------------------
// Power Functions (accurate_log2, generic_pow, unary_pow)
//----------------------------------------------------------------------

// This function computes log2(x) and returns the result as a double word.
template <typename Scalar>
struct accurate_log2 {
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Packet& x, Packet& log2_x_hi, Packet& log2_x_lo) const {
    log2_x_hi = plog2(x);
    log2_x_lo = pzero(x);
  }
};

// This specialization uses a more accurate algorithm to compute log2(x) for
// floats in [1/sqrt(2);sqrt(2)] with a relative accuracy of ~6.56508e-10.
// This additional accuracy is needed to counter the error-magnification
// inherent in multiplying by a potentially large exponent in pow(x,y).
// The minimax polynomial used was calculated using the Rminimax tool,
// see https://gitlab.inria.fr/sfilip/rminimax.
// Command line:
//   $ ratapprox --function="log2(1+x)/x"  --dom='[-0.2929,0.41422]'
//   --type=[10,0]
//       --numF="[D,D,SG]" --denF="[SG]" --log --dispCoeff="dec"
//
// The resulting implementation of pow(x,y) is accurate to 3 ulps.
template <>
struct accurate_log2<float> {
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Packet& z, Packet& log2_x_hi, Packet& log2_x_lo) const {
    // Split the two lowest order constant coefficient into double-word representation.
    constexpr double kC0 = 1.442695041742110273474963832995854318141937255859375e+00;
    constexpr float kC0_hi = static_cast<float>(kC0);
    constexpr float kC0_lo = static_cast<float>(kC0 - static_cast<double>(kC0_hi));
    const Packet c0_hi = pset1<Packet>(kC0_hi);
    const Packet c0_lo = pset1<Packet>(kC0_lo);

    constexpr double kC1 = -7.2134751588268664068692714863573201000690460205078125e-01;
    constexpr float kC1_hi = static_cast<float>(kC1);
    constexpr float kC1_lo = static_cast<float>(kC1 - static_cast<double>(kC1_hi));
    const Packet c1_hi = pset1<Packet>(kC1_hi);
    const Packet c1_lo = pset1<Packet>(kC1_lo);

    constexpr float c[] = {
        9.7010828554630279541015625e-02,  -1.6896486282348632812500000e-01, 1.7200836539268493652343750e-01,
        -1.7892272770404815673828125e-01, 2.0505344867706298828125000e-01,  -2.4046677350997924804687500e-01,
        2.8857553005218505859375000e-01,  -3.6067414283752441406250000e-01, 4.8089790344238281250000000e-01};

    // Evaluate the higher order terms in the polynomial using
    // standard arithmetic.
    const Packet one = pset1<Packet>(1.0f);
    const Packet x = psub(z, one);
    Packet p = ppolevl<Packet, 8>::run(x, c);
    // Evaluate the final two steps in Horner's rule using double-word
    // arithmetic.
    Packet p_hi, p_lo;
    twoprod(x, p, p_hi, p_lo);
    fast_twosum(c1_hi, c1_lo, p_hi, p_lo, p_hi, p_lo);
    twoprod(p_hi, p_lo, x, p_hi, p_lo);
    fast_twosum(c0_hi, c0_lo, p_hi, p_lo, p_hi, p_lo);
    // Multiply by x to recover log2(z).
    twoprod(p_hi, p_lo, x, log2_x_hi, log2_x_lo);
  }
};

// This specialization uses a more accurate algorithm to compute log2(x) for
// floats in [1/sqrt(2);sqrt(2)] with a relative accuracy of ~1.27e-18.
// This additional accuracy is needed to counter the error-magnification
// inherent in multiplying by a potentially large exponent in pow(x,y).
// The minimax polynomial used was calculated using the Sollya tool.
// See sollya.org.

template <>
struct accurate_log2<double> {
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(const Packet& x, Packet& log2_x_hi, Packet& log2_x_lo) const {
    // We use a transformation of variables:
    //    r = c * (x-1) / (x+1),
    // such that
    //    log2(x) = log2((1 + r/c) / (1 - r/c)) = f(r).
    // The function f(r) can be approximated well using an odd polynomial
    // of the form
    //   P(r) = ((Q(r^2) * r^2 + C) * r^2 + 1) * r,
    // For the implementation of log2<double> here, Q is of degree 6 with
    // coefficient represented in working precision (double), while C is a
    // constant represented in extra precision as a double word to achieve
    // full accuracy.
    //
    // The polynomial coefficients were computed by the Sollya script:
    //
    // c = 2 / log(2);
    // trans = c * (x-1)/(x+1);
    // itrans = (1+x/c)/(1-x/c);
    // interval=[trans(sqrt(0.5)); trans(sqrt(2))];
    // print(interval);
    // f = log2(itrans(x));
    // p=fpminimax(f,[|1,3,5,7,9,11,13,15,17|],[|1,DD,double...|],interval,relative,floating);
    const Packet q12 = pset1<Packet>(2.87074255468000586e-9);
    const Packet q10 = pset1<Packet>(2.38957980901884082e-8);
    const Packet q8 = pset1<Packet>(2.31032094540014656e-7);
    const Packet q6 = pset1<Packet>(2.27279857398537278e-6);
    const Packet q4 = pset1<Packet>(2.31271023278625638e-5);
    const Packet q2 = pset1<Packet>(2.47556738444535513e-4);
    const Packet q0 = pset1<Packet>(2.88543873228900172e-3);
    const Packet C_hi = pset1<Packet>(0.0400377511598501157);
    const Packet C_lo = pset1<Packet>(-4.77726582251425391e-19);
    const Packet one = pset1<Packet>(1.0);

    const Packet cst_2_log2e_hi = pset1<Packet>(2.88539008177792677);
    const Packet cst_2_log2e_lo = pset1<Packet>(4.07660016854549667e-17);
    // c * (x - 1)
    Packet t_hi, t_lo;
    // t = c * (x-1)
    twoprod(cst_2_log2e_hi, cst_2_log2e_lo, psub(x, one), t_hi, t_lo);
    // r = c * (x-1) / (x+1),
    Packet r_hi, r_lo;
    doubleword_div_fp(t_hi, t_lo, padd(x, one), r_hi, r_lo);

    // r2 = r * r
    Packet r2_hi, r2_lo;
    twoprod(r_hi, r_lo, r_hi, r_lo, r2_hi, r2_lo);
    // r4 = r2 * r2
    Packet r4_hi, r4_lo;
    twoprod(r2_hi, r2_lo, r2_hi, r2_lo, r4_hi, r4_lo);

    // Evaluate Q(r^2) in working precision. We evaluate it in two parts
    // (even and odd in r^2) to improve instruction level parallelism.
    Packet q_even = pmadd(q12, r4_hi, q8);
    Packet q_odd = pmadd(q10, r4_hi, q6);
    q_even = pmadd(q_even, r4_hi, q4);
    q_odd = pmadd(q_odd, r4_hi, q2);
    q_even = pmadd(q_even, r4_hi, q0);
    Packet q = pmadd(q_odd, r2_hi, q_even);

    // Now evaluate the low order terms of P(x) in double word precision.
    // In the following, due to the increasing magnitude of the coefficients
    // and r being constrained to [-0.5, 0.5] we can use fast_twosum instead
    // of the slower twosum.
    // Q(r^2) * r^2
    Packet p_hi, p_lo;
    twoprod(r2_hi, r2_lo, q, p_hi, p_lo);
    // Q(r^2) * r^2 + C
    Packet p1_hi, p1_lo;
    fast_twosum(C_hi, C_lo, p_hi, p_lo, p1_hi, p1_lo);
    // (Q(r^2) * r^2 + C) * r^2
    Packet p2_hi, p2_lo;
    twoprod(r2_hi, r2_lo, p1_hi, p1_lo, p2_hi, p2_lo);
    // ((Q(r^2) * r^2 + C) * r^2 + 1)
    Packet p3_hi, p3_lo;
    fast_twosum(one, p2_hi, p2_lo, p3_hi, p3_lo);

    // log2(x) ~= ((Q(r^2) * r^2 + C) * r^2 + 1) * r
    twoprod(p3_hi, p3_lo, r_hi, r_lo, log2_x_hi, log2_x_lo);
  }
};

// This function implements the non-trivial case of pow(x,y) where x is
// positive and y is (possibly) non-integer.
// Formally, pow(x,y) = exp2(y * log2(x)), where exp2(x) is shorthand for 2^x.
// TODO(rmlarsen): We should probably add this as a packet op 'ppow', to make it
// easier to specialize or turn off for specific types and/or backends.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet generic_pow_impl(const Packet& x, const Packet& y) {
  using Scalar = typename unpacket_traits<Packet>::type;
  // Split x into exponent e_x and mantissa m_x.
  Packet e_x;
  Packet m_x = pfrexp(x, e_x);

  // Adjust m_x to lie in [1/sqrt(2):sqrt(2)] to minimize absolute error in log2(m_x).
  constexpr Scalar sqrt_half = Scalar(0.70710678118654752440);
  const Packet m_x_scale_mask = pcmp_lt(m_x, pset1<Packet>(sqrt_half));
  m_x = pselect(m_x_scale_mask, pmul(pset1<Packet>(Scalar(2)), m_x), m_x);
  e_x = pselect(m_x_scale_mask, psub(e_x, pset1<Packet>(Scalar(1))), e_x);

  // Compute log2(m_x) with 6 extra bits of accuracy.
  Packet rx_hi, rx_lo;
  accurate_log2<Scalar>()(m_x, rx_hi, rx_lo);

  // Compute the two terms {y * e_x, y * r_x} in f = y * log2(x) with doubled
  // precision using double word arithmetic.
  Packet f1_hi, f1_lo, f2_hi, f2_lo;
  twoprod(e_x, y, f1_hi, f1_lo);
  twoprod(rx_hi, rx_lo, y, f2_hi, f2_lo);
  // Sum the two terms in f using double word arithmetic. We know
  // that |e_x| > |log2(m_x)|, except for the case where e_x==0.
  // This means that we can use fast_twosum(f1,f2).
  // In the case e_x == 0, e_x * y = f1 = 0, so we don't lose any
  // accuracy by violating the assumption of fast_twosum, because
  // it's a no-op.
  Packet f_hi, f_lo;
  fast_twosum(f1_hi, f1_lo, f2_hi, f2_lo, f_hi, f_lo);

  // Split f into integer and fractional parts.
  Packet n_z, r_z;
  absolute_split(f_hi, n_z, r_z);
  r_z = padd(r_z, f_lo);
  Packet n_r;
  absolute_split(r_z, n_r, r_z);
  n_z = padd(n_z, n_r);

  // We now have an accurate split of f = n_z + r_z and can compute
  //   x^y = 2**{n_z + r_z) = exp2(r_z) * 2**{n_z}.
  // Multiplication by the second factor can be done exactly using pldexp(), since
  // it is an integer power of 2.
  const Packet e_r = generic_exp2(r_z);

  // Since we know that e_r is in [1/sqrt(2); sqrt(2)], we can use the fast version
  // of pldexp to multiply by 2**{n_z} when |n_z| is sufficiently small.
  constexpr Scalar kPldExpThresh = std::numeric_limits<Scalar>::max_exponent - 2;
  const Packet pldexp_fast_unsafe = pcmp_lt(pset1<Packet>(kPldExpThresh), pabs(n_z));
  if (predux_any(pldexp_fast_unsafe)) {
    return pldexp(e_r, n_z);
  }
  return pldexp_fast(e_r, n_z);
}

// Generic implementation of pow(x,y).
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS std::enable_if_t<!is_scalar<Packet>::value, Packet> generic_pow(
    const Packet& x, const Packet& y) {
  using Scalar = typename unpacket_traits<Packet>::type;

  const Packet cst_inf = pinf<Packet>();
  const Packet cst_zero = pset1<Packet>(Scalar(0));
  const Packet cst_one = pset1<Packet>(Scalar(1));

  const Packet x_abs = pabs(x);
  Packet result = generic_pow_impl(x_abs, y);

  // In the following we enforce the special case handling prescribed in
  // https://en.cppreference.com/w/cpp/numeric/math/pow.

  // Predicates for sign and magnitude of x.
  const Packet x_is_negative = pcmp_lt(x, cst_zero);
  const Packet x_is_zero = pcmp_eq(x, cst_zero);
  const Packet x_is_one = pcmp_eq(x, cst_one);
  const Packet x_has_signbit = psignbit(x);
  const Packet x_abs_gt_one = pcmp_lt(cst_one, x_abs);
  const Packet x_abs_is_inf = pcmp_eq(x_abs, cst_inf);

  // Predicates for sign and magnitude of y.
  const Packet y_abs = pabs(y);
  const Packet y_abs_is_inf = pcmp_eq(y_abs, cst_inf);
  const Packet y_is_negative = pcmp_lt(y, cst_zero);
  const Packet y_is_zero = pcmp_eq(y, cst_zero);
  const Packet y_is_one = pcmp_eq(y, cst_one);
  // Predicates for whether y is integer and odd/even.
  const Packet y_is_int = pandnot(pcmp_eq(pfloor(y), y), y_abs_is_inf);
  const Packet y_div_2 = pmul(y, pset1<Packet>(Scalar(0.5)));
  const Packet y_is_even = pcmp_eq(pround(y_div_2), y_div_2);
  const Packet y_is_odd_int = pandnot(y_is_int, y_is_even);
  // Smallest exponent for which (1 + epsilon) overflows to infinity.
  constexpr Scalar huge_exponent =
      (NumTraits<Scalar>::max_exponent() * Scalar(EIGEN_LN2)) / NumTraits<Scalar>::epsilon();
  const Packet y_abs_is_huge = pcmp_le(pset1<Packet>(huge_exponent), y_abs);

  // *  pow(base, exp) returns NaN if base is finite and negative
  //    and exp is finite and non-integer.
  // A packet comparison mask is all ones, a NaN bit pattern, so por selects NaN and pand selects a constant
  // against zero. Scalar masks hold the value one instead, but this overload takes packets only.
  result = por(pandnot(x_is_negative, y_is_int), result);

  // * pow(±0, exp), where exp is negative, finite, and is an even integer or
  // a non-integer, returns +∞
  // * pow(±0, exp), where exp is positive non-integer or a positive even
  // integer, returns +0
  // * pow(+0, exp), where exp is a negative odd integer, returns +∞
  // * pow(-0, exp), where exp is a negative odd integer, returns -∞
  // * pow(+0, exp), where exp is a positive odd integer, returns +0
  // * pow(-0, exp), where exp is a positive odd integer, returns -0
  // Sign is flipped by the rule below.
  result = pselect(x_is_zero, pand(y_is_negative, cst_inf), result);

  // pow(base, exp) returns -pow(abs(base), exp) if base has the sign bit set,
  // and exp is an odd integer exponent.
  result = pselect(pand(x_has_signbit, y_is_odd_int), pnegate(result), result);

  // * pow(base, -∞) returns +∞ for any |base|<1
  // * pow(base, -∞) returns +0 for any |base|>1
  // * pow(base, +∞) returns +0 for any |base|<1
  // * pow(base, +∞) returns +∞ for any |base|>1
  // * pow(±0, -∞) returns +∞
  // * pow(-1, +-∞) = 1
  Packet inf_y_val = pand(pxor(y_is_negative, x_abs_gt_one), cst_inf);
  inf_y_val = pselect(pcmp_eq(x, pset1<Packet>(Scalar(-1.0))), cst_one, inf_y_val);
  result = pselect(y_abs_is_huge, inf_y_val, result);

  // * pow(+∞, exp) returns +0 for any negative exp
  // * pow(+∞, exp) returns +∞ for any positive exp
  // * pow(-∞, exp) returns -0 if exp is a negative odd integer.
  // * pow(-∞, exp) returns +0 if exp is a negative non-integer or negative
  //     even integer.
  // * pow(-∞, exp) returns -∞ if exp is a positive odd integer.
  // * pow(-∞, exp) returns +∞ if exp is a positive non-integer or positive
  //     even integer.
  auto x_pos_inf_value = pandnot(cst_inf, y_is_negative);
  auto x_neg_inf_value = pselect(y_is_odd_int, pnegate(x_pos_inf_value), x_pos_inf_value);
  result = pselect(x_abs_is_inf, pselect(x_is_negative, x_neg_inf_value, x_pos_inf_value), result);

  // All cases of NaN inputs return NaN, except the two below.
  result = por(por(pisnan(x), pisnan(y)), result);

  // * pow(base, 1) returns base.
  // * pow(base, +/-0) returns 1, regardless of base, even NaN.
  // * pow(+1, exp) returns 1, regardless of exponent, even NaN.
  result = pselect(y_is_one, x, pselect(por(x_is_one, y_is_zero), cst_one, result));

  return result;
}

template <typename Scalar>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS std::enable_if_t<is_scalar<Scalar>::value, Scalar> generic_pow(
    const Scalar& x, const Scalar& y) {
  return numext::pow(x, y);
}

namespace unary_pow {

// Integer exponents up to this magnitude use repeated squaring; larger ones take the log/exp path of generic_pow
// where that can represent them. The crossover is where a squaring step per exponent bit stops being cheaper
// than generic_pow, whose cost per element is about four times higher for double than for float (half the
// lanes, longer polynomials): measured at 2^12 for float and 2^20 for double on AVX2 with FMA. Complex bases
// take their real scalar's value; it only limits their scalar path, since no vectorized complex pow exists.
template <typename Scalar>
constexpr numext::uint64_t max_squaring_exponent() {
  return numext::uint64_t(1) << (std::is_same<typename NumTraits<Scalar>::Real, double>::value ? 20 : 12);
}

template <typename ScalarExponent, bool IsInteger = NumTraits<ScalarExponent>::IsInteger>
struct exponent_helper {
  using safe_abs_type = numext::uint64_t;
  // this routine assumes that exp is an integer of magnitude at most max_squaring_exponent() stored as a floating
  // point type
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_abs_type safe_abs(const ScalarExponent& exp) {
    eigen_assert(((numext::isfinite)(exp) && exp == numext::floor(exp)) && "exp must be an integer");
    return static_cast<safe_abs_type>(numext::abs(exp));
  }
};

template <typename ScalarExponent>
struct exponent_helper<ScalarExponent, true> {
  // if `exp` is a signed integer type, cast it to its unsigned counterpart to safely store its absolute value
  // consider the (rare) case where `exp` is an int32_t: abs(-2147483648) != 2147483648
  using safe_abs_type = typename numext::get_integer_by_size<sizeof(ScalarExponent)>::unsigned_type;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_abs_type safe_abs(const ScalarExponent& exp) {
    ScalarExponent mask = numext::signbit(exp);
    safe_abs_type result = safe_abs_type(exp ^ mask);
    return result + safe_abs_type(ScalarExponent(1) & mask);
  }
};

template <typename ScalarExponent, bool IsSigned = NumTraits<ScalarExponent>::IsSigned>
struct exponent_is_negative {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const ScalarExponent& exponent) {
    return exponent < ScalarExponent(0);
  }
};

template <typename ScalarExponent>
struct exponent_is_negative<ScalarExponent, false> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const ScalarExponent&) { return false; }
};

// Left-to-right binary exponentiation, so that the base is multiplied in as is: each step squares the running
// power and multiplies by the base when the next exponent bit is set.
template <typename AbsExponentType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AbsExponentType highest_set_bit(AbsExponentType m) {
  AbsExponentType bit = AbsExponentType(1);
  while ((m >> 1) >= bit) bit <<= 1;
  return bit;
}

// Repeated squaring for integer bases, wrapping on overflow like the underlying multiplication.
template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet int_pow_wrapping(const Packet& x, const ScalarExponent& exponent) {
  using Scalar = typename unpacket_traits<Packet>::type;
  using ExponentHelper = exponent_helper<ScalarExponent>;
  using AbsExponentType = typename ExponentHelper::safe_abs_type;
  Packet cst_pos_one = pset1<Packet>(Scalar(1));
  if (exponent == ScalarExponent(0)) return cst_pos_one;
  eigen_assert(!exponent_is_negative<ScalarExponent>::run(exponent));

  AbsExponentType m = ExponentHelper::safe_abs(exponent);
  Packet y = x;
  for (AbsExponentType bit = highest_set_bit(m) >> 1; bit != 0; bit >>= 1) {
    y = pmul(y, y);
    if ((m & bit) != 0) y = pmul(y, x);
  }
  return y;
}

// Double-word repeated squaring needs an exact two-product and the binary layout below; it serves binary32 and
// binary64 bases, real or complex. Other bases keep plain repeated squaring.
template <typename Scalar>
struct is_double_word_base
    : bool_constant<(std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value) &&
                    std::numeric_limits<Scalar>::is_iec559> {};
template <typename RealScalar>
struct is_double_word_base<std::complex<RealScalar>> : is_double_word_base<RealScalar> {};

// Whether exponent_bits_shift_right/exponent_bits_sub are available, not whether integer_packet itself is:
// AVX/PacketMath.h supplies all three for Packet4d without AVX2, where there is no Packet4l.
template <typename Packet, typename>
struct has_exponent_bit_ops : false_type {};
template <typename Packet>
struct has_exponent_bit_ops<Packet, void_t<typename unpacket_traits<Packet>::integer_packet>> : true_type {};

template <typename Packet, bool IsComplex = NumTraits<typename unpacket_traits<Packet>::type>::IsComplex,
          bool IsScalar = is_scalar<Packet>::value>
struct real_view {
  using type = Packet;
};
template <typename Packet>
struct real_view<Packet, true, false> {
  using type = typename unpacket_traits<Packet>::as_real;
};
template <typename Scalar>
struct real_view<Scalar, true, true> {
  using type = typename NumTraits<Scalar>::Real;
};

template <typename Packet>
struct use_double_word : bool_constant<is_double_word_base<typename unpacket_traits<Packet>::type>::value &&
                                       has_exponent_bit_ops<typename real_view<Packet>::type>::value> {};

// Declared in GenericPacketMathFunctionsFwd.h so that a backend header can override them.
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet exponent_bits_shift_right(const Packet& bits) {
  using PacketI = typename unpacket_traits<Packet>::integer_packet;
  constexpr int kMantissaBits = numext::numeric_limits<typename unpacket_traits<Packet>::type>::digits - 1;
  return preinterpret<Packet>(plogical_shift_right<kMantissaBits>(preinterpret<PacketI>(bits)));
}
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet exponent_bits_sub(const Packet& a_bits, const Packet& b_bits) {
  using PacketI = typename unpacket_traits<Packet>::integer_packet;
  return preinterpret<Packet>(psub(preinterpret<PacketI>(a_bits), preinterpret<PacketI>(b_bits)));
}

// Power-of-two scaling of a real packet through its exponent bits. A finite value's magnitude is never let far
// from one, so the residuals of the double-word arithmetic stay normal (a subnormal residual costs a microcode
// assist on x86 for every operation that touches it) and no intermediate overflows or underflows.
template <typename Packet>
struct binary_exponent_scaling {
  using Scalar = typename unpacket_traits<Packet>::type;
  using Bits = std::make_unsigned_t<typename make_integer<Scalar>::type>;
  static constexpr int kMantissaBits = numext::numeric_limits<Scalar>::digits - 1;
  static constexpr int kBias = numext::numeric_limits<Scalar>::max_exponent - 1;
  static constexpr Bits kExponentMask = ((Bits(1) << (CHAR_BIT * sizeof(Scalar) - kMantissaBits - 1)) - Bits(1))
                                        << kMantissaBits;
  // 2^kMantissaBits: or-ing an integer below 2^kMantissaBits into its low bits adds that integer to it.
  static constexpr Bits kMagicBits = Bits(kBias + kMantissaBits) << kMantissaBits;

  // For a normal x = m * 2^e with 2 <= |m| < 4, returns 2^-e and sets e (as a floating-point value). The
  // target [2, 4) rather than [1, 2) keeps 2^-e a normal number for every x.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet inverse_scale(const Packet& x, Packet& e) {
    Packet exponent_bits = pand(x, pset1frombits<Packet>(kExponentMask));
    Packet biased = exponent_bits_shift_right(exponent_bits);
    Packet magic = pset1frombits<Packet>(kMagicBits);
    e = psub(por(biased, magic), padd(magic, pset1<Packet>(Scalar(kBias + 1))));
    // 2^(bias + 1 - E) has biased exponent 2 * bias + 1 - E, in [1, 2 * bias] for a normal x.
    Packet two_bias_plus_one = pset1frombits<Packet>(Bits(2 * kBias + 1) << kMantissaBits);
    return exponent_bits_sub(two_bias_plus_one, exponent_bits);
  }

  // x * 2^e for the scaled power, whose |x| lies within 2^(+-62): beyond the clamp the result is infinite or zero
  // either way, and the scalar pldexp converts the exponent to int. Below kZeroBelow the result rounds to zero,
  // which multiplying by zero gives exactly, whereas pldexp would underflow through several subnormal
  // intermediates, each a microcode assist on x86.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet scale_result(const Packet& x, const Packet& e) {
    constexpr int kLimit = 4 * numext::numeric_limits<Scalar>::max_exponent;
    constexpr int kZeroBelow =
        numext::numeric_limits<Scalar>::min_exponent - numext::numeric_limits<Scalar>::digits - 64;
    Packet zero_below = pset1<Packet>(Scalar(kZeroBelow));
    Packet value = pselect(pcmp_lt(e, zero_below), pmul(x, pzero(x)), x);
    return pldexp(value, pmin(pmax(e, zero_below), pset1<Packet>(Scalar(kLimit))));
  }

  // Factors x = m * 2^e with 2 <= |m| < 4 for a finite, nonzero x, where m = (x * lift) * scale: lift is 2^digits
  // for a subnormal x and one otherwise, and both products are exact. Zero and infinity make m NaN (their scale
  // is infinite, respectively zero), which the callers resolve at the end; NaN stays NaN.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void input_scale(const Packet& x, Packet& lift, Packet& scale,
                                                                Packet& e) {
    constexpr int kDigits = numext::numeric_limits<Scalar>::digits;
    Packet is_subnormal = pcmp_lt(pabs(x), pset1<Packet>((numext::numeric_limits<Scalar>::min)()));
    lift = pselect(is_subnormal, pset1<Packet>(Scalar(Bits(1) << kDigits)), pset1<Packet>(Scalar(1)));
    scale = inverse_scale(pmul(x, lift), e);
    e = psub(e, pselect(is_subnormal, pset1<Packet>(Scalar(kDigits)), pzero(x)));
  }
};

// The running power of repeated squaring: a double word {hi, lo} times 2^exponent, with the exponent kept as a
// floating-point value that is exact while the result is finite, plus what a zero or infinite base turns into:
// itself, or its reciprocal, with the sign dropped for an even exponent. Such a base makes hi NaN from the input
// scaling on, and a finite nonzero base cannot make hi NaN or zero since the scaled power stays within 2^(+-62),
// so result() substitutes the special value exactly where hi is NaN or zero. (A zero hi would also lose its sign
// in the double-word sums.) NaN needs nothing: it propagates.
template <typename Packet, bool IsComplex = NumTraits<typename unpacket_traits<Packet>::type>::IsComplex>
struct repeated_squaring_ops {
  using Scalar = typename unpacket_traits<Packet>::type;
  using Scaling = binary_exponent_scaling<Packet>;
  using Bound = Packet;
  struct State {
    Packet hi, lo, exponent, special;
  };
  // Whether every lane lies within [1/bound, bound], where the power and its residuals stay normal without scaling.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool in_range(const Packet& x, const Packet& bound) {
    Packet abs_x = pabs(x);
    Packet out = por(pcmp_lt(pmul(abs_x, bound), pset1<Packet>(Scalar(1))), pcmp_lt_or_nan(bound, abs_x));
    return !predux_any(out);
  }
  // The double word of m or of 1/m = q + (1 - q*m)/m, where 1 - q*m is formed from the exact product
  // q*m = p_hi + p_lo (1 - p_hi is exact as p_hi is within rounding of 1); renormalizing makes hi the correctly
  // rounded reciprocal.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void power_base(const Packet& m, bool reciprocal, Packet& hi,
                                                               Packet& lo) {
    if (!reciprocal) {
      hi = m;
      lo = pzero(m);
      return;
    }
    Packet cst_pos_one = pset1<Packet>(Scalar(1));
    Packet q = pdiv(cst_pos_one, m);
    Packet p_hi, p_lo;
    twoprod(q, m, p_hi, p_lo);
    fast_twosum(q, pdiv(psub(psub(cst_pos_one, p_hi), p_lo), m), hi, lo);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE State base(const Packet& x, bool reciprocal, bool scaled,
                                                          const Scalar&) {
    State b;
    if (!scaled) {
      power_base(x, reciprocal, b.hi, b.lo);
      b.exponent = pzero(x);
      b.special = x;
      return b;
    }
    Packet lift, scale;
    Scaling::input_scale(x, lift, scale, b.exponent);
    power_base(pmul(pmul(x, lift), scale), reciprocal, b.hi, b.lo);
    if (reciprocal) {
      b.exponent = pnegate(b.exponent);
      b.special = pdiv(pset1<Packet>(Scalar(1)), x);
    } else {
      b.special = x;
    }
    return b;
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void multiply(State& y, const State& b) {
    fast_twoprod(y.hi, y.lo, b.hi, b.lo, y.hi, y.lo);
    y.exponent = padd(y.exponent, b.exponent);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void square(State& y) { multiply(y, y); }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void renormalize(State& y) {
    Packet e;
    Packet scale = Scaling::inverse_scale(y.hi, e);
    y.hi = pmul(y.hi, scale);
    y.lo = pmul(y.lo, scale);
    y.exponent = padd(y.exponent, e);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet result(const Packet&, const State& y, const State& b, bool odd,
                                                             bool scaled) {
    if (!scaled) return y.hi;
    Packet use_special = por(pisnan(y.hi), pcmp_eq(y.hi, pzero(y.hi)));
    return pselect(use_special, odd ? b.special : pabs(b.special), Scaling::scale_result(y.hi, y.exponent));
  }
  // With a scaled |base| in [1/4, 4) a step at most cubes the magnitude bound, so four steps keep it within 2^(+-62)
  // and the residuals, u times smaller, normal.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int renormalization_steps(const Scalar&) { return 4; }
  // Zero, infinite and NaN bases are resolved by the special value in result().
  struct SpecialBases {
    static constexpr bool any = false;
  };
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SpecialBases special_bases(const Packet&, bool) {
    return SpecialBases();
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet without_special_bases(const Packet& x, const SpecialBases&) {
    return x;
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet with_special_bases(const Packet& r, const SpecialBases&, bool) {
    return r;
  }
};

// The real and imaginary parts of a complex value or packet as two values of a real representation R, on which
// the complex algorithm below runs component-wise. For a complex packet R is its interleaved real view with both
// lanes of a pair holding the same component, so every lane operation applies to the pair at once.
template <typename Packet, bool IsScalar = is_scalar<Packet>::value>
struct complex_components {
  using Scalar = typename unpacket_traits<Packet>::type;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using R = typename unpacket_traits<Packet>::as_real;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R flip(const R& x) { return pcplxflip(Packet(x)).v; }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R odd_lanes() {
    return pcmp_eq(pset1<Packet>(Scalar(0, 1)).v, pset1<R>(RealScalar(1)));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void split(const Packet& z, R& re, R& im) {
    R odd = odd_lanes();
    re = pselect(odd, flip(z.v), z.v);
    im = pselect(odd, z.v, flip(z.v));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet join(const R& re, const R& im) {
    return Packet(pselect(odd_lanes(), im, re));
  }
  // max(|re|, |im|) in both lanes of each pair.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R magnitude(const Packet& z) {
    R abs_z = pabs(z.v);
    return pmax(abs_z, flip(abs_z));
  }
  // min(|re|, |im|) in both lanes of each pair.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R minor_magnitude(const Packet& z) {
    R abs_z = pabs(z.v);
    return pmin(abs_z, flip(abs_z));
  }
  // Whether a component exceeds bound or is NaN, in its own lane: pmax need not propagate NaN.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R exceeds(const Packet& z, const R& bound) {
    return pcmp_lt_or_nan(bound, pabs(z.v));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool any_zero(const Packet& z) {
    return predux_any(pcmp_eq(z.v, pzero(z.v)));
  }
  // In both lanes of each pair: both components zero; a component infinite; a component NaN and none infinite.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void special_masks(const Packet& z, R& zero, R& inf, R& nan) {
    R zero_lane = pcmp_eq(z.v, pzero(z.v));
    R inf_lane = pcmp_eq(pabs(z.v), pset1<R>(NumTraits<RealScalar>::infinity()));
    R nan_lane = pisnan(z.v);
    zero = pand(zero_lane, flip(zero_lane));
    inf = por(inf_lane, flip(inf_lane));
    nan = pandnot(por(nan_lane, flip(nan_lane)), inf);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet replace(const R& mask, const Scalar& value, const Packet& z) {
    return Packet(pselect(mask, pset1<Packet>(value).v, z.v));
  }
};

template <typename Scalar>
struct complex_components<Scalar, true> {
  using R = typename NumTraits<Scalar>::Real;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void split(const Scalar& z, R& re, R& im) {
    re = numext::real(z);
    im = numext::imag(z);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar join(const R& re, const R& im) { return Scalar(re, im); }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R magnitude(const Scalar& z) {
    return numext::maxi(numext::abs(numext::real(z)), numext::abs(numext::imag(z)));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R minor_magnitude(const Scalar& z) {
    return numext::mini(numext::abs(numext::real(z)), numext::abs(numext::imag(z)));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE R exceeds(const Scalar& z, const R& bound) {
    return por(pcmp_lt_or_nan(bound, numext::abs(numext::real(z))),
               pcmp_lt_or_nan(bound, numext::abs(numext::imag(z))));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool any_zero(const Scalar& z) {
    return numext::real(z) == R(0) || numext::imag(z) == R(0);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void special_masks(const Scalar& z, R& zero, R& inf, R& nan) {
    R re = numext::real(z), im = numext::imag(z);
    bool is_inf = (numext::isinf)(re) || (numext::isinf)(im);
    zero = re == R(0) && im == R(0) ? R(1) : R(0);
    inf = is_inf ? R(1) : R(0);
    nan = !is_inf && ((numext::isnan)(re) || (numext::isnan)(im)) ? R(1) : R(0);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar replace(const R& mask, const Scalar& value, const Scalar& z) {
    return mask != R(0) ? value : z;
  }
};

// Complex bases keep the real and imaginary parts as separate double words sharing one exponent, scaled by the
// larger component: z^2 = (a^2 - b^2) + 2ab i is three products and a product (a + bi)(c + di) four. Either
// component may legitimately be zero. Zero and infinite bases are the two points 0 and infinity of the extended
// complex plane, as std::proj sees it: every value with an infinite component, a NaN one included, is the same
// infinity, +inf + 0i. So a zero base gives +0 + 0i for n > 0 and +inf + 0i for n < 0, an infinite base the reverse,
// whatever the signs of their zeros; a base with a NaN component and no infinite one gives NaN + NaN i. std::pow
// leaves all of these to the implementation, and libstdc++'s follow its evaluation order: 1/0 = inf + nan i,
// (-inf + 2i)^3 = nan + inf i.
template <typename Packet>
struct repeated_squaring_ops<Packet, true> {
  using Scalar = typename unpacket_traits<Packet>::type;
  using Real = typename NumTraits<Scalar>::Real;
  using Components = complex_components<Packet>;
  using R = typename Components::R;
  using Scaling = binary_exponent_scaling<R>;
  using Bound = R;
  // Lanes marked `separated` hold only the larger component L (a, or bi) of z = L (1 + delta); the power is
  // L^n (1 + i c) with i c = n delta, c = (coefficient_hi + coefficient_lo) 2^coefficient_exponent (see base()).
  struct State {
    R re_hi, re_lo, im_hi, im_lo, exponent;
    R separated, coefficient_hi, coefficient_lo, coefficient_exponent;
    bool any_separated;
    bool negative, any_zero;
  };
  // Four steps between renormalizations let the power fall to 2^-62 of the scale when the base is a reciprocal in
  // [1/4, 1/2), and the smaller component, of order r = |s/L| times the power, keeps normal residuals only for
  // r >= 2^(min_exponent - 1 + digits + 62). Below that the base is separated (see base()), and first order is
  // exact to within (n r)^2 / 2 <= u/4 for |n| <= 2^kFirstOrderLog2Limit: every n for double, 2^25 for float.
  // Larger n renormalize at every step instead, which keeps the power above 2^-6 (a step from [2, 4), or from the
  // base itself) and lets the separation drop 56 binades, where first order holds for any n.
  static constexpr int kMinExponent = numext::numeric_limits<Real>::min_exponent;
  static constexpr int kDigits = numext::numeric_limits<Real>::digits;
  static constexpr int kSeparationExponent = kMinExponent + kDigits + 64;
  static constexpr int kFirstOrderLog2Limit = (1 - kDigits - 2 * kSeparationExponent) / 2;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool every_step(const Real& count) {
    return kFirstOrderLog2Limit < 64 &&
           count > Real(numext::uint64_t(1) << (kFirstOrderLog2Limit < 64 ? kFirstOrderLog2Limit : 0));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int renormalization_steps(const Real& count) {
    return every_step(count) ? 1 : 4;
  }
  // Whether every lane's max(|re|, |im|), and min(|re|, |im|) unless zero, lie within [1/bound, bound], where the
  // power and its residuals stay normal without scaling. A zero component stays exactly zero in every power.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool in_range(const Packet& x, const R& bound) {
    R magnitude = Components::magnitude(x);
    R minor = Components::minor_magnitude(x);
    R one = pset1<R>(Real(1));
    R out = por(pcmp_lt(pmul(magnitude, bound), one), Components::exceeds(x, bound));
    out = por(out, pandnot(pcmp_lt(pmul(minor, bound), one), pcmp_eq(minor, pzero(minor))));
    return predux_any(out) == false;
  }
  template <typename Count>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE State base(const Packet& x, bool reciprocal, bool scaled,
                                                          const Count& m) {
    State b;
    b.any_separated = false;
    R re, im;
    Components::split(x, re, im);
    b.negative = reciprocal;
    b.any_zero = Components::any_zero(x);
    if (!scaled) {
      power_base(re, im, reciprocal, b);
      b.exponent = pzero(re);
      return b;
    }
    R lift, scale;
    Scaling::input_scale(Components::magnitude(x), lift, scale, b.exponent);
    R wr = pmul(pmul(re, lift), scale), wi = pmul(pmul(im, lift), scale);
    // Both components take the larger one's scale, under which a nonzero smaller component s far below the larger
    // one L would lose its residuals (see kSeparationExponent). Then z = L (1 + delta) with |delta| tiny and
    // z^n = L^n (1 + n delta) to working precision: the power runs on L alone and result() adds L^n n delta, with
    // delta = i b/a, or a/(bi) = -i a/b, and -delta for the reciprocal base.
    int separation_exponent = every_step(Real(m)) ? kSeparationExponent - 56 : kSeparationExponent;
    R separation =
        pset1frombits<R>(typename Scaling::Bits(Scaling::kBias + separation_exponent) << Scaling::kMantissaBits);
    R zero = pzero(wr);
    R re_separated = pandnot(pcmp_lt(pabs(wr), separation), pcmp_eq(re, zero));
    R im_separated = pandnot(pcmp_lt(pabs(wi), separation), pcmp_eq(im, zero));
    b.separated = por(re_separated, im_separated);
    b.any_separated = predux_any(b.separated);
    if (b.any_separated) {
      R s_lift, s_scale, s_exponent, count_exponent;
      Scaling::input_scale(pselect(im_separated, im, re), s_lift, s_scale, s_exponent);
      R s = pmul(pmul(pselect(im_separated, im, re), s_lift), s_scale);
      // The count n is a double word too, as Real(n) is inexact beyond 2^digits: its leading digits, which Real
      // holds exactly, and the rest.
      numext::uint64_t n = numext::uint64_t(m);
      numext::uint64_t unit = highest_set_bit(n) >> (kDigits - 1);
      numext::uint64_t n_hi = unit > 1 ? n & ~(unit - 1) : n;
      R count_hi = pset1<R>(Real(n_hi)), count_lo = pset1<R>(Real(n - n_hi));
      R count_scale = Scaling::inverse_scale(count_hi, count_exponent);
      // The coefficient n s / L is a double word so that result() rounds the added component once: the ratio
      // q + (s - q L) / L, where s - q L = (s - p_hi) - p_lo is exact, times the count. The scaled |s / L| is in
      // (1/2, 2) and the scaled count in [2, 4); an eighth keeps |c hi| within 2^62.
      R l = pselect(im_separated, wr, wi);
      R q = pdiv(s, l);
      R p_hi, p_lo, c_hi, c_lo;
      twoprod(q, l, p_hi, p_lo);
      R q_lo = pdiv(psub(psub(s, p_hi), p_lo), l);
      R scaled_count = pmul(count_hi, count_scale);
      R scaled_count_lo = pmul(count_lo, count_scale);
      twoprod(scaled_count, q, c_hi, c_lo);
      fast_twosum(c_hi, pmadd(scaled_count, q_lo, pmadd(scaled_count_lo, q, c_lo)), p_hi, p_lo);
      R sign = pselect(im_separated, pset1<R>(Real(reciprocal ? -0.125 : 0.125)),
                       pset1<R>(Real(reciprocal ? 0.125 : -0.125)));
      b.coefficient_hi = pmul(p_hi, sign);
      b.coefficient_lo = pmul(p_lo, sign);
      b.coefficient_exponent = padd(padd(psub(s_exponent, b.exponent), count_exponent), pset1<R>(Real(3)));
      wr = pselect(re_separated, zero, wr);
      wi = pselect(im_separated, zero, wi);
    }
    power_base(wr, wi, reciprocal, b);
    if (reciprocal) b.exponent = pnegate(b.exponent);
    return b;
  }
  // The double words of w or of 1/w = q + e*q with q = conj(w)/|w|^2 and e = 1 - q*w = er - t i. The norm cannot
  // over- or underflow for max(|wr|, |wi|) in [2, 4) or within the unscaled range, and the few ulps of error in q
  // are what the residual corrects: e is of order u and is formed from exact products so that e*q carries the
  // correction to order u^2. 1 - s_hi is exact as s_hi is within rounding of 1.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void power_base(const R& wr, const R& wi, bool reciprocal, State& b) {
    b.re_lo = b.im_lo = pzero(wr);
    if (!reciprocal) {
      b.re_hi = wr;
      b.im_hi = wi;
      return;
    }
    R inv = pdiv(pset1<R>(typename NumTraits<Scalar>::Real(1)), pmadd(wr, wr, pmul(wi, wi)));
    R qr = pmul(wr, inv), qi = pnegate(pmul(wi, inv));
    R a_hi, a_lo, c_hi, c_lo, s_hi, s_lo, t_hi, t_lo;
    twoprod(qr, wr, a_hi, a_lo);
    twoprod(qi, wi, c_hi, c_lo);
    twodiff(a_hi, a_lo, c_hi, c_lo, s_hi, s_lo);
    twoprod(qr, wi, a_hi, a_lo);
    twoprod(qi, wr, c_hi, c_lo);
    twosum(a_hi, a_lo, c_hi, c_lo, t_hi, t_lo);
    R er = psub(psub(pset1<R>(typename NumTraits<Scalar>::Real(1)), s_hi), s_lo);
    R t = padd(t_hi, t_lo);
    fast_twosum(qr, pmadd(er, qr, pmul(t, qi)), b.re_hi, b.re_lo);
    fast_twosum(qi, pmsub(er, qi, pmul(t, qr)), b.im_hi, b.im_lo);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void square(State& y) {
    R a_hi, a_lo, c_hi, c_lo, p_hi, p_lo;
    fast_twoprod(y.re_hi, y.re_lo, y.re_hi, y.re_lo, a_hi, a_lo);
    fast_twoprod(y.im_hi, y.im_lo, y.im_hi, y.im_lo, c_hi, c_lo);
    fast_twoprod(y.re_hi, y.re_lo, y.im_hi, y.im_lo, p_hi, p_lo);
    twodiff(a_hi, a_lo, c_hi, c_lo, y.re_hi, y.re_lo);
    y.im_hi = padd(p_hi, p_hi);
    y.im_lo = padd(p_lo, p_lo);
    y.exponent = padd(y.exponent, y.exponent);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void multiply(State& y, const State& b) {
    R ac_hi, ac_lo, bd_hi, bd_lo, ad_hi, ad_lo, bc_hi, bc_lo;
    fast_twoprod(y.re_hi, y.re_lo, b.re_hi, b.re_lo, ac_hi, ac_lo);
    fast_twoprod(y.im_hi, y.im_lo, b.im_hi, b.im_lo, bd_hi, bd_lo);
    fast_twoprod(y.re_hi, y.re_lo, b.im_hi, b.im_lo, ad_hi, ad_lo);
    fast_twoprod(y.im_hi, y.im_lo, b.re_hi, b.re_lo, bc_hi, bc_lo);
    twodiff(ac_hi, ac_lo, bd_hi, bd_lo, y.re_hi, y.re_lo);
    twosum(ad_hi, ad_lo, bc_hi, bc_lo, y.im_hi, y.im_lo);
    y.exponent = padd(y.exponent, b.exponent);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void renormalize(State& y) {
    R e;
    R scale = Scaling::inverse_scale(pmax(pabs(y.re_hi), pabs(y.im_hi)), e);
    y.re_hi = pmul(y.re_hi, scale);
    y.re_lo = pmul(y.re_lo, scale);
    y.im_hi = pmul(y.im_hi, scale);
    y.im_lo = pmul(y.im_lo, scale);
    y.exponent = padd(y.exponent, e);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet result(const Packet& x, const State& y, const State& b, bool odd,
                                                             bool scaled) {
    if (!scaled) {
      R re = y.re_hi, im = y.im_hi;
      if (b.any_zero) zero_component_signs(x, b.negative, odd, re, im);
      return Components::join(re, im);
    }
    R re = Scaling::scale_result(y.re_hi, y.exponent);
    R im = Scaling::scale_result(y.im_hi, y.exponent);
    if (b.any_separated) {
      // L^n is real or imaginary, and i c L^n fills the component it leaves exactly zero.
      R e = padd(y.exponent, b.coefficient_exponent);
      R zero = pzero(re);
      R re_hi, re_lo, im_hi, im_lo;
      fast_twoprod(b.coefficient_hi, b.coefficient_lo, y.im_hi, y.im_lo, re_hi, re_lo);
      fast_twoprod(b.coefficient_hi, b.coefficient_lo, y.re_hi, y.re_lo, im_hi, im_lo);
      re = pselect(pand(b.separated, pcmp_eq(y.re_hi, zero)), Scaling::scale_result(pnegate(re_hi), e), re);
      im = pselect(pand(b.separated, pcmp_eq(y.im_hi, zero)), Scaling::scale_result(im_hi, e), im);
    }
    if (b.any_zero) zero_component_signs(x, b.negative, odd, re, im);
    return Components::join(re, im);
  }
  // Zero, infinite and NaN bases are replaced by 1 + i before the loop, so that they neither force the scaled loop on
  // their packet nor come back NaN, and their results are set at the end (see the comment above the struct). 1 + i
  // has no zero component, which would take the packet through zero_component_signs. Every one of them fails
  // in_range, so only a packet bound for the scaled loop is checked, and only whether there are any is carried through
  // the loop.
  struct SpecialBases {
    Packet x;
    bool any;
  };
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SpecialBases special_bases(const Packet& x, bool scaled) {
    SpecialBases special;
    special.x = x;
    special.any = false;
    if (scaled) {
      R zero, inf, nan;
      Components::special_masks(x, zero, inf, nan);
      special.any = predux_any(por(por(zero, inf), nan));
    }
    return special;
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet without_special_bases(const Packet& x,
                                                                            const SpecialBases& special) {
    if (!special.any) return x;
    R zero, inf, nan;
    Components::special_masks(x, zero, inf, nan);
    return Components::replace(por(por(zero, inf), nan), Scalar(1, 1), x);
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet with_special_bases(const Packet& r, const SpecialBases& special,
                                                                         bool negative) {
    if (!special.any) return r;
    R zero, inf, nan;
    Components::special_masks(special.x, zero, inf, nan);
    Real real_inf = NumTraits<Real>::infinity(), real_nan = NumTraits<Real>::quiet_NaN();
    Packet out = Components::replace(negative ? zero : inf, Scalar(real_inf, Real(0)), r);
    out = Components::replace(negative ? inf : zero, Scalar(0), out);
    return Components::replace(nan, Scalar(real_nan, real_nan), out);
  }
  // A base with one component s = +-0 is the limit of L (1 + i t), with t = s/L for a real L and -s/b for L = bi,
  // so z^n = L^n (1 + i n t): the zero component of the power has the sign of i n t L^n, which the double-word sums
  // lose (-0 + 0 = +0). Sign bits combine by xor; a scalar comparison mask is one rather than all ones, so masks
  // select.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void zero_component_signs(const Packet& x, bool negative, bool odd,
                                                                         R& re, R& im) {
    R zero = pzero(re);
    R x_re, x_im;
    Components::split(x, x_re, x_im);
    R imaginary = pcmp_eq(x_re, zero);
    R one_zero = pxor(imaginary, pcmp_eq(x_im, zero));
    R minus_zero = pset1<R>(Real(-0.0));
    // i^2 = -1 from t for L = bi and from i t L^n for an imaginary power cancel for odd n.
    R power_imaginary = odd ? imaginary : zero;
    R n_sign = negative ? minus_zero : zero;
    R extra = odd ? n_sign : pselect(imaginary, pxor(n_sign, minus_zero), n_sign);
    R sign = pxor(pand(pxor(pxor(x_re, x_im), pselect(power_imaginary, im, re)), minus_zero), extra);
    re = pselect(pand(one_zero, power_imaginary), sign, re);
    im = pselect(pandnot(one_zero, power_imaginary), sign, im);
  }
};

// Plain repeated squaring for the remaining floating-point bases.
template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet int_pow_plain(const Packet& x, const ScalarExponent& exponent) {
  using Scalar = typename unpacket_traits<Packet>::type;
  using ExponentHelper = exponent_helper<ScalarExponent>;
  using AbsExponentType = typename ExponentHelper::safe_abs_type;
  if (exponent == ScalarExponent(0)) return pset1<Packet>(Scalar(1));

  Packet base = exponent_is_negative<ScalarExponent>::run(exponent) ? pdiv(pset1<Packet>(Scalar(1)), x) : x;
  AbsExponentType m = ExponentHelper::safe_abs(exponent);
  Packet y = base;
  for (AbsExponentType bit = highest_set_bit(m) >> 1; bit != 0; bit >>= 1) {
    y = pmul(y, y);
    if ((m & bit) != 0) y = pmul(y, base);
  }
  return y;
}

// Repeated squaring for a binary32 or binary64 base, real or complex, carried in double-word arithmetic. Plain
// repeated squaring doubles the accumulated rounding error at every squaring, so its error grows like n * u for
// x^n. Keeping the running power as an unevaluated sum {hi, lo} bounds each step's relative error by about
// 7 * u^2 (fast_twoprod), so the result is correctly rounded unless the exact power lies within about 7 * n * u^2
// of a rounding boundary, and stays within 1 ulp up to n = max_squaring_exponent() for float. For a complex base
// the bound is normwise, relative to |z^n|, as a^2 - b^2 and ac - bd can cancel, and the n u^2 term passes u/2
// as |n| approaches 1/u: (0.6 + 0.8i)^n in complex<float> is 0.2 u off at n = 4096, 0.5 u at 2^24 and 4.2 u at
// 2^31 - 1. The power is scaled by powers of two throughout and only the final pldexp can overflow or underflow.
template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet int_pow_double_word(const Packet& x, const ScalarExponent& exponent) {
  using Scalar = typename unpacket_traits<Packet>::type;
  using ExponentHelper = exponent_helper<ScalarExponent>;
  using AbsExponentType = typename ExponentHelper::safe_abs_type;
  using Ops = repeated_squaring_ops<Packet>;
  if (exponent == ScalarExponent(0)) return pset1<Packet>(Scalar(1));

  bool negative = exponent_is_negative<ScalarExponent>::run(exponent);
  AbsExponentType m = ExponentHelper::safe_abs(exponent);
  bool odd = (m & AbsExponentType(1)) != 0;
  if (m == AbsExponentType(1) && !negative) return x;
  // A real x^-1 and x^2 are a single correctly rounded operation, which is what the double word would produce.
  EIGEN_IF_CONSTEXPR (!NumTraits<Scalar>::IsComplex) {
    if (m == AbsExponentType(1)) return pdiv(pset1<Packet>(Scalar(1)), x);
    if (m == AbsExponentType(2) && !negative) return pmul(x, x);
  }
  AbsExponentType top = highest_set_bit(m);

  // Bases whose magnitude lies within 2^(+-B) for B = budget / |n| keep every power up to x^n and its residuals
  // normal: |x^n| <= 2^budget < max and u * |x^n| >= 2^(min_exponent - 1), the smallest normal, so the loop
  // needs neither scaling nor exponents and the result is hi itself. A complex base is tested on both nonzero
  // components, as every term a^k b^(n-k) of its power must stay normal too; its magnitude exceeds the larger one
  // by up to sqrt(2), so B is one less to absorb sqrt(2)^n (an exhausted budget therefore admits nothing), and its
  // reciprocal forms |w|^2, which bounds B by half the exponent range. Zero, infinity and NaN fail the test.
  using Real = typename NumTraits<Scalar>::Real;
  using RealBits = std::make_unsigned_t<typename make_integer<Real>::type>;
  constexpr int kBudget = -(numext::numeric_limits<Real>::min_exponent + numext::numeric_limits<Real>::digits);
  constexpr int kMantissaBits = numext::numeric_limits<Real>::digits - 1;
  constexpr RealBits kBiasBits = RealBits(numext::numeric_limits<Real>::max_exponent - 1);
  int b = numext::uint64_t(m) > numext::uint64_t(kBudget) ? 0 : kBudget / int(m);
  EIGEN_IF_CONSTEXPR (NumTraits<Scalar>::IsComplex)
    b = numext::mini(b - 1, (numext::numeric_limits<Real>::max_exponent - 1) / 2);
  typename Ops::Bound bound =
      pset1frombits<typename Ops::Bound>(RealBits(kBiasBits + RealBits(b < 0 ? 0 : b)) << kMantissaBits);
  bool scaled = b < 0 || !Ops::in_range(x, bound);
  // Zero, infinite and NaN bases are never in range, so only a packet bound for the scaled loop can hold one.
  typename Ops::SpecialBases special = Ops::special_bases(x, scaled);
  Packet x_regular = Ops::without_special_bases(x, special);
  if (special.any) scaled = b < 0 || !Ops::in_range(x_regular, bound);

  typename Ops::State base = Ops::base(x_regular, negative, scaled, m);
  typename Ops::State y = base;
  int renormalization_steps = Ops::renormalization_steps(Real(m));
  int steps_since_renormalization = 0;
  for (AbsExponentType bit = top >> 1; bit != 0; bit >>= 1) {
    Ops::square(y);
    if ((m & bit) != 0) Ops::multiply(y, base);
    if (scaled && ++steps_since_renormalization == renormalization_steps) {
      Ops::renormalize(y);
      steps_since_renormalization = 0;
    }
  }
  return Ops::with_special_bases(Ops::result(x_regular, y, base, odd, scaled), special, negative);
}

template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet int_pow(const Packet& x, const ScalarExponent& exponent, true_type) {
  return int_pow_double_word(x, exponent);
}

template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet int_pow(const Packet& x, const ScalarExponent& exponent, false_type) {
  return int_pow_plain(x, exponent);
}

template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet int_pow(const Packet& x, const ScalarExponent& exponent) {
  return int_pow(x, exponent, use_double_word<Packet>());
}

// The largest integer-valued floating-point exponent that repeated squaring handles; beyond it generic_pow takes
// over. Plain squaring is accurate to a few ulps only for small exponents.
template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool use_repeated_squaring(const ScalarExponent& exponent) {
  using Scalar = typename unpacket_traits<Packet>::type;
  return use_double_word<Packet>::value ? numext::abs(exponent) <= ScalarExponent(max_squaring_exponent<Scalar>())
                                        : (exponent <= ScalarExponent(7) && exponent >= ScalarExponent(-3));
}

// The same for an exponent of integer type, which repeated squaring handles exactly whatever its size: generic_pow
// only takes over where the exponent converts exactly to the base type.
template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool use_repeated_squaring_for_integer(const ScalarExponent& exponent) {
  using Scalar = typename unpacket_traits<Packet>::type;
  constexpr numext::uint64_t kExactLimit = numext::uint64_t(1) << numext::numeric_limits<Scalar>::digits;
  return exponent_helper<ScalarExponent>::safe_abs(exponent) > kExactLimit ||
         use_repeated_squaring<Packet>(static_cast<Scalar>(exponent));
}

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::enable_if_t<!is_scalar<Packet>::value, Packet> gen_pow(
    const Packet& x, const typename unpacket_traits<Packet>::type& exponent) {
  const Packet exponent_packet = pset1<Packet>(exponent);
  // generic_pow_impl requires positive x; sign/error handling is done by the caller.
  return generic_pow_impl(pabs(x), exponent_packet);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::enable_if_t<is_scalar<Scalar>::value, Scalar> gen_pow(
    const Scalar& x, const Scalar& exponent) {
  return numext::pow(x, exponent);
}

// Handle special cases for pow(x, exponent) where both base and exponent are
// floating point and the exponent is a non-integer scalar (uniform across all
// SIMD lanes). This allows us to use scalar branches on exponent properties.
// Reached from packetOp only, so the comparison masks are all ones and select a NaN or an infinity bitwise.
template <typename Packet, typename ScalarExponent>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet handle_nonint_nonint_errors(const Packet& x, const Packet& powx,
                                                                         const ScalarExponent& exponent) {
  using Scalar = typename unpacket_traits<Packet>::type;
  const Packet cst_zero = pzero(x);
  const Packet cst_one = pset1<Packet>(Scalar(1));
  const Packet cst_inf = pinf<Packet>();
  const Packet cst_nan = pnan<Packet>();

  const Packet abs_x = pabs(x);

  // x < 0 with non-integer exponent -> NaN.
  Packet result = por(pcmp_lt(x, cst_zero), powx);

  if (!(numext::isfinite)(exponent)) {
    if (exponent != exponent) {
      // pow(x, NaN) = NaN, except pow(+1, NaN) = 1.
      result = pselect(pcmp_eq(x, cst_one), cst_one, cst_nan);
    } else {
      // Exponent is +inf or -inf.
      const Packet abs_x_is_one = pcmp_eq(abs_x, cst_one);
      if (exponent > ScalarExponent(0)) {
        // pow(x, +inf): |x| > 1 -> +inf, |x| < 1 -> 0, |x| == 1 -> 1.
        result = pand(pcmp_lt(cst_one, abs_x), cst_inf);
      } else {
        // pow(x, -inf): |x| < 1 -> +inf, |x| > 1 -> 0, |x| == 1 -> 1.
        result = pand(pcmp_lt(abs_x, cst_one), cst_inf);
      }
      // pow(+-1, +-inf) = 1.
      result = pselect(abs_x_is_one, cst_one, result);
    }
  } else {
    // Finite non-integer exponent.
    const Packet x_is_zero = pcmp_eq(x, cst_zero);
    const Packet abs_x_is_inf = pcmp_eq(abs_x, cst_inf);
    if (exponent < ScalarExponent(0)) {
      // pow(+-0, negative non-integer) = +inf. pow(+-inf, negative) = +0.
      result = pselect(x_is_zero, cst_inf, result);
      result = pselect(abs_x_is_inf, cst_zero, result);
    } else {
      // pow(+-0, positive non-integer) = +0. pow(+-inf, positive) = +inf.
      result = pselect(x_is_zero, cst_zero, result);
      result = pselect(abs_x_is_inf, cst_inf, result);
    }
  }

  // NaN base produces NaN. This overrides all cases above, but pow(NaN, 0) = 1
  // and pow(NaN, integer) are handled by the integer exponent path and never
  // reach this function.
  result = por(pisnan(x), result);

  return result;
}

template <typename Packet, typename ScalarExponent,
          std::enable_if_t<NumTraits<typename unpacket_traits<Packet>::type>::IsSigned, bool> = true>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet handle_negative_exponent(const Packet& x, const ScalarExponent& exponent) {
  using Scalar = typename unpacket_traits<Packet>::type;

  // signed integer base, signed integer exponent case

  // This routine handles negative exponents.
  // The return value is either 0, 1, or -1.
  Packet cst_pos_one = pset1<Packet>(Scalar(1));
  const bool exponent_is_odd = exponent % ScalarExponent(2) != ScalarExponent(0);
  const Packet exp_is_odd = exponent_is_odd ? ptrue<Packet>(x) : pzero<Packet>(x);

  const Packet abs_x = pabs(x);
  const Packet abs_x_is_one = pcmp_eq(abs_x, cst_pos_one);

  Packet result = pselect(exp_is_odd, x, abs_x);
  result = pselect(abs_x_is_one, result, pzero<Packet>(x));
  return result;
}

template <typename Packet, typename ScalarExponent,
          std::enable_if_t<!NumTraits<typename unpacket_traits<Packet>::type>::IsSigned, bool> = true>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet handle_negative_exponent(const Packet& x, const ScalarExponent&) {
  using Scalar = typename unpacket_traits<Packet>::type;

  // unsigned integer base, signed integer exponent case

  // This routine handles negative exponents.
  // The return value is either 0 or 1

  const Scalar pos_one = Scalar(1);

  const Packet cst_pos_one = pset1<Packet>(pos_one);

  const Packet x_is_one = pcmp_eq(x, cst_pos_one);

  return pand(x_is_one, x);
}

}  // end namespace unary_pow

template <typename Packet, typename ScalarExponent,
          bool BaseIsIntegerType = NumTraits<typename unpacket_traits<Packet>::type>::IsInteger,
          bool ExponentIsIntegerType = NumTraits<ScalarExponent>::IsInteger,
          bool ExponentIsSigned = NumTraits<ScalarExponent>::IsSigned>
struct unary_pow_impl;

template <typename Packet, typename ScalarExponent, bool ExponentIsSigned>
struct unary_pow_impl<Packet, ScalarExponent, false, false, ExponentIsSigned> {
  using Scalar = typename unpacket_traits<Packet>::type;
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet run(const Packet& x, const ScalarExponent& exponent) {
    const bool exponent_is_integer = (numext::isfinite)(exponent) && numext::round(exponent) == exponent;
    if (exponent_is_integer) {
      return unary_pow::use_repeated_squaring<Packet>(exponent) ? unary_pow::int_pow(x, exponent)
                                                                : generic_pow(x, pset1<Packet>(exponent));
    } else {
      Packet result = unary_pow::gen_pow(x, exponent);
      result = unary_pow::handle_nonint_nonint_errors(x, result, exponent);
      return result;
    }
  }
};

template <typename Packet, typename ScalarExponent, bool ExponentIsSigned>
struct unary_pow_impl<Packet, ScalarExponent, false, true, ExponentIsSigned> {
  using Scalar = typename unpacket_traits<Packet>::type;
  // Only real float and double bases with double-word support fall back to generic_pow for large exponents:
  // complex bases have no vectorized generic_pow, and the bases left without double-word support (half,
  // bfloat16, the GPU packets, the MSA and HVX float packets) have no packet generic_pow either, so their
  // squaring loop -- at most 64 steps -- is all there is.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet run(const Packet& x, const ScalarExponent& exponent) {
    return run(x, exponent,
               bool_constant < unary_pow::use_double_word<Packet>::value && !NumTraits<Scalar>::IsComplex > ());
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet run(const Packet& x, const ScalarExponent& exponent, true_type) {
    return unary_pow::use_repeated_squaring_for_integer<Packet>(exponent)
               ? unary_pow::int_pow(x, exponent)
               : generic_pow(x, pset1<Packet>(static_cast<Scalar>(exponent)));
  }
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet run(const Packet& x, const ScalarExponent& exponent, false_type) {
    return unary_pow::int_pow(x, exponent);
  }
};

template <typename Packet, typename ScalarExponent>
struct unary_pow_impl<Packet, ScalarExponent, true, true, true> {
  using Scalar = typename unpacket_traits<Packet>::type;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet run(const Packet& x, const ScalarExponent& exponent) {
    if (exponent < ScalarExponent(0)) {
      return unary_pow::handle_negative_exponent(x, exponent);
    } else {
      return unary_pow::int_pow_wrapping(x, exponent);
    }
  }
};

template <typename Packet, typename ScalarExponent>
struct unary_pow_impl<Packet, ScalarExponent, true, true, false> {
  using Scalar = typename unpacket_traits<Packet>::type;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet run(const Packet& x, const ScalarExponent& exponent) {
    return unary_pow::int_pow_wrapping(x, exponent);
  }
};

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_ARCH_GENERIC_PACKET_MATH_POW_H
