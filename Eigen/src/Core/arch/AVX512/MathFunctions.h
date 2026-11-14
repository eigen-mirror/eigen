// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Pedro Gonnet (pedro.gonnet@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_
#define THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet16f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet8d)

template <>
EIGEN_STRONG_INLINE Packet16h pfrexp(const Packet16h& a, Packet16h& exponent) {
  Packet16f fexponent;
  const Packet16h out = float2half(pfrexp<Packet16f>(half2float(a), fexponent));
  exponent = float2half(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet16h pldexp(const Packet16h& a, const Packet16h& exponent) {
  return float2half(pldexp<Packet16f>(half2float(a), half2float(exponent)));
}

template <>
EIGEN_STRONG_INLINE Packet16bf pfrexp(const Packet16bf& a, Packet16bf& exponent) {
  // Both results are exact: the mantissa keeps the input's significand, and the exponent is a
  // small integer.
  Packet16f fexponent;
  const Packet16bf out = F32ToBf16Truncate(pfrexp<Packet16f>(Bf16ToF32(a), fexponent));
  exponent = F32ToBf16Truncate(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet16bf pldexp(const Packet16bf& a, const Packet16bf& exponent) {
  return F32ToBf16(pldexp<Packet16f>(Bf16ToF32(a), Bf16ToF32(exponent)));
}

#if EIGEN_FAST_MATH
// Square root from the estimate r of 1 / sqrt(x) that rsqrt14 gives to 14 bits, finished on the square root itself
// (Karp and Markstein, High-precision division and square root, ACM TOMS 23(4), 1997): y = x r, then
// y + (r / 2) (x - y^2). The fused residual x - y^2 is exact up to its own rounding, so the result is within 0.57 ulp:
// correctly rounded for 99.5% of the floats and off by one ulp for the rest. A Newton step on r followed by x r, as in
// generic_sqrt_newton_step, rounds three times after the residual is formed, which left 37% of the results off by 1
// to 3 ulps for one operation more.
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet16f psqrt<Packet16f>(const Packet16f& x) {
  const Packet16f r = _mm512_rsqrt14_ps(x);
  const Packet16f y = _mm512_mul_ps(x, r);
  // r / 2 by its exponent field, which keeps the halving off the multiplier: r is normal wherever the result below is
  // used.
  const Packet16f half_r = _mm512_castsi512_ps(_mm512_sub_epi32(_mm512_castps_si512(r), _mm512_set1_epi32(0x00800000)));
  const Packet16f s = _mm512_fmadd_ps(half_r, _mm512_fnmadd_ps(y, y, x), y);
  // x r is NaN for x = +-0 and x = +inf, whose square root is x. A masked compare in place of two compares and an OR
  // of their masks leaves the multiplier's port to the arithmetic.
  const __mmask16 nonzero = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_NEQ_UQ);
  const __mmask16 keep = _mm512_mask_cmp_ps_mask(nonzero, x, _mm512_set1_ps(NumTraits<float>::infinity()), _CMP_NEQ_UQ);
  const Packet16f res = _mm512_mask_mov_ps(x, keep, s);
  // For a small x the residual, of the order of 2^-13 x, is subnormal: it loses the bits the correction needs, and
  // all of them when subnormal results are flushed to zero (FTZ), which left the result with the 14 bits of x r.
  // Below 2^-92 a flushed residual could move the result by more than 2^-11 ulp: those lanes take the square root
  // instruction. Zeros, which are common, are masked out and stay on the fast path.
  const __mmask16 redo = _mm512_mask_cmplt_epu32_mask(nonzero, _mm512_castps_si512(x), _mm512_set1_epi32(0x11800000));
  return EIGEN_PREDICT_TRUE(redo == 0) ? res : _mm512_mask_sqrt_ps(res, redo, x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet8d psqrt<Packet8d>(const Packet8d& x) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return generic_sqrt_newton_step<Packet8d, /*Steps=*/1>::run(x, _mm512_rsqrt28_pd(x));
#else
  // As for float, after one step that takes g = x r ~ sqrt(x) and h = r / 2 ~ 1 / (2 sqrt(x)) from 14 to 28 bits
  // together: with e = 1/2 - g h, g <- g + g e and h <- h + h e. The closing step, g + h (x - g^2), has the residual
  // exact up to its rounding, and the result is within 0.78 ulp: correctly rounded for 99.4% of the doubles and off by
  // one ulp for the rest. Two Newton steps on r followed by x r left 44% off by 1 to 3 ulps, for two operations more.
  const Packet8d r = _mm512_rsqrt14_pd(x);
  const Packet8d g0 = _mm512_mul_pd(x, r);
  const Packet8d h0 =
      _mm512_castsi512_pd(_mm512_sub_epi64(_mm512_castpd_si512(r), _mm512_set1_epi64(0x0010000000000000ll)));
  const Packet8d e = _mm512_fnmadd_pd(g0, h0, _mm512_set1_pd(0.5));
  const Packet8d g = _mm512_fmadd_pd(g0, e, g0);
  const Packet8d h = _mm512_fmadd_pd(h0, e, h0);
  const Packet8d s = _mm512_fmadd_pd(h, _mm512_fnmadd_pd(g, g, x), g);
  const __mmask8 nonzero = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_NEQ_UQ);
  const __mmask8 keep = _mm512_mask_cmp_pd_mask(nonzero, x, _mm512_set1_pd(NumTraits<double>::infinity()), _CMP_NEQ_UQ);
  const Packet8d res = _mm512_mask_mov_pd(x, keep, s);
  // Below 2^-956 a residual flushed to zero could move the result by more than 2^-14 ulp.
  const __mmask8 redo =
      _mm512_mask_cmplt_epu64_mask(nonzero, _mm512_castpd_si512(x), _mm512_set1_epi64(0x0430000000000000ll));
  return EIGEN_PREDICT_TRUE(redo == 0) ? res : _mm512_mask_sqrt_pd(res, redo, x);
#endif
}
#else
template <>
EIGEN_STRONG_INLINE Packet16f psqrt<Packet16f>(const Packet16f& x) {
  return _mm512_sqrt_ps(x);
}

template <>
EIGEN_STRONG_INLINE Packet8d psqrt<Packet8d>(const Packet8d& x) {
  return _mm512_sqrt_pd(x);
}
#endif

// prsqrt for float.
#if defined(EIGEN_VECTORIZE_AVX512ER)
template <>
EIGEN_STRONG_INLINE Packet16f prsqrt<Packet16f>(const Packet16f& x) {
  return _mm512_rsqrt28_ps(x);
}
#elif EIGEN_FAST_MATH

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet16f prsqrt<Packet16f>(const Packet16f& x) {
  return generic_rsqrt_newton_step<Packet16f, /*Steps=*/1>::run(x, _mm512_rsqrt14_ps(x));
}
#endif

// prsqrt for double.
#if EIGEN_FAST_MATH
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet8d prsqrt<Packet8d>(const Packet8d& x) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return generic_rsqrt_newton_step<Packet8d, /*Steps=*/1>::run(x, _mm512_rsqrt28_pd(x));
#else
  return generic_rsqrt_newton_step<Packet8d, /*Steps=*/2>::run(x, _mm512_rsqrt14_pd(x));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f preciprocal<Packet16f>(const Packet16f& a) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return _mm512_rcp28_ps(a);
#else
  // generic_reciprocal_newton_step, with its NaN-or-zero test as one compare into a mask register (r == 0 or
  // unordered), as for AVX: GCC compiles the generic predux_any(pcmp_lt_or_nan(...)) to a compare mask converted to a
  // vector and back.
  const Packet16f one = pset1<Packet16f>(1.0f);
  const Packet16f x = _mm512_rcp14_ps(a);
  const Packet16f refined = pmadd(x, pnmadd(a, x, one), x);
  const __mmask16 redo = _mm512_cmp_ps_mask(refined, _mm512_setzero_ps(), _CMP_EQ_UQ);
  return redo == 0 ? refined : _mm512_mask_div_ps(refined, redo, one, a);
#endif
}
#endif

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_BF16(Packet16f, Packet16bf)
EIGEN_INSTANTIATE_GENERIC_SQRT_FUNCS_BF16(Packet16f, Packet16bf)

#ifndef EIGEN_VECTORIZE_AVX512FP16
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_F16(Packet16f, Packet16h)
#endif  // EIGEN_VECTORIZE_AVX512FP16

}  // end namespace internal

}  // end namespace Eigen

#endif  // THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_
