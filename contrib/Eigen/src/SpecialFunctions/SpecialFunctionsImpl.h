// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPECIAL_FUNCTIONS_H
#define EIGEN_SPECIAL_FUNCTIONS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

//  Parts of this code are based on the Cephes Math Library.
//
//  Cephes Math Library Release 2.8:  June, 2000
//  Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
//
//  Permission has been kindly provided by the original author
//  to incorporate the Cephes software into the Eigen codebase:
//
//    From: Stephen Moshier
//    To: Eugene Brevdo
//    Subject: Re: Permission to wrap several cephes functions in Eigen
//
//    Hello Eugene,
//
//    Thank you for writing.
//
//    If your licensing is similar to BSD, the formal way that has been
//    handled is simply to add a statement to the effect that you are incorporating
//    the Cephes software by permission of the author.
//
//    Good luck with your project,
//    Steve

/****************************************************************************
 * Implementation of lgamma                                                 *
 ****************************************************************************/

template <typename Scalar>
struct lgamma_default_impl {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((!std::is_same<Scalar, Scalar>::value), THIS_TYPE_IS_NOT_SUPPORTED)
    return Scalar(0);
  }
};

template <typename Scalar>
struct lgamma_impl
    : special_function_overload_or<special_function_overloads::lgamma_fn, lgamma_default_impl<Scalar>, Scalar> {};

// Since glibc 2.19
#if defined(__GLIBC__) && ((__GLIBC__ >= 2 && __GLIBC_MINOR__ >= 19) || __GLIBC__ > 2) && \
    (defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE) || defined(_SVID_SOURCE))
#define EIGEN_HAS_LGAMMA_R
#endif

// Glibc versions before 2.19
#if defined(__GLIBC__) && ((__GLIBC__ == 2 && __GLIBC_MINOR__ < 19) || __GLIBC__ < 2) && \
    (defined(_BSD_SOURCE) || defined(_SVID_SOURCE))
#define EIGEN_HAS_LGAMMA_R
#endif

template <>
struct lgamma_impl<float> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float run(float x) {
#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__) && defined(EIGEN_HAS_LGAMMA_R) && \
    !defined(__APPLE__)
    int dummy;
    return ::lgammaf_r(x, &dummy);
#elif defined(EIGEN_SYCL_DEVICE_ONLY)
    return cl::sycl::lgamma(x);
#else
    return ::lgammaf(x);
#endif
  }
};

template <>
struct lgamma_impl<double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double run(double x) {
#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__) && defined(EIGEN_HAS_LGAMMA_R) && \
    !defined(__APPLE__)
    int dummy;
    return ::lgamma_r(x, &dummy);
#elif defined(EIGEN_SYCL_DEVICE_ONLY)
    return cl::sycl::lgamma(x);
#else
    return ::lgamma(x);
#endif
  }
};

template <>
struct lgamma_impl<long double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE long double run(long double x) {
#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__) && defined(EIGEN_HAS_LGAMMA_R) && \
    !defined(__APPLE__)
    int dummy;
    return ::lgammal_r(x, &dummy);
#else
    return std::lgamma(x);
#endif
  }
};

#undef EIGEN_HAS_LGAMMA_R

/****************************************************************************
 * Implementation of digamma (psi), based on Cephes                         *
 ****************************************************************************/

/*
 *
 * Polynomial evaluation helper for the Psi (digamma) function.
 *
 * digamma_impl_maybe_poly::run(s) evaluates the asymptotic Psi expansion for
 * input Scalar s, assuming s is above 10.0.
 *
 * If s is above a certain threshold for the given Scalar type, zero
 * is returned.  Otherwise the polynomial is evaluated with enough
 * coefficients for results matching Scalar machine precision.
 *
 *
 */
template <typename Scalar>
struct digamma_impl_maybe_poly {
  EIGEN_STATIC_ASSERT((std::is_same<Scalar, Scalar>::value == false), THIS_TYPE_IS_NOT_SUPPORTED)

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar run(const Scalar) { return Scalar(0); }
};

template <>
struct digamma_impl_maybe_poly<float> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float run(const float s) {
    constexpr float A[] = {-4.16666666666666666667E-3f, 3.96825396825396825397E-3f, -8.33333333333333333333E-3f,
                           8.33333333333333333333E-2f};

    float z;
    if (s < 1.0e8f) {
      z = 1.0f / (s * s);
      return z * internal::ppolevl<float, 3>::run(z, A);
    } else
      return 0.0f;
  }
};

template <>
struct digamma_impl_maybe_poly<double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double run(const double s) {
    constexpr double A[] = {8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
                            -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
                            8.33333333333333333333E-2};

    double z;
    if (s < 1.0e17) {
      z = 1.0 / (s * s);
      return z * internal::ppolevl<double, 6>::run(z, A);
    } else
      return 0.0;
  }
};

template <typename Scalar>
struct digamma_default_impl {
  EIGEN_DEVICE_FUNC static Scalar run(Scalar x) {
    /*
     *
     *     Psi (digamma) function (modified for Eigen)
     *
     *
     * SYNOPSIS:
     *
     * double x, y, psi();
     *
     * y = psi( x );
     *
     *
     * DESCRIPTION:
     *
     *              d      -
     *   psi(x)  =  -- ln | (x)
     *              dx
     *
     * is the logarithmic derivative of the gamma function.
     * For integer x,
     *                   n-1
     *                    -
     * psi(n) = -EUL  +   >  1/k.
     *                    -
     *                   k=1
     *
     * If x is negative, it is transformed to a positive argument by the
     * reflection formula  psi(1-x) = psi(x) + pi cot(pi x).
     * For general positive x, the argument is made greater than 10
     * using the recurrence  psi(x+1) = psi(x) + 1/x.
     * Then the following asymptotic expansion is applied:
     *
     *                           inf.   B
     *                            -      2k
     * psi(x) = log(x) - 1/2x -   >   -------
     *                            -        2k
     *                           k=1   2k x
     *
     * where the B2k are Bernoulli numbers.
     *
     * ACCURACY (float):
     *    Relative error (except absolute when |psi| < 1):
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       1.3e-15     1.4e-16
     *    IEEE      -30,0       40000       1.5e-15     2.2e-16
     *
     * ACCURACY (double):
     *    Absolute error,  relative when |psi| > 1 :
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      -33,0        30000      8.2e-7      1.2e-7
     *    IEEE      0,33        100000      7.3e-7      7.7e-8
     *
     * ERROR MESSAGES:
     *     message         condition      value returned
     * psi singularity    x integer <=0      INFINITY
     */

    Scalar p, q, nz, s, w, y;
    bool negative = false;

    const Scalar nan = NumTraits<Scalar>::quiet_NaN();
    const Scalar m_pi = Scalar(EIGEN_PI);

    const Scalar zero = Scalar(0);
    const Scalar one = Scalar(1);
    const Scalar half = Scalar(0.5);
    nz = zero;

    // Near 0, psi(x) ~ -1/x - gamma.
    // For x = +0.0: psi(+0.0) -> -infinity.
    // For x = -0.0: psi(-0.0) -> +infinity.
    if (x == zero) {
      return (std::signbit(x)) ? NumTraits<Scalar>::infinity() : -NumTraits<Scalar>::infinity();
    }
    if (x < zero) {
      negative = true;
      q = x;
      p = numext::floor(q);
      if (p == q) {
        return nan;
      }
      /* Remove the zeros of tan(m_pi x)
       * by subtracting the nearest integer from x
       */
      nz = q - p;
      if (nz != half) {
        if (nz > half) {
          p += one;
          nz = q - p;
        }
        nz = m_pi / numext::tan(m_pi * nz);
      } else {
        nz = zero;
      }
      x = one - x;
    }

    /* use the recurrence psi(x+1) = psi(x) + 1/x. */
    s = x;
    w = zero;
    while (s < Scalar(10)) {
      w += one / s;
      s += one;
    }

    y = digamma_impl_maybe_poly<Scalar>::run(s);

    y = numext::log(s) - (half / s) - y - w;

    return (negative) ? y - nz : y;
  }
};

template <typename Scalar>
struct digamma_impl
    : special_function_overload_or<special_function_overloads::digamma_fn, digamma_default_impl<Scalar>, Scalar> {};

template <typename T>
struct has_erf : has_special_function_overload<special_function_overloads::erf_fn, T> {};
template <typename T>
struct has_erfc : has_special_function_overload<special_function_overloads::erfc_fn, T> {};

/***************************************************************************
 * Implementation of erfc.
 ****************************************************************************/
template <typename Scalar>
struct generic_fast_erfc {
  template <typename T>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T run(const T& x_in);
};

template <>
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fast_erfc<float>::run(const T& x_in) {
  constexpr float kClamp = 11.0f;
  const T x = pmin<PropagateNaN>(pmax<PropagateNaN>(x_in, pset1<T>(-kClamp)), pset1<T>(kClamp));

  // erfc(x) = 1 + x * S(x^2), |x| <= 1.
  //
  // Coefficients for S and T generated with Rminimax command:
  // ./ratapprox --function="erfc(x)-1" --dom='[-1,1]' --type=[11,0] --num="odd"
  //   --numF="[SG]" --denF="[SG]" --log --dispCoeff="dec"
  constexpr float alpha[] = {5.61802298761904239654541015625e-04, -4.91381669417023658752441406250e-03,
                             2.67075151205062866210937500000e-02, -1.12800106406211853027343750000e-01,
                             3.76122951507568359375000000000e-01, -1.12837910652160644531250000000e+00};
  const T x2 = pmul(x, x);
  const T one = pset1<T>(1.0f);
  const T erfc_small = pmadd(x, ppolevl<T, 5>::run(x2, alpha), one);

  // Return early if we don't need the more expensive approximation for any
  // entry in a.
  const T x_abs_gt_one_mask = pcmp_lt(one, x2);
  if (!predux_any(x_abs_gt_one_mask)) return erfc_small;

  // erfc(x) = exp(-x^2) * 1/x * P(1/x^2) / Q(1/x^2), 1 < x < 9.
  //
  // Coefficients for P and Q generated with Rminimax command:
  //   ./ratapprox --function="erfc(1/sqrt(x))*exp(1/x)/sqrt(x)"
  //     --dom='[0.01,1]' --type=[3,4] --numF="[SG]" --denF="[SG]" --log
  //     --dispCoeff="dec"
  constexpr float gamma[] = {1.0208116471767425537109375e-01f, 4.2920666933059692382812500e-01f,
                             3.2379078865051269531250000e-01f, 5.3971976041793823242187500e-02f};
  constexpr float delta[] = {1.7251677811145782470703125e-02f, 3.9137163758277893066406250e-01f,
                             1.0000000000000000000000000e+00f, 6.2173241376876831054687500e-01f,
                             9.5662862062454223632812500e-02f};
  const T x2_lo = twoprod_low(x, x, x2);
  // Here we use that
  //   exp(-x^2) = exp(-(x2+x2_lo)^2) ~= exp(-x2)*exp(-x2_lo) ~= exp(-x2)*(1-x2_lo)
  // since x2_lo < kClamp * eps << 1 in the region we care about. This trick reduces the max error
  // from 34 ulps to below 5 ulps.
  const T exp2_hi = pexp(pnegate(x2));
  const T z = pnmadd(exp2_hi, x2_lo, exp2_hi);
  const T q2 = preciprocal(x2);
  const T num = ppolevl<T, 3>::run(q2, gamma);
  const T denom = pmul(x, ppolevl<T, 4>::run(q2, delta));
  const T r = pdiv(num, denom);
  const T maybe_two = pselect(pcmp_lt(x, pset1<T>(0.0f)), pset1<T>(2.0f), pset1<T>(0.0f));
  const T erfc_large = pmadd(z, r, maybe_two);
  return pselect(x_abs_gt_one_mask, erfc_large, erfc_small);
}

// Computes erf(x)/x for |x| <= 1. Used by both erf and erfc implementations.
// Takes x2 = x^2 as input.
//
// PRECONDITION: x2 <= 1.
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T erf_over_x_double_small(const T& x2) {
  // erf(x)/x =  S(x^2) / T(x^2), x^2 <= 1.
  //
  // Coefficients for S and T generated with Rminimax command:
  //  ./ratapprox --function="erf(x)" --dom='[-1,1]' --type=[9,10]
  //  --num="odd" --numF="[D]" --den="even" --denF="[D]" --log --dispCoeff="dec"
  constexpr double alpha[] = {1.9493725660006057018823477644531294572516344487667083740234375e-04,
                              1.8272566210022942682217328425053892715368419885635375976562500e-03,
                              4.5303363351690106863856044583371840417385101318359375000000000e-02,
                              1.4215015503619179981775744181504705920815467834472656250000000e-01,
                              1.1283791670955125585606992899556644260883331298828125000000000e+00};
  constexpr double beta[] = {2.0294484101083099089526257108317963684385176748037338256835938e-05,
                             6.8117805899186819641732970609382391558028757572174072265625000e-04,
                             1.0582026056098614921752165685120417037978768348693847656250000e-02,
                             9.3252603143757495374188692949246615171432495117187500000000000e-02,
                             4.5931062818368939559832142549566924571990966796875000000000000e-01,
                             1.0};
  const T num_small = ppolevl<T, 4>::run(x2, alpha);
  const T denom_small = ppolevl<T, 5>::run(x2, beta);
  return pdiv(num_small, denom_small);
}

// erfc(x) = exp(-x^2) * 1/x * P(1/x^2) / Q(1/x^2), 1 < x < 28.
//
// Coefficients for P and Q generated with Rminimax command:
//  ./ratapprox --function="erfc(1/sqrt(x))*exp(1/x)/sqrt(x)"  --dom='[0.0013717,1]' --type=[9,9] --numF="[D]"
//  --denF="[D]" --log --dispCoeff="dec"
//
// PRECONDITION: 1 < x < 28.
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T erfc_double_large(const T& x, const T& x2) {
  constexpr double gamma[] = {1.5252844933226974316088642158462107545346952974796295166015625e-04,
                              1.0909912393738931124520519233556115068495273590087890625000000e-02,
                              1.0628604636755033252537572252549580298364162445068359375000000e-01,
                              3.3492472973137982217295416376146022230386734008789062500000000e-01,
                              4.5065776215933289750026347064704168587923049926757812500000000e-01,
                              2.9433039130294824659017649537418037652969360351562500000000000e-01,
                              9.8792676360600226170838311645638896152377128601074218750000000e-02,
                              1.7095935395503719655962981960328761488199234008789062500000000e-02,
                              1.4249109729504577659398023570247460156679153442382812500000000e-03,
                              4.4567378313647954771875570045835956989321857690811157226562500e-05};
  constexpr double delta[] = {2.041985103115789845773520028160419315099716186523437500000000e-03,
                              5.316030659946043707142493417450168635696172714233398437500000e-02,
                              3.426242193784684864077405563875799998641014099121093750000000e-01,
                              8.565637124308049799026321124983951449394226074218750000000000e-01,
                              1.000000000000000000000000000000000000000000000000000000000000e+00,
                              5.968805280570776972126623149961233139038085937500000000000000e-01,
                              1.890922854723317836356244470152887515723705291748046875000000e-01,
                              3.152505418656005586885981983868987299501895904541015625000000e-02,
                              2.565085751861882583380047861965067568235099315643310546875000e-03,
                              7.899362131678837697403017248376499992446042597293853759765625e-05};
  // Compute exp(-x^2).
  const T x2_lo = twoprod_low(x, x, x2);
  // Here we use that
  //   exp(-x^2) = exp(-(x2+x2_lo)^2) ~= exp(-x2)*exp(-x2_lo) ~= exp(-x2)*(1-x2_lo)
  // since x2_lo < kClamp * eps << 1 in the region we care about. This trick reduces the max error
  // from 258 ulps to below 7 ulps.
  const T exp2_hi = pexp(pnegate(x2));
  const T z = pnmadd(exp2_hi, x2_lo, exp2_hi);
  // Compute r = P / Q.
  const T q2 = preciprocal(x2);
  const T num_large = ppolevl<T, 9>::run(q2, gamma);
  const T denom_large = pmul(x, ppolevl<T, 9>::run(q2, delta));
  const T r = pdiv(num_large, denom_large);
  const T maybe_two = pselect(pcmp_lt(x, pset1<T>(0.0)), pset1<T>(2.0), pset1<T>(0.0));
  return pmadd(z, r, maybe_two);
}

template <>
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fast_erfc<double>::run(const T& x_in) {
  // Clamp x to [-28:28] beyond which erfc(x) is either two or zero (below the underflow threshold).
  // This avoids having to deal with twoprod(x,x) producing NaN for sufficiently large x.
  constexpr double kClamp = 28.0;
  const T x = pmin<PropagateNaN>(pmax<PropagateNaN>(x_in, pset1<T>(-kClamp)), pset1<T>(kClamp));

  // For |x| < 1, we use erfc(x) = 1 - erf(x).
  const T x2 = pmul(x, x);
  const T one = pset1<T>(1.0);
  const T erfc_small = pnmadd(x, erf_over_x_double_small(x2), one);

  // Return early if we don't need the more expensive approximation for any
  // entry in a.
  const T x_abs_gt_one_mask = pcmp_lt(one, x2);
  if (!predux_any(x_abs_gt_one_mask)) return erfc_small;

  const T erfc_large = erfc_double_large(x, x2);
  return pselect(x_abs_gt_one_mask, erfc_large, erfc_small);
}

template <typename T>
struct erfc_impl {
  using Scalar = typename unpacket_traits<T>::type;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run(const T& x) { return run_impl(x, std::is_same<T, Scalar>()); }

 private:
  // Packets of float/double: vectorized rational approximation.
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_impl(const T& x, std::false_type) {
    return generic_fast_erfc<Scalar>::run(x);
  }
  // Any other scalar type: defer to an erfc found by argument-dependent lookup
  // (or std::erfc), keeping custom scalars on their own implementation instead
  // of the float/double-tuned polynomials.
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_impl(const T& x, std::true_type) {
    EIGEN_STATIC_ASSERT_NON_INTEGER(T)
    return run_scalar(x, has_erfc<T>());
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_scalar(const T& x, std::true_type) {
    return special_function_overloads::erfc_fn::call<T>(x);
  }
  // Reject the type here instead of letting overload resolution fail inside the
  // call above: the approximations in this file are tuned for float and double
  // and are not valid for an arbitrary scalar, so a scalar type that wants erfc
  // has to supply it.
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_scalar(const T& x, std::false_type) {
    EIGEN_STATIC_ASSERT(has_erfc<T>::value, SCALAR_TYPE_MUST_PROVIDE_AN_ERFC_OVERLOAD_FOUND_BY_ADL_OR_IN_NAMESPACE_STD)
    return x;
  }
};

template <>
struct erfc_impl<float> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float run(const float x) {
#if defined(EIGEN_SYCL_DEVICE_ONLY)
    return cl::sycl::erfc(x);
#else
    return generic_fast_erfc<float>::run(x);
#endif
  }
};

template <>
struct erfc_impl<double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double run(const double x) {
#if defined(EIGEN_SYCL_DEVICE_ONLY)
    return cl::sycl::erfc(x);
#else
    return generic_fast_erfc<double>::run(x);
#endif
  }
};

/****************************************************************************
 * Implementation of erf.
 ****************************************************************************/

template <typename Scalar>
struct generic_fast_erf {
  template <typename T>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T run(const T& x_in);
};

/** \internal \returns the error function of \a a (coeff-wise)
    This uses a 11/10-degree rational interpolant and is accurate to 3 ulp for
    normalized floats.

    This implementation works on both scalars and SIMD "packets".
*/
template <>
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fast_erf<float>::run(const T& x) {
  // The monomial coefficients of the numerator polynomial (odd).
  constexpr float alpha[] = {2.123732201653183437883853912353515625e-06f, 2.861979592125862836837768554687500000e-04f,
                             3.658048342913389205932617187500000000e-03f, 5.243302136659622192382812500000000000e-02f,
                             1.874160766601562500000000000000000000e-01f, 1.128379106521606445312500000000000000e+00f};

  // The monomial coefficients of the denominator polynomial (even).
  constexpr float beta[] = {3.89185734093189239501953125000e-05f, 1.14329601638019084930419921875e-03f,
                            1.47520881146192550659179687500e-02f, 1.12945675849914550781250000000e-01f,
                            4.99425798654556274414062500000e-01f, 1.0f};

  // Since the polynomials are odd/even, we need x^2.
  // Since erf(4) == 1 in float, we clamp x^2 to 16 to avoid computing Inf/Inf below.
  // NaN need not survive this clamp: multiplying by x below restores it.
  const T x2 = pmin(pset1<T>(16.0f), pmul(x, x));

  // Evaluate the numerator polynomial p.
  T p = ppolevl<T, 5>::run(x2, alpha);
  p = pmul(x, p);

  // Evaluate the denominator polynomial q.
  T q = ppolevl<T, 5>::run(x2, beta);
  const T r = pdiv(p, q);

  // Clamp to [-1:1].
  return pmax<PropagateNaN>(pmin<PropagateNaN>(r, pset1<T>(1.0f)), pset1<T>(-1.0f));
}

template <>
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fast_erf<double>::run(const T& x_in) {
  // Clamp x to [-28:28] beyond which erf(x) is ±1 within double precision.
  // This avoids NaN from twoprod and exp operations for infinite inputs.
  constexpr double kClamp = 28.0;
  const T x = pmin<PropagateNaN>(pmax<PropagateNaN>(x_in, pset1<T>(-kClamp)), pset1<T>(kClamp));
  T x2 = pmul(x, x);
  T erf_small = pmul(x, erf_over_x_double_small(x2));

  // Return early if we don't need the more expensive approximation for any
  // entry in a.
  const T one = pset1<T>(1.0);
  const T x_abs_gt_one_mask = pcmp_lt(one, x2);
  if (!predux_any(x_abs_gt_one_mask)) return erf_small;

  // For |x| > 1, use erf(x) = 1 - erfc(x).
  const T erf_large = psub(one, erfc_double_large(x, x2));
  return pselect(x_abs_gt_one_mask, erf_large, erf_small);
}

template <typename T>
struct erf_impl {
  using Scalar = typename unpacket_traits<T>::type;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run(const T& x) { return run_impl(x, std::is_same<T, Scalar>()); }

 private:
  // Packets of float/double: vectorized rational approximation.
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_impl(const T& x, std::false_type) {
    return generic_fast_erf<Scalar>::run(x);
  }
  // Any other scalar type: defer to an erf found by argument-dependent lookup
  // (or std::erf), keeping custom scalars on their own implementation instead
  // of the float/double-tuned polynomials.
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_impl(const T& x, std::true_type) {
    EIGEN_STATIC_ASSERT_NON_INTEGER(T)
    return run_scalar(x, has_erf<T>());
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_scalar(const T& x, std::true_type) {
    return special_function_overloads::erf_fn::call<T>(x);
  }
  // Reject the type here instead of letting overload resolution fail inside the
  // call above: the approximations in this file are tuned for float and double
  // and are not valid for an arbitrary scalar, so a scalar type that wants erf
  // has to supply it.
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run_scalar(const T& x, std::false_type) {
    EIGEN_STATIC_ASSERT(has_erf<T>::value, SCALAR_TYPE_MUST_PROVIDE_AN_ERF_OVERLOAD_FOUND_BY_ADL_OR_IN_NAMESPACE_STD)
    return x;
  }
};

template <>
struct erf_impl<float> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float run(const float x) {
#if defined(EIGEN_SYCL_DEVICE_ONLY)
    return cl::sycl::erf(x);
#else
    return generic_fast_erf<float>::run(x);
#endif
  }
};

template <>
struct erf_impl<double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double run(const double x) {
#if defined(EIGEN_SYCL_DEVICE_ONLY)
    return cl::sycl::erf(x);
#else
    return generic_fast_erf<double>::run(x);
#endif
  }
};

/***************************************************************************
 * Implementation of ndtri.                                                 *
 ****************************************************************************/

/* Inverse of Normal distribution function (modified for Eigen).
 *
 *
 * SYNOPSIS:
 *
 * double x, y, ndtri();
 *
 * x = ndtri( y );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the argument, x, for which the area under the
 * Gaussian probability density function (integrated from
 * minus infinity to x) is equal to y.
 *
 *
 * For small arguments 0 < y < exp(-2), the program computes
 * z = sqrt( -2.0 * log(y) );  then the approximation is
 * x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z).
 * There are two rational functions P/Q, one for 0 < y < exp(-32)
 * and the other for y up to exp(-2).  For larger arguments,
 * w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain        # trials      peak         rms
 *    DEC      0.125, 1         5500       9.5e-17     2.1e-17
 *    DEC      6e-39, 0.135     3500       5.7e-17     1.3e-17
 *    IEEE     0.125, 1        20000       7.2e-16     1.3e-16
 *    IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition    value returned
 * ndtri domain       x == 0        -INF
 * ndtri domain       x == 1         INF
 * ndtri domain       x < 0, x > 1   NAN
 */
/*
  Cephes Math Library Release 2.2: June, 1992
  Copyright 1985, 1987, 1992 by Stephen L. Moshier
  Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

// TODO: Add a cheaper approximation for float.

template <typename T, bool IsScalar = is_scalar<T>::value>
struct flipsign_impl;

template <typename T>
struct flipsign_impl<T, false> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T run(const T& should_flipsign, const T& x) {
    const T sign_mask = psignmask<T>();
    const T sign_bit = pand<T>(should_flipsign, sign_mask);
    return pxor<T>(sign_bit, x);
  }
};

template <typename T>
struct flipsign_impl<T, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T run(const T& should_flipsign, const T& x) {
    return should_flipsign == T(0) ? x : -x;
  }
};

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T flipsign(const T& should_flipsign, const T& x) {
  return flipsign_impl<T>::run(should_flipsign, x);
}

template <typename T, bool IsScalar = is_scalar<T>::value>
struct ndtri_negative_infinity_impl;

template <typename T>
struct ndtri_negative_infinity_impl<T, false> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T run(const T& positive_infinity) {
    return por(psignmask<T>(), positive_infinity);
  }
};

template <typename T>
struct ndtri_negative_infinity_impl<T, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T run(const T& positive_infinity) { return -positive_infinity; }
};

// We split this computation in to two so that in the scalar path
// only one branch is evaluated (due to our template specialization of pselect
// being an if statement.)

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_ndtri_gt_exp_neg_two(const T& b) {
  const ScalarType p0[] = {ScalarType(-5.99633501014107895267e1), ScalarType(9.80010754185999661536e1),
                           ScalarType(-5.66762857469070293439e1), ScalarType(1.39312609387279679503e1),
                           ScalarType(-1.23916583867381258016e0)};
  const ScalarType q0[] = {ScalarType(1.0),
                           ScalarType(1.95448858338141759834e0),
                           ScalarType(4.67627912898881538453e0),
                           ScalarType(8.63602421390890590575e1),
                           ScalarType(-2.25462687854119370527e2),
                           ScalarType(2.00260212380060660359e2),
                           ScalarType(-8.20372256168333339912e1),
                           ScalarType(1.59056225126211695515e1),
                           ScalarType(-1.18331621121330003142e0)};
  const T sqrt2pi = pset1<T>(ScalarType(2.50662827463100050242e0));
  const T half = pset1<T>(ScalarType(0.5));
  T c, c2, ndtri_gt_exp_neg_two;

  c = psub(b, half);
  c2 = pmul(c, c);
  ndtri_gt_exp_neg_two =
      pmadd(c, pmul(c2, pdiv(internal::ppolevl<T, 4>::run(c2, p0), internal::ppolevl<T, 8>::run(c2, q0))), c);
  return pmul(ndtri_gt_exp_neg_two, sqrt2pi);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_ndtri_lt_exp_neg_two(const T& b, const T& should_flipsign) {
  /* Approximation for interval z = sqrt(-2 log a ) between 2 and 8
   * i.e., a between exp(-2) = .135 and exp(-32) = 1.27e-14.
   */
  const ScalarType p1[] = {ScalarType(4.05544892305962419923e0),   ScalarType(3.15251094599893866154e1),
                           ScalarType(5.71628192246421288162e1),   ScalarType(4.40805073893200834700e1),
                           ScalarType(1.46849561928858024014e1),   ScalarType(2.18663306850790267539e0),
                           ScalarType(-1.40256079171354495875e-1), ScalarType(-3.50424626827848203418e-2),
                           ScalarType(-8.57456785154685413611e-4)};
  const ScalarType q1[] = {ScalarType(1.0),
                           ScalarType(1.57799883256466749731e1),
                           ScalarType(4.53907635128879210584e1),
                           ScalarType(4.13172038254672030440e1),
                           ScalarType(1.50425385692907503408e1),
                           ScalarType(2.50464946208309415979e0),
                           ScalarType(-1.42182922854787788574e-1),
                           ScalarType(-3.80806407691578277194e-2),
                           ScalarType(-9.33259480895457427372e-4)};
  /* Approximation for interval z = sqrt(-2 log a ) between 8 and 64
   * i.e., a between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
   */
  const ScalarType p2[] = {ScalarType(3.23774891776946035970e0),  ScalarType(6.91522889068984211695e0),
                           ScalarType(3.93881025292474443415e0),  ScalarType(1.33303460815807542389e0),
                           ScalarType(2.01485389549179081538e-1), ScalarType(1.23716634817820021358e-2),
                           ScalarType(3.01581553508235416007e-4), ScalarType(2.65806974686737550832e-6),
                           ScalarType(6.23974539184983293730e-9)};
  const ScalarType q2[] = {ScalarType(1.0),
                           ScalarType(6.02427039364742014255e0),
                           ScalarType(3.67983563856160859403e0),
                           ScalarType(1.37702099489081330271e0),
                           ScalarType(2.16236993594496635890e-1),
                           ScalarType(1.34204006088543189037e-2),
                           ScalarType(3.28014464682127739104e-4),
                           ScalarType(2.89247864745380683936e-6),
                           ScalarType(6.79019408009981274425e-9)};
  const T eight = pset1<T>(ScalarType(8.0));
  const T neg_two = pset1<T>(ScalarType(-2));
  T x, x0, correction, z;

  x = psqrt(pmul(neg_two, plog(b)));
  x0 = psub(x, pdiv(plog(x), x));
  z = preciprocal(x);
  correction =
      pselect(pcmp_lt(x, eight), pdiv(internal::ppolevl<T, 8>::run(z, p1), internal::ppolevl<T, 8>::run(z, q1)),
              pdiv(internal::ppolevl<T, 8>::run(z, p2), internal::ppolevl<T, 8>::run(z, q2)));
  return flipsign(should_flipsign, pnmadd(z, correction, x0));
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T generic_ndtri(const T& a) {
  const T maxnum = pinf<T>();
  const T neg_maxnum = ndtri_negative_infinity_impl<T>::run(maxnum);

  const T zero = pset1<T>(ScalarType(0));
  const T one = pset1<T>(ScalarType(1));
  // exp(-2)
  const T exp_neg_two = pset1<T>(ScalarType(0.13533528323661269189));
  T b, ndtri, should_flipsign;

  should_flipsign = pcmp_le(a, psub(one, exp_neg_two));
  b = pselect(should_flipsign, a, psub(one, a));

  ndtri = pselect(pcmp_lt(exp_neg_two, b), generic_ndtri_gt_exp_neg_two<T, ScalarType>(b),
                  generic_ndtri_lt_exp_neg_two<T, ScalarType>(b, should_flipsign));

  return pselect(pcmp_eq(a, zero), neg_maxnum, pselect(pcmp_eq(one, a), maxnum, ndtri));
}

template <typename Scalar>
struct ndtri_default_impl {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar run(const Scalar x) { return generic_ndtri<Scalar, Scalar>(x); }
};

template <typename Scalar>
struct ndtri_impl
    : special_function_overload_or<special_function_overloads::ndtri_fn, ndtri_default_impl<Scalar>, Scalar> {};

/**************************************************************************************************************
 * Implementation of igammac (complemented incomplete gamma integral), based on Cephes                       *
 **************************************************************************************************************/

// NOTE: cephes_helper is also used to implement zeta
template <typename Scalar>
struct cephes_helper {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar machep() {
    eigen_assert(false && "machep not supported for this type");
    return 0.0;
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar big() {
    eigen_assert(false && "big not supported for this type");
    return 0.0;
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar biginv() {
    eigen_assert(false && "biginv not supported for this type");
    return 0.0;
  }
};

template <>
struct cephes_helper<float> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float machep() {
    return NumTraits<float>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float big() {
    // use epsneg (1.0 - epsneg == 1.0)
    return 1.0f / (NumTraits<float>::epsilon() / 2);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float biginv() {
    // epsneg
    return machep();
  }
};

template <>
struct cephes_helper<double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double machep() {
    return NumTraits<double>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double big() { return 1.0 / NumTraits<double>::epsilon(); }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double biginv() {
    // inverse of eps
    return NumTraits<double>::epsilon();
  }
};

enum IgammaComputationMode { VALUE, DERIVATIVE, SAMPLE_DERIVATIVE };

template <typename Scalar>
EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar main_igamma_term(Scalar a, Scalar x) {
  /* Compute  x**a * exp(-x) / gamma(a)  */
  Scalar logax = a * numext::log(x) - x - lgamma_impl<Scalar>::run(a);
  if (logax < -numext::log(NumTraits<Scalar>::highest()) ||
      // Assuming x and a aren't Nan.
      (numext::isnan)(logax)) {
    return Scalar(0);
  }
  return numext::exp(logax);
}

/* Clamp a probability to [0, 1], passing NaN through: numext::maxi(0, NaN) is 0, which would turn a failed
 * evaluation into a confident answer. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar igamma_clamp_to_unit_interval(Scalar r) {
  return (numext::isnan)(r) ? r : numext::mini(Scalar(1), numext::maxi(Scalar(0), r));
}

template <typename Scalar, IgammaComputationMode mode>
EIGEN_DEVICE_FUNC constexpr int igamma_num_iterations() {
  /* Returns the maximum number of internal iterations for igamma computation.
   */
  return mode == VALUE                         ? 2000
         : std::is_same<Scalar, float>::value  ? 200
         : std::is_same<Scalar, double>::value ? 500
                                               : 2000;
}

/* log1p(d) - d for |d| < 1/4, with s = d / (2 + d), so that log1p(d) = 2 atanh(s):
 *   log1p(d) - d = -d s + 2 (s^3 / 3 + s^5 / 5 + ...),   |s| < 1/7: the tail past s^21 / 21 is below 2^-60 relative. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static Scalar igamma_log1p_minus_d(Scalar d) {
  const Scalar s = d / (Scalar(2) + d);
  const Scalar s2 = s * s;
  Scalar power = s * s2;
  Scalar sum = Scalar(0);
  for (int k = 3; k <= 21; k += 2) {
    sum += power / Scalar(k);
    power *= s2;
  }
  return Scalar(2) * sum - d * s;
}

/* Temme's uniform asymptotic expansion of P = igamma(a, x) and Q = igammac(a, x) = 1 - P for large a
 * (DLMF 8.12.3, 8.12.4 and 8.12.8; Temme, SIAM J. Math. Anal. 10, 1979):
 *
 *   Q = erfc(z) / 2 + R,   P = erfc(-z) / 2 - R,   R = exp(-a psi) / sqrt(2 pi a) (c_0(eta) + c_1(eta) / a + O(a^-2)),
 *
 * with d = (x - a) / a, psi = d - log1p(d) = eta^2 / 2, sign(eta) = sign(d), z = eta sqrt(a / 2) = sign(d) sqrt(a psi),
 *   c_0(eta) = 1 / d - 1 / eta,   c_1(eta) = 1 / eta^3 - 1 / d^3 - 1 / d^2 - 1 / (12 d).
 * Both cancel for small eta and are evaluated from their Taylor series at eta = 0, c_0 = -1/3 + eta/12 - 2 eta^2 / 135
 * + ..., c_1 = -1/540 - eta/288 + eta^2 / 378 - .... The coefficients were computed with GMP (mpq_class power series):
 * d = u(eta) from eta^2 / 2 = d - log1p(d) by the fixed-point iteration u = eta / sqrt(h(u)), h(u) = 2 sum_{k >= 2}
 * (-1)^k u^(k-2) / k, then the series of the closed forms above with d = u(eta). The leading terms of both match DLMF
 * 8.12(ii).
 *
 * Used for |eta| <= 0.15, where the truncation after eta^12 (c_0) and eta^8 (c_1) is 9e-20 and 9e-16 absolute (c_1 / a
 * is below 3e-19 relative to c_0), and a >= 1e5 (double), 1e4 (float). Outside, a psi > 1125 (112 in float), and P (x <
 * a) or Q (x > a) is <= exp(-a psi) (Chernoff bound) < denorm_min / 2, so it is 0 and the other is 1. The error of the
 * O(a^-2) term is the measured one: Q to 53 epsilon at a = 1e5 and |t|
 * <= 8, t = (x - a) / sqrt(a), against mpfr_gamma_inc, 0.7 of 2 (log(a) / 2 + t^2 + 2) epsilon. It takes no iterations
 * and is accurate near x = a for every a, where the continued fraction and the series need ~ sqrt(a) of them. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool igamma_use_temme(Scalar a) {
  return (numext::isfinite)(a) && a >= (sizeof(Scalar) == 4 ? Scalar(1e4) : Scalar(1e5));
}

template <typename Scalar>
EIGEN_DEVICE_FUNC static void igamma_temme(Scalar a, Scalar x, Scalar* p, Scalar* q) {
  const Scalar zero = 0, one = 1, half = Scalar(0.5);
  const Scalar d = (x - a) / a;
  // |d| >= 1/4 gives |eta| >= 0.23. The series for log1p(d) - d needs |d| < 1/4.
  const Scalar psi = numext::abs(d) < Scalar(0.25) ? -igamma_log1p_minus_d(d) : Scalar(1);
  if (!(Scalar(2) * psi <= Scalar(0.15) * Scalar(0.15))) {
    *p = d > zero ? one : zero;
    *q = one - *p;
    return;
  }
  const Scalar eta = (d < zero ? -one : one) * numext::sqrt(Scalar(2) * psi);
  const Scalar c0[] = {Scalar(-1) / Scalar(3),
                       Scalar(1) / Scalar(12),
                       Scalar(-2) / Scalar(135),
                       Scalar(1) / Scalar(864),
                       Scalar(1) / Scalar(2835),
                       Scalar(-139) / Scalar(777600),
                       Scalar(1) / Scalar(25515),
                       Scalar(-571) / Scalar(261273600),
                       Scalar(-281) / Scalar(151559100),
                       Scalar(163879) / Scalar(197522841600.0),
                       Scalar(-5221) / Scalar(29554024500.0),
                       Scalar(5246819) / Scalar(782190452736000.0),
                       Scalar(5459) / Scalar(531972441000.0)};
  const Scalar c1[] = {Scalar(-1) / Scalar(540),          Scalar(-1) / Scalar(288),
                       Scalar(1) / Scalar(378),           Scalar(-77) / Scalar(77760),
                       Scalar(1) / Scalar(4860),          Scalar(-1) / Scalar(2488320),
                       Scalar(-2743) / Scalar(151559100), Scalar(41969) / Scalar(5486745600.0),
                       Scalar(-11) / Scalar(6823440)};
  Scalar c0_eta = c0[12], c1_eta = c1[8];
  for (int k = 11; k >= 0; --k) c0_eta = c0_eta * eta + c0[k];
  for (int k = 7; k >= 0; --k) c1_eta = c1_eta * eta + c1[k];
  const Scalar a_psi = a * psi;
  const Scalar r = numext::exp(-a_psi) / (numext::sqrt(Scalar(2 * EIGEN_PI)) * numext::sqrt(a)) * (c0_eta + c1_eta / a);
  const Scalar half_erfc = half * erfc_impl<Scalar>::run(numext::sqrt(a_psi));
  if (d >= zero) {
    *q = igamma_clamp_to_unit_interval(half_erfc + r);
    *p = one - *q;
  } else {
    *p = igamma_clamp_to_unit_interval(half_erfc - r);
    *q = one - *p;
  }
}

/* The iteration limit of the power series of igamma(a, x) for x <= a in VALUE mode. The terms are
 * c_k = prod_{j <= k} x / (a + j) <= exp(-k^2 / (2 a)) for x <= a, so the sum has converged to machep (c_k <= machep,
 * and the sum is >= 1) once k >= sqrt(-2 a log(machep)), at most 8.6 sqrt(a) in double and 5.8 sqrt(a) in float, which
 * exceeds the limit 2000 of igamma_num_iterations for a >~ 5e4 (double) and 1.2e5 (float). The limit is 1.25 times
 * that, but at most 1e5. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static int igamma_series_num_iterations(Scalar a) {
  const Scalar n = Scalar(1.25) * numext::sqrt(-Scalar(2) * a * numext::log(cephes_helper<Scalar>::machep()));
  return !(n > Scalar(2000)) ? 2000 : n > Scalar(100000) ? 100000 : static_cast<int>(n);
}

/* x^a exp(-x) / Gamma(a) for x > 0 and a >> 1 from Stirling's series. a log(x) - x - lgamma(a) in main_igamma_term
 * cancels to an absolute error of about eps a log(a), more than 1 for a >~ 1 / eps. With d = (x - a) / a:
 *
 *   log(x^a exp(-x) / Gamma(a)) = a (log1p(d) - d) + (1/2) log(a / (2 pi)) - 1 / (12 a) + O(a^-3),
 *   with log1p(d) - d from igamma_log1p_minus_d for -1/4 < d < 1/4.
 * For d <= -1/4 (x <= 3a/4), 1 + d = x / a is formed by the division, not as 1 + d, which has the absolute error eps
 * and so the relative error eps / (1 + d) for x << a. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static Scalar main_igamma_term_stirling(Scalar a, Scalar x) {
  const Scalar d = (x - a) / a;
  Scalar log1pmx;
  if (d >= Scalar(0.25)) {
    log1pmx = numext::log1p(d) - d;
  } else if (d <= Scalar(-0.25)) {
    log1pmx = numext::log(x / a) - d;
  } else {
    log1pmx = igamma_log1p_minus_d(d);
  }
  const Scalar logax =
      a * log1pmx + Scalar(0.5) * (numext::log(a) - numext::log(Scalar(2 * EIGEN_PI))) - Scalar(1) / (Scalar(12) * a);
  // Written so that a NaN, from a = inf, gives 0 as main_igamma_term does.
  return !(logax >= -numext::log(NumTraits<Scalar>::highest())) ? Scalar(0) : numext::exp(logax);
}

/* Whether igammac_cf_impl evaluates igammac(a, x), x >= a, as main_igamma_term_stirling * igammac_cf_lentz rather
 * than main_igamma_term * recurrence: true once the error 1 / (360 a^3) of Stirling's series for log Gamma(a),
 * truncated after 1 / (12 a), is below eps, i.e. a >~ 2.3e4 (double), a >~ 29 (float), and when a^3 overflows.
 * The ordinary main term has the absolute error eps a log(a) in its exponent: better for a <~ 1500 (double),
 * a <~ 16 (float), worse above, and meaningless for a >~ 1 / eps. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool igamma_use_stirling_main_term(Scalar a) {
  return Scalar(360) * NumTraits<Scalar>::epsilon() * a * a * a >= Scalar(1);
}

/* The continued fraction of igammac(a, x) / main_igamma_term(a, x), for x >= a > 0, by the modified Lentz algorithm
 * (Thompson and Barnett, J. Comput. Phys. 64, 1986; Press et al., Numerical Recipes, 3rd ed., section 6.2):
 *
 *   K = 1 / (b_0 + a_1 / (b_1 + a_2 / (b_2 + ...))),   b_i = x + 2i + 1 - a,   a_i = -i (i - a).
 *
 * The forward recurrence of igammac_cf_impl evaluates the odd convergents of the Legendre fraction, Lentz's method
 * its even part (the same limit). The recurrence forms z x with z = x + 2 - a, which overflows once (x - a) x exceeds
 * the largest number; Lentz's method forms no product of two large quantities, and a_i * d is grouped as
 * -i * ((i - a) * d) so that it does not overflow for a near the largest number.
 *
 * The fraction needs ~ 355 / t^2 iterations (double), t = (x - a) / sqrt(a), for large a. It does not converge within
 * the iteration limit from (measured) a ~ 1.1e7 at t = 0, 1.8e7 at t = 0.1, 3.3e7 at t = 0.2, 1.4e8 at t = 0.3 and
 * 8e10 at t = 0.4 in double, and from a ~ 5.6e7 at t = 0 and 2.2e8 at t = 0.1 in float. Then *converged is false and
 * the returned value is meaningless; igammac_cf_impl returns NaN. igamma and igammac use igamma_temme from a = 1e5
 * (double), 1e4 (float), so this band is not reached through them. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static Scalar igammac_cf_lentz(Scalar a, Scalar x, bool* converged) {
  const Scalar one = 1;
  const Scalar tiny = (numext::numeric_limits<Scalar>::min)();
  const Scalar machep = cephes_helper<Scalar>::machep();
  Scalar b = (x - a) + one;  // x - a is exact for a <= x <= 2a; in (x + 1) - a the 1 is lost for x >= 2 / eps.
  Scalar c = one / tiny;
  Scalar d = one / b;
  Scalar h = d;
  for (int i = 1; i <= igamma_num_iterations<Scalar, VALUE>(); i++) {
    const Scalar ci = Scalar(i);
    b += Scalar(2);
    d = b - ci * ((ci - a) * d);
    if (numext::abs(d) < tiny) d = tiny;
    c = b - ci * ((ci - a) / c);
    if (numext::abs(c) < tiny) c = tiny;
    d = one / d;
    const Scalar delta = d * c;
    h *= delta;
    if (numext::abs(delta - one) <= machep) {
      *converged = true;
      return h;
    }
  }
  *converged = false;
  return h;
}

template <typename Scalar, IgammaComputationMode mode>
struct igammac_cf_impl {
  /* Computes igamc(a, x) or derivative (depending on the mode)
   * using the continued fraction expansion of the complementary
   * incomplete Gamma function.
   *
   * Preconditions:
   *   a > 0
   *   x >= 1
   *   x >= a
   */
  EIGEN_DEVICE_FUNC static Scalar run(Scalar a, Scalar x) {
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar two = 2;
    const Scalar machep = cephes_helper<Scalar>::machep();
    const Scalar big = cephes_helper<Scalar>::big();
    const Scalar biginv = cephes_helper<Scalar>::biginv();

    if ((numext::isinf)(x)) {
      return zero;
    }

    if (mode == VALUE && igamma_use_stirling_main_term(a)) {
      // Where the main term underflows so does Q. The fraction ~ 1 / x is below the smallest normal number for
      // x > 1 / min (8.5e37 in float, 4.5e307 in double), where Lentz's method clamps it and does not converge.
      const Scalar ax_stirling = main_igamma_term_stirling<Scalar>(a, x);
      if (ax_stirling == zero) return zero;
      bool converged;
      const Scalar h = igammac_cf_lentz<Scalar>(a, x, &converged);
      // Unconverged only in the band documented at igammac_cf_lentz, where igamma_temme is used instead. The recurrence
      // and the ordinary main term are wrong there (values above 1, or float igammac(1e8, 1e8) = 1.25e-4 for a true
      // value ~ 0.5), so the result is NaN: a converged fraction or NaN.
      if (!converged) return NumTraits<Scalar>::quiet_NaN();
      return ax_stirling * h;
    }

    Scalar ax = main_igamma_term<Scalar>(a, x);
    // This is independent of mode. If this value is zero,
    // then the function value is zero. If the function value is zero,
    // then we are in a neighborhood where the function value evaluates to zero,
    // so the derivative is zero.
    if (ax == zero) {
      return zero;
    }

    // continued fraction
    Scalar y = one - a;
    Scalar z = x + y + one;
    Scalar c = zero;
    Scalar pkm2 = one;
    Scalar qkm2 = x;
    Scalar pkm1 = x + one;
    Scalar qkm1 = z * x;
    Scalar ans = pkm1 / qkm1;

    Scalar dpkm2_da = zero;
    Scalar dqkm2_da = zero;
    Scalar dpkm1_da = zero;
    Scalar dqkm1_da = -x;
    Scalar dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;

    for (int i = 0; i < igamma_num_iterations<Scalar, mode>(); i++) {
      c += one;
      y += one;
      z += two;

      Scalar yc = y * c;
      Scalar pk = pkm1 * z - pkm2 * yc;
      Scalar qk = qkm1 * z - qkm2 * yc;

      Scalar dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c;
      Scalar dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c;

      if (qk != zero) {
        Scalar ans_prev = ans;
        ans = pk / qk;

        Scalar dans_da_prev = dans_da;
        dans_da = (dpk_da - ans * dqk_da) / qk;

        if (mode == VALUE) {
          if (numext::abs(ans_prev - ans) <= machep * numext::abs(ans)) {
            break;
          }
        } else {
          if (numext::abs(dans_da - dans_da_prev) <= machep) {
            break;
          }
        }
      }

      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      dpkm2_da = dpkm1_da;
      dpkm1_da = dpk_da;
      dqkm2_da = dqkm1_da;
      dqkm1_da = dqk_da;

      if (numext::abs(pk) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;

        dpkm2_da *= biginv;
        dpkm1_da *= biginv;
        dqkm2_da *= biginv;
        dqkm1_da *= biginv;
      }
    }

    /* Compute  x**a * exp(-x) / gamma(a)  */
    Scalar dlogax_da = numext::log(x) - digamma_impl<Scalar>::run(a);
    Scalar dax_da = ax * dlogax_da;

    switch (mode) {
      case VALUE: {
        // An overflow of the recurrence (inf - inf) is reported as NaN rather than returned as inf or a silent 0.
        const Scalar result = ans * ax;
        return EIGEN_PREDICT_FALSE(!(numext::isfinite)(result)) ? NumTraits<Scalar>::quiet_NaN() : result;
      }
      case DERIVATIVE:
        return ans * dax_da + dans_da * ax;
      case SAMPLE_DERIVATIVE:
      default:  // this is needed to suppress clang warning
        return -(dans_da + ans * dlogax_da) * x;
    }
  }
};

template <typename Scalar, IgammaComputationMode mode>
struct igamma_series_impl {
  /* Computes igam(a, x) or its derivative (depending on the mode)
   * using the series expansion of the incomplete Gamma function.
   *
   * Preconditions:
   *   x > 0
   *   a > 0
   *   !(x > 1 && x > a)
   */
  EIGEN_DEVICE_FUNC static Scalar run(Scalar a, Scalar x) {
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar machep = cephes_helper<Scalar>::machep();

    const bool stirling = mode == VALUE && igamma_use_stirling_main_term(a);
    Scalar ax = stirling ? main_igamma_term_stirling<Scalar>(a, x) : main_igamma_term<Scalar>(a, x);

    // This is independent of mode. If this value is zero,
    // then the function value is zero. If the function value is zero,
    // then we are in a neighborhood where the function value evaluates to zero,
    // so the derivative is zero.
    if (ax == zero) {
      return zero;
    }

    ax /= a;

    /* power series */
    Scalar r = a;
    Scalar c = one;
    Scalar ans = one;

    Scalar dc_da = zero;
    Scalar dans_da = zero;

    const int num_iterations = mode == VALUE ? igamma_series_num_iterations(a) : igamma_num_iterations<Scalar, mode>();
    bool converged = false;
    for (int i = 0; i < num_iterations; i++) {
      // r += 1 stalls once a >= 2^24 in float: a + 1 rounds to a.
      r = mode == VALUE ? a + Scalar(i + 1) : r + one;
      Scalar term = x / r;
      Scalar dterm_da = -x / (r * r);
      dc_da = term * dc_da + dterm_da * c;
      dans_da += dc_da;
      c *= term;
      ans += c;

      if (mode == VALUE) {
        if (c <= machep * ans) {
          converged = true;
          break;
        }
      } else {
        if (numext::abs(dc_da) <= machep * numext::abs(dans_da)) {
          break;
        }
      }
    }
    // A truncated sum is wrong by an unbounded amount (double igamma(1e6, 1e6) was 0.477, true 0.500): a converged sum
    // or NaN.
    if (mode == VALUE && !converged) return NumTraits<Scalar>::quiet_NaN();

    Scalar dlogax_da = numext::log(x) - digamma_impl<Scalar>::run(a + one);
    Scalar dax_da = ax * dlogax_da;

    switch (mode) {
      case VALUE:
        return ans * ax;
      case DERIVATIVE:
        return ans * dax_da + dans_da * ax;
      case SAMPLE_DERIVATIVE:
      default:  // this is needed to suppress clang warning
        return -(dans_da + ans * dlogax_da) * x / a;
    }
  }
};

template <typename Scalar>
struct igammac_default_impl {
  EIGEN_DEVICE_FUNC static Scalar run(Scalar a, Scalar x) {
    /*  igamc()
     *
     *	Incomplete gamma integral (modified for Eigen)
     *
     *
     *
     * SYNOPSIS:
     *
     * double a, x, y, igamc();
     *
     * y = igamc( a, x );
     *
     * DESCRIPTION:
     *
     * The function is defined by
     *
     *
     *  igamc(a,x)   =   1 - igam(a,x)
     *
     *                            inf.
     *                              -
     *                     1       | |  -t  a-1
     *               =   -----     |   e   t   dt.
     *                    -      | |
     *                   | (a)    -
     *                             x
     *
     *
     * In this implementation both arguments must be positive.
     * The integral is evaluated by either a power series or
     * continued fraction expansion, depending on the relative
     * values of a and x.
     *
     * ACCURACY (float):
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       7.8e-6      5.9e-7
     *
     *
     * ACCURACY (double):
     *
     * Tested at random a, x.
     *                a         x                      Relative error:
     * arithmetic   domain   domain     # trials      peak         rms
     *    IEEE     0.5,100   0,100      200000       1.9e-14     1.7e-15
     *    IEEE     0.01,0.5  0,100      200000       1.4e-13     1.6e-15
     *
     */
    /*
      Cephes Math Library Release 2.2: June, 1992
      Copyright 1985, 1987, 1992 by Stephen L. Moshier
      Direct inquiries to 30 Frost Street, Cambridge, MA 02140
    */
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    if ((x < zero) || (a <= zero)) {
      // domain error
      return nan;
    }

    if ((numext::isnan)(a) || (numext::isnan)(x)) {  // propagate nans
      return nan;
    }

    if (igamma_use_temme(a)) {
      Scalar p, q;
      igamma_temme(a, x, &p, &q);
      return q;
    }

    if ((x < one) || (x < a)) {
      // Clamp to [0,1] since 1-igamma can produce tiny negative values
      // due to floating-point cancellation for extreme arguments.
      return igamma_clamp_to_unit_interval(one - igamma_series_impl<Scalar, VALUE>::run(a, x));
    }

    // NaN where Lentz's fraction does not converge (igammac_cf_impl); the clamp keeps NaN.
    return igamma_clamp_to_unit_interval(igammac_cf_impl<Scalar, VALUE>::run(a, x));
  }
};

template <typename Scalar>
struct igammac_impl : special_function_overload_or<special_function_overloads::igammac_fn, igammac_default_impl<Scalar>,
                                                   Scalar, Scalar> {};

/************************************************************************************************
 * Implementation of igamma (incomplete gamma integral), based on Cephes                         *
 ************************************************************************************************/

template <typename Scalar, IgammaComputationMode mode>
struct igamma_generic_impl {
  EIGEN_DEVICE_FUNC static Scalar run(Scalar a, Scalar x) {
    /* Depending on the mode, returns
     * - VALUE: incomplete Gamma function igamma(a, x)
     * - DERIVATIVE: derivative of incomplete Gamma function d/da igamma(a, x)
     * - SAMPLE_DERIVATIVE: implicit derivative of a Gamma random variable
     * x ~ Gamma(x | a, 1), dx/da = -1 / Gamma(x | a, 1) * d igamma(a, x) / dx
     *
     * Derivatives are implemented by forward-mode differentiation.
     */
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    if ((numext::isnan)(a) || (numext::isnan)(x)) {  // propagate nans
      return nan;
    }

    if (x == zero) return zero;

    if ((x < zero) || (a <= zero)) {  // domain error
      return nan;
    }

    if (mode == VALUE && igamma_use_temme(a)) {
      Scalar p, q;
      igamma_temme(a, x, &p, &q);
      return p;
    }

    if ((x > one) && (x > a)) {
      Scalar ret = igammac_cf_impl<Scalar, mode>::run(a, x);
      if (mode == VALUE) {
        // Clamp to [0,1] since 1-igammac can produce tiny negative values
        // due to floating-point cancellation for extreme arguments.
        return igamma_clamp_to_unit_interval(one - ret);
      } else {
        return -ret;
      }
    }

    Scalar ret = igamma_series_impl<Scalar, mode>::run(a, x);
    if (mode == VALUE) {
      // Clamp to [0,1] since accumulated series terms can slightly exceed 1.0
      // due to floating-point rounding for extreme arguments.
      return igamma_clamp_to_unit_interval(ret);
    }
    return ret;
  }
};

template <typename Scalar>
struct igamma_impl : special_function_overload_or<special_function_overloads::igamma_fn,
                                                  igamma_generic_impl<Scalar, VALUE>, Scalar, Scalar> {
  /* igam()
   * Incomplete gamma integral.
   *
   * The CDF of Gamma(a, 1) random variable at the point x.
   *
   * Accuracy estimation. For each a in [10^-2, 10^-1...10^3] we sample
   * 50 Gamma random variables x ~ Gamma(x | a, 1), a total of 300 points.
   * The ground truth is computed by mpmath. Mean absolute error:
   * float: 1.26713e-05
   * double: 2.33606e-12
   *
   * Cephes documentation below.
   *
   * SYNOPSIS:
   *
   * double a, x, y, igam();
   *
   * y = igam( a, x );
   *
   * DESCRIPTION:
   *
   * The function is defined by
   *
   *                           x
   *                            -
   *                   1       | |  -t  a-1
   *  igam(a,x)  =   -----     |   e   t   dt.
   *                  -      | |
   *                 | (a)    -
   *                           0
   *
   *
   * In this implementation both arguments must be positive.
   * The integral is evaluated by either a power series or
   * continued fraction expansion, depending on the relative
   * values of a and x.
   *
   * ACCURACY (double):
   *
   *                      Relative error:
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      0,30       200000       3.6e-14     2.9e-15
   *    IEEE      0,100      300000       9.9e-14     1.5e-14
   *
   *
   * ACCURACY (float):
   *
   *                      Relative error:
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      0,30        20000       7.8e-6      5.9e-7
   *
   */
  /*
    Cephes Math Library Release 2.2: June, 1992
    Copyright 1985, 1987, 1992 by Stephen L. Moshier
    Direct inquiries to 30 Frost Street, Cambridge, MA 02140
  */

  /* left tail of incomplete gamma function:
   *
   *          inf.      k
   *   a  -x   -       x
   *  x  e     >   ----------
   *           -     -
   *          k=0   | (a+k+1)
   *
   */
};

template <typename Scalar>
struct igamma_der_a_impl : special_function_overload_or<special_function_overloads::igamma_der_a_fn,
                                                        igamma_generic_impl<Scalar, DERIVATIVE>, Scalar, Scalar> {
  /* Derivative of the incomplete Gamma function with respect to a.
   *
   * Computes d/da igamma(a, x) by forward differentiation of the igamma code.
   *
   * Accuracy estimation. For each a in [10^-2, 10^-1...10^3] we sample
   * 50 Gamma random variables x ~ Gamma(x | a, 1), a total of 300 points.
   * The ground truth is computed by mpmath. Mean absolute error:
   * float: 6.17992e-07
   * double: 4.60453e-12
   *
   * Reference:
   * R. Moore. "Algorithm AS 187: Derivatives of the incomplete gamma
   * integral". Journal of the Royal Statistical Society. 1982
   */
};

template <typename Scalar>
struct gamma_sample_der_alpha_impl
    : special_function_overload_or<special_function_overloads::gamma_sample_der_alpha_fn,
                                   igamma_generic_impl<Scalar, SAMPLE_DERIVATIVE>, Scalar, Scalar> {
  /* Derivative of a Gamma random variable sample with respect to alpha.
   *
   * Consider a sample of a Gamma random variable with the concentration
   * parameter alpha: sample ~ Gamma(alpha, 1). The reparameterization
   * derivative that we want to compute is dsample / dalpha =
   * d igammainv(alpha, u) / dalpha, where u = igamma(alpha, sample).
   * However, this formula is numerically unstable and expensive, so instead
   * we use implicit differentiation:
   *
   * igamma(alpha, sample) = u, where u ~ Uniform(0, 1).
   * Apply d / dalpha to both sides:
   * d igamma(alpha, sample) / dalpha
   *     + d igamma(alpha, sample) / dsample * dsample/dalpha  = 0
   * d igamma(alpha, sample) / dalpha
   *     + Gamma(sample | alpha, 1) dsample / dalpha = 0
   * dsample/dalpha = - (d igamma(alpha, sample) / dalpha)
   *                   / Gamma(sample | alpha, 1)
   *
   * Here Gamma(sample | alpha, 1) is the PDF of the Gamma distribution
   * (note that the derivative of the CDF w.r.t. sample is the PDF).
   * See the reference below for more details.
   *
   * The derivative of igamma(alpha, sample) is computed by forward
   * differentiation of the igamma code. Division by the Gamma PDF is performed
   * in the same code, increasing the accuracy and speed due to cancellation
   * of some terms.
   *
   * Accuracy estimation. For each alpha in [10^-2, 10^-1...10^3] we sample
   * 50 Gamma random variables sample ~ Gamma(sample | alpha, 1), a total of 300
   * points. The ground truth is computed by mpmath. Mean absolute error:
   * float: 2.1686e-06
   * double: 1.4774e-12
   *
   * Reference:
   * M. Figurnov, S. Mohamed, A. Mnih "Implicit Reparameterization Gradients".
   * 2018
   */
};

/*****************************************************************************
 * Implementation of Riemann zeta function of two arguments, based on Cephes *
 *****************************************************************************/

// For q < -kZetaMaxNegativeQ, zeta(x, q) is evaluated by a reflection rather than by summing about |q| terms.
constexpr int kZetaMaxNegativeQ = 1000;

template <typename Scalar>
struct zeta_impl_series {
  EIGEN_STATIC_ASSERT((std::is_same<Scalar, Scalar>::value == false), THIS_TYPE_IS_NOT_SUPPORTED)

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar run(const Scalar) { return Scalar(0); }
};

template <>
struct zeta_impl_series<float> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run(float& a, float& b, float& s, const float x, const float machep,
                                                        const float inv_scale) {
    int i = 0;
    while ((i < 9) || (a <= 9.0f)) {
      i += 1;
      a += 1.0f;
      b = numext::pow(a * inv_scale, -x);
      s += b;
      if (numext::abs(b / s) < machep) return true;
    }

    // Return whether we are done
    return false;
  }
};

template <>
struct zeta_impl_series<double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool run(double& a, double& b, double& s, const double x,
                                                        const double machep, const double inv_scale) {
    int i = 0;
    while ((i < 9) || (a <= 9.0)) {
      i += 1;
      a += 1.0;
      b = numext::pow(a * inv_scale, -x);
      s += b;
      if (numext::abs(b / s) < machep) return true;
    }

    // Return whether we are done
    return false;
  }
};

template <typename Scalar>
struct zeta_default_impl {
  /* Euler-Maclaurin summation of the Hurwitz zeta function (see run()), for x > 1 and q > 0 or q < 0 non-integer, and
   * multiplied by scale^x where inv_scale = 1 / scale. Every term (k + q)^-x is computed as (inv_scale (k + q))^-x,
   * which does not underflow when scale is of the size of q, so scale = q gives a result of order 1 where zeta(x, q)
   * itself underflows. */
  EIGEN_DEVICE_FUNC static Scalar sum(Scalar x, Scalar q, Scalar inv_scale) {
    int i;
    Scalar a, b, k, s, t, w;

    const Scalar A[] = {
        Scalar(12.0),
        Scalar(-720.0),
        Scalar(30240.0),
        Scalar(-1209600.0),
        Scalar(47900160.0),
        Scalar(-1.8924375803183791606e9), /*1.307674368e12/691*/
        Scalar(7.47242496e10),
        Scalar(-2.950130727918164224e12),  /*1.067062284288e16/3617*/
        Scalar(1.1646782814350067249e14),  /*5.109094217170944e18/43867*/
        Scalar(-4.5979787224074726105e15), /*8.028576626982912e20/174611*/
        Scalar(1.8152105401943546773e17),  /*1.5511210043330985984e23/854513*/
        Scalar(-7.1661652561756670113e18)  /*1.6938241367317436694528e27/236364091*/
    };

    const Scalar zero = Scalar(0.0), half = Scalar(0.5), one = Scalar(1.0);
    const Scalar machep = cephes_helper<Scalar>::machep();

    /* Permit negative q but continue sum until n+q > +9 . For q < -kZetaMaxNegativeQ, run() uses the reflection
     * below instead, so this runs for at most kZetaMaxNegativeQ + 10 iterations. */
    s = numext::pow(q * inv_scale, -x);
    a = q;
    b = zero;
    // Run the summation in a helper function that is specific to the floating precision
    if (zeta_impl_series<Scalar>::run(a, b, s, x, machep, inv_scale)) {
      return s;
    }

    // If b is zero, then the tail sum will also end up being zero.
    // Exiting early here can prevent NaNs for some large inputs, where
    // the tail sum computed below has term `a` which can overflow to `inf`.
    if (numext::equal_strict(b, zero)) {
      return s;
    }

    w = a;
    s += b * w / (x - one);
    s -= half * b;
    a = one;
    k = zero;

    for (i = 0; i < 12; i++) {
      a *= x + k;
      b /= w;
      t = a * b / A[i];
      s = s + t;
      t = numext::abs(t / s);
      if (t < machep) {
        break;
      }
      k += one;
      a *= x + k;
      b /= w;
      k += one;
    }
    return s;
  }

  /* zeta(x, q) = exp(log(sum(x, q, 1 / q)) - x log(q)), for q >= 1. The terms are scaled to ~ 1 and the sum to
   * ~ q / (x - 1), which is finite for x >= 2 and q <= max / 16; 1 / q is then also normal under flush to zero. */
  EIGEN_DEVICE_FUNC static Scalar log_domain_sum_scaled_by_q(Scalar x, Scalar q) {
    return numext::exp(numext::log(sum(x, q, Scalar(1) / q)) - x * numext::log(q));
  }

  /* The same for any finite q >= 1 and x > 1, with the terms scaled to ~ 1 / q_c, q_c = min(q, 1 / (16 min)), so
   * that neither the first term nor the sum ~ q / (q_c (x - 1)) <= 64 / (x - 1) leaves the normal range, however
   * close x is to 1 and up to q = max. With L = log(q), L_c = log(q_c) and e = L_c - x L = (L_c - L) - (x - 1) L:
   *   zeta(x, q) = exp(log(sum(x, q, exp(e / x))) + e).
   * The relative error is ~ eps (x - 1) log(q). Where e < log(denorm_min) - 6, x - 1 > 1 and the sum is < 64, so zeta
   * rounds to 0; returning it there also avoids exp(e / x) ~ 1 / q, which is subnormal or flushed to zero. */
  EIGEN_DEVICE_FUNC static Scalar log_domain_sum_rescaled(Scalar x, Scalar q) {
    const Scalar log_q = numext::log(q);
    const Scalar q_c = numext::mini(q, Scalar(1) / (Scalar(16) * (numext::numeric_limits<Scalar>::min)()));
    const Scalar e = (numext::log(q_c) - log_q) - (x - Scalar(1)) * log_q;
    const Scalar log_denorm_min = numext::log((numext::numeric_limits<Scalar>::min)()) -
                                  Scalar(numext::numeric_limits<Scalar>::digits - 1) * Scalar(0.6931471805599453);
    if (e < log_denorm_min - Scalar(6)) return Scalar(0);
    return numext::exp(numext::log(sum(x, q, numext::exp(e / x))) + e);
  }

  /* sum(x, q, 1) for q >= 1, also where q^-x is subnormal or zero but the sum ~ q^(1-x) / (x - 1) is normal. The
   * first term b = q^-x = min 2^-k then has the relative error ~ eps 2^k and dominates the sum for x near 1. Once
   * that exceeds the error ~ eps (1 + 2 (x - 1) log(q)) of a log-domain form, use one. For x >= 2 and q <= max / 16
   * that is the scaling by 1 / q; otherwise the rescaled form, since the sum ~ q / (x - 1) of the first overflows for
   * x - 1 < 1/16 and its 1 / q is subnormal, or flushed to zero, for q > max / 16. q^-x < min is x log(q) > -log(min);
   * this also excludes q = 1. */
  EIGEN_DEVICE_FUNC static Scalar sum_without_underflow(Scalar x, Scalar q) {
    if (q >= Scalar(1) && !(numext::isinf)(x)) {
      const Scalar log_q = numext::log(q);
      const Scalar x_log_q = x * log_q;
      const Scalar log_min = numext::log((numext::numeric_limits<Scalar>::min)());
      if (x_log_q > -log_min && x_log_q + log_min > numext::log(Scalar(2) * (Scalar(1) + (x - Scalar(1)) * log_q))) {
        if (x >= Scalar(2) && q <= (numext::numeric_limits<Scalar>::max)() / Scalar(16)) {
          return log_domain_sum_scaled_by_q(x, q);
        }
        return log_domain_sum_rescaled(x, q);
      }
    }
    return sum(x, q, Scalar(1));
  }

  EIGEN_DEVICE_FUNC static Scalar run(Scalar x, Scalar q) {
    /*							zeta.c
     *
     *	Riemann zeta function of two arguments
     *
     *
     *
     * SYNOPSIS:
     *
     * double x, q, y, zeta();
     *
     * y = zeta( x, q );
     *
     *
     *
     * DESCRIPTION:
     *
     *
     *
     *                 inf.
     *                  -        -x
     *   zeta(x,q)  =   >   (k+q)
     *                  -
     *                 k=0
     *
     * where x > 1 and q is not a negative integer or zero.
     * The Euler-Maclaurin summation formula is used to obtain
     * the expansion
     *
     *                n
     *                -       -x
     * zeta(x,q)  =   >  (k+q)
     *                -
     *               k=1
     *
     *           1-x                 inf.  B   x(x+1)...(x+2j)
     *      (n+q)           1         -     2j
     *  +  ---------  -  -------  +   >    --------------------
     *        x-1              x      -                   x+2j+1
     *                   2(n+q)      j=1       (2j)! (n+q)
     *
     * where the B2j are Bernoulli numbers.  Note that (see zetac.c)
     * zeta(x,1) = zetac(x) + 1.
     *
     *
     *
     * ACCURACY:
     *
     * Relative error for single precision:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,25        10000       6.9e-7      1.0e-7
     *
     * Large arguments may produce underflow in powf(), in which
     * case the results are inaccurate.
     *
     * REFERENCE:
     *
     * Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
     * Series, and Products, p. 1073; Academic Press, 1980.
     *
     */

    Scalar p, r;

    const Scalar maxnum = NumTraits<Scalar>::infinity();
    const Scalar zero = Scalar(0.0), half = Scalar(0.5), one = Scalar(1.0);
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    if (x == one) return maxnum;

    if (x < one) {
      return nan;
    }

    if (q <= zero) {
      if (q == numext::floor(q)) {
        if (numext::rint(Scalar(0.5) * x) == Scalar(0.5) * x) {
          return maxnum;
        } else {
          return nan;
        }
      }
      p = x;
      r = numext::floor(p);
      if (p != r) return nan;
    }

    /* The sum for q < 0 has about |q| terms. Beyond kZetaMaxNegativeQ split it at the sign change of k + q, with
     * q = f - M, f in (0, 1) and M a positive integer, for integer x (checked above):
     *
     *   zeta(x, q) = zeta(x, f) + sum_{j=1}^M (f - j)^-x
     *              = zeta(x, f) + (-1)^x (zeta(x, 1 - f) - zeta(x, 1 - q)),
     *
     * using sum_{j=1}^M (j - f)^-x = zeta(x, 1 - f) - zeta(x, M + 1 - f) and M + 1 - f = 1 - q. */
    if (q < -Scalar(kZetaMaxNegativeQ)) {
      const Scalar f = q - numext::floor(q);
      const Scalar x_is_even = (numext::rint(half * x) == half * x) ? one : -one;
      return (sum(x, f, one) + x_is_even * sum(x, one - f, one)) - x_is_even * sum_without_underflow(x, one - q);
    }

    return sum_without_underflow(x, q);
  }
};

template <typename Scalar>
struct zeta_impl
    : special_function_overload_or<special_function_overloads::zeta_fn, zeta_default_impl<Scalar>, Scalar, Scalar> {};

/****************************************************************************
 * Implementation of polygamma function                                      *
 ****************************************************************************/

template <typename Scalar>
struct polygamma_default_impl {
  EIGEN_DEVICE_FUNC static Scalar run(Scalar n, Scalar x) {
    Scalar zero = 0.0, one = 1.0;
    Scalar nplus = n + one;
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    // Check that n is a non-negative integer
    if (numext::floor(n) != n || n < zero || (numext::isinf)(n)) {
      return nan;
    }
    // Just return the digamma function for n = 0
    else if (n == zero) {
      return digamma_impl<Scalar>::run(x);
    }
    // Use the same implementation as scipy
    else {
      // (-1)^(n+1) from the parity of n: n + 1 is not representable for n >= 2^24 (float), 2^53 (double).
      const Scalar half = Scalar(0.5);
      const bool n_is_even = numext::floor(half * n) == half * n;
      const Scalar sign = n_is_even ? -one : one;
      // For odd n + 1 and x = -M - 1/2 < -kZetaMaxNegativeQ the reflection in zeta_impl reduces exactly to
      // zeta(n + 1, x) = zeta(n + 1, 1 - x); with 1 - x > 0 the log-domain evaluation below applies.
      if (n_is_even && x < -Scalar(kZetaMaxNegativeQ) && x - numext::floor(x) == half) x = one - x;
      Scalar factorial = numext::exp(lgamma_impl<Scalar>::run(nplus));
      Scalar zeta = zeta_impl<Scalar>::run(nplus, x);
      // The limit at +inf is 0, with the sign of (-1)^(n+1); n! zeta(n + 1, inf) would be NaN for n! = inf.
      if (x == NumTraits<Scalar>::infinity()) return sign * zero;
      // 1 / x must be a normal number, also when subnormals are flushed to zero: for x <= max / 16 it is >= 16 / max,
      // which is above the smallest normal number (4.7e-38 vs 1.2e-38 in float, 8.9e-308 vs 2.2e-308 in double).
      if (x > zero && x <= (numext::numeric_limits<Scalar>::max)() / Scalar(16) &&
          ((numext::isinf)(factorial) || !(zeta >= (numext::numeric_limits<Scalar>::min)()))) {
        // n! overflows, or zeta(n + 1, x) ~ x^-(n+1) underflows, while n! zeta(n + 1, x) can be representable (a
        // product of inf and 0 is NaN). With Z = x^(n+1) zeta(n + 1, x), which is >= 1 and ~ x / n for x >> n, form
        // the product in the log domain:
        //   n! zeta(n + 1, x) = exp(lgamma(n + 1) - (n + 1) log x + log Z).
        const Scalar log_z = numext::log(zeta_default_impl<Scalar>::sum(nplus, x, one / x));
        if ((numext::isfinite)(log_z)) {
          return sign * numext::exp(lgamma_impl<Scalar>::run(nplus) - nplus * numext::log(x) + log_z);
        }
      }
      // x < 0: zeta(n + 1, x) = 0 where the sums cancel exactly (x = -N - 1/2) and the true value, n! zeta(n + 1, 1 -
      // x) with 1 - x > 1000, underflows; likewise for x > max / 16, where n! x^-n underflows. inf * 0 would be NaN.
      if ((numext::isinf)(factorial) && zeta == zero) return sign * zero;
      return sign * factorial * zeta;
    }
  }
};

template <typename Scalar>
struct polygamma_impl : special_function_overload_or<special_function_overloads::polygamma_fn,
                                                     polygamma_default_impl<Scalar>, Scalar, Scalar> {};

/************************************************************************************************
 * Implementation of betainc (incomplete beta integral), based on Cephes                         *
 ************************************************************************************************/

template <typename Scalar>
struct betainc_default_impl {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar run(Scalar, Scalar, Scalar) {
    EIGEN_STATIC_ASSERT((!std::is_same<Scalar, Scalar>::value), THIS_TYPE_IS_NOT_SUPPORTED)
    /*	betaincf.c
     *
     *	Incomplete beta integral
     *
     *
     * SYNOPSIS:
     *
     * float a, b, x, y, betaincf();
     *
     * y = betaincf( a, b, x );
     *
     *
     * DESCRIPTION:
     *
     * Returns incomplete beta integral of the arguments, evaluated
     * from zero to x.  The function is defined as
     *
     *                  x
     *     -            -
     *    | (a+b)      | |  a-1     b-1
     *  -----------    |   t   (1-t)   dt.
     *   -     -     | |
     *  | (a) | (b)   -
     *                 0
     *
     * The domain of definition is 0 <= x <= 1.  In this
     * implementation a and b are restricted to positive values.
     * The integral from x to 1 may be obtained by the symmetry
     * relation
     *
     *    1 - betainc( a, b, x )  =  betainc( b, a, 1-x ).
     *
     * The integral is evaluated by a continued fraction expansion.
     * If a < 1, the function calls itself recursively after a
     * transformation to increase a to a+1.
     *
     * ACCURACY (float):
     *
     * Tested at random points (a,b,x) with a and b in the indicated
     * interval and x between 0 and 1.
     *
     * arithmetic   domain     # trials      peak         rms
     * Relative error:
     *    IEEE       0,30       10000       3.7e-5      5.1e-6
     *    IEEE       0,100      10000       1.7e-4      2.5e-5
     * The useful domain for relative error is limited by underflow
     * of the single precision exponential function.
     * Absolute error:
     *    IEEE       0,30      100000       2.2e-5      9.6e-7
     *    IEEE       0,100      10000       6.5e-5      3.7e-6
     *
     * Larger errors may occur for extreme ratios of a and b.
     *
     * ACCURACY (double):
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,5         10000       6.9e-15     4.5e-16
     *    IEEE      0,85       250000       2.2e-13     1.7e-14
     *    IEEE      0,1000      30000       5.3e-12     6.3e-13
     *    IEEE      0,10000    250000       9.3e-11     7.1e-12
     *    IEEE      0,100000    10000       8.7e-10     4.8e-11
     * Outputs smaller than the IEEE gradual underflow threshold
     * were excluded from these statistics.
     *
     * ERROR MESSAGES:
     *   message         condition      value returned
     * incbet domain      x<0, x>1          nan
     * incbet underflow                     nan
     */
    return Scalar(0);
  }
};

template <typename Scalar>
struct betainc_impl : special_function_overload_or<special_function_overloads::betainc_fn, betainc_default_impl<Scalar>,
                                                   Scalar, Scalar, Scalar> {};

/* Continued fraction expansion #1 for incomplete beta integral (small_branch = True)
 * Continued fraction expansion #2 for incomplete beta integral (small_branch = False)
 */
template <typename Scalar>
struct incbeta_cfe {
  EIGEN_STATIC_ASSERT((std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value),
                      THIS_TYPE_IS_NOT_SUPPORTED)

  // xc is 1 - x: for x close to 1 it cannot be recovered from x, so the caller may supply it.
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar run(Scalar a, Scalar b, Scalar x, Scalar xc, bool small_branch) {
    const Scalar big = cephes_helper<Scalar>::big();
    const Scalar machep = cephes_helper<Scalar>::machep();
    const Scalar biginv = cephes_helper<Scalar>::biginv();

    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar two = 2;

    Scalar xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    Scalar k1, k2, k3, k4, k5, k6, k7, k8, k26update;
    Scalar ans;
    int n;

    constexpr int num_iters = (std::is_same<Scalar, float>::value) ? 100 : 300;
    const Scalar thresh = (std::is_same<Scalar, float>::value) ? machep : Scalar(3) * machep;
    Scalar r = (std::is_same<Scalar, float>::value) ? zero : one;

    if (small_branch) {
      k1 = a;
      k2 = a + b;
      k3 = a;
      k4 = a + one;
      k5 = one;
      k6 = b - one;
      k7 = k4;
      k8 = a + two;
      k26update = one;
    } else {
      k1 = a;
      k2 = b - one;
      k3 = a;
      k4 = a + one;
      k5 = one;
      k6 = a + b;
      k7 = a + one;
      k8 = a + two;
      k26update = -one;
      x = x / xc;
    }

    pkm2 = zero;
    qkm2 = one;
    pkm1 = one;
    qkm1 = one;
    ans = one;
    n = 0;

    do {
      xk = -(x * k1 * k2) / (k3 * k4);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = (x * k5 * k6) / (k7 * k8);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if (qk != zero) {
        r = pk / qk;
        if (numext::abs(ans - r) < numext::abs(r) * thresh) {
          return r;
        }
        ans = r;
      }

      k1 += one;
      k2 += k26update;
      k3 += two;
      k4 += two;
      k5 += one;
      k6 -= k26update;
      k7 += two;
      k8 += two;

      if ((numext::abs(qk) + numext::abs(pk)) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;
      }
      if ((numext::abs(qk) < biginv) || (numext::abs(pk) < biginv)) {
        pkm2 *= big;
        pkm1 *= big;
        qkm2 *= big;
        qkm1 *= big;
      }
    } while (++n < num_iters);

    return ans;
  }
};

/* log(Gamma(a + b) / (Gamma(a) Gamma(b))) is evaluated as lgamma(a + b) - lgamma(a) - lgamma(b), whose error is about
 * eps M (log M - 1) for M = max(a, b), the rounding of lgamma(M), which the caller's exp() turns into relative error.
 * Where that reaches ~0.1, i.e. M >= 1 / (64 eps) (1.3e5 in float, 7e13 in double), the result is garbage and
 * betainc_stirling_ratio, which has no such cancellation when m = min(a, b) <= M / 4, is used instead. Below that
 * threshold the sum is kept: its rounding errors are correlated with those of the continued fraction, which uses the
 * same a + b, and the Stirling form is not systematically better there. The callers apply this only where 1 - x
 * rounds to 1, the inputs that otherwise hang; every other input keeps the original arithmetic. */
template <typename Scalar>
EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE bool betainc_use_stirling_ratio(Scalar a, Scalar b) {
  const Scalar big = numext::maxi(a, b);
  return big * NumTraits<Scalar>::epsilon() >= Scalar(1.0 / 64) && Scalar(4) * numext::mini(a, b) <= big;
}

/* Stirling's series (DLMF 5.11.1) with the large terms combined, for L = M + m, M >= 1 / (64 eps):
 *   lgamma(L) - lgamma(M) = m log M + (L - 1/2) log1p(m / M) - m + (1/12)(1/L - 1/M) - (1/360)(1/L^3 - 1/M^3) + O(M^-5)
 * and the remaining term is -lgamma(m). */
template <typename Scalar>
EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar betainc_stirling_ratio(Scalar a, Scalar b) {
  const Scalar M = numext::maxi(a, b);
  const Scalar m = numext::mini(a, b);
  const Scalar sum = M + m;
  const Scalar sum_inv = Scalar(1) / sum;
  const Scalar M_inv = Scalar(1) / M;
  Scalar d = m * numext::log(M) + (sum - Scalar(0.5)) * numext::log1p(m / M) - m;
  d += (Scalar(1) / Scalar(12)) * (sum_inv - M_inv) -
       (Scalar(1) / Scalar(360)) * (sum_inv * sum_inv * sum_inv - M_inv * M_inv * M_inv);
  return d - lgamma_impl<Scalar>::run(m);
}

/* Helper functions depending on the Scalar type */
template <typename Scalar>
struct betainc_helper {};

template <>
struct betainc_helper<float> {
  /* Core implementation, assumes a large (> 1.0) */
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float incbsa(float aa, float bb, float xx) {
    float ans, a, b, t, x, onemx, xc, logx;
    bool reversed_a_b = false;

    onemx = 1.0f - xx;
    // 1 - xx rounds to 1: after reversing, x would be 1 and its complement x / (1 - x) infinite.
    const bool onemx_is_one = (onemx == 1.0f);

    /* see if x is greater than the mean */
    if (xx > (aa / (aa + bb))) {
      reversed_a_b = true;
      a = bb;
      b = aa;
      t = xx;
      x = onemx;
    } else {
      a = aa;
      b = bb;
      t = onemx;
      x = xx;
    }

    // In the degenerate case use the exact 1 - x = xx (reversed) or log(1 - xx) = log1p(-xx) in place of the rounded 1.
    const bool degenerate = reversed_a_b && onemx_is_one;
    xc = degenerate ? t : 1.0f - x;
    logx = degenerate ? numext::log1p(-xx) : numext::log(x);

    /* Choose expansion for optimal convergence */
    if (b > 10.0f) {
      if (numext::abs(b * x / a) < 0.3f) {
        t = betainc_helper<float>::incbps(a, b, x, xc, logx, degenerate ? numext::log(t) : numext::log1p(-x),
                                          onemx_is_one);
        if (reversed_a_b) t = 1.0f - t;
        return t;
      }
    }

    const float log_t = (onemx_is_one && !reversed_a_b) ? numext::log1p(-xx) : numext::log(t);
    ans = x * (a + b - 2.0f) / (a - 1.0f);
    if (ans < 1.0f) {
      ans = incbeta_cfe<float>::run(a, b, x, xc, true /* small_branch */);
      t = b * log_t;
    } else {
      ans = incbeta_cfe<float>::run(a, b, x, xc, false /* small_branch */);
      t = (b - 1.0f) * log_t;
    }

    if (onemx_is_one && betainc_use_stirling_ratio(a, b)) {
      t += a * logx + betainc_stirling_ratio(a, b);
    } else {
      t += a * logx + lgamma_impl<float>::run(a + b) - lgamma_impl<float>::run(a) - lgamma_impl<float>::run(b);
    }
    t += numext::log(ans / a);
    t = numext::exp(t);

    if (reversed_a_b) t = 1.0f - t;
    return t;
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float incbps(float a, float b, float x, float xc, float logx,
                                                            float logxc, bool xc_rounds_to_one) {
    float t, u, y, s;
    const float machep = cephes_helper<float>::machep();

    y = a * logx + (b - 1.0f) * logxc - numext::log(a);
    if (xc_rounds_to_one && betainc_use_stirling_ratio(a, b)) {
      y += betainc_stirling_ratio(a, b);
    } else {
      y -= lgamma_impl<float>::run(a) + lgamma_impl<float>::run(b);
      y += lgamma_impl<float>::run(a + b);
    }

    t = x / xc;
    s = 0.0f;
    u = 1.0f;
    // Safety net against an endless loop: for b >= 2^24, b -= 1 changes nothing, and a ratio u_{k+1} / u_k of 1 never
    // decays. Convergent sums need at most b + 1 < 2^24 iterations (thousands for b ~ 5e4 with t b / a ~ 1).
    int n = 0;
    do {
      b -= 1.0f;
      if (b == 0.0f) {
        break;
      }
      a += 1.0f;
      u *= t * b / a;
      s += u;
    } while (numext::abs(u) > machep && ++n < (1 << 24));

    return numext::exp(y) * (1.0f + s);
  }
};

template <>
struct betainc_impl<float> {
  EIGEN_DEVICE_FUNC static float run(float a, float b, float x) {
    const float nan = NumTraits<float>::quiet_NaN();
    float ans, t;

    if ((numext::isnan)(a) || (numext::isnan)(b) || (numext::isnan)(x)) return nan;
    if (a == 0.0f && b == 0.0f) return nan;
    if (x < 0.0f || x > 1.0f) return nan;
    if (a < 0.0f) return nan;
    if (b < 0.0f) return nan;
    if (a == 0.0f) return 1.0f;
    if (b == 0.0f) return 0.0f;
    if (x == 0.0f) return 0.0f;
    if (x == 1.0f) return 1.0f;
    // mtherr("betaincf", DOMAIN);

    /* transformation for small aa */
    if (a <= 1.0f) {
      ans = betainc_helper<float>::incbsa(a + 1.0f, b, x);
      if (1.0f - x == 1.0f && betainc_use_stirling_ratio(a + 1.0f, b)) {
        // Gamma(a + b) / (Gamma(a + 1) Gamma(b)) = Gamma(a + 1 + b) / (Gamma(a + 1) Gamma(b)) / (a + b).
        t = a * numext::log(x) + b * numext::log1p(-x) + betainc_stirling_ratio(a + 1.0f, b) - numext::log(a + b);
      } else {
        t = a * numext::log(x) + b * numext::log1p(-x) + lgamma_impl<float>::run(a + b) -
            lgamma_impl<float>::run(a + 1.0f) - lgamma_impl<float>::run(b);
      }
      return (ans + numext::exp(t));
    } else {
      return betainc_helper<float>::incbsa(a, b, x);
    }
  }
};

template <>
struct betainc_helper<double> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE double incbps(double a, double b, double x) {
    const double machep = cephes_helper<double>::machep();

    double s, t, u, v, n, t1, z, ai;

    ai = 1.0 / a;
    u = (1.0 - b) * x;
    v = u / (a + 1.0);
    t1 = v;
    t = u;
    n = 2.0;
    s = 0.0;
    z = machep * ai;
    while (numext::abs(v) > z) {
      u = (n - b) * x / n;
      t *= u;
      v = t / (a + n);
      s += v;
      n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * numext::log(x);
    // TODO: gamma() is not directly implemented in Eigen.
    /*
    if ((a + b) < maxgam && numext::abs(u) < maxlog) {
      t = gamma(a + b) / (gamma(a) * gamma(b));
      s = s * t * pow(x, a);
    }
    */
    if (1.0 - x == 1.0 && betainc_use_stirling_ratio(a, b)) {
      t = betainc_stirling_ratio(a, b) + u + numext::log(s);
    } else {
      t = lgamma_impl<double>::run(a + b) - lgamma_impl<double>::run(a) - lgamma_impl<double>::run(b) + u +
          numext::log(s);
    }
    return numext::exp(t);
  }
};

template <>
struct betainc_impl<double> {
  EIGEN_DEVICE_FUNC static double run(double aa, double bb, double xx) {
    const double nan = NumTraits<double>::quiet_NaN();
    const double machep = cephes_helper<double>::machep();
    double a, b, t, x, xc, w, y;
    bool reversed_a_b = false;

    if ((numext::isnan)(aa) || (numext::isnan)(bb) || (numext::isnan)(xx)) return nan;
    if (aa == 0.0 && bb == 0.0) return nan;
    if (xx < 0.0 || xx > 1.0) return nan;
    if (aa < 0.0) return nan;
    if (bb < 0.0) return nan;
    if (aa == 0.0) return 1.0;
    if (bb == 0.0) return 0.0;
    if (xx == 0.0) return 0.0;
    if (xx == 1.0) return 1.0;
    // mtherr("incbet", DOMAIN);

    if ((bb * xx) <= 1.0 && xx <= 0.95) {
      return betainc_helper<double>::incbps(aa, bb, xx);
    }

    w = 1.0 - xx;

    /* Reverse a and b if x is greater than the mean. If 1 - xx rounds to 1, x = w = 1 below and its complement
     * x / (1 - x) is infinite: use the exact xc = xx instead. */
    const bool w_is_one = (w == 1.0);
    if (xx > (aa / (aa + bb))) {
      reversed_a_b = true;
      a = bb;
      b = aa;
      xc = xx;
      x = w;
    } else {
      a = aa;
      b = bb;
      xc = w;
      x = xx;
    }

    if (reversed_a_b && (b * x) <= 1.0 && x <= 0.95) {
      t = betainc_helper<double>::incbps(a, b, x);
      if (t <= machep) {
        t = 1.0 - machep;
      } else {
        t = 1.0 - t;
      }
      return t;
    }

    /* Choose expansion for better convergence. */
    y = x * (a + b - 2.0) - (a - 1.0);
    if (y < 0.0) {
      w = incbeta_cfe<double>::run(a, b, x, reversed_a_b && w_is_one ? xc : 1.0 - x, true /* small_branch */);
    } else {
      w = incbeta_cfe<double>::run(a, b, x, reversed_a_b && w_is_one ? xc : 1.0 - x, false /* small_branch */) / xc;
    }

    /* Multiply w by the factor
         a      b   _             _     _
        x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

    y = a * ((reversed_a_b && w_is_one) ? numext::log1p(-xx) : numext::log(x));
    t = b * ((xc < 1.0) ? numext::log(xc) : numext::log1p(-xx));
    // TODO: gamma is not directly implemented in Eigen.
    /*
    if ((a + b) < maxgam && numext::abs(y) < maxlog && numext::abs(t) < maxlog)
    {
      t = pow(xc, b);
      t *= pow(x, a);
      t /= a;
      t *= w;
      t *= gamma(a + b) / (gamma(a) * gamma(b));
    } else {
    */
    /* Resort to logarithms.  */
    if (w_is_one && betainc_use_stirling_ratio(a, b)) {
      y += t + betainc_stirling_ratio(a, b);
    } else {
      y += t + lgamma_impl<double>::run(a + b) - lgamma_impl<double>::run(a) - lgamma_impl<double>::run(b);
    }
    y += numext::log(w / a);
    t = numext::exp(y);

    /* } */
    // done:

    if (reversed_a_b) {
      if (t <= machep) {
        t = 1.0 - machep;
      } else {
        t = 1.0 - t;
      }
    }
    return t;
  }
};

}  // end namespace internal

namespace numext {

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto lgamma(const Scalar& x) -> decltype(EIGEN_MATHFUNC_IMPL(lgamma, Scalar)::run(x)) {
  return EIGEN_MATHFUNC_IMPL(lgamma, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto digamma(const Scalar& x) -> decltype(EIGEN_MATHFUNC_IMPL(digamma, Scalar)::run(x)) {
  return EIGEN_MATHFUNC_IMPL(digamma, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto zeta(const Scalar& x, const Scalar& q)
    -> decltype(EIGEN_MATHFUNC_IMPL(zeta, Scalar)::run(x, q)) {
  return EIGEN_MATHFUNC_IMPL(zeta, Scalar)::run(x, q);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto polygamma(const Scalar& n, const Scalar& x)
    -> decltype(EIGEN_MATHFUNC_IMPL(polygamma, Scalar)::run(n, x)) {
  return EIGEN_MATHFUNC_IMPL(polygamma, Scalar)::run(n, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto erf(const Scalar& x) -> decltype(EIGEN_MATHFUNC_IMPL(erf, Scalar)::run(x)) {
  return EIGEN_MATHFUNC_IMPL(erf, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto erfc(const Scalar& x) -> decltype(EIGEN_MATHFUNC_IMPL(erfc, Scalar)::run(x)) {
  return EIGEN_MATHFUNC_IMPL(erfc, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto ndtri(const Scalar& x) -> decltype(EIGEN_MATHFUNC_IMPL(ndtri, Scalar)::run(x)) {
  return EIGEN_MATHFUNC_IMPL(ndtri, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto igamma(const Scalar& a, const Scalar& x)
    -> decltype(EIGEN_MATHFUNC_IMPL(igamma, Scalar)::run(a, x)) {
  return EIGEN_MATHFUNC_IMPL(igamma, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto igamma_der_a(const Scalar& a, const Scalar& x)
    -> decltype(EIGEN_MATHFUNC_IMPL(igamma_der_a, Scalar)::run(a, x)) {
  return EIGEN_MATHFUNC_IMPL(igamma_der_a, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto gamma_sample_der_alpha(const Scalar& a, const Scalar& x)
    -> decltype(EIGEN_MATHFUNC_IMPL(gamma_sample_der_alpha, Scalar)::run(a, x)) {
  return EIGEN_MATHFUNC_IMPL(gamma_sample_der_alpha, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto igammac(const Scalar& a, const Scalar& x)
    -> decltype(EIGEN_MATHFUNC_IMPL(igammac, Scalar)::run(a, x)) {
  return EIGEN_MATHFUNC_IMPL(igammac, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline auto betainc(const Scalar& a, const Scalar& b, const Scalar& x)
    -> decltype(EIGEN_MATHFUNC_IMPL(betainc, Scalar)::run(a, b, x)) {
  return EIGEN_MATHFUNC_IMPL(betainc, Scalar)::run(a, b, x);
}

}  // end namespace numext
}  // end namespace Eigen

#endif  // EIGEN_SPECIAL_FUNCTIONS_H
