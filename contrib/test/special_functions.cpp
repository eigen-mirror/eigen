// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include <limits.h>
#include "main.h"
#include "fp_control.h"
#include "../Eigen/SpecialFunctions"

// Workaround to allow "implicit" conversions from double to Scalar via comma-initialization.
template <typename Derived>
Eigen::CommaInitializer<Derived> operator<<(Eigen::DenseBase<Derived>& dense, double v) {
  return (dense << static_cast<typename Derived::Scalar>(v));
}

template <typename XprType>
Eigen::CommaInitializer<XprType>& operator,(Eigen::CommaInitializer<XprType>& ci, double v) {
  return (ci, static_cast<typename XprType::Scalar>(v));
}

template <typename X, typename Y>
void verify_component_wise(const X& x, const Y& y) {
  for (Index i = 0; i < x.size(); ++i) {
    if ((numext::isfinite)(y(i)))
      VERIFY_IS_APPROX(x(i), y(i));
    else if ((numext::isnan)(y(i)))
      VERIFY((numext::isnan)(x(i)));
    else
      VERIFY_IS_EQUAL(x(i), y(i));
  }
}

template <typename ArrayType>
void array_special_functions() {
  using std::abs;
  using std::sqrt;
  using Scalar = typename ArrayType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Scalar plusinf = std::numeric_limits<Scalar>::infinity();
  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();

  Index rows = internal::random<Index>(1, 30);
  Index cols = 1;

  // API
  {
    ArrayType m1 = ArrayType::Random(rows, cols);
    VERIFY_IS_APPROX(m1.lgamma(), lgamma(m1));
    VERIFY_IS_APPROX(m1.digamma(), digamma(m1));
    VERIFY_IS_APPROX(m1.erf(), erf(m1));
    VERIFY_IS_APPROX(m1.erfc(), erfc(m1));
  }

  // check special functions (comparing against numpy implementation)
  if (!NumTraits<Scalar>::IsComplex) {
    {
      ArrayType m1 = ArrayType::Random(rows, cols);
      ArrayType m2 = ArrayType::Random(rows, cols);

      // Test various properties of igamma & igammac.  These are normalized
      // gamma integrals where
      //   igammac(a, x) = Gamma(a, x) / Gamma(a)
      //   igamma(a, x) = gamma(a, x) / Gamma(a)
      // where Gamma and gamma are considered the standard unnormalized
      // upper and lower incomplete gamma functions, respectively.
      ArrayType a = m1.abs() + Scalar(2);
      ArrayType x = m2.abs() + Scalar(2);
      ArrayType zero = ArrayType::Zero(rows, cols);
      ArrayType one = ArrayType::Constant(rows, cols, Scalar(1.0));
      ArrayType a_m1 = a - one;
      ArrayType Gamma_a_x = Eigen::igammac(a, x) * a.lgamma().exp();
      ArrayType Gamma_a_m1_x = Eigen::igammac(a_m1, x) * a_m1.lgamma().exp();
      ArrayType gamma_a_x = Eigen::igamma(a, x) * a.lgamma().exp();
      ArrayType gamma_a_m1_x = Eigen::igamma(a_m1, x) * a_m1.lgamma().exp();

      // Gamma(a, 0) == Gamma(a)
      VERIFY_IS_APPROX(Eigen::igammac(a, zero), one);

      // Gamma(a, x) + gamma(a, x) == Gamma(a)
      VERIFY_IS_APPROX(Gamma_a_x + gamma_a_x, a.lgamma().exp());

      // Gamma(a, x) == (a - 1) * Gamma(a-1, x) + x^(a-1) * exp(-x)
      VERIFY_IS_APPROX(Gamma_a_x, (a - Scalar(1)) * Gamma_a_m1_x + x.pow(a - Scalar(1)) * (-x).exp());

      // gamma(a, x) == (a - 1) * gamma(a-1, x) - x^(a-1) * exp(-x)
      VERIFY_IS_APPROX(gamma_a_x, (a - Scalar(1)) * gamma_a_m1_x - x.pow(a - Scalar(1)) * (-x).exp());
    }
    {
      // Verify for large a and x that values are between 0 and 1.
      ArrayType m1 = ArrayType::Random(rows, cols);
      ArrayType m2 = ArrayType::Random(rows, cols);
      int max_exponent = std::numeric_limits<Scalar>::max_exponent10;
      ArrayType a = m1.abs() * Scalar(pow(10., max_exponent - 1));
      ArrayType x = m2.abs() * Scalar(pow(10., max_exponent - 1));
      for (int i = 0; i < a.size(); ++i) {
        Scalar igam = numext::igamma(a(i), x(i));
        VERIFY(0 <= igam);
        VERIFY(igam <= 1);
      }
    }

    {
      // Check exact values of igamma and igammac against a third party calculation.
      Scalar a_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};
      Scalar x_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};

      // location i*6+j corresponds to a_s[i], x_s[j].
      Scalar igamma_s[][6] = {
          {Scalar(0.0), nan, nan, nan, nan, nan},
          {Scalar(0.0), Scalar(0.6321205588285578), Scalar(0.7768698398515702), Scalar(0.9816843611112658),
           Scalar(9.999500016666262e-05), Scalar(1.0)},
          {Scalar(0.0), Scalar(0.4275932955291202), Scalar(0.608374823728911), Scalar(0.9539882943107686),
           Scalar(7.522076445089201e-07), Scalar(1.0)},
          {Scalar(0.0), Scalar(0.01898815687615381), Scalar(0.06564245437845008), Scalar(0.5665298796332909),
           Scalar(4.166333347221828e-18), Scalar(1.0)},
          {Scalar(0.0), Scalar(0.9999780593618628), Scalar(0.9999899967080838), Scalar(0.9999996219837988),
           Scalar(0.9991370418689945), Scalar(1.0)},
          {Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.5042041932513908)}};
      Scalar igammac_s[][6] = {
          {nan, nan, nan, nan, nan, nan},
          {Scalar(1.0), Scalar(0.36787944117144233), Scalar(0.22313016014842982), Scalar(0.018315638888734182),
           Scalar(0.9999000049998333), Scalar(0.0)},
          {Scalar(1.0), Scalar(0.5724067044708798), Scalar(0.3916251762710878), Scalar(0.04601170568923136),
           Scalar(0.9999992477923555), Scalar(0.0)},
          {Scalar(1.0), Scalar(0.9810118431238462), Scalar(0.9343575456215499), Scalar(0.4334701203667089), Scalar(1.0),
           Scalar(0.0)},
          {Scalar(1.0), Scalar(2.1940638138146658e-05), Scalar(1.0003291916285e-05), Scalar(3.7801620118431334e-07),
           Scalar(0.0008629581310054535), Scalar(0.0)},
          {Scalar(1.0), Scalar(1.0), Scalar(1.0), Scalar(1.0), Scalar(1.0), Scalar(0.49579580674813944)}};

      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          if ((std::isnan)(igamma_s[i][j])) {
            VERIFY((std::isnan)(numext::igamma(a_s[i], x_s[j])));
          } else {
            VERIFY_IS_APPROX(numext::igamma(a_s[i], x_s[j]), igamma_s[i][j]);
          }

          if ((std::isnan)(igammac_s[i][j])) {
            VERIFY((std::isnan)(numext::igammac(a_s[i], x_s[j])));
          } else {
            VERIFY_IS_APPROX(numext::igammac(a_s[i], x_s[j]), igammac_s[i][j]);
          }
        }
      }
    }
  }

  // Check the ndtri function against scipy.special.ndtri
  {
    ArrayType x(11), res(11), ref(11);
    x << 0.5, 0.2, 0.8, 0.9, 0.1, 0.99, 0.01, 0, 1, -0.01, 1.01;
    ref << 0., -0.8416212335729142, 0.8416212335729142, 1.2815515655446004, -1.2815515655446004, 2.3263478740408408,
        -2.3263478740408408, -plusinf, plusinf, nan, nan;
    CALL_SUBTEST(verify_component_wise(ref, ref););
    CALL_SUBTEST(res = x.ndtri(); verify_component_wise(res, ref););
    CALL_SUBTEST(res = ndtri(x); verify_component_wise(res, ref););

    // ndtri(normal_cdf(x)) ~= x. A rounding error of eps * p in p = normal_cdf(x) moves ndtri(p) by eps * p / pdf(x),
    // at most 3.5 eps on [-1, 1], so the error is bounded absolutely rather than relative to x, which may be near 0.
    // The rest of the 16 eps covers the rounding of erf and ndtri themselves.
    CALL_SUBTEST(ArrayType m1 = ArrayType::Random(32); using std::sqrt;

                 ArrayType cdf_val = (m1 / Scalar(sqrt(2.))).erf(); cdf_val = (cdf_val + Scalar(1)) / Scalar(2);
                 VERIFY(max_abs_coeff(cdf_val.ndtri() - m1) <= RealScalar(16) * NumTraits<RealScalar>::epsilon()););
  }

  // Check the zeta function against scipy.special.zeta
  {
    ArrayType x(11), q(11), res(11), ref(11);
    x << 1.5, 4, 10.5, 10000.5, 3, 1, 0.9, 2, 3, 4, 2000;
    q << 2, 1.5, 3, 1.0001, -2.5, 1.2345, 1.2345, -1, -2, -3, 2000;
    ref << 1.61237534869, 0.234848505667, 1.03086757337e-5, 0.367879440865, 0.054102025820864097, plusinf, nan, plusinf,
        nan, plusinf, 0;
    CALL_SUBTEST(verify_component_wise(ref, ref););
    CALL_SUBTEST(res = x.zeta(q); verify_component_wise(res, ref););
    CALL_SUBTEST(res = zeta(x, q); verify_component_wise(res, ref););
  }

  // digamma
  {
    ArrayType x(10), res(10), ref(10);
    x << 1, 1.5, 4, -10.5, 10000.5, 0.0, -0.0, -1, -2, -3;
    ref << -0.5772156649015329, 0.03648997397857645, 1.2561176684318, 2.398239129535781, 9.210340372392849, -plusinf,
        plusinf, nan, nan, nan;
    CALL_SUBTEST(verify_component_wise(ref, ref););

    CALL_SUBTEST(res = x.digamma(); verify_component_wise(res, ref););
    CALL_SUBTEST(res = digamma(x); verify_component_wise(res, ref););
  }

  {
    ArrayType n(16), x(16), res(16), ref(16);
    n << 1, 1, 1, 1.5, 17, 31, 28, 8, 42, 147, 170, -1, 0, 1, 2, 3;
    x << 2, 3, 25.5, 1.5, 4.7, 11.8, 17.7, 30.2, 15.8, 54.1, 64, -1, -2, -3, -4, -5;
    ref << 0.644934066848, 0.394934066848, 0.0399946696496, nan, 293.334565435, 0.445487887616, -2.47810300902e-07,
        -8.29668781082e-09, -0.434562276666, 0.567742190178, -0.0108615497927, nan, nan, plusinf, nan, plusinf;
    CALL_SUBTEST(verify_component_wise(ref, ref););

    if (sizeof(RealScalar) >= 8) {  // double
      // Reason for commented line: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1232
      //       CALL_SUBTEST( res = x.polygamma(n); verify_component_wise(res, ref); );
      CALL_SUBTEST(res = polygamma(n, x); verify_component_wise(res, ref););
    } else {
      //       CALL_SUBTEST( res = x.polygamma(n); verify_component_wise(res.head(8), ref.head(8)); );
      CALL_SUBTEST(res = polygamma(n, x); verify_component_wise(res.head(8), ref.head(8)););
    }
  }

  {
    // Inputs and ground truth generated with scipy via:
    //   a = np.logspace(-3, 3, 5) - 1e-3
    //   b = np.logspace(-3, 3, 5) - 1e-3
    //   x = np.linspace(-0.1, 1.1, 5)
    //   (full_a, full_b, full_x) = np.vectorize(lambda a, b, x: (a, b, x))(*np.ix_(a, b, x))
    //   full_a = full_a.flatten().tolist()
    //   full_b = full_b.flatten().tolist()
    //   full_x = full_x.flatten().tolist()
    //   v = scipy.special.betainc(full_a, full_b, full_x).flatten().tolist()
    // Note in Eigen, we call betainc with arguments in the order (a, b, x);
    ArrayType a(125);
    ArrayType b(125);
    ArrayType x(125);
    ArrayType v(125);
    ArrayType res(125);

    a << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999,
        0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999, 999.999, 999.999, 999.999,
        999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999,
        999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999;

    b << 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999, 31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 999.999, 999.999, 999.999, 999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999,
        0.999, 0.999, 0.999, 0.999, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 999.999, 999.999, 999.999, 999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999, 0.999, 0.999, 0.999,
        0.999, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999,
        999.999, 999.999, 999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999, 999.999,
        999.999, 999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999, 999.999,
        999.999, 999.999, 999.999;

    x << -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1,
        0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2,
        0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5,
        0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8,
        1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1,
        -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1;

    v << nan, nan, nan, nan, nan, nan, 1.0, 1.0, 1.0, nan, nan, 1.0, 1.0, 1.0, nan, nan, 1.0, 1.0, 1.0, nan, nan, 1.0,
        1.0, 1.0, nan, nan, 0.0, 0.0, 0.0, nan, nan, 0.47972119876364683, 0.4999999999999997, 0.5202788012363533, nan,
        nan, 0.9518683957740043, 0.9789663010413745, 0.993172918807344, nan, nan, 0.999995949033062, 0.99999999999937,
        1.0, nan, nan, 1.0, 1.0, 1.0, nan, nan, 0.0, 0.0, 0.0, nan, nan, 0.0068270811926558735, 0.0210336989586256,
        0.048131604225995696, nan, nan, 0.20014344256217687, 0.5000000000000002, 0.7998565574378237, nan, nan,
        0.9991401428435834, 0.9999999996984031, 1.0, nan, nan, 1.0, 1.0, 1.0, nan, nan, 0.0, 0.0, 0.0, nan, nan,
        1.0646600232370862e-25, 6.30172287782625e-13, 4.050966937974916e-06, nan, nan, 7.864342668429712e-23,
        3.015969667594142e-10, 0.0008598571564165379, nan, nan, 6.031987710123845e-08, 0.5000000000000001,
        0.9999999396801229, nan, nan, 1.0, 1.0, 1.0, nan, nan, 0.0, 0.0, 0.0, nan, nan, 0.0, 7.029920380993552e-306,
        2.245072820861537e-101, nan, nan, 0.0, 9.275871147875753e-302, 1.223291302616039e-97, nan, nan, 0.0,
        3.089139308197642e-252, 2.9303043666230076e-60, nan, nan, 2.248913486881819e-196, 0.5000000000000036, 1.0, nan;

    CALL_SUBTEST(res = betainc(a, b, x); verify_component_wise(res, v););
  }

  // Test various properties of betainc
  {
    ArrayType m1 = ArrayType::Random(32);
    ArrayType m2 = ArrayType::Random(32);
    ArrayType m3 = ArrayType::Random(32);
    ArrayType one = ArrayType::Constant(32, Scalar(1.0));
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    ArrayType a = (m1 * Scalar(4)).exp();
    ArrayType b = (m2 * Scalar(4)).exp();
    ArrayType x = m3.abs();

    // ARMv7 NEON flushes subnormal float intermediates, unlike the scalar betainc implementation. Form the
    // reference in at least double precision so x**a and the recurrence term retain subnormal float results.
    using ReferenceScalar = std::conditional_t<(sizeof(Scalar) < sizeof(double)), double, Scalar>;
    using ReferenceArray = Array<ReferenceScalar, Dynamic, 1>;
    const ReferenceArray& ar = a.template cast<ReferenceScalar>();
    const ReferenceArray& br = b.template cast<ReferenceScalar>();
    const ReferenceArray& xr = x.template cast<ReferenceScalar>();
    const Scalar tiny = (std::numeric_limits<Scalar>::min)();

    // betainc(a, 1, x) == x**a. The absolute floor covers rounding of subnormal results.
    CALL_SUBTEST(ArrayType test = betainc(a, one, x); ArrayType expected = xr.pow(ar).template cast<Scalar>();
                 for (Index i = 0; i < test.size(); ++i)
                     VERIFY_IS_APPROX_SCALED(test.segment(i, 1), expected.segment(i, 1), expected(i) + tiny););

    // betainc(1, b, x) == 1 - (1 - x)**b, evaluated as -expm1(b * log1p(-x)): for small b * x the direct form cancels
    // to a relative error near eps / (b * x).
    CALL_SUBTEST(ArrayType test = betainc(one, b, x); ArrayType expected = -(b * (-x).log1p()).expm1();
                 verify_component_wise(test, expected););

    // betainc(a, b, x) == 1 - betainc(b, a, 1-x)
    CALL_SUBTEST(ArrayType test = betainc(a, b, x) + betainc(b, a, one - x); ArrayType expected = one;
                 verify_component_wise(test, expected););

    // betainc(a+1, b, x) = betainc(a, b, x) - x**a * (1 - x)**b / (a * beta(a, b)). Bound cancellation relative
    // to the larger term. Log space avoids underflow of the numerator before division by a small beta(a, b).
    CALL_SUBTEST(ReferenceArray log_beta = ar.lgamma() + br.lgamma() - (ar + br).lgamma();
                 ReferenceArray term = (ar * xr.log() + br * (-xr).log1p() - ar.log() - log_beta).exp();
                 ArrayType lhs = betainc(a, b, x); ArrayType scale = (lhs.max)(term.template cast<Scalar>()) + tiny;
                 ArrayType expected = (lhs.template cast<ReferenceScalar>() - term).template cast<Scalar>();
                 ArrayType test = betainc(a + one, b, x); for (Index i = 0; i < test.size(); ++i)
                     VERIFY_IS_APPROX_SCALED(test.segment(i, 1), expected.segment(i, 1), scale(i)););

    // betainc(a, b+1, x) = betainc(a, b, x) + x**a * (1 - x)**b / (b * beta(a, b))
    CALL_SUBTEST(
        // Add eps to rhs and lhs so that component-wise test doesn't result in
        // nans when both outputs are zeros.
        ArrayType num = x.pow(a) * (one - x).pow(b);
        ArrayType denom = b * (a.lgamma() + b.lgamma() - (a + b).lgamma()).exp();
        ArrayType expected = betainc(a, b, x) + num / denom + eps; ArrayType test = betainc(a, b + one, x) + eps;
        verify_component_wise(test, expected););
  }

  /* Code to generate the data for the following two test cases.
  N = 5
  np.random.seed(3)

  a = np.logspace(-2, 3, 6)
  a = np.ravel(np.tile(np.reshape(a, [-1, 1]), [1, N]))
  x = np.random.gamma(a, 1.0)
  x = np.maximum(x, np.finfo(np.float32).tiny)

  def igamma(a, x):
    return mpmath.gammainc(a, 0, x, regularized=True)

  def igamma_der_a(a, x):
    res = mpmath.diff(lambda a_prime: igamma(a_prime, x), a)
    return np.float64(res)

  def gamma_sample_der_alpha(a, x):
    igamma_x = igamma(a, x)
    def igammainv_of_igamma(a_prime):
      return mpmath.findroot(lambda x_prime: igamma(a_prime, x_prime) -
          igamma_x, x, solver='newton')
    return np.float64(mpmath.diff(igammainv_of_igamma, a))

  v_igamma_der_a = np.vectorize(igamma_der_a)(a, x)
  v_gamma_sample_der_alpha = np.vectorize(gamma_sample_der_alpha)(a, x)
*/

  // Test igamma_der_a
  {
    ArrayType a(30);
    ArrayType x(30);
    ArrayType res(30);
    ArrayType v(30);

    a << 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0;

    x << 1.25668890405e-26, 1.17549435082e-38, 1.20938905072e-05, 1.17549435082e-38, 1.17549435082e-38,
        5.66572070696e-16, 0.0132865061065, 0.0200034203853, 6.29263709118e-17, 1.37160367764e-06, 0.333412038288,
        1.18135687766, 0.580629033777, 0.170631439426, 0.786686768458, 7.63873279537, 13.1944344379, 11.896042354,
        10.5830172417, 10.5020942233, 92.8918587747, 95.003720371, 86.3715926467, 96.0330217672, 82.6389930677,
        968.702906754, 969.463546828, 1001.79726022, 955.047416547, 1044.27458568;

    v << -32.7256441441, -36.4394150514, -9.66467612263, -36.4394150514, -36.4394150514, -1.0891900302, -2.66351229645,
        -2.48666868596, -0.929700494428, -3.56327722764, -0.455320135314, -0.391437214323, -0.491352055991,
        -0.350454834292, -0.471773162921, -0.104084440522, -0.0723646747909, -0.0992828975532, -0.121638215446,
        -0.122619605294, -0.0317670267286, -0.0359974812869, -0.0154359225363, -0.0375775365921, -0.00794899153653,
        -0.00777303219211, -0.00796085782042, -0.0125850719397, -0.00455500206958, -0.00476436993148;

    CALL_SUBTEST(res = igamma_der_a(a, x); verify_component_wise(res, v););
  }

  // Test gamma_sample_der_alpha
  {
    ArrayType alpha(30);
    ArrayType sample(30);
    ArrayType res(30);
    ArrayType v(30);

    alpha << 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0;

    sample << 1.25668890405e-26, 1.17549435082e-38, 1.20938905072e-05, 1.17549435082e-38, 1.17549435082e-38,
        5.66572070696e-16, 0.0132865061065, 0.0200034203853, 6.29263709118e-17, 1.37160367764e-06, 0.333412038288,
        1.18135687766, 0.580629033777, 0.170631439426, 0.786686768458, 7.63873279537, 13.1944344379, 11.896042354,
        10.5830172417, 10.5020942233, 92.8918587747, 95.003720371, 86.3715926467, 96.0330217672, 82.6389930677,
        968.702906754, 969.463546828, 1001.79726022, 955.047416547, 1044.27458568;

    v << 7.42424742367e-23, 1.02004297287e-34, 0.0130155240738, 1.02004297287e-34, 1.02004297287e-34, 1.96505168277e-13,
        0.525575786243, 0.713903991771, 2.32077561808e-14, 0.000179348049886, 0.635500453302, 1.27561284917,
        0.878125852156, 0.41565819538, 1.03606488534, 0.885964824887, 1.16424049334, 1.10764479598, 1.04590810812,
        1.04193666963, 0.965193152414, 0.976217589464, 0.93008035061, 0.98153216096, 0.909196397698, 0.98434963993,
        0.984738050206, 1.00106492525, 0.97734200649, 1.02198794179;

    CALL_SUBTEST(res = gamma_sample_der_alpha(alpha, sample); verify_component_wise(res, v););
  }
}

template <typename Scalar>
void ndtri_tail_accuracy() {
  const int exponents[] = {4, 20, 46, 52, 120, 1000};
  // sqrt(2) * erfinv(2 * 2^-exponent - 1), evaluated with mpmath 1.3.0 at 340 decimal digits.
  const long double reference[] = {-1.5341205443525463117084L,  -4.76300103426781395698855L,
                                   -7.60531213194874895463731L, -8.12589066470190685850799L,
                                   -12.6266965604913946489664L, -37.1110119371647914101147L};
  const Index count = 2 * internal::packet_traits<Scalar>::size + 7;
  Array<Scalar, Dynamic, 1> input(count), result;
  for (Index i = 0; i < input.size(); ++i) input(i) = std::ldexp(Scalar(1), -exponents[i % 6]);
  result = input.ndtri();
  for (Index i = 0; i < input.size(); ++i) {
    if (input(i) == Scalar(0)) {
      VERIFY_IS_EQUAL(result(i), -NumTraits<Scalar>::infinity());
    } else {
      const Scalar expected = Scalar(reference[i % 6]);
      const Scalar bound = Scalar(4) * NumTraits<Scalar>::epsilon() * numext::abs(expected);
      VERIFY(numext::abs(result(i) - expected) <= bound);
      VERIFY(numext::abs(numext::ndtri(input(i)) - expected) <= bound);
    }
  }
}

template <typename Scalar>
void scalar_ndtri() {
  const Scalar sign_mask = internal::psignmask<Scalar>();
  VERIFY_IS_EQUAL(sign_mask, Scalar(0));
  VERIFY((std::signbit)(sign_mask));

  const Scalar packet_inf = internal::pinf<Scalar>();
  const Scalar packet_nan = internal::pnan<Scalar>();
  VERIFY((numext::isinf)(packet_inf));
  VERIFY(packet_inf > Scalar(0));
  VERIFY((numext::isnan)(packet_nan));

  const Scalar expected_quartile = Scalar(0.6744897501960817432L);
  const Scalar lower_quartile = numext::ndtri(Scalar(0.25L));
  const Scalar upper_quartile = numext::ndtri(Scalar(0.75L));
  VERIFY_IS_APPROX(lower_quartile, -expected_quartile);
  VERIFY_IS_APPROX(upper_quartile, expected_quartile);

  // These exactly complementary binary values exercise the scalar flipsign path in the tail approximation.
  const Scalar lower_tail = numext::ndtri(Scalar(0.125L));
  const Scalar upper_tail = numext::ndtri(Scalar(0.875L));
  VERIFY(lower_tail < Scalar(0));
  VERIFY(upper_tail > Scalar(0));
  VERIFY_IS_APPROX(lower_tail, -upper_tail);

  VERIFY_IS_EQUAL(numext::ndtri(Scalar(0.5L)), Scalar(0));
  const Scalar negative_inf = numext::ndtri(Scalar(0));
  const Scalar positive_inf = numext::ndtri(Scalar(1));
  VERIFY((numext::isinf)(negative_inf));
  VERIFY((numext::isinf)(positive_inf));
  VERIFY(negative_inf < Scalar(0));
  VERIFY(positive_inf > Scalar(0));
}

// A scalar type without a vectorized erf/erfc implementation must defer to
// std::erf/std::erfc instead of the float/double-tuned polynomials.
template <typename Scalar>
void scalar_erf_erfc() {
  EIGEN_USING_STD(erf);
  EIGEN_USING_STD(erfc);
  // Approximate comparison: the compiler may constant-fold one side to the
  // correctly rounded value while the other runs the libm implementation.
  for (Scalar x : {Scalar(-2.5L), Scalar(-0.5L), Scalar(0.0L), Scalar(0.25L), Scalar(3.0L)}) {
    VERIFY_IS_APPROX(numext::erf(x), Scalar(erf(x)));
    VERIFY_IS_APPROX(numext::erfc(x), Scalar(erfc(x)));
  }
}

namespace custom_scalar {
// Minimal custom real scalar providing erf/erfc via argument-dependent lookup,
// the way multiprecision types do. numext::erf/erfc must route to these instead
// of the float/double-tuned polynomials (issue #3023).
struct CustomReal {
  double value;
  CustomReal() : value(0) {}
  explicit CustomReal(double v) : value(v) {}
};
inline CustomReal erf(const CustomReal& x) { return CustomReal(std::erf(x.value)); }
inline CustomReal erfc(const CustomReal& x) { return CustomReal(std::erfc(x.value)); }

// The shape of boost::numeric::interval: neither erf nor erfc. Eigen's
// approximations are not valid for such a type, so it must not be routed into
// them; see failtest/erf_no_scalar_overload.cpp for the resulting error.
struct NoErfReal {
  double value;
};
}  // namespace custom_scalar

namespace Eigen {
template <>
struct NumTraits<custom_scalar::CustomReal> : NumTraits<double> {
  using Real = custom_scalar::CustomReal;
  using NonInteger = custom_scalar::CustomReal;
  using Nested = custom_scalar::CustomReal;
};
}  // namespace Eigen

void custom_scalar_erf_erfc() {
  using custom_scalar::CustomReal;
  using custom_scalar::NoErfReal;
  // Pin the routing decision in both directions.
  STATIC_CHECK((internal::has_erf<CustomReal>::value));
  STATIC_CHECK((internal::has_erfc<CustomReal>::value));
  STATIC_CHECK((internal::has_erf<long double>::value));
  STATIC_CHECK((internal::has_erfc<long double>::value));
  STATIC_CHECK((!internal::has_erf<NoErfReal>::value));
  STATIC_CHECK((!internal::has_erfc<NoErfReal>::value));

  Eigen::Array<CustomReal, 4, 1> x;
  x(0) = CustomReal(-2.5);
  x(1) = CustomReal(-0.5);
  x(2) = CustomReal(0.25);
  x(3) = CustomReal(3.0);
  Eigen::Array<CustomReal, 4, 1> e = x.erf();
  Eigen::Array<CustomReal, 4, 1> c = x.erfc();
  // The polynomial paths cannot even be instantiated for CustomReal, so
  // compiling proves the routing; the value check guards the plumbing.
  for (int i = 0; i < 4; ++i) {
    VERIFY_IS_APPROX(e(i).value, std::erf(x(i).value));
    VERIFY_IS_APPROX(c(i).value, std::erfc(x(i).value));
  }
}

// betainc for b so large that 1 - x rounds to 1 in Scalar (x < eps / 2): these used to loop forever or return NaN.
// References: long double continued fraction (DLMF 8.17.22) with the exact complement of x; a = 1 and a = 1/2 also
// follow from I_x(1, b) = 1 - (1 - x)^b and I_x(1/2, b) -> erf(sqrt(b x)).
template <typename Scalar>
void test_betainc_large_b() {
  struct Case {
    double a, b, x, ref;
  };
  std::vector<Case> cases;
  if (sizeof(Scalar) == 4) {
    cases = {{20.5, 1e10, 3e-9, 0.97205905127192506},
             {20, 1e10, 3e-9, 0.97812653455614376},
             {2.5, 1e9, 1e-8, 0.99875026943340965},
             {30.5, 1e12, 5e-11, 0.99878628135546261},
             {2.5, 1e9, 2e-9, 0.45058403336738364},
             {1, 1e9, 1e-9, 0.63212054860815611},
             {0.5, 1e8, 2e-8, 0.95449973625714398},
             {11141.5, 7.8e32, 1.26e-20, 1},
             {3.2e10, 1.05e28, 5.4e-17, 1}};
  } else {
    cases = {{20.5, 1e20, 3e-18, 1}, {20, 1e20, 3e-18, 1}, {2.5, 1e17, 2e-17, 0.45058404864721982}};
  }
  // The exponent a log x + ... + (b - 1) log1p(-x) + log(Gamma(a + b) / (Gamma(a) Gamma(b))) sums terms of size
  // a |log x| (and m log M with m = a) that nearly cancel, so the relative error is eps (c0 + c a |log x|), with
  // c0 = 8 (the cases with value 1 measure 0) and c = 1.5. The measured error is at most 0.12 of this bound (8.9 eps
  // vs 83 for float (2.5, 1e9, 2e-9) with clang++ -march=x86-64-v3, 14.4 vs 152 for double (2.5, 1e17, 2e-17); g++ and
  // -ffp-contract=off measure the same or less), i.e. the bound is at least 8 times the error.
  const double eps = static_cast<double>(NumTraits<Scalar>::epsilon());
  Array<Scalar, Dynamic, 1> a(cases.size()), b(cases.size()), x(cases.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    a(i) = Scalar(cases[i].a);
    b(i) = Scalar(cases[i].b);
    x(i) = Scalar(cases[i].x);
    const Scalar v = numext::betainc(a(i), b(i), x(i));
    VERIFY(!((numext::isnan)(v)));
    const Scalar ref = Scalar(cases[i].ref);
    const double tol = eps * (8 + (cases[i].ref == 1 ? 0.0 : 1.5 * cases[i].a * std::abs(std::log(double(x(i))))));
    VERIFY(numext::abs(v - ref) <= Scalar(tol) * ref);
  }
  Array<Scalar, Dynamic, 1> res = betainc(a, b, x);
  for (size_t i = 0; i < cases.size(); ++i) {
    const Scalar ref = Scalar(cases[i].ref);
    const double tol = eps * (8 + (cases[i].ref == 1 ? 0.0 : 1.5 * cases[i].a * std::abs(std::log(double(x(i))))));
    VERIFY(numext::abs(res(i) - ref) <= Scalar(tol) * ref);
  }
}

// polygamma(n, x) = (-1)^(n+1) n! zeta(n + 1, x) for n! overflowing, or zeta(n + 1, x) ~ x^-(n+1) underflowing, where
// the product is representable: these used to return NaN, inf or 0. References: long double sum of (1 + k / x)^-(n+1)
// combined with n! in the log domain.
template <typename Scalar>
void test_polygamma_large_order() {
  struct Case {
    double n, x, ref;
  };
  std::vector<Case> cases;
  if (sizeof(Scalar) == 4) {
    cases = {{40, 50, -3.261035087314967e-22},  {35, 100, 3.499912459241533e-32},   {30, 30, -6.8043629582669342e-14},
             {34, 50, -2.0576445694243446e-21}, {20, 171, -2.8206264083990782e-28}, {36, 10, -38339.459700163335}};
  } else {
    cases = {{180, 100, -2.40802169974166e-33},
             {171, 171, 1.6510804421365765e-75},
             {300, 100, -3222042538708.5125},
             {200, 30, -9.9109053899111735e+77},
             {170, 100, -8.8841193781856056e-36}};
  }
  // The relative error is that of the exponent lgamma(n + 1) - (n + 1) log x, about |exponent| eps. Largest measured:
  // 40 eps (float), 620 eps (double); the margin is a factor of 10 and 8.
  const Scalar tol = Scalar(sizeof(Scalar) == 4 ? 400 : 5000) * NumTraits<Scalar>::epsilon();
  Array<Scalar, Dynamic, 1> n(cases.size()), x(cases.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    n(i) = Scalar(cases[i].n);
    x(i) = Scalar(cases[i].x);
  }
  const Array<Scalar, Dynamic, 1> res = polygamma(n, x);
  for (size_t i = 0; i < cases.size(); ++i) {
    const Scalar ref = Scalar(cases[i].ref);
    const Scalar v = numext::polygamma(n(i), x(i));
    VERIFY(numext::abs(v - ref) <= tol * numext::abs(ref));
    VERIFY(numext::abs(res(i) - ref) <= tol * numext::abs(ref));
  }
  // Subnormal x: 1 / x overflows and n! zeta(n + 1, x) is infinite; it was never 0.
  VERIFY_IS_EQUAL(numext::polygamma(Scalar(sizeof(Scalar) == 4 ? 35 : 171), std::numeric_limits<Scalar>::denorm_min()),
                  std::numeric_limits<Scalar>::infinity());
  // x < 0 where zeta(n + 1, x) cancels to exactly 0 and n! overflows: inf * 0 is not the answer. The true value
  // (-1)^(n+1) n! zeta(n + 1, 1 - x) underflows to a zero with the sign of (-1)^(n+1), here -0 for n = 200. (In float
  // zeta(201, 1/2) overflows and the result is NaN.)
  if (sizeof(Scalar) == 8) {
    const Scalar v = numext::polygamma(Scalar(200), Scalar(-1e6 - 0.5));
    VERIFY_IS_EQUAL(v, Scalar(0));
    VERIFY((std::signbit)(v));
  }
  // Representable in double but not in float.
  if (sizeof(Scalar) == 4) {
    VERIFY_IS_EQUAL(numext::polygamma(Scalar(170), Scalar(2)), -std::numeric_limits<Scalar>::infinity());
  }
}

// polygamma(n, x) = (-1)^(n+1) n! zeta(n + 1, x) at x = +inf (limit 0, with the sign of (-1)^(n+1); inf * 0 was NaN)
// and near the largest number, where 1 / x is subnormal or, with flush to zero, 0, and the true value underflows to 0
// or a subnormal number with the sign of (-1)^(n+1). Scalar and array, with and without flush to zero.
template <typename Scalar>
void test_polygamma_extreme_x() {
  const Scalar inf = std::numeric_limits<Scalar>::infinity();
  const Scalar largest = (std::numeric_limits<Scalar>::max)();
  const Scalar smallest_normal = (std::numeric_limits<Scalar>::min)();
  forEachFlushToZeroMode([&](FlushToZeroMode) {
    const int orders[] = {1, 2, 3, 5, 35, 171};
    const Scalar xs[] = {inf, largest, largest / Scalar(2)};
    for (int n : orders) {
      Array<Scalar, Dynamic, 1> nv = Array<Scalar, Dynamic, 1>::Constant(3, Scalar(n));
      Array<Scalar, Dynamic, 1> xv(3);
      xv << xs[0], xs[1], xs[2];
      const Array<Scalar, Dynamic, 1> av = polygamma(nv, xv);
      for (int k = 0; k < 3; ++k) {
        for (int path = 0; path < 2; ++path) {
          const Scalar v = path ? av(k) : numext::polygamma(Scalar(n), xs[k]);
          VERIFY((numext::isfinite)(v));
          VERIFY(numext::abs(v) <= smallest_normal);
          if (k == 0 || v != Scalar(0)) VERIFY((std::signbit)(v) == (n % 2 == 0));
          if (k == 0) VERIFY_IS_EQUAL(v, Scalar(0));
        }
      }
    }
  });
}

// zeta(x, q) for integer x and large negative non-integer q used to sum about |q| terms (seconds for |q| = 1e9, years
// for 1e15). For q = -N - 1/2 and even x, zeta(x, q) = 2 zeta(x, 1/2) - zeta(x, N + 3/2), with
// zeta(2, 1/2) = pi^2 / 2, zeta(4, 1/2) = pi^4 / 6 and zeta(x, z) = z^(1-x) / (x - 1) + z^-x / 2 + x z^(-x-1) / 12 +
// ...
template <typename Scalar>
void test_zeta_large_negative_q() {
  const double pi2 = EIGEN_PI * EIGEN_PI;
  // Largest measured relative error: 1 eps (float and double); the margin is a factor of 32.
  const Scalar tol = Scalar(32) * NumTraits<Scalar>::epsilon();
  const double n_values[] = {999, 1e3, 1e5, 1e6, 1e9, 1e12};
  for (double n : n_values) {
    if (sizeof(Scalar) == 4 && n > 1e6) continue;  // N + 1/2 is not representable
    const double z = n + 1.5;
    const double ref2 = pi2 - (1 / z + 1 / (2 * z * z) + 1 / (6 * z * z * z));
    const double ref4 = pi2 * pi2 / 3 - (1 / (3 * z * z * z) + 1 / (2 * z * z * z * z));
    const Scalar q = Scalar(-n - 0.5);
    VERIFY(numext::abs(numext::zeta(Scalar(2), q) - Scalar(ref2)) <= tol * Scalar(ref2));
    VERIFY(numext::abs(numext::zeta(Scalar(4), q) - Scalar(ref4)) <= tol * Scalar(ref4));
  }
  // polygamma(1, x) = zeta(2, x)
  {
    const double z = 1e6 + 1.5;
    const double ref = pi2 - (1 / z + 1 / (2 * z * z) + 1 / (6 * z * z * z));
    VERIFY(numext::abs(numext::polygamma(Scalar(1), Scalar(-1e6 - 0.5)) - Scalar(ref)) <= tol * Scalar(ref));
  }
  // Odd x: zeta(x, q) = zeta(x, f) - zeta(x, 1 - f) + zeta(x, 1 - q) is a difference of O(1) numbers, which is exact at
  // f = 1/2: zeta(3, -N - 1/2) = zeta(3, N + 3/2) ~ 5.0e-13 for N = 1e6, from z^-2 / 2 + z^-3 / 2 + z^-4 / 4 - z^-6
  // / 12.
  {
    const long double z = 1e6L + 1.5L;
    const long double ref =
        1 / (2 * z * z) + 1 / (2 * z * z * z) + 1 / (4 * z * z * z * z) - 1 / (12 * z * z * z * z * z * z);
    VERIFY(numext::abs(numext::zeta(Scalar(3), Scalar(-1e6 - 0.5)) - Scalar(ref)) <= tol * Scalar(ref));
  }
  // x = inf: the terms with |k + q| < 1 are infinite.
  VERIFY_IS_EQUAL(numext::zeta(std::numeric_limits<Scalar>::infinity(), Scalar(-1e6 - 0.5)),
                  std::numeric_limits<Scalar>::infinity());
}

// The sign of polygamma(n, x) = (-1)^(n+1) n! zeta(n + 1, x) is that of n even: n + 1 is not representable for
// n >= 2^24 (float), 2^53 (double), where n is even and (-1)^(n+1) = -1. The values are -inf where n! zeta(n + 1, x)
// overflows (x = 1, 2) and -0 in the limit x = +inf, scalar and array. n = +inf is not an integer: NaN.
template <typename Scalar>
void test_polygamma_sign_huge_order() {
  const Scalar inf = std::numeric_limits<Scalar>::infinity();
  const Scalar n0 = Scalar(sizeof(Scalar) == 4 ? 16777216.0 : 9007199254740992.0);
  const int num = 9;
  Array<Scalar, Dynamic, 1> n(num), x(num);
  n << n0, n0, n0, n0 * 2, n0 * 2, n0 * 2, inf, inf, inf;
  x << Scalar(1), Scalar(2), inf, Scalar(1), Scalar(2), inf, Scalar(1), Scalar(2), inf;
  const Array<Scalar, Dynamic, 1> res = polygamma(n, x);
  for (int i = 0; i < num; ++i) {
    for (int path = 0; path < 2; ++path) {
      const Scalar v = path ? res(i) : numext::polygamma(n(i), x(i));
      if (n(i) == inf) {
        VERIFY((numext::isnan)(v));
        continue;
      }
      VERIFY((std::signbit)(v));
      if (x(i) == inf) {
        VERIFY_IS_EQUAL(v, Scalar(0));
      } else {
        VERIFY_IS_EQUAL(v, -inf);
      }
    }
  }
}

// For q >= 1 and q^-x below the smallest normal number, the terms of zeta(x, q) ~ q^(1-x) / (x - 1) underflow: the sum
// was 0 or had the relative error of subnormals (zeta(27, 1e12 + 1.5) = 3.8e-314 was 0). For q = -N - 1/2 and odd x,
// zeta(x, q) = zeta(x, N + 3/2), and for even n polygamma(n, q) = -n! zeta(n + 1, N + 3/2) is normal. References:
// Euler-Maclaurin with 8 tail terms in 2000-bit MPFR. The log-domain form has the relative error ~ 2 eps x log(q); the
// bounds are 2x that for zeta and 4x for polygamma, and the measured error is at most 0.3 of the bound. The float
// cases are past the switch to the log-domain form (x log(q) > 92.6). polygamma(10, -2e4 - 1/2) is normal although
// zeta(11, 2e4 + 3/2) = 9.8e-45 (7 denorm_min) is subnormal, through polygamma's substitution x -> 1 - x.
template <typename Scalar>
void test_zeta_underflowing_terms() {
  struct Case {
    double x, q, ref;
  };
  const bool is_float = sizeof(Scalar) == 4;
  // zeta(x, q) = ref; for polygamma, n = x - 1 and q < 0 with ref = -n! zeta(x, 1 - q).
  const Case zeta_cases_f[] = {{5, 1e9, 2.5000000050000001e-37}, {7, 1e6, 1.6666716666725001e-37}};
  const Case zeta_cases_d[] = {{20, 1e16, 5.2631578947368468e-306}, {27, 1e12 + 1.5, 3.8461538460149259e-314}};
  const Case poly_cases_f[] = {{9, -1e5 - 0.5, -5.039596816631546e-37}, {11, -2e4 - 0.5, -3.5419785715870746e-38}};
  const Case poly_cases_d[] = {{27, -1e6 - 0.5, -1.5510806756860553e-131}, {27, -1e12 - 0.5, -1.5511210042927696e-287}};
  const Case* zc = is_float ? zeta_cases_f : zeta_cases_d;
  const Case* pc = is_float ? poly_cases_f : poly_cases_d;
  const int num_cases = 2;
  const double eps = double(NumTraits<Scalar>::epsilon());
  // A subnormal result is rounded to a multiple of denorm_min.
  const double denorm_min = double(std::numeric_limits<Scalar>::denorm_min());
  for (int i = 0; i < num_cases; ++i) {
    const double tol = 4 * eps * zc[i].x * std::log(zc[i].q);
    const double v = double(numext::zeta(Scalar(zc[i].x), Scalar(zc[i].q)));
    VERIFY(std::abs(v - zc[i].ref) <= denorm_min + tol * zc[i].ref);
  }
  Array<Scalar, Dynamic, 1> n(num_cases), x(num_cases);
  for (int i = 0; i < num_cases; ++i) {
    n(i) = Scalar(pc[i].x - 1);
    x(i) = Scalar(pc[i].q);
  }
  const Array<Scalar, Dynamic, 1> res = polygamma(n, x);
  for (int i = 0; i < num_cases; ++i) {
    const double tol = 8 * eps * pc[i].x * std::log(-pc[i].q);
    for (int path = 0; path < 2; ++path) {
      const double v = double(path ? res(i) : numext::polygamma(n(i), x(i)));
      VERIFY(std::abs(v - pc[i].ref) <= tol * std::abs(pc[i].ref));
    }
  }
  if (!is_float) {
    // zeta(27, -1e12 - 1/2) = zeta(27, 1e12 + 3/2), subnormal.
    const double ref = 3.8461538460149259e-314;
    const double tol = 4 * eps * 27 * std::log(1e12);
    const double v = double(numext::zeta(Scalar(27), Scalar(-1e12 - 0.5)));
    VERIFY(std::abs(v - ref) <= denorm_min + tol * ref);
  }
  // q = 1: x log(q) = 0, so zeta(x, 1) is the plain sum, bit for bit. Its relative error near the pole is 1 eps at
  // x = 1 + 1e-10 (double); the log-domain form had 14.
  const double one_plus = is_float ? 1.001 : 1.0000000001;
  const double ref_pole = is_float ? 1000.5305606213946 : 9999999173.1735744;
  VERIFY(std::abs(double(numext::zeta(Scalar(one_plus), Scalar(1))) - ref_pole) <= 8 * eps * ref_pole);
  const double xs[] = {1.0000001, 1.5, 2, 3.5, 20, 100};
  for (double xv : xs) {
    VERIFY_IS_EQUAL(numext::zeta(Scalar(xv), Scalar(1)),
                    (internal::zeta_impl<Scalar>::sum(Scalar(xv), Scalar(1), Scalar(1))));
  }
}

// The log-domain forms of zeta(x, q) for q^-x below the smallest normal number: the sum scaled by 1 / q is
// ~ q / (x - 1), which overflows for x - 1 < 1/16 and q near max / 16 (double zeta(1.03, 1e307) was inf), and 1 / q is
// subnormal for q > 1 / min and flushed to zero for q > max / 16, where the plain sum returned 0 or had the error of
// subnormal terms (zeta(1.0624, max / 16 + 1ulp) = 0, true 1.1e-18; zeta(1.05, max) = 0, true 7.7e-15). The rescaled
// form, selected for x < 2 or q > max / 16, forms neither. References: Euler-Maclaurin in 3000-bit MPFR at the
// arguments rounded to Scalar; the error of the forms is ~ eps (1 + (x - 1) log(q)), measured at most 0.5 of the bound.
// Cases marked 'N' are those where the plain sum is within the tolerance of the log-domain form, so the choice of
// form is not visible (float x log(q) just above 87.3, x within 1e-4 of 1, q^-x normal), and are checked against the
// reference only. The tolerance is the bound 8 eps (1 + (x - 1) log(q)) also for the forms' agreement, since the
// rounding of their exponent differs between compilers and targets.
template <typename Scalar>
void test_zeta_log_domain_range() {
  struct Case {
    double x, q, ref;
    char form;  // 'L': the log-domain form, which differs from the plain sum by more than the tolerance; 'N': not
                // checked.
  };
  const bool is_float = sizeof(Scalar) == 4;
  const Case cases_f[] = {{1.0623999834060669, 2.1267647932558654e+37, 0.075087293531669665, 'N'},
                          {1.0499999523162842, 3.4028234663852886e+38, 0.23683194202994449, 'L'},
                          {1.0499999523162842, 2.4999999200714231e+37, 0.26985793539306468, 'N'},
                          {1.0299999713897705, 2.1267647932558654e+37, 2.5295813085892365, 'N'},
                          {1.0269999504089355, 1.9999999867631625e+37, 3.643435398682416, 'N'},
                          {2, 3.4028234663852886e+38, 2.9387360522180373e-39, 'L'},
                          {1.5, 3.4028234663852886e+38, 1.0842022047972486e-19, 'L'},
                          {1.0010000467300415, 3.0000000054977558e+38, 915.16791680630282, 'N'},
                          {1.2000000476837158, 9.9999996802856925e+37, 1.2559376842600161e-07, 'L'},
                          {30, 3.4028234663852886e+38, 0, 'N'},
                          {2, 1.9999999867631625e+37, 5.000000033092094e-38, 'L'},
                          {1.0000001192092896, 3.4028234663852886e+38, 8388519.2776301387, 'N'}};
  const Case cases_d[] = {{1.0624, 1.1235582092889474e+307, 1.1088128397196225e-18, 'L'},
                          {1.05, 2.2500000000000001e+307, 8.5786879325781094e-15, 'L'},
                          {1.05, 1.7976931348623157e+308, 7.7320422357738033e-15, 'L'},
                          {1.03, 1.1235582092889474e+307, 2.0481458411441312e-08, 'L'},
                          {1.03, 1e307, 2.05531667287156683e-08, 'L'},
                          {1.02, 1e307, 3.62217980037490158e-05, 'L'},
                          {1.05, 1.1e307, 8.89119963076898069e-15, 'L'},
                          {1.0124, 9.9999999999999999e+306, 0.012582829586234699, 'L'},
                          {2, 1.7976931348623157e+308, 5.5626846462680041e-309, 'L'},
                          {1.5, 1.7976931348623157e+308, 1.4916681462400414e-154, 'L'},
                          {1.0001, 1.7976931348623157e+308, 9314.8213181353866, 'N'},
                          {1.2, 4.5000000000000002e+307, 1.4734231924042045e-61, 'L'},
                          {27, 1.7976931348623157e+308, 0, 'N'},
                          {1.0000001000000001, 1.7976931348623157e+308, 9999290.2366374139, 'N'},
                          {2, 9.9999999999999999e+306, 1e-307, 'L'}};
  const Case* cases = is_float ? cases_f : cases_d;
  const int num_cases = is_float ? 12 : 15;
  Array<Scalar, Dynamic, 1> xv(num_cases), qv(num_cases);
  for (int i = 0; i < num_cases; ++i) {
    xv(i) = Scalar(cases[i].x);
    qv(i) = Scalar(cases[i].q);
  }
  const Array<Scalar, Dynamic, 1> zv = zeta(xv, qv);
  const double eps = double(NumTraits<Scalar>::epsilon());
  const double denorm_min = double(std::numeric_limits<Scalar>::denorm_min());
  for (int i = 0; i < num_cases; ++i) {
    const Case& c = cases[i];
    const Scalar x = Scalar(c.x), q = Scalar(c.q);
    const double tol = 8 * eps * (1 + (c.x - 1) * std::log(c.q));
    if (c.form == 'L') {
      // The result is the rescaled form up to the rounding of its exponent, not the plain sum, which is off by more.
      const double form = double(internal::zeta_impl<Scalar>::log_domain_sum_rescaled(x, q));
      const double plain = double(internal::zeta_impl<Scalar>::sum(x, q, Scalar(1)));
      VERIFY(std::abs(double(numext::zeta(x, q)) - form) <= denorm_min + tol * form);
      VERIFY(std::abs(plain - form) > tol * form);
    }
    for (int path = 0; path < 2; ++path) {
      const double v = double(path ? zv(i) : numext::zeta(x, q));
      VERIFY(std::abs(v - c.ref) <= denorm_min + tol * c.ref);
    }
    // polygamma(1, q) = zeta(2, q); for larger n the product n! zeta(n + 1, q) underflows to a signed zero.
    if (c.x == 2) {
      const double v = double(numext::polygamma(Scalar(1), q));
      VERIFY(std::abs(v - c.ref) <= denorm_min + tol * c.ref);
    }
    if (c.x == 2 && c.form == 'L') {
      const Scalar even = numext::polygamma(Scalar(2), q), odd = numext::polygamma(Scalar(3), q);
      VERIFY_IS_EQUAL(even, Scalar(0));
      VERIFY((std::signbit)(even));
      VERIFY_IS_EQUAL(odd, Scalar(0));
      VERIFY(!(std::signbit)(odd));
    }
  }
}

// For huge a and x one ulp apart the continued fraction overflows and produces an internal NaN, which the clamping of
// igamma to [0, 1] used to turn into 0 (igammac returned the NaN). The true values are P = 1 and Q = 0 (x - a is
// ~1e8 standard deviations): the answer must be that or NaN, in the scalar and the array paths, never a silent 0.
template <typename Scalar>
void test_igamma_internal_nan() {
  const Scalar a = Scalar(sizeof(Scalar) == 4 ? 1e30 : 1e300);
  const Scalar x = std::nextafter(a, std::numeric_limits<Scalar>::infinity());
  const Array<Scalar, Dynamic, 1> av = Array<Scalar, Dynamic, 1>::Constant(5, a),
                                  xv = Array<Scalar, Dynamic, 1>::Constant(5, x);
  const Array<Scalar, Dynamic, 1> pv = igamma(av, xv), qv = igammac(av, xv);
  const Scalar p[2] = {numext::igamma(a, x), pv(4)}, q[2] = {numext::igammac(a, x), qv(4)};
  for (int i = 0; i < 2; ++i) {
    VERIFY((numext::isnan)(p[i]) || p[i] == Scalar(1));
    VERIFY((numext::isnan)(q[i]) || q[i] == Scalar(0));
  }
}

// For x > 1 / min (8.5e37 in float, 4.5e307 in double) the continued fraction of igammac, ~ 1 / x, is subnormal and
// Lentz's method, which clamps it at min, does not converge. The main term underflows there, so Q = 0 and P = 1, which
// was reported as NaN (the check for convergence preceded that for the main term).
template <typename Scalar>
void test_igamma_huge_x() {
  const bool is_float = sizeof(Scalar) == 4;
  const double a_f[] = {30, 79.6292877, 5000, 9999}, a_d[] = {24000, 50000, 99000, 99999};
  const Scalar x = is_float ? Scalar(9.41301965e37) : Scalar(1.7e308);
  for (int i = 0; i < 4; ++i) {
    const Scalar a = Scalar(is_float ? a_f[i] : a_d[i]);
    bool converged = true;
    internal::igammac_cf_lentz<Scalar>(a, x, &converged);
    VERIFY(!converged);
    VERIFY_IS_EQUAL(numext::igammac(a, x), Scalar(0));
    VERIFY_IS_EQUAL(numext::igamma(a, x), Scalar(1));
  }
}

// For a ~ x >~ 1 / eps, ax = x^a exp(-x) / Gamma(a) is inaccurate, and the igammac recurrence overflows once
// (x - a) x > max (inf - inf = NaN, which igamma turned into 0). For finite arguments with x != a, P and Q must not be
// NaN (they are clamped to [0, 1]), with P + Q = 1 where both use the same evaluation. Where the recurrence overflows
// here, t = (x - a) / sqrt(a) >~ 40, so Q underflows: P = 1 and Q = 0 exactly. At x = a Lentz's fraction does not
// converge for a >~ 1e7 (a >~ 5e7 in float).
template <typename Scalar>
void test_igamma_large_arguments() {
  using ArrayType = Array<Scalar, Dynamic, 1>;
  const bool is_float = sizeof(Scalar) == 4;
  // Arguments of similar size, a <= x, a >= 1e15 (float) or a >= 1e56 (double).
  const double a_f[] = {5.91395965e35, 1.35730605e30, 8.87793298e23, 5.8135072e18,  6.47320813e18,
                        7.71149728e15, 2.65364438e18, 3.17027468e36, 1.89243253e30, 4.04864182e36};
  const double x_f[] = {5.91395965e35, 1.35730605e30, 8.87793298e23, 5.81351325e18, 6.47320813e18,
                        7.71149728e15, 2.65364438e18, 3.17948384e36, 1.89432495e30, 8.23012518e36};
  const double a_d[] = {3.11504927e56,  3.044355e117,   4.27351094e138, 2.23741782e108, 1.44216162e111,
                        1.86758896e108, 2.41798259e105, 5.52553743e287, 1.20190782e199, 3.896103e300};
  const double x_d[] = {3.1150493e56,   3.044355e117,   4.27351094e138, 2.23741784e108, 1.44216162e111,
                        1.86758898e108, 2.41798259e105, 5.52553748e287, 1.20190784e199, 5e300};
  // The last three pairs have (x - a) * x > max.
  const int n = 10, num_overflow = 3;
  ArrayType a(n), x(n);
  for (int i = 0; i < n; ++i) {
    a(i) = Scalar(is_float ? a_f[i] : a_d[i]);
    x(i) = Scalar(is_float ? x_f[i] : x_d[i]);
  }
  const ArrayType p_array = igamma(a, x), q_array = igammac(a, x);
  const Scalar eps = NumTraits<Scalar>::epsilon();
  for (int i = 0; i < n; ++i) {
    if (x(i) == a(i)) {
      bool converged = true;
      internal::igammac_cf_lentz<Scalar>(a(i), x(i), &converged);
      VERIFY(!converged);
    }
    for (int path = 0; path < 2; ++path) {
      const Scalar p = path ? p_array(i) : numext::igamma(a(i), x(i));
      const Scalar q = path ? q_array(i) : numext::igammac(a(i), x(i));
      // Written so that a NaN fails; no claim at x = a, where neither evaluation converges.
      if (x(i) != a(i)) {
        VERIFY(p >= Scalar(0) && p <= Scalar(1));
        VERIFY(q >= Scalar(0) && q <= Scalar(1));
        VERIFY(numext::abs(p + q - Scalar(1)) <= 2 * eps);
      }
      if (i >= n - num_overflow) {
        VERIFY_IS_EQUAL(p, Scalar(1));
        VERIFY_IS_EQUAL(q, Scalar(0));
      }
    }
  }
}

// Temme's uniform expansion (igamma_temme) is used for a >= 1e5 (double), 1e4 (float). It covers the band near x = a
// where Lentz's fraction does not converge (|t| <~ 0.4 for a >~ 1e7 in double, 5e7 in float, where the recurrence gave
// values above 1: igammac(1.1959741656683808e15, 1.1959741715277768e15) was 842.8, true 0.4327; float igammac(1e8, 1e8)
// was 1.25e-4, true 0.5) and the series side, with no iterations. References as in test_igamma_stirling_regime (the
// terms c0 and c1 of the same expansion in 800-bit MPFR above 2e6, for which this checks the double evaluation of
// c0(eta), c1(eta) and of the cancelling forms; mpfr_gamma_inc below, an independent reference at a = 1e5). The
// bound is the 2 (log(a) / 2 + t^2 + 2) eps of that test, measured at most 0.7 of it. The last two points of each type
// are at |eta| ~ 0.1, in the Taylor series of c0 and c1 (P and Q ~ 1e-206 and 1e-235 in double), and the one before is
// just below the threshold, which takes the continued fraction and the series. P is relative only where it is tiny; it
// is otherwise 1 - Q, or the series, whose error is that of the sum of ~ sqrt(a) terms.
template <typename Scalar>
void test_igamma_temme_regime() {
  struct Reference {
    double a, x, p, q;
  };
  const Reference refs_f[] = {{1e8, 100000008, 0.50033245185326008, 0.49966754814673992},
                              {1e9, 1000001024, 0.51292037981548138, 0.48707962018451862},
                              {3e8, 300000512, 0.5117988173178956, 0.4882011826821044},
                              {999999995904.0, 1000000126976.0, 0.55214095428130504, 0.44785904571869496},
                              {1e4, 10050, 0.69234244070256556, 0.30765755929743444},
                              {1e4, 10000, 0.5013298083399552, 0.4986701916600448},
                              {9999.9990234375, 9949.9990234375, 0.30941787626932313, 0.69058212373067687},
                              {10000, 10900, 1, 1.0988648788660544e-18},
                              {10000, 9100, 8.3344447234757691e-21, 1}};
  const Reference refs_d[] = {{1195974165668380.8, 1195974171527776.8, 0.56727104130332627, 0.43272895869667373},
                              {11000000, 11000008, 0.50100237890445836, 0.49899762109554164},
                              {1000000000000.0, 1000000300000.0, 0.61791153787658355, 0.38208846212341645},
                              {10000000000.0, 10000010000.0, 0.53982914722011697, 0.46017085277988303},
                              {100000, 100100, 0.6244451307044055, 0.3755548692955945},
                              {100000, 99684, 0.15882956241747924, 0.84117043758252076},
                              {99999.999999999985, 99899.999999999985, 0.3762748927543433, 0.6237251072456567},
                              {100000, 110000, 1, 2.6554004793766945e-206},
                              {100000, 90000, 1.9782570322356405e-235, 1}};
  const bool is_float = sizeof(Scalar) == 4;
  const Reference* refs = is_float ? refs_f : refs_d;
  const int n = 9;
  using ArrayType = Array<Scalar, Dynamic, 1>;
  ArrayType a(n), x(n);
  for (int i = 0; i < n; ++i) {
    a(i) = Scalar(refs[i].a);
    x(i) = Scalar(refs[i].x);
  }
  const ArrayType p_array = igamma(a, x), q_array = igammac(a, x);
  const double eps = double(NumTraits<Scalar>::epsilon());
  for (int i = 0; i < n; ++i) {
    const double t = (double(x(i)) - double(a(i))) / std::sqrt(double(a(i)));
    const double tol = 2 * (0.5 * std::log(double(a(i))) + t * t + 2) * eps;
    if (i < 4) {  // The band of the continued fraction.
      bool converged = true;
      internal::igammac_cf_lentz<Scalar>(a(i), x(i), &converged);
      VERIFY(!converged);
    }
    for (int path = 0; path < 2; ++path) {
      const double p = double(path ? p_array(i) : numext::igamma(a(i), x(i)));
      const double q = double(path ? q_array(i) : numext::igammac(a(i), x(i)));
      VERIFY(numext::abs(q - refs[i].q) <= tol * refs[i].q);
      VERIFY(numext::abs(p - refs[i].p) <=
             (refs[i].p < 1e-3 ? tol * refs[i].p : 2 * eps * refs[i].p + tol * refs[i].q));
    }
  }
  // |eta| > 0.15 (here x / a < 0.8 or > 1.2 or a, x far apart): the smaller of P and Q is below denorm_min / 2.
  const Scalar a0 = Scalar(is_float ? 1e4 : 1e5), inf = std::numeric_limits<Scalar>::infinity();
  const Scalar xs[] = {a0 * Scalar(1.3), a0 * Scalar(0.7), inf, Scalar(0), Scalar(1)};
  const Scalar p_ref[] = {1, 0, 1, 0, 0}, q_ref[] = {0, 1, 0, 1, 1};
  for (int i = 0; i < 5; ++i) {
    VERIFY_IS_EQUAL(numext::igamma(a0, xs[i]), p_ref[i]);
    VERIFY_IS_EQUAL(numext::igammac(a0, xs[i]), q_ref[i]);
  }
}

// Q = igammac(a, x) and P = igamma(a, x), x = a + t sqrt(a), through the scalar and array functions, for a above
// igamma_use_stirling_main_term's threshold (a >~ 29 in float, a >~ 2.3e4 in double). The ordinary main term would
// have the error eps a log(a) in its exponent: 1e3 eps at a = 1e3, O(1) for a >~ 1 / eps. References: MPFR (>= 600
// bits) at the arguments rounded to Scalar, mpfr_gamma_inc for a < 1e6 and Temme's uniform expansion (DLMF 8.12.3,
// 8.12.8, terms c0 and c1) above. The relative error of Q is ~ (log(a) / 2 + t^2) eps, from log Gamma(a) and
// exp(-t^2 / 2). The bound at these test points is 2 (log(a) / 2 + t^2 + 2) eps, the 2 eps from truncating Stirling's
// series at the threshold; measured <= 0.5 of it in g++ and clang++, with and without FMA. It is not a general bound:
// near x = a at large a the error is larger (63 eps at a = 4.40e6, t = 4.6e-4, where the bound is 15 eps).
// |P - P_ref| <= 2 eps P_ref + the bound on |Q - Q_ref| for x != a; at x = a igamma uses the series, not tested here.
template <typename Scalar>
void test_igamma_stirling_regime() {
  using ArrayType = Array<Scalar, Dynamic, 1>;
  struct Reference {
    double a, x, q;
  };
  // The first three of each are at the threshold: a = 29 (float), 23300 (double), for x = a, t = 1 and t = 4.
  const Reference refs_f[] = {{29, 29, 0.47530172814254118},
                              {29, 34.385166168212891, 0.15742604999135354},
                              {29, 50.540657043457031, 0.00039717534593627728},
                              {40, 46.3245544, 0.157748569},
                              {1000, 1015.8114, 0.305755924},
                              {1000000, 1002000, 0.0228040959},
                              {1e+09, 1.00003162e+09, 0.158707111},
                              {9.99999996e+11, 1.00000052e+12, 0.300039037},
                              {9.99999996e+11, 1.00000203e+12, 0.0210963254},
                              {1e+14, 1.00000009e+14, 0.200773711}};
  const Reference refs_d[] = {{23300, 23300, 0.49912881386668523},
                              {23300, 23452.643375224736, 0.15865353112085209},
                              {23300, 23910.57350089895, 3.6275102017670049e-05},
                              {30000, 30173.205080756888, 0.15865391514519289},
                              {100000, 100632.45553203368, 0.022920595778153613},
                              {100000000, 100010000, 0.15865525352820117},
                              {1000000000000, 1000000500000, 0.30853745070965705},
                              {10000000000000000, 10000000050000000, 0.30853753784582361},
                              {1e+17, 1.0000000031622776e+17, 0.15865525853541282},
                              {1e+18, 1.000000001e+18, 0.15865525393145705},
                              {1e+20, 1.00000000005e+20, 0.30853741253698347},
                              {1e+20, 1.0000000001000001e+20, 0.15865508048690388},
                              {1e+20, 1.0000000002e+20, 0.02275014301093051},
                              {1e+20, 1.0000000007999999e+20, 6.2210020687104885e-16},
                              {9.9999999999999998e+23, 1.000000000001e+24, 0.15864163346140472},
                              {9.9999999999999998e+23, 1.00000000002e+24, 2.7507500517774874e-89}};
  const bool is_float = sizeof(Scalar) == 4;
  const Reference* refs = is_float ? refs_f : refs_d;
  const int n = is_float ? 10 : 16;
  ArrayType a(n), x(n);
  for (int i = 0; i < n; ++i) {
    a(i) = Scalar(refs[i].a);
    x(i) = Scalar(refs[i].x);
  }
  const ArrayType p_array = igamma(a, x), q_array = igammac(a, x);
  const double eps = double(NumTraits<Scalar>::epsilon());
  const double factor = 2;
  for (int i = 0; i < n; ++i) {
    const double q_ref = refs[i].q, p_ref = 1 - q_ref;
    const double t = (double(x(i)) - double(a(i))) / std::sqrt(double(a(i)));
    const double tol = factor * (0.5 * std::log(double(a(i))) + t * t + 2) * eps;
    for (int path = 0; path < 2; ++path) {
      const double q = double(path ? q_array(i) : numext::igammac(a(i), x(i)));
      const double p = double(path ? p_array(i) : numext::igamma(a(i), x(i)));
      VERIFY(numext::abs(q - q_ref) <= tol * q_ref);
      if (x(i) != a(i)) VERIFY(numext::abs(p - p_ref) <= 2 * eps * p_ref + tol * q_ref);
    }
  }
}

// P = igamma(a, x) for x <= a (and igamma at x = a), below the threshold of igamma_temme (a < 1e5 in double, 1e4 in
// float) and above that of the Stirling main term (a >~ 2.3e4, 29): the power series stopped after 2000 terms and used
// the ordinary main term with the error eps a log(a) in its exponent. The terms decay as exp(-k^2 / (2 a)), so
// ~ 8.6 sqrt(a) (double) are needed near x = a, 2700 at a = 1e5. Double igamma(99990, 99990) had the relative error
// 2.7e-10 (1.2e6 eps), float igamma(5000, 4990) 2.4e-3. The series takes the Stirling main term and a limit that grows
// with sqrt(a). References as in test_igamma_stirling_regime, here all mpfr_gamma_inc. The error of P is ~ eps sqrt(a),
// from the sum of ~ sqrt(a) terms; the bound is (10 + sqrt(a) / 4) eps, measured at most 0.5 of it.
template <typename Scalar>
void test_igamma_series_side() {
  struct Reference {
    double a, x, p, q;
  };
  const Reference refs_f[] = {
      {29, 25, 0.23659925813359777, 0.76340074186640223},     {29, 29, 0.52469827185745882, 0.47530172814254118},
      {100, 90, 0.15822098918643017, 0.84177901081356983},    {1000, 1000, 0.50420524418021551, 0.49579475581978449},
      {5000, 4990, 0.44559408878966122, 0.55440591121033878}, {9999, 9999, 0.50132987483543325, 0.49867012516456675}};
  const Reference refs_d[] = {{23300, 23150, 0.16289860911857687, 0.83710139088142313},
                              {23300, 23300, 0.50087118613331477, 0.49912881386668523},
                              {50000, 50000, 0.50059470810479331, 0.49940529189520669},
                              {99000, 99000, 0.50042264062330677, 0.49957735937669323},
                              {99990, 99990, 0.50042054313805012, 0.49957945686194988},
                              {99990, 99390, 0.02870242637511175, 0.97129757362488825}};
  const bool is_float = sizeof(Scalar) == 4;
  const Reference* refs = is_float ? refs_f : refs_d;
  const int n = 6;
  using ArrayType = Array<Scalar, Dynamic, 1>;
  ArrayType a(n), x(n);
  for (int i = 0; i < n; ++i) {
    a(i) = Scalar(refs[i].a);
    x(i) = Scalar(refs[i].x);
  }
  const ArrayType p_array = igamma(a, x), q_array = igammac(a, x);
  const double eps = double(NumTraits<Scalar>::epsilon());
  for (int i = 0; i < n; ++i) {
    const double tol = (10 + std::sqrt(refs[i].a) / 4) * eps;
    for (int path = 0; path < 2; ++path) {
      const double p = double(path ? p_array(i) : numext::igamma(a(i), x(i)));
      const double q = double(path ? q_array(i) : numext::igammac(a(i), x(i)));
      VERIFY(numext::abs(p - refs[i].p) <= tol * refs[i].p);
      VERIFY(numext::abs(q - refs[i].q) <= tol * refs[i].q);
      VERIFY(numext::abs(p + q - 1) <= 2 * tol);
    }
  }
}

// Limiting values, in the scalar and the array code paths. NaN must not be turned into a number by the clamping of
// igamma to [0, 1].
template <typename Scalar>
void test_igamma_limits() {
  const Scalar inf = std::numeric_limits<Scalar>::infinity();
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  const Scalar a_in[] = {Scalar(1), Scalar(1), inf, inf, Scalar(3), nan, Scalar(3), nan};
  const Scalar x_in[] = {inf, Scalar(0), Scalar(1), Scalar(0), inf, Scalar(2), nan, nan};
  // igamma, igammac
  const Scalar p_ref[] = {Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1), nan, nan, nan};
  const Scalar q_ref[] = {Scalar(0), Scalar(1), Scalar(1), Scalar(1), Scalar(0), nan, nan, nan};
  const int n = 8;
  Array<Scalar, Dynamic, 1> a = Map<const Array<Scalar, Dynamic, 1>>(a_in, n);
  Array<Scalar, Dynamic, 1> x = Map<const Array<Scalar, Dynamic, 1>>(x_in, n);
  const Array<Scalar, Dynamic, 1> p = igamma(a, x), q = igammac(a, x);
  for (int i = 0; i < n; ++i) {
    for (int path = 0; path < 2; ++path) {
      const Scalar ps = path ? p(i) : numext::igamma(a_in[i], x_in[i]);
      const Scalar qs = path ? q(i) : numext::igammac(a_in[i], x_in[i]);
      if ((numext::isnan)(p_ref[i])) {
        VERIFY((numext::isnan)(ps));
        VERIFY((numext::isnan)(qs));
      } else {
        VERIFY_IS_EQUAL(ps, p_ref[i]);
        VERIFY_IS_EQUAL(qs, q_ref[i]);
      }
    }
  }
}

template <typename Scalar>
void test_special_functions_edge_cases() {
  const Scalar plusinf = std::numeric_limits<Scalar>::infinity();
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  const Scalar max_val = (std::numeric_limits<Scalar>::max)();
  const Scalar denorm_min = std::numeric_limits<Scalar>::denorm_min();
  const Scalar min_val = (std::numeric_limits<Scalar>::min)();

  // Test scalar digamma edge cases
  VERIFY_IS_EQUAL(numext::digamma(Scalar(0.0)), -plusinf);
  VERIFY_IS_EQUAL(numext::digamma(Scalar(-0.0)), plusinf);
  VERIFY((numext::isnan)(numext::digamma(Scalar(-1.0))));
  VERIFY((numext::isnan)(numext::digamma(Scalar(-2.0))));
  VERIFY_IS_EQUAL(numext::digamma(plusinf), plusinf);
  VERIFY((numext::isnan)(numext::digamma(-plusinf)));
  VERIFY((numext::isnan)(numext::digamma(nan)));

  // Test polygamma at (0, 0)
  VERIFY_IS_EQUAL(numext::polygamma(Scalar(0.0), Scalar(0.0)), -plusinf);
  VERIFY_IS_EQUAL(numext::polygamma(Scalar(0.0), Scalar(-0.0)), plusinf);

  // Test scalar erf/erfc edge cases
  VERIFY_IS_EQUAL(numext::erf(Scalar(0.0)), Scalar(0.0));
  VERIFY_IS_EQUAL(numext::erf(Scalar(-0.0)), Scalar(-0.0));
  VERIFY((std::signbit)(numext::erf(Scalar(-0.0))));
  VERIFY_IS_EQUAL(numext::erf(Scalar(10.0)), Scalar(1.0));
  VERIFY_IS_EQUAL(numext::erf(Scalar(-10.0)), Scalar(-1.0));
  VERIFY_IS_EQUAL(numext::erf(Scalar(28.0)), Scalar(1.0));
  VERIFY_IS_EQUAL(numext::erf(Scalar(-28.0)), Scalar(-1.0));
  VERIFY_IS_EQUAL(numext::erf(Scalar(100.0)), Scalar(1.0));
  VERIFY_IS_EQUAL(numext::erf(Scalar(-100.0)), Scalar(-1.0));
  VERIFY_IS_EQUAL(numext::erf(max_val), Scalar(1.0));
  VERIFY_IS_EQUAL(numext::erf(-max_val), Scalar(-1.0));
  VERIFY_IS_EQUAL(numext::erf(plusinf), Scalar(1.0));
  VERIFY_IS_EQUAL(numext::erf(-plusinf), Scalar(-1.0));
  VERIFY((numext::isnan)(numext::erf(nan)));

  VERIFY_IS_EQUAL(numext::erfc(Scalar(0.0)), Scalar(1.0));
  VERIFY_IS_EQUAL(numext::erfc(Scalar(28.0)), Scalar(0.0));
  VERIFY_IS_EQUAL(numext::erfc(Scalar(-28.0)), Scalar(2.0));
  VERIFY_IS_EQUAL(numext::erfc(Scalar(100.0)), Scalar(0.0));
  VERIFY_IS_EQUAL(numext::erfc(Scalar(-100.0)), Scalar(2.0));
  VERIFY_IS_EQUAL(numext::erfc(max_val), Scalar(0.0));
  VERIFY_IS_EQUAL(numext::erfc(-max_val), Scalar(2.0));
  VERIFY_IS_EQUAL(numext::erfc(plusinf), Scalar(0.0));
  VERIFY_IS_EQUAL(numext::erfc(-plusinf), Scalar(2.0));
  VERIFY((numext::isnan)(numext::erfc(nan)));

  // NaN propagates through the x = 0 shortcut of igamma and its derivatives.
  VERIFY((numext::isnan)(numext::igamma(nan, Scalar(0))));
  VERIFY((numext::isnan)(numext::igamma_der_a(nan, Scalar(0))));
  VERIFY((numext::isnan)(numext::gamma_sample_der_alpha(nan, Scalar(0))));

  // betainc propagates NaN past its a = 0 and b = 0 shortcuts.
  VERIFY((numext::isnan)(numext::betainc(Scalar(0), Scalar(0.5), nan)));
  VERIFY((numext::isnan)(numext::betainc(Scalar(0.5), Scalar(0), nan)));
  VERIFY((numext::isnan)(numext::betainc(nan, Scalar(0.5), Scalar(0))));

  if (sizeof(Scalar) < sizeof(double)) {
    // MPFR-backed Boost.Math ibeta, 100 decimal digits. The result is exp(t) with
    // t = b log(1 - x) + a' log(x) + lgamma(a' + b) - lgamma(a') - lgamma(b) + log(cf / a'), a' = a + 1, summed in
    // Scalar: |t| ~ 90, but its first five terms add up to S ~ 330 in magnitude, so t carries an absolute rounding
    // error of order S eps, which exp turns into relative error. How much of it is reached depends on the libm (the
    // result is off by 62 eps with glibc 2.43), hence S eps rather than a measured constant, plus one subnormal ulp for
    // the final rounding.
    const Scalar a = Scalar(38.325092315673828), b = Scalar(5.0568113327026367), x = Scalar(0.07500767707824707);
    const double ref = 6.17294757793914262e-40;
    const double actual = static_cast<double>(numext::betainc(a + Scalar(1), b, x));
    const double ap = static_cast<double>(a) + 1, bd = static_cast<double>(b), xd = static_cast<double>(x);
    const double S = bd * std::abs(std::log1p(-xd)) + ap * std::abs(std::log(xd)) + std::lgamma(ap + bd) +
                     std::lgamma(ap) + std::abs(std::lgamma(bd));
    VERIFY(std::abs(actual - ref) <= S * static_cast<double>(NumTraits<Scalar>::epsilon()) * ref +
                                         static_cast<double>((std::numeric_limits<Scalar>::denorm_min)()));
  }

  // A negative q sums until n + q > 9. zeta(2, -20.5) = pi^2/2 + sum_{j=0}^{20} (j + 1/2)^-2.
  {
    double ref = EIGEN_PI * EIGEN_PI / 2;
    for (int j = 0; j <= 20; ++j) ref += 1 / ((j + 0.5) * (j + 0.5));
    VERIFY_IS_APPROX(numext::zeta(Scalar(2), Scalar(-20.5)), Scalar(ref));
  }

  if (sizeof(Scalar) >= 8) {
    VERIFY_IS_EQUAL(numext::erf(Scalar(1e200)), Scalar(1.0));
    VERIFY_IS_EQUAL(numext::erf(Scalar(-1e200)), Scalar(-1.0));
    VERIFY_IS_EQUAL(numext::erfc(Scalar(1e200)), Scalar(0.0));
    VERIFY_IS_EQUAL(numext::erfc(Scalar(-1e200)), Scalar(2.0));
  } else {
    VERIFY_IS_EQUAL(numext::erf(Scalar(1e30f)), Scalar(1.0));
    VERIFY_IS_EQUAL(numext::erf(Scalar(-1e30f)), Scalar(-1.0));
    VERIFY_IS_EQUAL(numext::erfc(Scalar(1e30f)), Scalar(0.0));
    VERIFY_IS_EQUAL(numext::erfc(Scalar(-1e30f)), Scalar(2.0));
  }

  // Denormals / subnormals
  if (denorm_min > Scalar(0)) {
    VERIFY(numext::erf(denorm_min) > Scalar(0));
    VERIFY(numext::erf(-denorm_min) < Scalar(0));
  }
  VERIFY(numext::erf(min_val) > Scalar(0));
  VERIFY(numext::erf(-min_val) < Scalar(0));

  // Array / vectorized edge cases
  using ArrayType = Array<Scalar, Dynamic, 1>;
  ArrayType x(14), erf_ref(14), erfc_ref(14);
  Scalar large_val = (sizeof(Scalar) >= 8) ? Scalar(1e200) : Scalar(1e30f);
  x << Scalar(0.0), Scalar(-0.0), denorm_min, min_val, Scalar(2.0), Scalar(-2.0), Scalar(28.0), Scalar(-28.0),
      large_val, -large_val, max_val, -max_val, plusinf, -plusinf;
  erf_ref << Scalar(0.0), Scalar(-0.0), numext::erf(denorm_min), numext::erf(min_val), Scalar(std::erf(2.0)),
      Scalar(std::erf(-2.0)), Scalar(1.0), Scalar(-1.0), Scalar(1.0), Scalar(-1.0), Scalar(1.0), Scalar(-1.0),
      Scalar(1.0), Scalar(-1.0);
  erfc_ref << Scalar(1.0), Scalar(1.0), numext::erfc(denorm_min), numext::erfc(min_val), Scalar(std::erfc(2.0)),
      Scalar(std::erfc(-2.0)), Scalar(0.0), Scalar(2.0), Scalar(0.0), Scalar(2.0), Scalar(0.0), Scalar(2.0),
      Scalar(0.0), Scalar(2.0);

  ArrayType erf_res = x.erf();
  verify_component_wise(erf_res, erf_ref);
  erf_res = erf(x);
  verify_component_wise(erf_res, erf_ref);

  ArrayType erfc_res = x.erfc();
  verify_component_wise(erfc_res, erfc_ref);
  erfc_res = erfc(x);
  verify_component_wise(erfc_res, erfc_ref);

  // Array with NaN
  ArrayType x_nan(3), erf_nan_ref(3);
  x_nan << Scalar(1.0), nan, Scalar(-1.0);
  erf_nan_ref << Scalar(std::erf(1.0)), nan, Scalar(std::erf(-1.0));
  ArrayType erf_nan_res = x_nan.erf();
  verify_component_wise(erf_nan_res, erf_nan_ref);
}

EIGEN_DECLARE_TEST(special_functions) {
  CALL_SUBTEST_1(ndtri_tail_accuracy<float>());
  CALL_SUBTEST_1(array_special_functions<ArrayXf>());
  CALL_SUBTEST_1(test_special_functions_edge_cases<float>());
  CALL_SUBTEST_1(test_betainc_large_b<float>());
  CALL_SUBTEST_1(test_polygamma_large_order<float>());
  CALL_SUBTEST_1(test_zeta_large_negative_q<float>());
  CALL_SUBTEST_1(test_polygamma_extreme_x<float>());
  CALL_SUBTEST_1(test_polygamma_sign_huge_order<float>());
  CALL_SUBTEST_1(test_zeta_underflowing_terms<float>());
  CALL_SUBTEST_1(test_zeta_log_domain_range<float>());
  CALL_SUBTEST_1(test_igamma_limits<float>());
  CALL_SUBTEST_1(test_igamma_internal_nan<float>());
  CALL_SUBTEST_1(test_igamma_huge_x<float>());
  CALL_SUBTEST_1(test_igamma_large_arguments<float>());
  CALL_SUBTEST_1(test_igamma_temme_regime<float>());
  CALL_SUBTEST_1(test_igamma_series_side<float>());
  CALL_SUBTEST_1(test_igamma_stirling_regime<float>());
  CALL_SUBTEST_2(ndtri_tail_accuracy<double>());
  CALL_SUBTEST_2(array_special_functions<ArrayXd>());
  CALL_SUBTEST_2(test_special_functions_edge_cases<double>());
  CALL_SUBTEST_2(test_betainc_large_b<double>());
  CALL_SUBTEST_2(test_polygamma_large_order<double>());
  CALL_SUBTEST_2(test_zeta_large_negative_q<double>());
  CALL_SUBTEST_2(test_polygamma_extreme_x<double>());
  CALL_SUBTEST_2(test_polygamma_sign_huge_order<double>());
  CALL_SUBTEST_2(test_zeta_underflowing_terms<double>());
  CALL_SUBTEST_2(test_zeta_log_domain_range<double>());
  CALL_SUBTEST_2(test_igamma_limits<double>());
  CALL_SUBTEST_2(test_igamma_internal_nan<double>());
  CALL_SUBTEST_2(test_igamma_huge_x<double>());
  CALL_SUBTEST_2(test_igamma_large_arguments<double>());
  CALL_SUBTEST_2(test_igamma_temme_regime<double>());
  CALL_SUBTEST_2(test_igamma_series_side<double>());
  CALL_SUBTEST_2(test_igamma_stirling_regime<double>());
  CALL_SUBTEST_3(scalar_ndtri<long double>());
  CALL_SUBTEST_4(scalar_erf_erfc<long double>());
  CALL_SUBTEST_5(custom_scalar_erf_erfc());
  // TODO(cantonios): half/bfloat16 don't have enough precision to reproduce results above.
  // CALL_SUBTEST_4(array_special_functions<ArrayX<Eigen::half>>());
  // CALL_SUBTEST_5(array_special_functions<ArrayX<Eigen::bfloat16>>());
}
