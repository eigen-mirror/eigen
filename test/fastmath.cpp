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
  typedef std::complex<RealScalar> Scalar;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMatrix;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> ColMatrix;

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
  typedef std::complex<RealScalar> Scalar;
  typedef Matrix<Scalar, 2, 1> Vector2;

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
}

template <typename RealScalar>
void check_complex_packet_math_functions() {
  typedef std::complex<RealScalar> Scalar;
  typedef Matrix<Scalar, 4, 1> Vector4;

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
  typedef std::complex<RealScalar> Scalar;
  typedef Matrix<Scalar, 3, 2> Matrix32;
  typedef Matrix<Scalar, 2, 2> Matrix22;

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

template <typename RealScalar>
void check_complex_fastmath() {
  check_complex_rowmajor_adjoint_product<RealScalar>();
  check_complex_packet_arithmetic<RealScalar>();
  check_complex_packet_math_functions<RealScalar>();
  check_complex_householder_qr<RealScalar>();
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
}
