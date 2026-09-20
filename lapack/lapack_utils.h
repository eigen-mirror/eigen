// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_LAPACK_UTILS_H
#define EIGEN_LAPACK_UTILS_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <limits>

#include "../Eigen/Core"

namespace Eigen {
namespace internal {

// Compute sqrt(x*x + y*y) taking care not to cause unnecessary overflow.
template <typename T>
T xlapy2(const T x, const T y) {
  return numext::hypot(x, y);
}

// Compute sqrt(x*x + y*y + z*z) taking care not to cause unnecessary overflow.
template <typename T>
T xlapy3(const T x, const T y, const T z) {
  if (x == T(0) && y == T(0) && z == T(0)) {
    return T(0);
  }

  const T xabs = numext::abs(x);
  const T yabs = numext::abs(y);
  const T zabs = numext::abs(z);
  const T w = numext::maxi(numext::maxi(xabs, yabs), zabs);

  if (numext::is_exactly_zero(w)) {
    return (xabs + yabs + zabs);
  }

  const T tx = xabs / w;
  const T ty = yabs / w;
  const T tz = zabs / w;
  return w * numext::sqrt(tx * tx + ty * ty + tz * tz);
}

// Determines machine parameters according to LAPACK specification.
template <typename T>
T xlamch(const char c) {
  switch (std::tolower(static_cast<unsigned char>(c))) {
    case 'e':
      return numext::numeric_limits<T>::epsilon() / numext::numeric_limits<T>::radix;
    case 's': {
      EIGEN_CONSTEXPR T tiny = (numext::numeric_limits<T>::min)();
      EIGEN_CONSTEXPR T small = T(1) / (numext::numeric_limits<T>::max)();
      EIGEN_CONSTEXPR T small_scaled = small * (T(1) + numext::numeric_limits<T>::epsilon());
      return (small >= tiny) ? small_scaled : tiny;
    }
    case 'b':
      return numext::numeric_limits<T>::radix;
    case 'p':
      return numext::numeric_limits<T>::epsilon();
    case 'n':
      return numext::numeric_limits<T>::digits;
    case 'r':
      return 1;
    case 'm':
      return numext::numeric_limits<T>::min_exponent;
    case 'u':
      return (numext::numeric_limits<T>::min)();
    case 'l':
      return numext::numeric_limits<T>::max_exponent;
    case 'o':
      return (numext::numeric_limits<T>::max)();
    default:
      return 0;
  }
}

// Performs complex division in real arithmetic, avoiding unnecessary intermediate overflow.
// Consistent with LAPACK auxiliary routines SLADIV / DLADIV and CLADIV / ZLADIV
// (algorithm due to Robert L. Smith, D. Knuth, The Art of Computer Programming, Vol. 2, p. 195).
template <typename RealScalar>
EIGEN_STRONG_INLINE std::complex<RealScalar> xladiv(const std::complex<RealScalar>& num,
                                                    const std::complex<RealScalar>& den) {
  const RealScalar a = numext::real(num);
  const RealScalar b = numext::imag(num);
  const RealScalar c = numext::real(den);
  const RealScalar d = numext::imag(den);

  RealScalar p, q;
  if (numext::abs(d) < numext::abs(c)) {
    const RealScalar e = d / c;
    const RealScalar f = c + d * e;
    p = (a + b * e) / f;
    q = (b - a * e) / f;
  } else {
    const RealScalar e = c / d;
    const RealScalar f = d + c * e;
    p = (b + a * e) / f;
    q = (-a + b * e) / f;
  }
  return std::complex<RealScalar>(p, q);
}

template <typename RealScalar>
EIGEN_STRONG_INLINE std::complex<RealScalar> xladiv(const RealScalar a, const RealScalar b, const RealScalar c,
                                                    const RealScalar d) {
  return xladiv(std::complex<RealScalar>(a, b), std::complex<RealScalar>(c, d));
}

// Conjugate entries of vector x.
template <typename T>
void xlacgv(const int n, T* x, const int incx) {
  using StridedVector = Map<Matrix<T, Dynamic, 1>, Unaligned, InnerStride<Dynamic> >;
  StridedVector xvec(x, n, InnerStride<Dynamic>(numext::abs(incx)));
  xvec = xvec.conjugate();
}

// Scan a matrix for its last non-zero column.
template <typename T>
int ilaxlc(const int m, const int n, const T* a, const int lda) {
  if (m == 0 || n == 0) return 0;
  using MatrixType = Map<const Matrix<T, Dynamic, Dynamic, ColMajor>, Unaligned, OuterStride<> >;
  MatrixType a_matrix(a, m, n, OuterStride<>(lda));

  // Quick test for the common case where one corner is non-zero.
  if (!numext::is_exactly_zero(a_matrix(0, n - 1)) || !numext::is_exactly_zero(a_matrix(m - 1, n - 1))) {
    return n;
  }

  // Now scan each column from the end, returning with the first non-zero.
  int c = n - 1;
  while (c >= 0 && (a_matrix.array().col(c) == T(0)).all()) {
    --c;
  }
  return c + 1;
}

// Scan a matrix for its last non-zero row.
template <typename T>
int ilaxlr(const int m, const int n, const T* a, const int lda) {
  if (m == 0 || n == 0) return 0;
  using MatrixType = Map<const Matrix<T, Dynamic, Dynamic, ColMajor>, Unaligned, OuterStride<> >;
  MatrixType a_matrix(a, m, n, OuterStride<>(lda));

  // Quick test for the common case where one corner is non-zero.
  if (!numext::is_exactly_zero(a_matrix(m - 1, 0)) || !numext::is_exactly_zero(a_matrix(m - 1, n - 1))) {
    return m;
  }

  // Now scan each row from the end, returning with the first non-zero.
  int r = m - 1;
  while (r >= 0 && (a_matrix.array().row(r) == T(0)).all()) {
    --r;
  }
  return r + 1;
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_LAPACK_UTILS_H
