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

#include "../blas/blas.h"

#include "../Eigen/Core"

namespace Eigen {
namespace internal {

// Compute sqrt(x*x + y*y) taking care not to cause unnecessary overflow.
// A NaN argument is returned, y's first, as in reference xLAPY2; hypot would return +inf for (inf, NaN).
template <typename T>
T xlapy2(const T x, const T y) {
  if ((numext::isnan)(y)) return y;
  if ((numext::isnan)(x)) return x;
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

  if (numext::is_exactly_zero(w) || w > (numext::numeric_limits<T>::max)()) {
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

// Computes p + iq = (a + ib) / (c + id) without unnecessary overflow or underflow, as LAPACK's xLADIV does: the
// robust form of Smith's algorithm from M. Baudin and R. L. Smith, "A Robust Complex Division in Scilab",
// arXiv:1210.4539 (2012), applied after scaling operands near the overflow or underflow threshold.
template <typename RealScalar>
std::complex<RealScalar> xladiv(RealScalar a, RealScalar b, RealScalar c, RealScalar d) {
  // (x + y r) t with r = d / c and t = 1 / (c + d r), evaluated so that an underflowing y r does not lose y.
  const auto robust_term = [](RealScalar x, RealScalar y, RealScalar c_, RealScalar d_, RealScalar r, RealScalar t) {
    if (r != RealScalar(0)) {
      const RealScalar yr = y * r;
      return yr != RealScalar(0) ? (x + yr) * t : x * t + (y * t) * r;
    }
    return (x + d_ * (y / c_)) * t;
  };
  // Smith's division for |d| <= |c|.
  const auto smith = [&](RealScalar a_, RealScalar b_, RealScalar c_, RealScalar d_) {
    const RealScalar r = d_ / c_;
    const RealScalar t = RealScalar(1) / (c_ + d_ * r);
    return std::complex<RealScalar>(robust_term(a_, b_, c_, d_, r, t), robust_term(b_, -a_, c_, d_, r, t));
  };

  const RealScalar ab = numext::maxi(numext::abs(a), numext::abs(b));
  const RealScalar cd = numext::maxi(numext::abs(c), numext::abs(d));
  const RealScalar half_overflow = RealScalar(0.5) * xlamch<RealScalar>('O');
  const RealScalar eps = xlamch<RealScalar>('E');
  const RealScalar tiny = xlamch<RealScalar>('S') * RealScalar(2) / eps;
  const RealScalar big_scale = RealScalar(2) / (eps * eps);
  const bool swap = !(numext::abs(d) <= numext::abs(c));  // true for a NaN, as in the reference
  RealScalar s = 1;
  if (ab >= half_overflow) {
    a *= RealScalar(0.5);
    b *= RealScalar(0.5);
    s *= RealScalar(2);
  }
  if (cd >= half_overflow) {
    c *= RealScalar(0.5);
    d *= RealScalar(0.5);
    s *= RealScalar(0.5);
  }
  if (ab <= tiny) {
    a *= big_scale;
    b *= big_scale;
    s /= big_scale;
  }
  if (cd <= tiny) {
    c *= big_scale;
    d *= big_scale;
    s *= big_scale;
  }
  // For |d| > |c|, divide (b + ia) by (d + ic) instead and conjugate: (a + ib) / (c + id) = conj((b + ia) / (d + ic)).
  const std::complex<RealScalar> pq = swap ? numext::conj(smith(b, a, d, c)) : smith(a, b, c, d);
  return std::complex<RealScalar>(numext::real(pq) * s, numext::imag(pq) * s);
}

template <typename RealScalar>
EIGEN_STRONG_INLINE std::complex<RealScalar> xladiv(const std::complex<RealScalar>& num,
                                                    const std::complex<RealScalar>& den) {
  return xladiv(numext::real(num), numext::imag(num), numext::real(den), numext::imag(den));
}

// Conjugate entries of vector x.
template <typename T>
void xlacgv(const EIGEN_BLAS_INT n, T* x, const EIGEN_BLAS_INT incx) {
  if (n <= 0) return;
  using StridedVector = Map<Matrix<T, Dynamic, 1>, Unaligned, InnerStride<Dynamic> >;
  StridedVector xvec(x, n, InnerStride<Dynamic>(numext::abs(incx)));
  xvec = xvec.conjugate();
}

// Scan a matrix for its last non-zero column.
template <typename T>
EIGEN_BLAS_INT ilaxlc(const EIGEN_BLAS_INT m, const EIGEN_BLAS_INT n, const T* a, const EIGEN_BLAS_INT lda) {
  if (m == 0 || n == 0) return 0;
  using MatrixType = Map<const Matrix<T, Dynamic, Dynamic, ColMajor>, Unaligned, OuterStride<> >;
  MatrixType a_matrix(a, m, n, OuterStride<>(lda));

  // Quick test for the common case where one corner is non-zero.
  if (!numext::is_exactly_zero(a_matrix(0, n - 1)) || !numext::is_exactly_zero(a_matrix(m - 1, n - 1))) {
    return n;
  }

  // Now scan each column from the end, returning with the first non-zero.
  EIGEN_BLAS_INT c = n - 1;
  while (c >= 0 && (a_matrix.array().col(c) == T(0)).all()) {
    --c;
  }
  return c + 1;
}

// Scan a matrix for its last non-zero row.
template <typename T>
EIGEN_BLAS_INT ilaxlr(const EIGEN_BLAS_INT m, const EIGEN_BLAS_INT n, const T* a, const EIGEN_BLAS_INT lda) {
  if (m == 0 || n == 0) return 0;
  using MatrixType = Map<const Matrix<T, Dynamic, Dynamic, ColMajor>, Unaligned, OuterStride<> >;
  MatrixType a_matrix(a, m, n, OuterStride<>(lda));

  // Quick test for the common case where one corner is non-zero.
  if (!numext::is_exactly_zero(a_matrix(m - 1, 0)) || !numext::is_exactly_zero(a_matrix(m - 1, n - 1))) {
    return m;
  }

  // Now scan each row from the end, returning with the first non-zero.
  EIGEN_BLAS_INT r = m - 1;
  while (r >= 0 && (a_matrix.array().row(r) == T(0)).all()) {
    --r;
  }
  return r + 1;
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_LAPACK_UTILS_H
