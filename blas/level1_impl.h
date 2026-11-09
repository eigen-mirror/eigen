// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "common.h"

EIGEN_BLAS_FUNC(axpy)
(const EIGEN_BLAS_INT *n_in, const RealScalar *palpha, const RealScalar *px, const EIGEN_BLAS_INT *incx_in,
 RealScalar *py, const EIGEN_BLAS_INT *incy_in) {
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Eigen::Index n = *n_in;

  if (n <= 0 || alpha == Scalar(0)) return;
  apply_blas_vectors(n, x, *incx_in, y, *incy_in, [&](auto vx, auto vy) { vy += alpha * vx; });
}

EIGEN_BLAS_FUNC(axpby)
(const EIGEN_BLAS_INT *pn, const RealScalar *palpha, const RealScalar *px, const EIGEN_BLAS_INT *pincx,
 const RealScalar *pbeta, RealScalar *py, const EIGEN_BLAS_INT *pincy) {
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);
  const Eigen::Index n = *pn;

  if (n <= 0) return;
  // alpha == 0 does not read x, and beta == 0 does not read y.
  apply_blas_vectors(n, x, *pincx, y, *pincy, [&](auto vx, auto vy) {
    if (beta == Scalar(0)) {
      if (alpha == Scalar(0))
        vy.setZero();
      else
        vy = alpha * vx;
    } else if (alpha == Scalar(0)) {
      if (beta != Scalar(1)) vy *= beta;
    } else if (beta == Scalar(1)) {
      vy += alpha * vx;
    } else {
      vy = alpha * vx + beta * vy;
    }
  });
}

EIGEN_BLAS_FUNC(copy)
(EIGEN_BLAS_INT *n_in, RealScalar *px, EIGEN_BLAS_INT *incx_in, RealScalar *py, EIGEN_BLAS_INT *incy_in) {
  const Eigen::Index n = *n_in;
  if (n <= 0) return;
  apply_blas_vectors(n, reinterpret_cast<const Scalar *>(px), *incx_in, reinterpret_cast<Scalar *>(py), *incy_in,
                     [](auto vx, auto vy) { vy = vx; });
}

EIGEN_BLAS_FUNC(rotg)(RealScalar *pa, RealScalar *pb, RealScalar *pc, RealScalar *ps) {
  using std::abs;

  Scalar &a = *reinterpret_cast<Scalar *>(pa);
  Scalar &b = *reinterpret_cast<Scalar *>(pb);
  RealScalar *c = pc;
  Scalar *s = reinterpret_cast<Scalar *>(ps);

#if !ISCOMPLEX
  Scalar r, z;
  Scalar aa = abs(a);
  Scalar ab = abs(b);
  if ((aa + ab) == Scalar(0)) {
    *c = 1;
    *s = 0;
    r = 0;
    z = 0;
  } else {
    r = Eigen::numext::hypot(a, b);
    Scalar amax = aa > ab ? a : b;
    r = amax > 0 ? r : -r;
    *c = a / r;
    *s = b / r;
    z = 1;
    if (aa > ab) z = *s;
    if (ab >= aa && *c != RealScalar(0)) z = Scalar(1) / *c;
  }
  *pa = r;
  *pb = z;
#else
  // The conventions of Reference BLAS 3.10 and later: c = 1, s = 0 for b = 0, and r = |b|, s = conj(b)/|b| for
  // a = 0. Dividing by max(|a|, |b|) first keeps the quotients accurate for subnormal inputs and r finite whenever it
  // is representable.
  const RealScalar fa = abs(a), ga = abs(b);
  if (ga == RealScalar(0)) {
    *c = RealScalar(1);
    *s = Scalar(0);
  } else if (fa == RealScalar(0)) {
    *c = RealScalar(0);
    *s = Eigen::numext::conj(b) / ga;
    a = Scalar(ga);
  } else {
    const RealScalar scale = Eigen::numext::maxi(fa, ga);
    const RealScalar norm = Eigen::numext::hypot(fa / scale, ga / scale);
    const Scalar phase = a / fa;
    *c = (fa / scale) / norm;
    *s = phase * Eigen::numext::conj(b / scale) / norm;
    a = phase * (scale * norm);
  }
#endif
}

EIGEN_BLAS_FUNC(scal)(EIGEN_BLAS_INT *n_in, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  Scalar *x = reinterpret_cast<Scalar *>(px);
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  if (n <= 0 || incx <= 0 || alpha == Scalar(1)) return;

  if (incx == 1)
    make_vector(x, n) *= alpha;
  else
    make_vector(x, n, incx) *= alpha;
}

EIGEN_BLAS_FUNC(swap)
(EIGEN_BLAS_INT *n_in, RealScalar *px, EIGEN_BLAS_INT *incx_in, RealScalar *py, EIGEN_BLAS_INT *incy_in) {
  const Eigen::Index n = *n_in;
  if (n <= 0) return;
  apply_blas_vectors(n, reinterpret_cast<Scalar *>(px), *incx_in, reinterpret_cast<Scalar *>(py), *incy_in,
                     [](auto vx, auto vy) { vy.swap(vx); });
}
