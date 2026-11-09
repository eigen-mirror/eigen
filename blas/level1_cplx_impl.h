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

namespace {
// |Re(z)| + |Im(z)|; local to each complex translation unit, whose Scalar it depends on.
struct scalar_norm1_op {
  using result_type = RealScalar;
  inline RealScalar operator()(const Scalar &a) const { return Eigen::numext::norm1(a); }
};
}  // namespace
namespace Eigen {
namespace internal {
template <>
struct functor_traits<scalar_norm1_op> {
  enum { Cost = 3 * NumTraits<Scalar>::AddCost, PacketAccess = 0 };
};
}  // namespace internal
}  // namespace Eigen

// computes |Re(z)| + |Im(z)| of a complex number z.
extern "C" EIGEN_BLAS_API RealScalar EIGEN_CAT(REAL_SCALAR_SUFFIX,
                                               EIGEN_CAT(cabs1, EIGEN_BLAS_FUNC_SUFFIX))(Complex *z) {
  return Eigen::numext::norm1(*z);
}

// computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
// res = |Rex1| + |Imx1| + |Rex2| + |Imx2| + ... + |Rexn| + |Imxn|, where x is a vector of order n
extern "C" RealScalar EIGEN_CAT(REAL_SCALAR_SUFFIX, EIGEN_BLAS_FUNC_NAME(asum))(EIGEN_BLAS_INT *n_in, RealScalar *px,
                                                                                EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0 || incx <= 0) return 0;

  // A unit-stride complex vector is a real vector of length 2n, whose |.| sum vectorizes.
  if (incx == 1) return make_vector(px, 2 * n).cwiseAbs().sum();
  return make_vector(reinterpret_cast<Scalar *>(px), n, incx).unaryExpr(scalar_norm1_op()).sum();
}

extern "C" EIGEN_BLAS_INT EIGEN_CAT(i, EIGEN_BLAS_FUNC_NAME(amax))(EIGEN_BLAS_INT *n_in, RealScalar *px,
                                                                   EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0 || incx <= 0) return 0;
  return blas_index_of_max(make_vector(reinterpret_cast<Scalar *>(px), n, incx).unaryExpr(scalar_norm1_op()));
}

extern "C" EIGEN_BLAS_INT EIGEN_CAT(i, EIGEN_BLAS_FUNC_NAME(amin))(EIGEN_BLAS_INT *n_in, RealScalar *px,
                                                                   EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0 || incx <= 0) return 0;
  return blas_index_of_min(make_vector(reinterpret_cast<Scalar *>(px), n, incx).unaryExpr(scalar_norm1_op()));
}

// computes a dot product of a conjugated vector with another vector.
EIGEN_BLAS_FUNC(dotcw)
(EIGEN_BLAS_INT *n_in, RealScalar *px, EIGEN_BLAS_INT *incx_in, RealScalar *py, EIGEN_BLAS_INT *incy_in,
 RealScalar *pres) {
  const Eigen::Index n = *n_in;
  Scalar *res = reinterpret_cast<Scalar *>(pres);
  *res = n <= 0 ? Scalar(0)
                : apply_blas_vectors(n, reinterpret_cast<Scalar *>(px), *incx_in, reinterpret_cast<Scalar *>(py),
                                     *incy_in, [](auto vx, auto vy) { return vx.dot(vy); });
}

// computes a vector-vector dot product without complex conjugation.
EIGEN_BLAS_FUNC(dotuw)
(EIGEN_BLAS_INT *n_in, RealScalar *px, EIGEN_BLAS_INT *incx_in, RealScalar *py, EIGEN_BLAS_INT *incy_in,
 RealScalar *pres) {
  const Eigen::Index n = *n_in;
  Scalar *res = reinterpret_cast<Scalar *>(pres);
  *res = n <= 0 ? Scalar(0)
                : apply_blas_vectors(n, reinterpret_cast<Scalar *>(px), *incx_in, reinterpret_cast<Scalar *>(py),
                                     *incy_in, [](auto vx, auto vy) { return vx.cwiseProduct(vy).sum(); });
}

// The Euclidean norm; see the real nrm2 for the increment convention.
extern "C" RealScalar EIGEN_CAT(REAL_SCALAR_SUFFIX, EIGEN_BLAS_FUNC_NAME(nrm2))(EIGEN_BLAS_INT *n_in, RealScalar *px,
                                                                                EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0) return 0;

  Scalar *x = reinterpret_cast<Scalar *>(px);
  if (incx == 1) return make_vector(x, n).stableNorm();
  return make_vector(x, n, std::abs(incx)).stableNorm();
}

// Applies the real rotation (c, s) to complex vectors: the real and imaginary parts are rotated independently, as
// real vectors of length 2n when both increments are one.
EIGEN_BLAS_FUNC(EIGEN_CAT(REAL_SCALAR_SUFFIX, rot))
(EIGEN_BLAS_INT *n_in, RealScalar *px, EIGEN_BLAS_INT *incx_in, RealScalar *py, EIGEN_BLAS_INT *incy_in, RealScalar *pc,
 RealScalar *ps) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  const Eigen::Index incy = *incy_in;
  if (n <= 0) return;

  if (incx == 1 && incy == 1) {
    apply_real_rotation(2 * n, px, 1, py, 1, *pc, *ps);
  } else {
    apply_real_rotation(n, px, 2 * incx, py, 2 * incy, *pc, *ps);
    apply_real_rotation(n, px + 1, 2 * incx, py + 1, 2 * incy, *pc, *ps);
  }
}

EIGEN_BLAS_FUNC(EIGEN_CAT(REAL_SCALAR_SUFFIX, scal))
(EIGEN_BLAS_INT *n_in, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  const RealScalar alpha = *palpha;
  if (n <= 0 || incx <= 0 || alpha == RealScalar(1)) return;

  // A real factor scales each component, so it never forms 0 * Inf from the other one.
  if (incx == 1)
    make_vector(px, 2 * n) *= alpha;
  else
    make_vector(reinterpret_cast<Scalar *>(px), n, incx) *= alpha;
}
