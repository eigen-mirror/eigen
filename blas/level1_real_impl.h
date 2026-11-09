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

// computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
// res = |Rex1| + |Imx1| + |Rex2| + |Imx2| + ... + |Rexn| + |Imxn|, where x is a vector of order n
extern "C" RealScalar EIGEN_BLAS_FUNC_NAME(asum)(EIGEN_BLAS_INT *n_in, Scalar *x, EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0 || incx <= 0) return 0;

  if (incx == 1) return make_vector(x, n).cwiseAbs().sum();
  return make_vector(x, n, incx).cwiseAbs().sum();
}

extern "C" EIGEN_BLAS_INT EIGEN_CAT(i, EIGEN_BLAS_FUNC_NAME(amax))(EIGEN_BLAS_INT *n_in, Scalar *x,
                                                                   EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0 || incx <= 0) return 0;

  if (incx == 1) return blas_index_of_max(make_vector(x, n).cwiseAbs());
  return blas_index_of_max(make_vector(x, n, incx).cwiseAbs());
}

extern "C" EIGEN_BLAS_INT EIGEN_CAT(i, EIGEN_BLAS_FUNC_NAME(amin))(EIGEN_BLAS_INT *n_in, Scalar *x,
                                                                   EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0 || incx <= 0) return 0;

  if (incx == 1) return blas_index_of_min(make_vector(x, n).cwiseAbs());
  return blas_index_of_min(make_vector(x, n, incx).cwiseAbs());
}

// computes a vector-vector dot product.
extern "C" Scalar EIGEN_BLAS_FUNC_NAME(dot)(EIGEN_BLAS_INT *n_in, Scalar *x, EIGEN_BLAS_INT *incx_in, Scalar *y,
                                            EIGEN_BLAS_INT *incy_in) {
  const Eigen::Index n = *n_in;
  if (n <= 0) return 0;
  return apply_blas_vectors(n, x, *incx_in, y, *incy_in, [](auto vx, auto vy) { return vx.dot(vy); });
}

// computes the Euclidean norm of a vector. As in Reference BLAS 3.10 and later, a negative increment walks the
// vector backwards and a zero one counts x(1) n times, so |incx| gives the same norm.
extern "C" Scalar EIGEN_BLAS_FUNC_NAME(nrm2)(EIGEN_BLAS_INT *n_in, Scalar *x, EIGEN_BLAS_INT *incx_in) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  if (n <= 0) return 0;

  if (incx == 1) return make_vector(x, n).stableNorm();
  return make_vector(x, n, std::abs(incx)).stableNorm();
}

EIGEN_BLAS_FUNC(rot)
(EIGEN_BLAS_INT *n_in, Scalar *x, EIGEN_BLAS_INT *incx_in, Scalar *y, EIGEN_BLAS_INT *incy_in, Scalar *pc, Scalar *ps) {
  const Eigen::Index n = *n_in;
  if (n <= 0) return;
  apply_real_rotation(n, x, *incx_in, y, *incy_in, *pc, *ps);
}

// Applies modified Givens rotation H to vectors x and y.
//   param[0] = flag:
//     -1: H = [[h11, h12], [h21, h22]]         (all 4 elements from param)
//      0: H = [[1, h12], [h21, 1]]              (h12, h21 from param)
//      1: H = [[h11, 1], [-1, h22]]             (h11, h22 from param)
//     -2: H = identity                           (no-op)
//   param[1..4] = h11, h21, h12, h22
EIGEN_BLAS_FUNC(rotm)
(EIGEN_BLAS_INT *n_in, Scalar *x, EIGEN_BLAS_INT *incx_in, Scalar *y, EIGEN_BLAS_INT *incy_in, Scalar *param) {
  const Eigen::Index n = *n_in;
  const Eigen::Index incx = *incx_in;
  const Eigen::Index incy = *incy_in;

  Scalar flag = param[0];
  if (n <= 0 || flag == Scalar(-2)) return;

  Scalar h11, h12, h21, h22;
  if (flag < Scalar(0)) {
    h11 = param[1];
    h21 = param[2];
    h12 = param[3];
    h22 = param[4];
  } else if (flag == Scalar(0)) {
    h11 = Scalar(1);
    h21 = param[2];
    h12 = param[3];
    h22 = Scalar(1);
  } else {
    h11 = param[1];
    h21 = Scalar(-1);
    h12 = Scalar(1);
    h22 = param[4];
  }

  Eigen::Index kx = incx > 0 ? 0 : (1 - n) * incx;
  Eigen::Index ky = incy > 0 ? 0 : (1 - n) * incy;

  for (Eigen::Index i = 0; i < n; ++i) {
    Scalar w = x[kx];
    Scalar z = y[ky];
    x[kx] = h11 * w + h12 * z;
    y[ky] = h21 * w + h22 * z;
    kx += incx;
    ky += incy;
  }
}

// Constructs the modified Givens transformation matrix H which zeros the second
// component of (sqrt(d1)*x1, sqrt(d2)*y1)^T.
EIGEN_BLAS_FUNC(rotmg)(Scalar *d1, Scalar *d2, Scalar *x1, Scalar *y1, Scalar *param) {
  using std::abs;

  const Scalar gam = Scalar(4096);
  const Scalar gamsq = gam * gam;
  const Scalar rgamsq = Scalar(1) / gamsq;

  Scalar flag, h11 = Scalar(0), h12 = Scalar(0), h21 = Scalar(0), h22 = Scalar(0);

  if (*d1 < Scalar(0)) {
    // Negative d1: zero everything.
    flag = Scalar(-1);
    *d1 = *d2 = *x1 = Scalar(0);
  } else {
    Scalar p2 = *d2 * *y1;
    if (p2 == Scalar(0)) {
      // d2*y1 == 0: identity transform.
      param[0] = Scalar(-2);
      return;
    }

    Scalar p1 = *d1 * *x1;
    Scalar q2 = p2 * *y1;
    Scalar q1 = p1 * *x1;
    bool do_scale = true;

    if (abs(q1) > abs(q2)) {
      h21 = -(*y1) / *x1;
      h12 = p2 / p1;
      Scalar u = Scalar(1) - h12 * h21;
      if (u <= Scalar(0)) {
        flag = Scalar(-1);
        h11 = h12 = h21 = h22 = Scalar(0);
        *d1 = *d2 = *x1 = Scalar(0);
        do_scale = false;
      } else {
        flag = Scalar(0);
        *d1 /= u;
        *d2 /= u;
        *x1 *= u;
      }
    } else if (q2 < Scalar(0)) {
      flag = Scalar(-1);
      h11 = h12 = h21 = h22 = Scalar(0);
      *d1 = *d2 = *x1 = Scalar(0);
      do_scale = false;
    } else {
      flag = Scalar(1);
      h11 = p1 / p2;
      h22 = *x1 / *y1;
      Scalar u = Scalar(1) + h11 * h22;
      Scalar temp = *d2 / u;
      *d2 = *d1 / u;
      *d1 = temp;
      *x1 = *y1 * u;
    }

    if (do_scale) {
      // Converts compact H representation (flag 0 or 1) to full form (flag -1)
      // so that scaling factors can be absorbed into all four elements.
      auto fix_h = [&]() {
        if (flag >= Scalar(0)) {
          if (flag == Scalar(0)) {
            h11 = Scalar(1);
            h22 = Scalar(1);
          } else {
            h21 = Scalar(-1);
            h12 = Scalar(1);
          }
          flag = Scalar(-1);
        }
      };

      // Rescale d1 and d2 into [rgamsq, gamsq]. The loops stop at an infinite d, which the reference never leaves,
      // and at a d1 that came out negative because q2 underflowed to -0.
      while (*d1 > Scalar(0) && *d1 <= rgamsq) {
        fix_h();
        *d1 *= gamsq;
        *x1 /= gam;
        h11 /= gam;
        h12 /= gam;
      }
      while (*d1 >= gamsq && (Eigen::numext::isfinite)(*d1)) {
        fix_h();
        *d1 /= gamsq;
        *x1 *= gam;
        h11 *= gam;
        h12 *= gam;
      }
      while (abs(*d2) <= rgamsq && *d2 != Scalar(0)) {
        fix_h();
        *d2 *= gamsq;
        h21 /= gam;
        h22 /= gam;
      }
      while (abs(*d2) >= gamsq && (Eigen::numext::isfinite)(*d2)) {
        fix_h();
        *d2 /= gamsq;
        h21 *= gam;
        h22 *= gam;
      }
    }
  }

  // Store result in param array.
  if (flag < Scalar(0)) {
    param[1] = h11;
    param[2] = h21;
    param[3] = h12;
    param[4] = h22;
  } else if (flag == Scalar(0)) {
    param[2] = h21;
    param[3] = h12;
  } else {
    param[1] = h11;
    param[4] = h22;
  }
  param[0] = flag;
}
