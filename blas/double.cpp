// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define SCALAR double
#define SCALAR_SUFFIX d
#define SCALAR_SUFFIX_UP "D"
#define ISCOMPLEX 0

#include "level1_impl.h"
#include "level1_real_impl.h"
#include "level2_impl.h"
#include "level2_real_impl.h"
#include "level3_impl.h"

// dsdot: the dot product of two float vectors, accumulated in double.
extern "C" double EIGEN_BLAS_FUNC_NAME(sdot)(EIGEN_BLAS_INT* n_in, float* x, EIGEN_BLAS_INT* incx_in, float* y,
                                             EIGEN_BLAS_INT* incy_in) {
  const Eigen::Index n = *n_in;
  if (n <= 0) return 0;
  return apply_blas_vectors(n, x, *incx_in, y, *incy_in, [](auto vx, auto vy) {
    return vx.template cast<double>().cwiseProduct(vy.template cast<double>()).sum();
  });
}
