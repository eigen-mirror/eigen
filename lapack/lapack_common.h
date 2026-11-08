// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_LAPACK_COMMON_H
#define EIGEN_LAPACK_COMMON_H

#include "../blas/common.h"
#include "lapack.h"
#include "lapack_utils.h"

#define EIGEN_LAPACK_FUNC(FUNC) EIGEN_BLAS_FUNC(FUNC)

using TranspositionsType = Eigen::Transpositions<Eigen::Dynamic, Eigen::Dynamic, EIGEN_BLAS_INT>;
using PivotsType = Eigen::Map<TranspositionsType>;

namespace {
// WORK(1) for a workspace size: RealScalar(lwork), rounded up where RealScalar cannot represent lwork, as the
// reference's xROUNDUP_LWORK does, so that a caller allocating INT(WORK(1)) passes the LWORK check.
inline RealScalar roundup_lwork(Eigen::Index lwork) {
  RealScalar work = RealScalar(lwork);
  if (static_cast<Eigen::Index>(work) < lwork) work *= RealScalar(1) + Eigen::NumTraits<RealScalar>::epsilon();
  return work;
}
}  // namespace

#endif  // EIGEN_LAPACK_COMMON_H
