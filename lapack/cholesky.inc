// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"
#include <Eigen/Cholesky>

// POTRF computes the Cholesky factorization of a symmetric (Hermitian) positive definite matrix A.
EIGEN_LAPACK_FUNC(potrf)(const char *uplo, int *n, RealScalar *pa, int *lda, int *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*lda < std::max(1, *n))
    *info = -4;
  if (*info != 0) {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "POTRF", &e, kBlasNameLength);
  }

  Scalar *a = reinterpret_cast<Scalar *>(pa);
  MatrixType A(a, *n, *n, *lda);
  int ret;
  if (UPLO(*uplo) == UP)
    ret = int(Eigen::internal::llt_inplace<Scalar, Eigen::Upper>::blocked(A));
  else
    ret = int(Eigen::internal::llt_inplace<Scalar, Eigen::Lower>::blocked(A));

  // The kernel stops only at a pivot <= 0; xPOTRF also stops at a NaN pivot. Before the kernel's stop the diagonal of
  // the factor holds the square roots of the pivots, so its first NaN there is the pivot xPOTRF reports. This does not
  // rely on the NaN reaching the later pivots, which a kernel that skips zero operands would prevent.
  const int stop = ret >= 0 ? ret : *n;
  for (int j = 0; j < stop; ++j) {
    if ((Eigen::numext::isnan)(Eigen::numext::real(A(j, j)))) {
      *info = j + 1;
      return;
    }
  }
  if (ret >= 0) *info = ret + 1;
}

// POTRS solves A X = B for a symmetric (Hermitian) positive definite matrix A, using the Cholesky factorization
// A = U^H U or A = L L^H computed by POTRF.
EIGEN_LAPACK_FUNC(potrs)
(const char *uplo, int *n, int *nrhs, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, int *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*nrhs < 0)
    *info = -3;
  else if (*lda < std::max(1, *n))
    *info = -5;
  else if (*ldb < std::max(1, *n))
    *info = -7;
  if (*info != 0) {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "POTRS", &e, kBlasNameLength);
  }

  Scalar *a = reinterpret_cast<Scalar *>(pa);
  Scalar *b = reinterpret_cast<Scalar *>(pb);
  MatrixType A(a, *n, *n, *lda);
  MatrixType B(b, *n, *nrhs, *ldb);

  if (UPLO(*uplo) == UP) {
    A.triangularView<Eigen::Upper>().adjoint().solveInPlace(B);
    A.triangularView<Eigen::Upper>().solveInPlace(B);
  } else {
    A.triangularView<Eigen::Lower>().solveInPlace(B);
    A.triangularView<Eigen::Lower>().adjoint().solveInPlace(B);
  }
}
