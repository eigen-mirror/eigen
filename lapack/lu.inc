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
#include <Eigen/LU>

// GETRF computes an LU factorization of a general M-by-N matrix A using partial pivoting with row interchanges.
EIGEN_LAPACK_FUNC(getrf)
(EIGEN_BLAS_INT *m, EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv,
 EIGEN_BLAS_INT *info) {
  *info = 0;
  if (*m < 0)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *m))
    *info = -4;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GETRF", &e, kBlasNameLength);
  }

  if (*m == 0 || *n == 0) return;

  Scalar *a = reinterpret_cast<Scalar *>(pa);
  EIGEN_BLAS_INT nb_transpositions;
  EIGEN_BLAS_INT ret =
      EIGEN_BLAS_INT(Eigen::internal::partial_lu_impl<Scalar, Eigen::ColMajor, EIGEN_BLAS_INT>::blocked_lu(
          *m, *n, a, *lda, ipiv, nb_transpositions));

  for (EIGEN_BLAS_INT i = 0; i < std::min(*m, *n); ++i) ipiv[i]++;

  if (ret >= 0) *info = ret + 1;
}

// GETRS solves op(A) X = B, where op(A) is A, A^T or A^H, with a general N-by-N matrix A using the LU
// factorization computed by GETRF.
EIGEN_LAPACK_FUNC(getrs)
(const char *trans, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *nrhs, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv,
 RealScalar *pb, EIGEN_BLAS_INT *ldb, EIGEN_BLAS_INT *info) {
  *info = 0;
  if (OP(*trans) == INVALID)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*nrhs < 0)
    *info = -3;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -5;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -8;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GETRS", &e, kBlasNameLength);
  }

  Scalar *a = reinterpret_cast<Scalar *>(pa);
  Scalar *b = reinterpret_cast<Scalar *>(pb);
  MatrixType lu(a, *n, *n, *lda);
  MatrixType B(b, *n, *nrhs, *ldb);

  using Eigen::UnitLower;
  using Eigen::Upper;
  // IPIV is input-only (and may be shared by concurrent solves): convert to 0-based indices in a copy.
  TranspositionsType pivots(*n);
  for (EIGEN_BLAS_INT i = 0; i < *n; ++i) pivots.indices()(i) = ipiv[i] - 1;
  if (OP(*trans) == NOTR) {
    B = pivots * B;
    lu.triangularView<UnitLower>().solveInPlace(B);
    lu.triangularView<Upper>().solveInPlace(B);
  } else if (OP(*trans) == TR) {
    lu.triangularView<Upper>().transpose().solveInPlace(B);
    lu.triangularView<UnitLower>().transpose().solveInPlace(B);
    B = pivots.transpose() * B;
  } else if (OP(*trans) == ADJ) {
    lu.triangularView<Upper>().adjoint().solveInPlace(B);
    lu.triangularView<UnitLower>().adjoint().solveInPlace(B);
    B = pivots.transpose() * B;
  }
}

// GETRI computes the inverse of a matrix using the LU factorization computed by GETRF: inv(A) is found by solving
// inv(A) L = inv(U). WORK(1) returns the optimal LWORK on a workspace query (LWORK = -1) and the workspace used
// otherwise; LWORK >= N suffices, and LWORK >= 64 * N lets the solve run on block columns.
EIGEN_LAPACK_FUNC(getri)
(EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv, RealScalar *pwork, EIGEN_BLAS_INT *lwork,
 EIGEN_BLAS_INT *info) {
  constexpr Eigen::Index kGetriBlockSize = 64;
  constexpr Eigen::Index kNbMin = 2;
  *info = 0;
  Scalar *work = reinterpret_cast<Scalar *>(pwork);
  work[0] = Scalar(roundup_lwork(std::max<Eigen::Index>(1, Eigen::Index(*n) * kGetriBlockSize)));
  const bool query = *lwork == -1;
  if (*n < 0)
    *info = -1;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -3;
  else if (*lwork < std::max<EIGEN_BLAS_INT>(1, *n) && !query)
    *info = -6;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GETRI", &e, kBlasNameLength);
  }
  if (query || *n == 0) return;

  // Form inv(U). If U is singular (INFO > 0), A is left unchanged.
  EIGEN_BLAS_FUNC_NAME(trtri)("U", "N", n, pa, lda, info);
  if (*info > 0) return;

  const Eigen::Index size = *n;
  MatrixType A(reinterpret_cast<Scalar *>(pa), size, size, *lda);
  Eigen::Index nb = kGetriBlockSize;
  Eigen::Index iws = size;
  if (nb < size) {
    iws = nb * size;
    if (*lwork < iws) nb = *lwork / size;
  }

  // Column j of inv(A) is column j of inv(U) minus inv(A)(:, j+1:n) L(j+1:n, j), with the strictly lower part of L
  // copied to WORK and zeroed in A so that the columns to the right already hold inv(A) when they are used.
  // Block columns of width nb take the same step with a GEMM and a unit-lower right triangular solve.
  if (nb < kNbMin || nb >= size) {
    CompactVectorType w(work, size);
    for (Eigen::Index j = size - 1; j >= 0; --j) {
      const Eigen::Index rest = size - j - 1;
      w.tail(rest) = A.col(j).tail(rest);
      A.col(j).tail(rest).setZero();
      if (rest > 0) A.col(j).noalias() -= A.middleCols(j + 1, rest) * w.tail(rest);
    }
  } else {
    MatrixType W(work, size, nb, size);
    for (Eigen::Index j = ((size - 1) / nb) * nb; j >= 0; j -= nb) {
      const Eigen::Index jb = std::min(nb, size - j);
      const Eigen::Index rest = size - j - jb;
      for (Eigen::Index jj = 0; jj < jb; ++jj) {
        const Eigen::Index below = size - j - jj - 1;
        W.col(jj).tail(below) = A.col(j + jj).tail(below);
        A.col(j + jj).tail(below).setZero();
      }
      auto Aj = A.middleCols(j, jb);
      if (rest > 0) Aj.noalias() -= A.middleCols(j + jb, rest) * W.block(j + jb, 0, rest, jb);
      W.block(j, 0, jb, jb).triangularView<Eigen::UnitLower>().solveInPlace<Eigen::OnTheRight>(Aj);
    }
  }

  // inv(A) = inv(U) inv(L) P^T, where A = P L U.
  for (Eigen::Index j = size - 2; j >= 0; --j) {
    const Eigen::Index jp = ipiv[j] - 1;
    if (jp != j) A.col(j).swap(A.col(jp));
  }
  work[0] = Scalar(roundup_lwork(iws));
}
