// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"
#include <Eigen/SVD>

#if ISCOMPLEX
#define EIGEN_LAPACK_ARG_IF_COMPLEX(X) X,
#else
#define EIGEN_LAPACK_ARG_IF_COMPLEX(X)
#endif

namespace {
// Writes the singular values of `svd` to s, U to u and V^H to vt, skipping a null destination, and returns INFO.
// Eigen's SVDs fill no output for a non-finite A (InvalidInput) or after a numerical failure (NumericalIssue).
// Reference xGESDD and xGESVD scale an Inf in A into NaN singular values, with finite but meaningless vectors, and
// return INFO = 0. Here NaN goes to every requested output instead, so that no vector reads as a result. INFO = 1
// reports a failure other than a non-finite A, as INFO > 0 reports a failure to converge in the reference.
template <typename SVD>
EIGEN_BLAS_INT store_svd(const SVD &svd, EIGEN_BLAS_INT diag_size, RealScalar *s, Scalar *u, EIGEN_BLAS_INT ldu,
                         Scalar *vt, EIGEN_BLAS_INT ldvt) {
  const Eigen::ComputationInfo status = svd.info();
  const bool unfilled = status != Eigen::Success && status != Eigen::NoConvergence;
  const Scalar nan(Eigen::NumTraits<RealScalar>::quiet_NaN());
  if (unfilled)
    make_vector(s, diag_size).setConstant(Eigen::numext::real(nan));
  else
    make_vector(s, diag_size) = svd.singularValues().head(diag_size);
  if (u != nullptr) {
    auto dst = matrix(u, svd.matrixU().rows(), svd.matrixU().cols(), ldu);
    if (unfilled)
      dst.setConstant(nan);
    else
      dst = svd.matrixU();
  }
  if (vt != nullptr) {
    auto dst = matrix(vt, svd.matrixV().cols(), svd.matrixV().rows(), ldvt);
    if (unfilled)
      dst.setConstant(nan);
    else
      dst = svd.matrixV().adjoint();
  }
  return status == Eigen::Success || status == Eigen::InvalidInput ? 0 : 1;
}
}  // namespace

// GESDD computes the singular values and vectors of a general M-by-N matrix A by divide and conquer.
EIGEN_LAPACK_FUNC(gesdd)
(const char *jobz, EIGEN_BLAS_INT *m, EIGEN_BLAS_INT *n, RealScalar *ra, EIGEN_BLAS_INT *lda, RealScalar *s,
 RealScalar *ru, EIGEN_BLAS_INT *ldu, RealScalar *rvt, EIGEN_BLAS_INT *ldvt, RealScalar *work, EIGEN_BLAS_INT *lwork,
 EIGEN_LAPACK_ARG_IF_COMPLEX(RealScalar * /*rwork*/) EIGEN_BLAS_INT * /*iwork*/, EIGEN_BLAS_INT *info) {
  // TODO exploit the work buffer
  bool query_size = *lwork == -1;
  const char jobz_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobz)));
  jobz = &jobz_upper;
  EIGEN_BLAS_INT diag_size = (std::min)(*m, *n);

  *info = 0;
  if (*jobz != 'A' && *jobz != 'S' && *jobz != 'O' && *jobz != 'N')
    *info = -1;
  else if (*m < 0)
    *info = -2;
  else if (*n < 0)
    *info = -3;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *m))
    *info = -5;
  else if (*ldu < 1 || ((*jobz == 'A' || *jobz == 'S') && *ldu < *m) || (*jobz == 'O' && *m < *n && *ldu < *m))
    *info = -8;
  else if (*ldvt < 1 || (*jobz == 'A' && *ldvt < *n) || (*jobz == 'S' && *ldvt < diag_size) ||
           (*jobz == 'O' && *m >= *n && *ldvt < *n))
    *info = -10;

  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GESDD ", &e, kBlasNameLength);
  }

  if (query_size) {
    work[0] = RealScalar(1);
    return;
  }

  if (*n == 0 || *m == 0) return;

  Scalar *a = reinterpret_cast<Scalar *>(ra);
  Scalar *u = reinterpret_cast<Scalar *>(ru);
  Scalar *vt = reinterpret_cast<Scalar *>(rvt);

  PlainMatrixType mat = matrix(a, *m, *n, *lda);

  // As reference xGESDD documents: INFO = -4 if A has a NaN entry, without a call to XERBLA.
  if (mat.hasNaN()) {
    *info = -4;
    return;
  }

  if (*jobz == 'A') {
    Eigen::BDCSVD<PlainMatrixType, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(mat);
    *info = store_svd(svd, diag_size, s, u, *ldu, vt, *ldvt);
  } else if (*jobz == 'N') {
    Eigen::BDCSVD<PlainMatrixType> svd(mat);
    *info = store_svd(svd, diag_size, s, nullptr, 0, nullptr, 0);
  } else {
    // JOBZ = 'S' or 'O'; 'O' overwrites A with U if M >= N and with V^H otherwise.
    Eigen::BDCSVD<PlainMatrixType, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(mat);
    if (*jobz == 'S')
      *info = store_svd(svd, diag_size, s, u, *ldu, vt, *ldvt);
    else if (*m >= *n)
      *info = store_svd(svd, diag_size, s, a, *lda, vt, *ldvt);
    else
      *info = store_svd(svd, diag_size, s, u, *ldu, a, *lda);
  }
}

// GESVD computes the singular values and vectors of a general M-by-N matrix A by the two-sided Jacobi algorithm.
EIGEN_LAPACK_FUNC(gesvd)
(const char *jobu, const char *jobv, EIGEN_BLAS_INT *m, EIGEN_BLAS_INT *n, RealScalar *ra, EIGEN_BLAS_INT *lda,
 RealScalar *s, RealScalar *ru, EIGEN_BLAS_INT *ldu, RealScalar *rvt, EIGEN_BLAS_INT *ldvt, RealScalar *work,
 EIGEN_BLAS_INT *lwork, EIGEN_LAPACK_ARG_IF_COMPLEX(RealScalar * /*rwork*/) EIGEN_BLAS_INT *info) {
  // TODO exploit the work buffer
  bool query_size = *lwork == -1;
  const char jobu_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobu)));
  jobu = &jobu_upper;
  const char jobv_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobv)));
  jobv = &jobv_upper;
  EIGEN_BLAS_INT diag_size = (std::min)(*m, *n);

  *info = 0;
  if (*jobu != 'A' && *jobu != 'S' && *jobu != 'O' && *jobu != 'N')
    *info = -1;
  else if ((*jobv != 'A' && *jobv != 'S' && *jobv != 'O' && *jobv != 'N') || (*jobu == 'O' && *jobv == 'O'))
    *info = -2;
  else if (*m < 0)
    *info = -3;
  else if (*n < 0)
    *info = -4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *m))
    *info = -6;
  else if (*ldu < 1 || ((*jobu == 'A' || *jobu == 'S') && *ldu < *m))
    *info = -9;
  else if (*ldvt < 1 || (*jobv == 'A' && *ldvt < *n) || (*jobv == 'S' && *ldvt < diag_size))
    *info = -11;

  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GESVD ", &e, kBlasNameLength);
  }

  if (query_size) {
    work[0] = RealScalar(1);
    return;
  }

  if (*n == 0 || *m == 0) return;

  Scalar *a = reinterpret_cast<Scalar *>(ra);
  Scalar *u = reinterpret_cast<Scalar *>(ru);
  Scalar *vt = reinterpret_cast<Scalar *>(rvt);

  PlainMatrixType mat = matrix(a, *m, *n, *lda);

  EIGEN_BLAS_INT option = 0;
  if (*jobu == 'A')
    option |= Eigen::ComputeFullU;
  else if (*jobu == 'S' || *jobu == 'O')
    option |= Eigen::ComputeThinU;
  if (*jobv == 'A')
    option |= Eigen::ComputeFullV;
  else if (*jobv == 'S' || *jobv == 'O')
    option |= Eigen::ComputeThinV;

  Eigen::JacobiSVD<PlainMatrixType> svd(mat, option);

  // 'O' overwrites A with U or V^H.
  Scalar *u_dst = *jobu == 'N' ? nullptr : *jobu == 'O' ? a : u;
  Scalar *vt_dst = *jobv == 'N' ? nullptr : *jobv == 'O' ? a : vt;
  *info = store_svd(svd, diag_size, s, u_dst, *jobu == 'O' ? *lda : *ldu, vt_dst, *jobv == 'O' ? *lda : *ldvt);
}

#undef EIGEN_LAPACK_ARG_IF_COMPLEX
