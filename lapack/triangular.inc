// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"
#include <Eigen/Core>

namespace {

// LAPACK's singularity test: the 1-based index of the first exactly zero diagonal entry of A, or 0 if there is none.
EIGEN_BLAS_INT first_zero_diagonal_entry(const MatrixType &A) {
  for (Eigen::Index i = 0; i < A.rows(); ++i) {
    if (Eigen::numext::is_exactly_zero(A(i, i))) return EIGEN_BLAS_INT(i + 1);
  }
  return 0;
}

// Solves op(T) X = B in place, where op(T) is T (NOTR), T^T (TR) or T^H (ADJ).
template <typename TriangularViewType>
void triangular_solve_in_place(TriangularViewType T, EIGEN_BLAS_INT op, MatrixType &B) {
  if (op == NOTR)
    T.solveInPlace(B);
  else if (op == TR)
    T.transpose().solveInPlace(B);
  else
    T.adjoint().solveInPlace(B);
}

}  // namespace

// TRTRI computes the inverse of a real or complex upper or lower triangular matrix A in place. If DIAG = 'U' the
// diagonal is not referenced and assumed to be one. INFO = i > 0 means A(i,i) is exactly zero, found before any
// entry of A is modified.
EIGEN_LAPACK_FUNC(trtri)
(const char *uplo, const char *diag, EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (DIAG(*diag) == INVALID)
    *info = -2;
  else if (*n < 0)
    *info = -3;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -5;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "TRTRI", &e, kBlasNameLength);
  }

  if (*n == 0) return;

  MatrixType A(reinterpret_cast<Scalar *>(pa), *n, *n, *lda);
  const bool unit = DIAG(*diag) == UNIT;
  if (!unit) {
    *info = first_zero_diagonal_entry(A);
    if (*info > 0) return;
  }

  if (UPLO(*uplo) == UP) {
    if (unit)
      A.triangularView<Eigen::UnitUpper>().inverseInPlace();
    else
      A.triangularView<Eigen::Upper>().inverseInPlace();
  } else {
    if (unit)
      A.triangularView<Eigen::UnitLower>().inverseInPlace();
    else
      A.triangularView<Eigen::Lower>().inverseInPlace();
  }
}

// LAUUM computes the product U U^H (UPLO = 'U') or L^H L (UPLO = 'L'), overwriting the triangular factor
// stored in the same triangle of A with the corresponding triangle of the Hermitian result. The other triangle is
// not referenced. The computed diagonal is real.
EIGEN_LAPACK_FUNC(lauum)
(const char *uplo, EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -4;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "LAUUM", &e, kBlasNameLength);
  }

  if (*n == 0) return;

  MatrixType A(reinterpret_cast<Scalar *>(pa), *n, *n, *lda);
  if (UPLO(*uplo) == UP)
    Eigen::internal::triangular_adjoint_square_in_place<Eigen::Upper>(A);
  else
    Eigen::internal::triangular_adjoint_square_in_place<Eigen::Lower>(A);
}

// TRTRS solves op(A) X = B with a triangular matrix A, where op(A) is A, A^T or A^H, overwriting B with X.
// INFO = i > 0 means A(i,i) is exactly zero (DIAG = 'N'); B is then not modified.
EIGEN_LAPACK_FUNC(trtrs)
(const char *uplo, const char *trans, const char *diag, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *nrhs, RealScalar *pa,
 EIGEN_BLAS_INT *lda, RealScalar *pb, EIGEN_BLAS_INT *ldb, EIGEN_BLAS_INT *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (OP(*trans) == INVALID)
    *info = -2;
  else if (DIAG(*diag) == INVALID)
    *info = -3;
  else if (*n < 0)
    *info = -4;
  else if (*nrhs < 0)
    *info = -5;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -7;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -9;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "TRTRS", &e, kBlasNameLength);
  }

  if (*n == 0) return;

  MatrixType A(reinterpret_cast<Scalar *>(pa), *n, *n, *lda);
  MatrixType B(reinterpret_cast<Scalar *>(pb), *n, *nrhs, *ldb);
  const bool unit = DIAG(*diag) == UNIT;
  if (!unit) {
    *info = first_zero_diagonal_entry(A);
    if (*info > 0) return;
  }

  if (*nrhs == 0) return;

  const EIGEN_BLAS_INT op = OP(*trans);
  if (UPLO(*uplo) == UP) {
    if (unit)
      triangular_solve_in_place(A.triangularView<Eigen::UnitUpper>(), op, B);
    else
      triangular_solve_in_place(A.triangularView<Eigen::Upper>(), op, B);
  } else {
    if (unit)
      triangular_solve_in_place(A.triangularView<Eigen::UnitLower>(), op, B);
    else
      triangular_solve_in_place(A.triangularView<Eigen::Lower>(), op, B);
  }
}
