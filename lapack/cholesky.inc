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
EIGEN_LAPACK_FUNC(potrf)
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
    return xerbla_(SCALAR_SUFFIX_UP "POTRF", &e, kBlasNameLength);
  }

  MatrixType A(reinterpret_cast<Scalar *>(pa), *n, *n, *lda);
  EIGEN_BLAS_INT ret;
  if (UPLO(*uplo) == UP)
    ret = EIGEN_BLAS_INT(Eigen::internal::llt_inplace<Scalar, Eigen::Upper>::blocked(A));
  else
    ret = EIGEN_BLAS_INT(Eigen::internal::llt_inplace<Scalar, Eigen::Lower>::blocked(A));

  // The kernel stops only at a pivot <= 0; xPOTRF also stops at a NaN pivot. Before the kernel's stop the diagonal of
  // the factor holds the square roots of the pivots, so its first NaN there is the pivot xPOTRF reports. This does not
  // rely on the NaN reaching the later pivots, which a kernel that skips zero operands would prevent.
  const EIGEN_BLAS_INT stop = ret >= 0 ? ret : *n;
  for (EIGEN_BLAS_INT j = 0; j < stop; ++j) {
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
(const char *uplo, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *nrhs, RealScalar *pa, EIGEN_BLAS_INT *lda, RealScalar *pb,
 EIGEN_BLAS_INT *ldb, EIGEN_BLAS_INT *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*nrhs < 0)
    *info = -3;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -5;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -7;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "POTRS", &e, kBlasNameLength);
  }

  ConstMatrixType A(reinterpret_cast<const Scalar *>(pa), *n, *n, *lda);
  MatrixType B(reinterpret_cast<Scalar *>(pb), *n, *nrhs, *ldb);

  if (UPLO(*uplo) == UP) {
    A.triangularView<Eigen::Upper>().adjoint().solveInPlace(B);
    A.triangularView<Eigen::Upper>().solveInPlace(B);
  } else {
    A.triangularView<Eigen::Lower>().solveInPlace(B);
    A.triangularView<Eigen::Lower>().adjoint().solveInPlace(B);
  }
}

namespace {

using ScalarVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// LAPACK factors A = U D U^* from its last column, which is the lower factorization of J A J, J the reversal, with
// U = J L J. IPIV entry i of one is entry n-1-i of the other, its index p mapped to n+1-p and its sign kept.
EIGEN_BLAS_INT reversed_pivot(EIGEN_BLAS_INT p, EIGEN_BLAS_INT n) { return p > 0 ? n + 1 - p : -(n + 1 + p); }

// Solves D_k x = b for D_k = [a conj(c); c b] in rows i < j of B, scaled by c as xSYTRS/xHETRS do: denom =
// det(D_k) / |c|^2 stays O(1) where det(D_k) itself over- or underflows.
template <typename Mat>
void solve_block_2x2(Mat &B, Eigen::Index i, Eigen::Index j, RealScalar a, RealScalar b, const Scalar &c) {
  const Scalar conj_c = Eigen::numext::conj(c);
  const Scalar akm1 = Eigen::numext::divide(Scalar(a), conj_c);
  const Scalar ak = Eigen::numext::divide(Scalar(b), c);
  const RealScalar denom = Eigen::numext::real(akm1 * ak) - RealScalar(1);
  for (Eigen::Index col = 0; col < B.cols(); ++col) {
    const Scalar bkm1 = Eigen::numext::divide(B(i, col), conj_c);
    const Scalar bk = Eigen::numext::divide(B(j, col), c);
    B(i, col) = (ak * bkm1 - bk) / denom;
    B(j, col) = (akm1 * bk - bkm1) / denom;
  }
}

// Eigen's P A P^T = L D L^* -> xSYTRF's A = (P_1 L_1) ... (P_m L_m) D (...)^*, whose interchange s acts on rows s:n
// only, so later interchanges are undone on earlier columns, last block first. IPIV(k) = p: 1x1 block, rows k, p
// swapped. IPIV(k) = IPIV(k+1) = -p: 2x2 block, rows k+1, p swapped, D(k+1,k) in A(k+1,k). Returns xSYTRF's INFO.
template <typename Mat>
EIGEN_BLAS_INT bunch_kaufman_to_lapack(Mat &A, const TranspositionsType &transpositions, const ScalarVector &subdiag,
                                       EIGEN_BLAS_INT *ipiv) {
  const EIGEN_BLAS_INT n = EIGEN_BLAS_INT(A.rows());
  EIGEN_BLAS_INT info = 0;
  for (EIGEN_BLAS_INT k = 0; k < n;) {
    if (k + 1 < n && !Eigen::numext::is_exactly_zero(subdiag.coeff(k))) {
      A(k + 1, k) = subdiag.coeff(k);
      ipiv[k] = ipiv[k + 1] = -(transpositions.coeff(k + 1) + 1);
      k += 2;
    } else {
      const RealScalar d = Eigen::numext::real(A(k, k));
      if (info == 0 && (d == RealScalar(0) || (Eigen::numext::isnan)(d))) info = k + 1;
      ipiv[k] = transpositions.coeff(k) + 1;
      k += 1;
    }
  }
  for (EIGEN_BLAS_INT k = n - 1; k >= 0;) {
    const EIGEN_BLAS_INT first = ipiv[k] < 0 ? k - 1 : k;
    const EIGEN_BLAS_INT p = std::abs(ipiv[k]) - 1;
    if (p != k) A.row(k).head(first).swap(A.row(p).head(first));
    k = first - 1;
  }
  return info;
}

// SYTRF / HETRF.
void bunch_kaufman_factor(const char *name, const char *uplo, EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda,
                          EIGEN_BLAS_INT *ipiv, RealScalar *pwork, EIGEN_BLAS_INT *lwork, EIGEN_BLAS_INT *info) {
  *info = 0;
  const bool query = *lwork == -1;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -4;
  else if (*lwork < 1 && !query)
    *info = -7;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(name, &e, kBlasNameLength);
  }

  // The factorization allocates its own workspace, so the minimum LWORK is also the optimal one.
  *reinterpret_cast<Scalar *>(pwork) = Scalar(1);
  if (query || *n == 0) return;

  MatrixType A(reinterpret_cast<Scalar *>(pa), *n, *n, *lda);
  TranspositionsType transpositions(*n);
  ScalarVector subdiag(*n);
  PlainMatrixType workspace;
  if (UPLO(*uplo) == LO) {
    Eigen::internal::bunch_kaufman<Eigen::Lower>::blocked(A, transpositions, subdiag, workspace);
    *info = bunch_kaufman_to_lapack(A, transpositions, subdiag, ipiv);
  } else {
    // J A J in place: the upper triangle becomes the lower one that the kernel factors, and the strictly lower
    // triangle, which the kernel does not touch, returns to its place when the reversal is undone. The last of equal
    // pivot candidates in the reversed order is xSYTRF's first in the original order.
    A.reverseInPlace();
    Eigen::internal::bunch_kaufman<Eigen::Lower>::blocked</*LastOnTies=*/true>(A, transpositions, subdiag, workspace);
    const EIGEN_BLAS_INT reversed_info = bunch_kaufman_to_lapack(A, transpositions, subdiag, ipiv);
    A.reverseInPlace();
    std::reverse(ipiv, ipiv + *n);
    for (EIGEN_BLAS_INT i = 0; i < *n; ++i) ipiv[i] = reversed_pivot(ipiv[i], *n);
    *info = reversed_info == 0 ? 0 : *n + 1 - reversed_info;
  }
}

// SYTRS / HETRS.
void bunch_kaufman_solve(const char *name, const char *uplo, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *nrhs, RealScalar *pa,
                         EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv, RealScalar *pb, EIGEN_BLAS_INT *ldb,
                         EIGEN_BLAS_INT *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
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
    return xerbla_(name, &e, kBlasNameLength);
  }
  if (*n == 0 || *nrhs == 0) return;

  // The blocks of xSYTRF's product form one at a time, as xSYTRS does, without workspace:
  // A = (P_1 L_1) ... D ... (P_1 L_1)^* for UPLO = 'L', blocks first to last; A = (P_n U_n) ... for 'U', last to first.
  const EIGEN_BLAS_INT size = *n;
  ConstMatrixType A(reinterpret_cast<const Scalar *>(pa), size, size, *lda);
  MatrixType B(reinterpret_cast<Scalar *>(pb), size, *nrhs, *ldb);
  const auto swap_rows = [&B](EIGEN_BLAS_INT i, EIGEN_BLAS_INT j) {
    if (i != j) B.row(i).swap(B.row(j));
  };
  const auto scale_row = [&A, &B](EIGEN_BLAS_INT k) { B.row(k) *= RealScalar(1) / Eigen::numext::real(A(k, k)); };
  if (UPLO(*uplo) == LO) {
    // B <- D^-1 L^-1 B.
    for (EIGEN_BLAS_INT k = 0; k < size;) {
      const EIGEN_BLAS_INT s = ipiv[k] > 0 ? 1 : 2, tail = size - k - s;
      swap_rows(k + s - 1, std::abs(ipiv[k]) - 1);
      B.bottomRows(tail).noalias() -= A.block(k + s, k, tail, s) * B.middleRows(k, s);
      if (s == 1)
        scale_row(k);
      else
        solve_block_2x2(B, k, k + 1, Eigen::numext::real(A(k, k)), Eigen::numext::real(A(k + 1, k + 1)), A(k + 1, k));
      k += s;
    }
    // B <- L^-* B.
    for (EIGEN_BLAS_INT k = size - 1; k >= 0;) {
      const EIGEN_BLAS_INT s = ipiv[k] > 0 ? 1 : 2, first = k - s + 1, tail = size - k - 1;
      B.middleRows(first, s).noalias() -= A.block(k + 1, first, tail, s).adjoint() * B.bottomRows(tail);
      swap_rows(k, std::abs(ipiv[k]) - 1);
      k -= s;
    }
  } else {
    // B <- D^-1 U^-1 B.
    for (EIGEN_BLAS_INT k = size - 1; k >= 0;) {
      const EIGEN_BLAS_INT s = ipiv[k] > 0 ? 1 : 2, first = k - s + 1;
      swap_rows(first, std::abs(ipiv[k]) - 1);
      B.topRows(first).noalias() -= A.block(0, first, first, s) * B.middleRows(first, s);
      if (s == 1)
        scale_row(k);
      else
        solve_block_2x2(B, first, k, Eigen::numext::real(A(first, first)), Eigen::numext::real(A(k, k)),
                        Eigen::numext::conj(A(first, k)));
      k -= s;
    }
    // B <- U^-* B.
    for (EIGEN_BLAS_INT k = 0; k < size;) {
      const EIGEN_BLAS_INT s = ipiv[k] > 0 ? 1 : 2;
      B.middleRows(k, s).noalias() -= A.block(0, k, k, s).adjoint() * B.topRows(k);
      swap_rows(k, std::abs(ipiv[k]) - 1);
      k += s;
    }
  }
}

// P A P^T = L L^* from LDLT's panel kernel, pivoting on max diag(S_k) as xPSTRF does. Unlike
// ldlt_inplace<Lower>::blocked() it stops after the panel holding the first pivot d_k <= dstop or NaN, k > 0 (the
// first pivot is the positive max diag(A), taken whatever dstop): O(n^2 (r + nb)) for rank r. L = l sqrt(d) is formed
// from LDLT's l = w / d, which underflows where xPSTRF's w / sqrt(d) does not only if diag(A) spans the exponent
// range, and is NaN where an infinite pivot makes xPSTRF's zero. Returns rank = that k, and applies the interchanges to
// piv, which holds 1..n on entry.
template <typename Mat>
EIGEN_BLAS_INT pivoted_cholesky_lower(Mat &mat, RealScalar dstop, EIGEN_BLAS_INT *piv) {
  using Eigen::Index;
  const Index size = mat.rows();
  TranspositionsType transpositions(size);
  ScalarVector temp = mat.diagonal().real().template cast<Scalar>();
  PlainMatrixType workspace;
  Eigen::internal::SignMatrix sign = Eigen::internal::ZeroSign;
  const Index blocksize = Eigen::internal::ldlt_blocksize<PlainMatrixType>(size);
  const Index nb = blocksize > 0 ? blocksize : size;
  if (blocksize > 0) workspace.resize(size, nb);

  Index rank = size;
  Index factored = 0;
  while (factored < size && rank == size) {
    const Index k0 = factored;
    const Index kb = Eigen::numext::mini(nb, size - k0);
    const Index k1 = k0 + kb;
    Eigen::internal::ldlt_inplace<Eigen::Lower>::factor_panel</*PivotOnLargestValue=*/true>(mat, k0, kb, transpositions,
                                                                                            temp, sign);
    for (Index k = k0; k < k1; ++k) {
      const Index p = transpositions.coeff(k);
      if (p != k) mat.row(k).head(k0).swap(mat.row(p).head(k0));
    }
    for (Index k = k0; k < k1 && rank == size; ++k) {
      const RealScalar d = Eigen::numext::real(mat.coeff(k, k));
      if ((Eigen::numext::isnan)(d) || (k > 0 && d <= dstop)) rank = k;
    }
    factored = k1;
    const Index rs = size - k1;
    if (rank == size && rs > 0) {
      auto L21 = mat.block(k1, k0, rs, kb);
      auto W = workspace.topLeftCorner(rs, kb);
      W.noalias() = L21 * mat.diagonal().real().segment(k0, kb).asDiagonal();
      mat.block(k1, k1, rs, rs).template triangularView<Eigen::Lower>() -= L21 * W.adjoint();
    }
  }

  for (Index k = 0; k < factored; ++k) Eigen::numext::swap(piv[k], piv[transpositions.coeff(k)]);
  for (Index k = 0; k < rank; ++k) {
    const RealScalar s = Eigen::numext::sqrt(Eigen::numext::real(mat.coeff(k, k)));
    mat.coeffRef(k, k) = Scalar(s);
    mat.col(k).tail(size - k - 1) *= s;
  }
  return EIGEN_BLAS_INT(rank);
}

}  // namespace

// PSTRF computes the Cholesky factorization with complete pivoting of a positive semidefinite matrix,
// P^T A P = U^* U or L L^*, stopping at the first pivot <= TOL (N * eps * max(diag(A)) if TOL < 0). RANK is the
// number of pivots taken, and INFO = 1 if it is less than N. Rows and columns RANK+1:N of the factor are
// unspecified, as in LAPACK.
EIGEN_LAPACK_FUNC(pstrf)
(const char *uplo, EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *piv, EIGEN_BLAS_INT *rank,
 RealScalar *tol, RealScalar * /*work*/, EIGEN_BLAS_INT *info) {
  *info = 0;
  if (UPLO(*uplo) == INVALID)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    *info = -4;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "PSTRF", &e, kBlasNameLength);
  }
  *rank = 0;
  if (*n == 0) return;

  MatrixType A(reinterpret_cast<Scalar *>(pa), *n, *n, *lda);
  for (EIGEN_BLAS_INT i = 0; i < *n; ++i) piv[i] = i + 1;
  // LAPACK factors nothing unless the largest diagonal entry, its first pivot, is positive. The real routines scan
  // from A(1,1), so a NaN there stops them; the complex ones take MAXLOC, which skips NaNs.
  RealScalar max_diagonal = Eigen::numext::real(A(0, 0));
  for (EIGEN_BLAS_INT i = 1; i < *n; ++i) {
    const RealScalar d = Eigen::numext::real(A(i, i));
    if (d > max_diagonal || (ISCOMPLEX && (Eigen::numext::isnan)(max_diagonal))) max_diagonal = d;
  }
  if (!(max_diagonal > RealScalar(0))) {
    *info = 1;
    return;
  }
  const RealScalar dstop =
      *tol < RealScalar(0) ? RealScalar(*n) * Eigen::internal::xlamch<RealScalar>('e') * max_diagonal : *tol;

  if (UPLO(*uplo) == LO) {
    *rank = pivoted_cholesky_lower(A, dstop, piv);
  } else {
    // The lower factorization of A^T stores U = L^T in the upper triangle of A: P^T A P = U^* U, as LDLT<Upper>.
    Eigen::Transpose<MatrixType> At(A);
    *rank = pivoted_cholesky_lower(At, dstop, piv);
  }
  if (*rank < *n) *info = 1;
}

#if ISCOMPLEX
// HETRF computes the Bunch-Kaufman factorization A = U D U^H or L D L^H of a complex Hermitian matrix.
EIGEN_LAPACK_FUNC(hetrf)
(const char *uplo, EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv, RealScalar *pwork,
 EIGEN_BLAS_INT *lwork, EIGEN_BLAS_INT *info) {
  bunch_kaufman_factor(SCALAR_SUFFIX_UP "HETRF", uplo, n, pa, lda, ipiv, pwork, lwork, info);
}

// HETRS solves A X = B with the factorization computed by HETRF.
EIGEN_LAPACK_FUNC(hetrs)
(const char *uplo, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *nrhs, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv,
 RealScalar *pb, EIGEN_BLAS_INT *ldb, EIGEN_BLAS_INT *info) {
  bunch_kaufman_solve(SCALAR_SUFFIX_UP "HETRS", uplo, n, nrhs, pa, lda, ipiv, pb, ldb, info);
}
#else
// SYTRF computes the Bunch-Kaufman factorization A = U D U^T or L D L^T of a real symmetric matrix.
EIGEN_LAPACK_FUNC(sytrf)
(const char *uplo, EIGEN_BLAS_INT *n, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv, RealScalar *pwork,
 EIGEN_BLAS_INT *lwork, EIGEN_BLAS_INT *info) {
  bunch_kaufman_factor(SCALAR_SUFFIX_UP "SYTRF", uplo, n, pa, lda, ipiv, pwork, lwork, info);
}

// SYTRS solves A X = B with the factorization computed by SYTRF.
EIGEN_LAPACK_FUNC(sytrs)
(const char *uplo, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *nrhs, RealScalar *pa, EIGEN_BLAS_INT *lda, EIGEN_BLAS_INT *ipiv,
 RealScalar *pb, EIGEN_BLAS_INT *ldb, EIGEN_BLAS_INT *info) {
  bunch_kaufman_solve(SCALAR_SUFFIX_UP "SYTRS", uplo, n, nrhs, pa, lda, ipiv, pb, ldb, info);
}
#endif

// POTRI computes the inverse of a symmetric (Hermitian) positive definite matrix from its Cholesky factor computed by
// POTRF: inv(A) = inv(U) inv(U)^H (UPLO = 'U') or inv(L)^H inv(L) (UPLO = 'L'), stored in the same triangle.
// INFO = i > 0 means the factor's diagonal entry (i,i) is zero, and the inverse is not computed.
EIGEN_LAPACK_FUNC(potri)
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
    return xerbla_(SCALAR_SUFFIX_UP "POTRI", &e, kBlasNameLength);
  }

  if (*n == 0) return;

  EIGEN_BLAS_FUNC_NAME(trtri)(uplo, "N", n, pa, lda, info);
  if (*info > 0) return;
  EIGEN_BLAS_FUNC_NAME(lauum)(uplo, n, pa, lda, info);
}
