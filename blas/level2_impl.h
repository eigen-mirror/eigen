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

template <typename Index, typename Scalar, int StorageOrder, bool ConjugateLhs, bool ConjugateRhs>
struct general_matrix_vector_product_wrapper {
  static void run(Index rows, Index cols, const Scalar *lhs, Index lhsStride, const Scalar *rhs, Index rhsIncr,
                  Scalar *res, Index resIncr, Scalar alpha) {
    using LhsMapper = Eigen::internal::const_blas_data_mapper<Scalar, Index, StorageOrder>;
    using RhsMapper = Eigen::internal::const_blas_data_mapper<Scalar, Index, Eigen::RowMajor>;

    Eigen::internal::general_matrix_vector_product<Index, Scalar, LhsMapper, StorageOrder, ConjugateLhs, Scalar,
                                                   RhsMapper, ConjugateRhs>::run(rows, cols, LhsMapper(lhs, lhsStride),
                                                                                 RhsMapper(rhs, rhsIncr), res, resIncr,
                                                                                 alpha);
  }
};

namespace {

// y += alpha A x for the n-by-n selfadjoint band matrix A with k super-diagonals in BLAS band storage: the upper
// triangle holds A(i, j) at a[(k + i - j) + j * lda], the lower one at a[(i - j) + j * lda]. The imaginary part of
// the diagonal is not read.
void selfadjoint_band_product(bool upper, Eigen::Index n, Eigen::Index k, const Scalar *a, Eigen::Index lda,
                              const Scalar *x, Scalar *y, const Scalar &alpha) {
  using Eigen::Index;
  if (k >= 8) {
    // Vectorized path: use Eigen Map segments for the inner band operations.
    ConstMatrixType band(a, k + 1, n, lda);
    if (upper) {
      for (Index j = 0; j < n; ++j) {
        Index start = (std::max)(Index(0), j - k);
        Index len = j - start;
        Index offset = k - (j - start);
        Scalar temp1 = alpha * x[j];
        y[j] += Scalar(Eigen::numext::real(band(k, j))) * temp1;
        if (len > 0) {
          make_vector(y + start, len) += temp1 * band.col(j).segment(offset, len);
          y[j] += alpha * band.col(j).segment(offset, len).dot(make_vector(x + start, len));
        }
      }
    } else {
      for (Index j = 0; j < n; ++j) {
        Index len = (std::min)(n - 1, j + k) - j;
        Scalar temp1 = alpha * x[j];
        y[j] += Scalar(Eigen::numext::real(band(0, j))) * temp1;
        if (len > 0) {
          make_vector(y + j + 1, len) += temp1 * band.col(j).segment(1, len);
          y[j] += alpha * band.col(j).segment(1, len).dot(make_vector(x + j + 1, len));
        }
      }
    }
  } else {
    // Scalar path: for narrow bandwidth, avoid Map overhead.
    if (upper) {
      for (Index j = 0; j < n; ++j) {
        Scalar temp1 = alpha * x[j];
        Scalar temp2 = Scalar(0);
        for (Index i = (std::max)(Index(0), j - k); i < j; ++i) {
          Scalar aij = a[(k + i - j) + j * lda];
          y[i] += temp1 * aij;
          temp2 += Eigen::numext::conj(aij) * x[i];
        }
        y[j] += Scalar(Eigen::numext::real(a[k + j * lda])) * temp1 + alpha * temp2;
      }
    } else {
      for (Index j = 0; j < n; ++j) {
        Scalar temp1 = alpha * x[j];
        Scalar temp2 = Scalar(0);
        y[j] += Scalar(Eigen::numext::real(a[j * lda])) * temp1;
        for (Index i = j + 1; i <= (std::min)(n - 1, j + k); ++i) {
          Scalar aij = a[(i - j) + j * lda];
          y[i] += temp1 * aij;
          temp2 += Eigen::numext::conj(aij) * x[i];
        }
        y[j] += alpha * temp2;
      }
    }
  }
}

// y += alpha A x for the n-by-n selfadjoint A packed column by column: column j holds rows 0..j of the upper triangle
// or rows j..n-1 of the lower one. The imaginary part of the diagonal is not read.
void selfadjoint_packed_product(bool upper, Eigen::Index n, const Scalar *ap, const Scalar *x, Scalar *y,
                                const Scalar &alpha) {
  using Eigen::Index;
  Index kk = 0;  // the start of column j
  if (upper) {
    for (Index j = 0; j < n; ++j) {
      Scalar temp1 = alpha * x[j];
      y[j] += Scalar(Eigen::numext::real(ap[kk + j])) * temp1;
      if (j > 0) {
        make_vector(y, j) += temp1 * make_vector(ap + kk, j);
        y[j] += alpha * make_vector(ap + kk, j).dot(make_vector(x, j));
      }
      kk += j + 1;
    }
  } else {
    for (Index j = 0; j < n; ++j) {
      Index len = n - j - 1;
      Scalar temp1 = alpha * x[j];
      y[j] += Scalar(Eigen::numext::real(ap[kk])) * temp1;
      if (len > 0) {
        make_vector(y + j + 1, len) += temp1 * make_vector(ap + kk + 1, len);
        y[j] += alpha * make_vector(ap + kk + 1, len).dot(make_vector(x + j + 1, len));
      }
      kk += n - j;
    }
  }
}

}  // namespace

EIGEN_BLAS_FUNC(gemv)
(const char *opa, const EIGEN_BLAS_INT *m, const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *pa,
 const EIGEN_BLAS_INT *lda, const RealScalar *pb, const EIGEN_BLAS_INT *incb, const RealScalar *pbeta, RealScalar *pc,
 const EIGEN_BLAS_INT *incc) {
  using functype = void (*)(Eigen::Index, Eigen::Index, const Scalar *, Eigen::Index, const Scalar *, Eigen::Index,
                            Scalar *, Eigen::Index, Scalar);
  static const functype func[4] = {
      // array index: NOTR
      (general_matrix_vector_product_wrapper<Eigen::Index, Scalar, Eigen::ColMajor, false, false>::run),
      // array index: TR
      (general_matrix_vector_product_wrapper<Eigen::Index, Scalar, Eigen::RowMajor, false, false>::run),
      // array index: ADJ
      (general_matrix_vector_product_wrapper<Eigen::Index, Scalar, Eigen::RowMajor, Conj, false>::run), 0};

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *b = reinterpret_cast<const Scalar *>(pb);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  // check arguments
  EIGEN_BLAS_INT info = 0;
  if (OP(*opa) == INVALID)
    info = 1;
  else if (*m < 0)
    info = 2;
  else if (*n < 0)
    info = 3;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 6;
  else if (*incb == 0)
    info = 8;
  else if (*incc == 0)
    info = 11;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GEMV ", &info, kBlasNameLength);

  if (*m == 0 || *n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  Eigen::Index actual_m = *m;
  Eigen::Index actual_n = *n;
  int code = OP(*opa);
  if (code != NOTR) std::swap(actual_m, actual_n);

  compact_vector<Scalar> actual_c(c, actual_m, *incc);
  scale_vector(actual_c.data(), actual_m, beta);
  if (alpha == Scalar(0)) return;

  compact_vector<const Scalar> actual_b(b, actual_n, *incb);
  func[code](actual_m, actual_n, a, *lda, actual_b.data(), 1, actual_c.data(), 1, alpha);
}

EIGEN_BLAS_FUNC(trsv)
(const char *uplo, const char *opa, const char *diag, const EIGEN_BLAS_INT *n, const RealScalar *pa,
 const EIGEN_BLAS_INT *lda, RealScalar *pb, const EIGEN_BLAS_INT *incb) {
  using functype = void (*)(Eigen::Index, const Scalar *, Eigen::Index, Scalar *);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::OnTheLeft;
  using Eigen::RowMajor;
  using Eigen::UnitDiag;
  using Eigen::Upper;
  static const functype func[16] = {// array index: NOTR  | (UP << 2) | (NUNIT << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Upper | 0, false, ColMajor>::run),
                                    // array index: TR    | (UP << 2) | (NUNIT << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Lower | 0, false, RowMajor>::run),
                                    // array index: ADJ   | (UP << 2) | (NUNIT << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Lower | 0, Conj, RowMajor>::run),
                                    0,
                                    // array index: NOTR  | (LO << 2) | (NUNIT << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Lower | 0, false, ColMajor>::run),
                                    // array index: TR    | (LO << 2) | (NUNIT << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Upper | 0, false, RowMajor>::run),
                                    // array index: ADJ   | (LO << 2) | (NUNIT << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Upper | 0, Conj, RowMajor>::run),
                                    0,
                                    // array index: NOTR  | (UP << 2) | (UNIT  << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Upper | UnitDiag, false, ColMajor>::run),
                                    // array index: TR    | (UP << 2) | (UNIT  << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Lower | UnitDiag, false, RowMajor>::run),
                                    // array index: ADJ   | (UP << 2) | (UNIT  << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Lower | UnitDiag, Conj, RowMajor>::run),
                                    0,
                                    // array index: NOTR  | (LO << 2) | (UNIT  << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Lower | UnitDiag, false, ColMajor>::run),
                                    // array index: TR    | (LO << 2) | (UNIT  << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Upper | UnitDiag, false, RowMajor>::run),
                                    // array index: ADJ   | (LO << 2) | (UNIT  << 3)
                                    (Eigen::internal::triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft,
                                                                              Upper | UnitDiag, Conj, RowMajor>::run),
                                    0};

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  Scalar *b = reinterpret_cast<Scalar *>(pb);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*opa) == INVALID)
    info = 2;
  else if (DIAG(*diag) == INVALID)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 6;
  else if (*incb == 0)
    info = 8;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TRSV ", &info, kBlasNameLength);

  if (*n == 0) return;

  compact_vector<Scalar> actual_b(b, *n, *incb);
  int code = OP(*opa) | (UPLO(*uplo) << 2) | (DIAG(*diag) << 3);
  func[code](*n, a, *lda, actual_b.data());
}

EIGEN_BLAS_FUNC(trmv)
(const char *uplo, const char *opa, const char *diag, const EIGEN_BLAS_INT *n, const RealScalar *pa,
 const EIGEN_BLAS_INT *lda, RealScalar *pb, const EIGEN_BLAS_INT *incb) {
  using functype = void (*)(Eigen::Index, Eigen::Index, const Scalar *, Eigen::Index, const Scalar *, Eigen::Index,
                            Scalar *, Eigen::Index, const Scalar &);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::RowMajor;
  using Eigen::UnitDiag;
  using Eigen::Upper;
  static const functype func[16] = {
      // array index: NOTR  | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Upper | 0, Scalar, false, Scalar, false,
                                                         ColMajor>::run),
      // array index: TR    | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Lower | 0, Scalar, false, Scalar, false,
                                                         RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Lower | 0, Scalar, Conj, Scalar, false,
                                                         RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Lower | 0, Scalar, false, Scalar, false,
                                                         ColMajor>::run),
      // array index: TR    | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Upper | 0, Scalar, false, Scalar, false,
                                                         RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Upper | 0, Scalar, Conj, Scalar, false,
                                                         RowMajor>::run),
      0,
      // array index: NOTR  | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Upper | UnitDiag, Scalar, false, Scalar, false,
                                                         ColMajor>::run),
      // array index: TR    | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Lower | UnitDiag, Scalar, false, Scalar, false,
                                                         RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Lower | UnitDiag, Scalar, Conj, Scalar, false,
                                                         RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Lower | UnitDiag, Scalar, false, Scalar, false,
                                                         ColMajor>::run),
      // array index: TR    | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Upper | UnitDiag, Scalar, false, Scalar, false,
                                                         RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::triangular_matrix_vector_product<Eigen::Index, Upper | UnitDiag, Scalar, Conj, Scalar, false,
                                                         RowMajor>::run),
      0};

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  Scalar *b = reinterpret_cast<Scalar *>(pb);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*opa) == INVALID)
    info = 2;
  else if (DIAG(*diag) == INVALID)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 6;
  else if (*incb == 0)
    info = 8;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TRMV ", &info, kBlasNameLength);

  if (*n == 0) return;

  compact_vector<Scalar> actual_b(b, *n, *incb);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(*n);
  int code = OP(*opa) | (UPLO(*uplo) << 2) | (DIAG(*diag) << 3);
  func[code](*n, *n, a, *lda, actual_b.data(), 1, res.data(), 1, Scalar(1));
  make_vector(actual_b.data(), *n) = res;
}

/**  GBMV  performs one of the matrix-vector operations
 *
 *     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are vectors and A is an
 *  m by n band matrix, with kl sub-diagonals and ku super-diagonals.
 */
EIGEN_BLAS_FUNC(gbmv)
(char *trans, EIGEN_BLAS_INT *m, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *kl, EIGEN_BLAS_INT *ku, RealScalar *palpha,
 RealScalar *pa, EIGEN_BLAS_INT *lda, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *pbeta, RealScalar *py,
 EIGEN_BLAS_INT *incy) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  EIGEN_BLAS_INT info = 0;
  if (OP(*trans) == INVALID)
    info = 1;
  else if (*m < 0)
    info = 2;
  else if (*n < 0)
    info = 3;
  else if (*kl < 0)
    info = 4;
  else if (*ku < 0)
    info = 5;
  else if (*lda <= *kl || *lda - *kl <= *ku)  // lda < kl + ku + 1, without overflow
    info = 8;
  else if (*incx == 0)
    info = 10;
  else if (*incy == 0)
    info = 13;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GBMV ", &info, kBlasNameLength);

  if (*m == 0 || *n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  Eigen::Index actual_m = *m;
  Eigen::Index actual_n = *n;
  if (OP(*trans) != NOTR) std::swap(actual_m, actual_n);

  compact_vector<Scalar> compact_y(y, actual_m, *incy);
  Scalar *actual_y = compact_y.data();
  scale_vector(actual_y, actual_m, beta);
  if (alpha == Scalar(0)) return;

  compact_vector<const Scalar> compact_x(x, actual_n, *incx);
  const Scalar *actual_x = compact_x.data();
  ConstMatrixType mat_coeffs(a, Eigen::Index(*kl) + Eigen::Index(*ku) + 1, *n, *lda);

  Eigen::Index nb = (std::min)(Eigen::Index(*n), Eigen::Index(*m) + Eigen::Index(*ku));
  for (Eigen::Index j = 0; j < nb; ++j) {
    Eigen::Index start = (std::max)(Eigen::Index(0), j - Eigen::Index(*ku));
    Eigen::Index end = (std::min)(Eigen::Index(*m) - 1, j + Eigen::Index(*kl));
    Eigen::Index len = end - start + 1;
    Eigen::Index offset = Eigen::Index(*ku) - j + start;
    if (OP(*trans) == NOTR)
      make_vector(actual_y + start, len) += (alpha * actual_x[j]) * mat_coeffs.col(j).segment(offset, len);
    else if (OP(*trans) == TR)
      actual_y[j] +=
          alpha * (mat_coeffs.col(j).segment(offset, len).transpose() * make_vector(actual_x + start, len)).value();
    else
      actual_y[j] +=
          alpha * (mat_coeffs.col(j).segment(offset, len).adjoint() * make_vector(actual_x + start, len)).value();
  }
}

/**  TBMV  performs one of the matrix-vector operations
 *
 *     x := A*x,   or   x := A'*x,   or   x := conjg(A')*x,
 *
 *  where x is an n element vector and A is an n by n unit, or non-unit,
 *  upper or lower triangular band matrix, with ( k + 1 ) diagonals.
 *
 *  Band storage: upper triangle stores A[i,j] at a[(k+i-j) + j*lda],
 *  lower triangle stores A[i,j] at a[(i-j) + j*lda].
 */
EIGEN_BLAS_FUNC(tbmv)
(char *uplo, char *opa, char *diag, EIGEN_BLAS_INT *n_in, EIGEN_BLAS_INT *k_in, RealScalar *pa, EIGEN_BLAS_INT *lda_in,
 RealScalar *px, EIGEN_BLAS_INT *incx) {
  Scalar *a = reinterpret_cast<Scalar *>(pa);
  Scalar *x = reinterpret_cast<Scalar *>(px);
  const Eigen::Index n = *n_in;
  const Eigen::Index k = *k_in;
  const Eigen::Index lda = *lda_in;

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*opa) == INVALID)
    info = 2;
  else if (DIAG(*diag) == INVALID)
    info = 3;
  else if (n < 0)
    info = 4;
  else if (k < 0)
    info = 5;
  else if (lda <= k)
    info = 7;
  else if (*incx == 0)
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TBMV ", &info, kBlasNameLength);

  if (n == 0) return;

  compact_vector<Scalar> compact_x(x, n, *incx);
  Scalar *actual_x = compact_x.data();

  bool upper = (UPLO(*uplo) == UP);
  int op = OP(*opa);
  bool unit = (DIAG(*diag) == UNIT);

  if (k >= 8) {
    // Vectorized path: use Eigen Map segments for the inner band operations.
    ConstMatrixType band(a, k + 1, n, lda);
    if (op == NOTR) {
      if (upper) {
        for (Eigen::Index j = 0; j < n; ++j) {
          if (actual_x[j] != Scalar(0)) {
            Eigen::Index start = (std::max)(Eigen::Index(0), j - k);
            Eigen::Index len = j - start;
            Eigen::Index offset = k - (j - start);
            Scalar temp = actual_x[j];
            if (len > 0) make_vector(actual_x + start, len) += temp * band.col(j).segment(offset, len);
            if (!unit) actual_x[j] = temp * band(k, j);
          }
        }
      } else {
        for (Eigen::Index j = n - 1; j >= 0; --j) {
          if (actual_x[j] != Scalar(0)) {
            Eigen::Index len = (std::min)(n - 1, j + k) - j;
            Scalar temp = actual_x[j];
            if (len > 0) make_vector(actual_x + j + 1, len) += temp * band.col(j).segment(1, len);
            if (!unit) actual_x[j] = temp * band(0, j);
          }
        }
      }
    } else if (op == TR) {
      if (upper) {
        for (Eigen::Index j = n - 1; j >= 0; --j) {
          Eigen::Index start = (std::max)(Eigen::Index(0), j - k);
          Eigen::Index len = j - start;
          Eigen::Index offset = k - (j - start);
          Scalar temp = actual_x[j];
          if (!unit) temp *= band(k, j);
          if (len > 0)
            temp += (band.col(j).segment(offset, len).cwiseProduct(make_vector(actual_x + start, len))).sum();
          actual_x[j] = temp;
        }
      } else {
        for (Eigen::Index j = 0; j < n; ++j) {
          Eigen::Index len = (std::min)(n - 1, j + k) - j;
          Scalar temp = actual_x[j];
          if (!unit) temp *= band(0, j);
          if (len > 0) temp += (band.col(j).segment(1, len).cwiseProduct(make_vector(actual_x + j + 1, len))).sum();
          actual_x[j] = temp;
        }
      }
    } else {
      // Conjugate transpose: .dot() computes conj(lhs) . rhs.
      if (upper) {
        for (Eigen::Index j = n - 1; j >= 0; --j) {
          Eigen::Index start = (std::max)(Eigen::Index(0), j - k);
          Eigen::Index len = j - start;
          Eigen::Index offset = k - (j - start);
          Scalar temp = actual_x[j];
          if (!unit) temp *= Eigen::numext::conj(band(k, j));
          if (len > 0) temp += band.col(j).segment(offset, len).dot(make_vector(actual_x + start, len));
          actual_x[j] = temp;
        }
      } else {
        for (Eigen::Index j = 0; j < n; ++j) {
          Eigen::Index len = (std::min)(n - 1, j + k) - j;
          Scalar temp = actual_x[j];
          if (!unit) temp *= Eigen::numext::conj(band(0, j));
          if (len > 0) temp += band.col(j).segment(1, len).dot(make_vector(actual_x + j + 1, len));
          actual_x[j] = temp;
        }
      }
    }
  } else {
    // Scalar path: for narrow bandwidth, avoid Map overhead.
    if (op == NOTR) {
      if (upper) {
        for (Eigen::Index j = 0; j < n; ++j) {
          if (actual_x[j] != Scalar(0)) {
            Scalar temp = actual_x[j];
            for (Eigen::Index i = (std::max)(Eigen::Index(0), j - k); i < j; ++i)
              actual_x[i] += temp * a[(k + i - j) + j * lda];
            if (!unit) actual_x[j] = temp * a[k + j * lda];
          }
        }
      } else {
        for (Eigen::Index j = n - 1; j >= 0; --j) {
          if (actual_x[j] != Scalar(0)) {
            Scalar temp = actual_x[j];
            for (Eigen::Index i = j + 1; i <= (std::min)(n - 1, j + k); ++i) actual_x[i] += temp * a[(i - j) + j * lda];
            if (!unit) actual_x[j] = temp * a[j * lda];
          }
        }
      }
    } else {
      // Transpose or conjugate transpose.
      auto maybe_conj = [op](Scalar val) -> Scalar { return op == ADJ ? Eigen::numext::conj(val) : val; };
      if (upper) {
        for (Eigen::Index j = n - 1; j >= 0; --j) {
          Scalar temp = actual_x[j];
          if (!unit) temp *= maybe_conj(a[k + j * lda]);
          for (Eigen::Index i = (std::max)(Eigen::Index(0), j - k); i < j; ++i)
            temp += maybe_conj(a[(k + i - j) + j * lda]) * actual_x[i];
          actual_x[j] = temp;
        }
      } else {
        for (Eigen::Index j = 0; j < n; ++j) {
          Scalar temp = actual_x[j];
          if (!unit) temp *= maybe_conj(a[j * lda]);
          for (Eigen::Index i = j + 1; i <= (std::min)(n - 1, j + k); ++i)
            temp += maybe_conj(a[(i - j) + j * lda]) * actual_x[i];
          actual_x[j] = temp;
        }
      }
    }
  }
}

/**  DTBSV  solves one of the systems of equations
 *
 *     A*x = b,   or   A'*x = b,
 *
 *  where b and x are n element vectors and A is an n by n unit, or
 *  non-unit, upper or lower triangular band matrix, with ( k + 1 )
 *  diagonals.
 *
 *  No test for singularity or near-singularity is included in this
 *  routine. Such tests must be performed before calling this routine.
 */
EIGEN_BLAS_FUNC(tbsv)
(char *uplo, char *op, char *diag, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *k, RealScalar *pa, EIGEN_BLAS_INT *lda,
 RealScalar *px, EIGEN_BLAS_INT *incx) {
  using functype = void (*)(Eigen::Index, Eigen::Index, const Scalar *, Eigen::Index, Scalar *);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::RowMajor;
  using Eigen::UnitDiag;
  using Eigen::Upper;
  static const functype func[16] = {
      // array index: NOTR  | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Upper | 0, Scalar, false, Scalar, ColMajor>::run),
      // array index: TR    | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Lower | 0, Scalar, false, Scalar, RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Lower | 0, Scalar, Conj, Scalar, RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Lower | 0, Scalar, false, Scalar, ColMajor>::run),
      // array index: TR    | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Upper | 0, Scalar, false, Scalar, RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Upper | 0, Scalar, Conj, Scalar, RowMajor>::run),
      0,
      // array index: NOTR  | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Upper | UnitDiag, Scalar, false, Scalar,
                                                       ColMajor>::run),
      // array index: TR    | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Lower | UnitDiag, Scalar, false, Scalar,
                                                       RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Lower | UnitDiag, Scalar, Conj, Scalar,
                                                       RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Lower | UnitDiag, Scalar, false, Scalar,
                                                       ColMajor>::run),
      // array index: TR    | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Upper | UnitDiag, Scalar, false, Scalar,
                                                       RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::band_solve_triangular_selector<Eigen::Index, Upper | UnitDiag, Scalar, Conj, Scalar,
                                                       RowMajor>::run),
      0,
  };

  Scalar *a = reinterpret_cast<Scalar *>(pa);
  Scalar *x = reinterpret_cast<Scalar *>(px);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*op) == INVALID)
    info = 2;
  else if (DIAG(*diag) == INVALID)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*k < 0)
    info = 5;
  else if (*lda <= *k)
    info = 7;
  else if (*incx == 0)
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TBSV ", &info, kBlasNameLength);

  if (*n == 0 || (*k == 0 && DIAG(*diag) == UNIT)) return;

  compact_vector<Scalar> actual_x(x, *n, *incx);
  int code = OP(*op) | (UPLO(*uplo) << 2) | (DIAG(*diag) << 3);
  func[code](*n, *k, a, *lda, actual_x.data());
}

/**  DTPMV  performs one of the matrix-vector operations
 *
 *     x := A*x,   or   x := A'*x,
 *
 *  where x is an n element vector and  A is an n by n unit, or non-unit,
 *  upper or lower triangular matrix, supplied in packed form.
 */
EIGEN_BLAS_FUNC(tpmv)
(char *uplo, char *opa, char *diag, EIGEN_BLAS_INT *n, RealScalar *pap, RealScalar *px, EIGEN_BLAS_INT *incx) {
  using functype = void (*)(Eigen::Index, const Scalar *, const Scalar *, Scalar *, Scalar);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::RowMajor;
  using Eigen::UnitDiag;
  using Eigen::Upper;
  static const functype func[16] = {
      // array index: NOTR  | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Upper | 0, Scalar, false, Scalar, false,
                                                                ColMajor>::run),
      // array index: TR    | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Lower | 0, Scalar, false, Scalar, false,
                                                                RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Lower | 0, Scalar, Conj, Scalar, false,
                                                                RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Lower | 0, Scalar, false, Scalar, false,
                                                                ColMajor>::run),
      // array index: TR    | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Upper | 0, Scalar, false, Scalar, false,
                                                                RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Upper | 0, Scalar, Conj, Scalar, false,
                                                                RowMajor>::run),
      0,
      // array index: NOTR  | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Upper | UnitDiag, Scalar, false, Scalar,
                                                                false, ColMajor>::run),
      // array index: TR    | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Lower | UnitDiag, Scalar, false, Scalar,
                                                                false, RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Lower | UnitDiag, Scalar, Conj, Scalar,
                                                                false, RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Lower | UnitDiag, Scalar, false, Scalar,
                                                                false, ColMajor>::run),
      // array index: TR    | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Upper | UnitDiag, Scalar, false, Scalar,
                                                                false, RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_matrix_vector_product<Eigen::Index, Upper | UnitDiag, Scalar, Conj, Scalar,
                                                                false, RowMajor>::run),
      0};

  Scalar *ap = reinterpret_cast<Scalar *>(pap);
  Scalar *x = reinterpret_cast<Scalar *>(px);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*opa) == INVALID)
    info = 2;
  else if (DIAG(*diag) == INVALID)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*incx == 0)
    info = 7;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TPMV ", &info, kBlasNameLength);

  if (*n == 0) return;

  compact_vector<Scalar> actual_x(x, *n, *incx);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(*n);
  int code = OP(*opa) | (UPLO(*uplo) << 2) | (DIAG(*diag) << 3);
  func[code](*n, ap, actual_x.data(), res.data(), Scalar(1));
  make_vector(actual_x.data(), *n) = res;
}

/**  DTPSV  solves one of the systems of equations
 *
 *     A*x = b,   or   A'*x = b,
 *
 *  where b and x are n element vectors and A is an n by n unit, or
 *  non-unit, upper or lower triangular matrix, supplied in packed form.
 *
 *  No test for singularity or near-singularity is included in this
 *  routine. Such tests must be performed before calling this routine.
 */
EIGEN_BLAS_FUNC(tpsv)
(char *uplo, char *opa, char *diag, EIGEN_BLAS_INT *n, RealScalar *pap, RealScalar *px, EIGEN_BLAS_INT *incx) {
  using functype = void (*)(Eigen::Index, const Scalar *, Scalar *);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::OnTheLeft;
  using Eigen::RowMajor;
  using Eigen::UnitDiag;
  using Eigen::Upper;
  static const functype func[16] = {
      // array index: NOTR  | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Upper | 0, false,
                                                       ColMajor>::run),
      // array index: TR    | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Lower | 0, false,
                                                       RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Lower | 0, Conj,
                                                       RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Lower | 0, false,
                                                       ColMajor>::run),
      // array index: TR    | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Upper | 0, false,
                                                       RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (NUNIT << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Upper | 0, Conj,
                                                       RowMajor>::run),
      0,
      // array index: NOTR  | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Upper | UnitDiag, false,
                                                       ColMajor>::run),
      // array index: TR    | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Lower | UnitDiag, false,
                                                       RowMajor>::run),
      // array index: ADJ   | (UP << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Lower | UnitDiag, Conj,
                                                       RowMajor>::run),
      0,
      // array index: NOTR  | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Lower | UnitDiag, false,
                                                       ColMajor>::run),
      // array index: TR    | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Upper | UnitDiag, false,
                                                       RowMajor>::run),
      // array index: ADJ   | (LO << 2) | (UNIT  << 3)
      (Eigen::internal::packed_triangular_solve_vector<Scalar, Scalar, Eigen::Index, OnTheLeft, Upper | UnitDiag, Conj,
                                                       RowMajor>::run),
      0};

  Scalar *ap = reinterpret_cast<Scalar *>(pap);
  Scalar *x = reinterpret_cast<Scalar *>(px);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*opa) == INVALID)
    info = 2;
  else if (DIAG(*diag) == INVALID)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*incx == 0)
    info = 7;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TPSV ", &info, kBlasNameLength);

  if (*n == 0) return;

  compact_vector<Scalar> actual_x(x, *n, *incx);
  int code = OP(*opa) | (UPLO(*uplo) << 2) | (DIAG(*diag) << 3);
  func[code](*n, ap, actual_x.data());
}
