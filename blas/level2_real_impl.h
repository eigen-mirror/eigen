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

// y = alpha*A*x + beta*y
EIGEN_BLAS_FUNC(symv)
(const char *uplo, const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *pa, const EIGEN_BLAS_INT *lda,
 const RealScalar *px, const EIGEN_BLAS_INT *incx, const RealScalar *pbeta, RealScalar *py,
 const EIGEN_BLAS_INT *incy) {
  typedef void (*functype)(EIGEN_BLAS_INT, const Scalar *, EIGEN_BLAS_INT, const Scalar *, Scalar *, Scalar);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::Upper;
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::selfadjoint_matrix_vector_product<Scalar, EIGEN_BLAS_INT, ColMajor, Upper, false, false>::run),
      // array index: LO
      (Eigen::internal::selfadjoint_matrix_vector_product<Scalar, EIGEN_BLAS_INT, ColMajor, Lower, false, false>::run),
  };

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  // check arguments
  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 5;
  else if (*incx == 0)
    info = 7;
  else if (*incy == 0)
    info = 10;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYMV ", &info, kBlasNameLength);

  if (*n == 0) return;

  const Scalar *actual_x = get_compact_vector(x, *n, *incx);
  Scalar *actual_y = get_compact_vector(y, *n, *incy);

  if (beta != Scalar(1)) {
    if (beta == Scalar(0))
      make_vector(actual_y, *n).setZero();
    else
      make_vector(actual_y, *n) *= beta;
  }

  int code = UPLO(*uplo);
  if (code >= 2 || func[code] == 0) return;

  func[code](*n, a, *lda, actual_x, actual_y, alpha);

  if (actual_x != x) delete[] actual_x;
  if (actual_y != y) delete[] copy_back(actual_y, y, *n, *incy);
}

// C := alpha*x*x' + C
EIGEN_BLAS_FUNC(syr)
(const char *uplo, const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *px, const EIGEN_BLAS_INT *incx,
 RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  typedef void (*functype)(EIGEN_BLAS_INT, Scalar *, EIGEN_BLAS_INT, const Scalar *, const Scalar *, const Scalar &);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::Upper;
  static const functype func[2] = {
      // array index: UP
      (Eigen::selfadjoint_rank1_update<Scalar, EIGEN_BLAS_INT, ColMajor, Upper, false, Conj>::run),
      // array index: LO
      (Eigen::selfadjoint_rank1_update<Scalar, EIGEN_BLAS_INT, ColMajor, Lower, false, Conj>::run),
  };

  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 7;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYR  ", &info, kBlasNameLength);

  if (*n == 0 || alpha == Scalar(0)) return;

  // if the increment is not 1, let's copy it to a temporary vector to enable vectorization
  const Scalar *x_cpy = get_compact_vector(x, *n, *incx);

  int code = UPLO(*uplo);
  if (code >= 2 || func[code] == 0) return;

  func[code](*n, c, *ldc, x_cpy, x_cpy, alpha);

  if (x_cpy != x) delete[] x_cpy;
}

// C := alpha*x*y' + alpha*y*x' + C
EIGEN_BLAS_FUNC(syr2)
(const char *uplo, const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *px, const EIGEN_BLAS_INT *incx,
 const RealScalar *py, const EIGEN_BLAS_INT *incy, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  typedef void (*functype)(EIGEN_BLAS_INT, Scalar *, EIGEN_BLAS_INT, const Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::rank2_update_selector<Scalar, EIGEN_BLAS_INT, Eigen::Upper>::run),
      // array index: LO
      (Eigen::internal::rank2_update_selector<Scalar, EIGEN_BLAS_INT, Eigen::Lower>::run),
  };

  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  const Scalar *y = reinterpret_cast<const Scalar *>(py);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*incy == 0)
    info = 7;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYR2 ", &info, kBlasNameLength);

  if (alpha == Scalar(0)) return;

  const Scalar *x_cpy = get_compact_vector(x, *n, *incx);
  const Scalar *y_cpy = get_compact_vector(y, *n, *incy);

  int code = UPLO(*uplo);
  if (code >= 2 || func[code] == 0) return;

  func[code](*n, c, *ldc, x_cpy, y_cpy, alpha);

  if (x_cpy != x) delete[] x_cpy;
  if (y_cpy != y) delete[] y_cpy;

  //   int code = UPLO(*uplo);
  //   if(code>=2 || func[code]==0)
  //     return 0;

  //   func[code](*n, a, *inca, b, *incb, c, *ldc, alpha);
}

/**  SBMV  performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n symmetric band matrix, with k super-diagonals.
 *
 *  Band storage: upper triangle stores A[i,j] at a[(k+i-j) + j*lda],
 *  lower triangle stores A[i,j] at a[(i-j) + j*lda].
 */
EIGEN_BLAS_FUNC(sbmv)
(char *uplo, EIGEN_BLAS_INT *n, EIGEN_BLAS_INT *k, RealScalar *palpha, RealScalar *pa, EIGEN_BLAS_INT *lda,
 RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *pbeta, RealScalar *py, EIGEN_BLAS_INT *incy) {
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*k < 0)
    info = 3;
  else if (*lda < *k + 1)
    info = 6;
  else if (*incx == 0)
    info = 8;
  else if (*incy == 0)
    info = 11;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SBMV ", &info, kBlasNameLength);

  if (*n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  const Scalar *actual_x = get_compact_vector(x, *n, *incx);
  Scalar *actual_y = get_compact_vector(y, *n, *incy);

  // First form y := beta*y.
  if (beta != Scalar(1)) {
    if (beta == Scalar(0))
      make_vector(actual_y, *n).setZero();
    else
      make_vector(actual_y, *n) *= beta;
  }

  if (alpha == Scalar(0)) {
    if (actual_x != x) delete[] actual_x;
    if (actual_y != y) delete[] copy_back(actual_y, y, *n, *incy);
    return;
  }

  if (*k >= 8) {
    // Vectorized path: use Eigen Map segments for the inner band operations.
    ConstMatrixType band(a, *k + 1, *n, *lda);
    if (UPLO(*uplo) == UP) {
      for (EIGEN_BLAS_INT j = 0; j < *n; ++j) {
        EIGEN_BLAS_INT start = std::max<EIGEN_BLAS_INT>(0, j - *k);
        EIGEN_BLAS_INT len = j - start;
        EIGEN_BLAS_INT offset = *k - (j - start);
        Scalar temp1 = alpha * actual_x[j];
        actual_y[j] += temp1 * band(*k, j);
        if (len > 0) {
          make_vector(actual_y + start, len) += temp1 * band.col(j).segment(offset, len);
          actual_y[j] += alpha * band.col(j).segment(offset, len).dot(make_vector(actual_x + start, len));
        }
      }
    } else {
      for (EIGEN_BLAS_INT j = 0; j < *n; ++j) {
        EIGEN_BLAS_INT len = std::min(*n - 1, j + *k) - j;
        Scalar temp1 = alpha * actual_x[j];
        actual_y[j] += temp1 * band(0, j);
        if (len > 0) {
          make_vector(actual_y + j + 1, len) += temp1 * band.col(j).segment(1, len);
          actual_y[j] += alpha * band.col(j).segment(1, len).dot(make_vector(actual_x + j + 1, len));
        }
      }
    }
  } else {
    // Scalar path: for narrow bandwidth, avoid Map overhead.
    if (UPLO(*uplo) == UP) {
      for (EIGEN_BLAS_INT j = 0; j < *n; ++j) {
        Scalar temp1 = alpha * actual_x[j];
        Scalar temp2 = Scalar(0);
        for (EIGEN_BLAS_INT i = std::max<EIGEN_BLAS_INT>(0, j - *k); i < j; ++i) {
          Scalar aij = a[(*k + i - j) + j * *lda];
          actual_y[i] += temp1 * aij;
          temp2 += aij * actual_x[i];
        }
        actual_y[j] += temp1 * a[*k + j * *lda] + alpha * temp2;
      }
    } else {
      for (EIGEN_BLAS_INT j = 0; j < *n; ++j) {
        Scalar temp1 = alpha * actual_x[j];
        Scalar temp2 = Scalar(0);
        actual_y[j] += temp1 * a[j * *lda];
        for (EIGEN_BLAS_INT i = j + 1; i <= std::min(*n - 1, j + *k); ++i) {
          Scalar aij = a[(i - j) + j * *lda];
          actual_y[i] += temp1 * aij;
          temp2 += aij * actual_x[i];
        }
        actual_y[j] += alpha * temp2;
      }
    }
  }

  if (actual_x != x) delete[] actual_x;
  if (actual_y != y) delete[] copy_back(actual_y, y, *n, *incy);
}

/**  SPMV  performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n symmetric matrix, supplied in packed form.
 *
 *  Packed storage: upper triangle stores columns sequentially so that
 *  column j occupies positions kk..kk+j (where kk = j*(j+1)/2),
 *  lower triangle stores column j at positions kk..kk+(n-j-1).
 */
EIGEN_BLAS_FUNC(spmv)
(char *uplo, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *pap, RealScalar *px, EIGEN_BLAS_INT *incx,
 RealScalar *pbeta, RealScalar *py, EIGEN_BLAS_INT *incy) {
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);
  const Scalar *ap = reinterpret_cast<const Scalar *>(pap);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 6;
  else if (*incy == 0)
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SPMV ", &info, kBlasNameLength);

  if (*n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  const Scalar *actual_x = get_compact_vector(x, *n, *incx);
  Scalar *actual_y = get_compact_vector(y, *n, *incy);

  // First form y := beta*y.
  if (beta != Scalar(1)) {
    if (beta == Scalar(0))
      make_vector(actual_y, *n).setZero();
    else
      make_vector(actual_y, *n) *= beta;
  }

  if (alpha == Scalar(0)) {
    if (actual_x != x) delete[] actual_x;
    if (actual_y != y) delete[] copy_back(actual_y, y, *n, *incy);
    return;
  }

  EIGEN_BLAS_INT kk = 0;
  if (UPLO(*uplo) == UP) {
    // Upper triangle packed: column j occupies ap[kk..kk+j].
    for (EIGEN_BLAS_INT j = 0; j < *n; ++j) {
      Scalar temp1 = alpha * actual_x[j];
      actual_y[j] += temp1 * ap[kk + j];
      if (j > 0) {
        make_vector(actual_y, j) += temp1 * make_vector(ap + kk, j);
        actual_y[j] += alpha * make_vector(ap + kk, j).dot(make_vector(actual_x, j));
      }
      kk += j + 1;
    }
  } else {
    // Lower triangle packed: column j occupies ap[kk..kk+(n-j-1)].
    for (EIGEN_BLAS_INT j = 0; j < *n; ++j) {
      EIGEN_BLAS_INT len = *n - j - 1;
      Scalar temp1 = alpha * actual_x[j];
      actual_y[j] += temp1 * ap[kk];
      if (len > 0) {
        make_vector(actual_y + j + 1, len) += temp1 * make_vector(ap + kk + 1, len);
        actual_y[j] += alpha * make_vector(ap + kk + 1, len).dot(make_vector(actual_x + j + 1, len));
      }
      kk += *n - j;
    }
  }

  if (actual_x != x) delete[] actual_x;
  if (actual_y != y) delete[] copy_back(actual_y, y, *n, *incy);
}

/**  DSPR    performs the symmetric rank 1 operation
 *
 *     A := alpha*x*x' + A,
 *
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n symmetric matrix, supplied in packed form.
 */
EIGEN_BLAS_FUNC(spr)(char *uplo, EIGEN_BLAS_INT *n, Scalar *palpha, Scalar *px, EIGEN_BLAS_INT *incx, Scalar *pap) {
  typedef void (*functype)(EIGEN_BLAS_INT, Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::selfadjoint_packed_rank1_update<Scalar, EIGEN_BLAS_INT, Eigen::ColMajor, Eigen::Upper, false,
                                                        false>::run),
      // array index: LO
      (Eigen::internal::selfadjoint_packed_rank1_update<Scalar, EIGEN_BLAS_INT, Eigen::ColMajor, Eigen::Lower, false,
                                                        false>::run),
  };

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *ap = reinterpret_cast<Scalar *>(pap);
  Scalar alpha = *reinterpret_cast<Scalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SPR  ", &info, kBlasNameLength);

  if (alpha == Scalar(0)) return;

  Scalar *x_cpy = get_compact_vector(x, *n, *incx);

  int code = UPLO(*uplo);
  if (code >= 2 || func[code] == 0) return;

  func[code](*n, ap, x_cpy, alpha);

  if (x_cpy != x) delete[] x_cpy;
}

/**  DSPR2  performs the symmetric rank 2 operation
 *
 *     A := alpha*x*y' + alpha*y*x' + A,
 *
 *  where alpha is a scalar, x and y are n element vectors and A is an
 *  n by n symmetric matrix, supplied in packed form.
 */
EIGEN_BLAS_FUNC(spr2)
(char *uplo, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py,
 EIGEN_BLAS_INT *incy, RealScalar *pap) {
  typedef void (*functype)(EIGEN_BLAS_INT, Scalar *, const Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::packed_rank2_update_selector<Scalar, EIGEN_BLAS_INT, Eigen::Upper>::run),
      // array index: LO
      (Eigen::internal::packed_rank2_update_selector<Scalar, EIGEN_BLAS_INT, Eigen::Lower>::run),
  };

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar *ap = reinterpret_cast<Scalar *>(pap);
  Scalar alpha = *reinterpret_cast<Scalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*incy == 0)
    info = 7;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SPR2 ", &info, kBlasNameLength);

  if (alpha == Scalar(0)) return;

  Scalar *x_cpy = get_compact_vector(x, *n, *incx);
  Scalar *y_cpy = get_compact_vector(y, *n, *incy);

  int code = UPLO(*uplo);
  if (code >= 2 || func[code] == 0) return;

  func[code](*n, ap, x_cpy, y_cpy, alpha);

  if (x_cpy != x) delete[] x_cpy;
  if (y_cpy != y) delete[] y_cpy;
}

/**  DGER   performs the rank 1 operation
 *
 *     A := alpha*x*y' + A,
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 */
EIGEN_BLAS_FUNC(ger)
(EIGEN_BLAS_INT *m, EIGEN_BLAS_INT *n, Scalar *palpha, Scalar *px, EIGEN_BLAS_INT *incx, Scalar *py,
 EIGEN_BLAS_INT *incy, Scalar *pa, EIGEN_BLAS_INT *lda) {
  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar *a = reinterpret_cast<Scalar *>(pa);
  Scalar alpha = *reinterpret_cast<Scalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (*m < 0)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*incy == 0)
    info = 7;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GER  ", &info, kBlasNameLength);

  if (alpha == Scalar(0)) return;

  Scalar *x_cpy = get_compact_vector(x, *m, *incx);
  Scalar *y_cpy = get_compact_vector(y, *n, *incy);

  Eigen::internal::general_rank1_update<Scalar, EIGEN_BLAS_INT, Eigen::ColMajor, false, false>::run(
      *m, *n, a, *lda, x_cpy, y_cpy, alpha);

  if (x_cpy != x) delete[] x_cpy;
  if (y_cpy != y) delete[] y_cpy;
}
