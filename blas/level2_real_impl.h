// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "common.h"

// y = alpha*A*x + beta*y
EIGEN_BLAS_FUNC(symv)
(const char *uplo, const int *n, const RealScalar *palpha, const RealScalar *pa, const int *lda, const RealScalar *px,
 const int *incx, const RealScalar *pbeta, RealScalar *py, const int *incy) {
  typedef void (*functype)(int, const Scalar *, int, const Scalar *, Scalar *, Scalar);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::Upper;
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::selfadjoint_matrix_vector_product<Scalar, int, ColMajor, Upper, false, false>::run),
      // array index: LO
      (Eigen::internal::selfadjoint_matrix_vector_product<Scalar, int, ColMajor, Lower, false, false>::run),
  };

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  // check arguments
  int info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*lda < std::max(1, *n))
    info = 5;
  else if (*incx == 0)
    info = 7;
  else if (*incy == 0)
    info = 10;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYMV ", &info);

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
(const char *uplo, const int *n, const RealScalar *palpha, const RealScalar *px, const int *incx, RealScalar *pc,
 const int *ldc) {
  typedef void (*functype)(int, Scalar *, int, const Scalar *, const Scalar *, const Scalar &);
  using Eigen::ColMajor;
  using Eigen::Lower;
  using Eigen::Upper;
  static const functype func[2] = {
      // array index: UP
      (Eigen::selfadjoint_rank1_update<Scalar, int, ColMajor, Upper, false, Conj>::run),
      // array index: LO
      (Eigen::selfadjoint_rank1_update<Scalar, int, ColMajor, Lower, false, Conj>::run),
  };

  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);

  int info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*ldc < std::max(1, *n))
    info = 7;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYR  ", &info);

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
(const char *uplo, const int *n, const RealScalar *palpha, const RealScalar *px, const int *incx, const RealScalar *py,
 const int *incy, RealScalar *pc, const int *ldc) {
  typedef void (*functype)(int, Scalar *, int, const Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::rank2_update_selector<Scalar, int, Eigen::Upper>::run),
      // array index: LO
      (Eigen::internal::rank2_update_selector<Scalar, int, Eigen::Lower>::run),
  };

  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  const Scalar *y = reinterpret_cast<const Scalar *>(py);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);

  int info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*incy == 0)
    info = 7;
  else if (*ldc < std::max(1, *n))
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYR2 ", &info);

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
(char *uplo, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *px, int *incx, RealScalar *pbeta,
 RealScalar *py, int *incy) {
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  int info = 0;
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
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SBMV ", &info);

  if (*n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  int kx = *incx > 0 ? 0 : (1 - *n) * *incx;
  int ky = *incy > 0 ? 0 : (1 - *n) * *incy;

  // First form y := beta*y.
  if (beta != Scalar(1)) {
    int iy = ky;
    for (int i = 0; i < *n; ++i) {
      y[iy] = (beta == Scalar(0)) ? Scalar(0) : beta * y[iy];
      iy += *incy;
    }
  }

  if (alpha == Scalar(0)) return;

  if (UPLO(*uplo) == UP) {
    // Upper triangle: A[i,j] at a[(k+i-j) + j*lda], diagonal at row k.
    int jx = kx, jy = ky;
    for (int j = 0; j < *n; ++j) {
      Scalar temp1 = alpha * x[jx];
      Scalar temp2 = Scalar(0);
      int ix = kx, iy = ky;
      for (int i = std::max(0, j - *k); i < j; ++i) {
        Scalar aij = a[(*k + i - j) + j * *lda];
        y[iy] += temp1 * aij;
        temp2 += aij * x[ix];
        ix += *incx;
        iy += *incy;
      }
      y[jy] += temp1 * a[*k + j * *lda] + alpha * temp2;
      jx += *incx;
      jy += *incy;
      if (j >= *k) {
        kx += *incx;
        ky += *incy;
      }
    }
  } else {
    // Lower triangle: A[i,j] at a[(i-j) + j*lda], diagonal at row 0.
    int jx = kx, jy = ky;
    for (int j = 0; j < *n; ++j) {
      Scalar temp1 = alpha * x[jx];
      Scalar temp2 = Scalar(0);
      y[jy] += temp1 * a[j * *lda];
      int ix = jx, iy = jy;
      for (int i = j + 1; i <= std::min(*n - 1, j + *k); ++i) {
        ix += *incx;
        iy += *incy;
        Scalar aij = a[(i - j) + j * *lda];
        y[iy] += temp1 * aij;
        temp2 += aij * x[ix];
      }
      y[jy] += alpha * temp2;
      jx += *incx;
      jy += *incy;
    }
  }
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
(char *uplo, int *n, RealScalar *palpha, RealScalar *pap, RealScalar *px, int *incx, RealScalar *pbeta, RealScalar *py,
 int *incy) {
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);
  const Scalar *ap = reinterpret_cast<const Scalar *>(pap);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  int info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 6;
  else if (*incy == 0)
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SPMV ", &info);

  if (*n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  int kx = *incx > 0 ? 0 : (1 - *n) * *incx;
  int ky = *incy > 0 ? 0 : (1 - *n) * *incy;

  // First form y := beta*y.
  if (beta != Scalar(1)) {
    int iy = ky;
    for (int i = 0; i < *n; ++i) {
      y[iy] = (beta == Scalar(0)) ? Scalar(0) : beta * y[iy];
      iy += *incy;
    }
  }

  if (alpha == Scalar(0)) return;

  int kk = 0;
  if (UPLO(*uplo) == UP) {
    // Upper triangle packed.
    int jx = kx, jy = ky;
    for (int j = 0; j < *n; ++j) {
      Scalar temp1 = alpha * x[jx];
      Scalar temp2 = Scalar(0);
      int ix = kx, iy = ky;
      for (int i = 0; i < j; ++i) {
        y[iy] += temp1 * ap[kk + i];
        temp2 += ap[kk + i] * x[ix];
        ix += *incx;
        iy += *incy;
      }
      y[jy] += temp1 * ap[kk + j] + alpha * temp2;
      jx += *incx;
      jy += *incy;
      kk += j + 1;
    }
  } else {
    // Lower triangle packed.
    int jx = kx, jy = ky;
    for (int j = 0; j < *n; ++j) {
      Scalar temp1 = alpha * x[jx];
      Scalar temp2 = Scalar(0);
      y[jy] += temp1 * ap[kk];
      int ix = jx, iy = jy;
      for (int i = 1; i < *n - j; ++i) {
        ix += *incx;
        iy += *incy;
        y[iy] += temp1 * ap[kk + i];
        temp2 += ap[kk + i] * x[ix];
      }
      y[jy] += alpha * temp2;
      jx += *incx;
      jy += *incy;
      kk += *n - j;
    }
  }
}

/**  DSPR    performs the symmetric rank 1 operation
 *
 *     A := alpha*x*x' + A,
 *
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n symmetric matrix, supplied in packed form.
 */
EIGEN_BLAS_FUNC(spr)(char *uplo, int *n, Scalar *palpha, Scalar *px, int *incx, Scalar *pap) {
  typedef void (*functype)(int, Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::selfadjoint_packed_rank1_update<Scalar, int, Eigen::ColMajor, Eigen::Upper, false, false>::run),
      // array index: LO
      (Eigen::internal::selfadjoint_packed_rank1_update<Scalar, int, Eigen::ColMajor, Eigen::Lower, false, false>::run),
  };

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *ap = reinterpret_cast<Scalar *>(pap);
  Scalar alpha = *reinterpret_cast<Scalar *>(palpha);

  int info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SPR  ", &info);

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
(char *uplo, int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pap) {
  typedef void (*functype)(int, Scalar *, const Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::packed_rank2_update_selector<Scalar, int, Eigen::Upper>::run),
      // array index: LO
      (Eigen::internal::packed_rank2_update_selector<Scalar, int, Eigen::Lower>::run),
  };

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar *ap = reinterpret_cast<Scalar *>(pap);
  Scalar alpha = *reinterpret_cast<Scalar *>(palpha);

  int info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*incy == 0)
    info = 7;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SPR2 ", &info);

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
(int *m, int *n, Scalar *palpha, Scalar *px, int *incx, Scalar *py, int *incy, Scalar *pa, int *lda) {
  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar *a = reinterpret_cast<Scalar *>(pa);
  Scalar alpha = *reinterpret_cast<Scalar *>(palpha);

  int info = 0;
  if (*m < 0)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*incy == 0)
    info = 7;
  else if (*lda < std::max(1, *m))
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GER  ", &info);

  if (alpha == Scalar(0)) return;

  Scalar *x_cpy = get_compact_vector(x, *m, *incx);
  Scalar *y_cpy = get_compact_vector(y, *n, *incy);

  Eigen::internal::general_rank1_update<Scalar, int, Eigen::ColMajor, false, false>::run(*m, *n, a, *lda, x_cpy, y_cpy,
                                                                                         alpha);

  if (x_cpy != x) delete[] x_cpy;
  if (y_cpy != y) delete[] y_cpy;
}
