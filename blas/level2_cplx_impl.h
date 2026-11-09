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

/**  ZHEMV  performs the matrix-vector  operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n hermitian matrix.
 */
EIGEN_BLAS_FUNC(hemv)
(const char *uplo, const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *pa, const EIGEN_BLAS_INT *lda,
 const RealScalar *px, const EIGEN_BLAS_INT *incx, const RealScalar *pbeta, RealScalar *py,
 const EIGEN_BLAS_INT *incy) {
  using functype = void (*)(Eigen::Index, const Scalar *, Eigen::Index, const Scalar *, Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::selfadjoint_matrix_vector_product<Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Upper, false,
                                                          false>::run),
      // array index: LO
      (Eigen::internal::selfadjoint_matrix_vector_product<Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Lower, false,
                                                          false>::run),
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
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HEMV ", &info, kBlasNameLength);

  if (*n == 0) return;

  compact_vector<Scalar> compact_y(y, *n, *incy);
  Scalar *actual_y = compact_y.data();
  scale_vector(actual_y, *n, beta);
  if (alpha == Scalar(0)) return;

  compact_vector<const Scalar> compact_x(x, *n, *incx);
  const Scalar *actual_x = compact_x.data();

  int code = UPLO(*uplo);

  func[code](*n, a, *lda, actual_x, actual_y, alpha);
}

/**  HBMV  performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n hermitian band matrix, with k super-diagonals.
 *  Diagonal elements are real; off-diagonal contributions use conjugation.
 */
EIGEN_BLAS_FUNC(hbmv)
(char *uplo, EIGEN_BLAS_INT *n_in, EIGEN_BLAS_INT *k_in, RealScalar *palpha, RealScalar *pa, EIGEN_BLAS_INT *lda_in,
 RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *pbeta, RealScalar *py, EIGEN_BLAS_INT *incy) {
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  const Eigen::Index n = *n_in;
  const Eigen::Index k = *k_in;
  const Eigen::Index lda = *lda_in;

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (n < 0)
    info = 2;
  else if (k < 0)
    info = 3;
  else if (lda <= k)
    info = 6;
  else if (*incx == 0)
    info = 8;
  else if (*incy == 0)
    info = 11;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HBMV ", &info, kBlasNameLength);

  if (n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  compact_vector<Scalar> compact_y(y, n, *incy);
  scale_vector(compact_y.data(), n, beta);
  if (alpha == Scalar(0)) return;

  compact_vector<const Scalar> compact_x(x, n, *incx);
  selfadjoint_band_product(UPLO(*uplo) == UP, n, k, a, lda, compact_x.data(), compact_y.data(), alpha);
}

/**  HPMV  performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n hermitian matrix, supplied in packed form.
 *  Diagonal elements are real; off-diagonal contributions use conjugation.
 */
EIGEN_BLAS_FUNC(hpmv)
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
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HPMV ", &info, kBlasNameLength);

  if (*n == 0 || (alpha == Scalar(0) && beta == Scalar(1))) return;

  compact_vector<Scalar> compact_y(y, *n, *incy);
  scale_vector(compact_y.data(), *n, beta);
  if (alpha == Scalar(0)) return;

  compact_vector<const Scalar> compact_x(x, *n, *incx);
  selfadjoint_packed_product(UPLO(*uplo) == UP, *n, ap, compact_x.data(), compact_y.data(), alpha);
}

/**  ZHPR    performs the hermitian rank 1 operation
 *
 *     A := alpha*x*conjg( x' ) + A,
 *
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n hermitian matrix, supplied in packed form.
 */
EIGEN_BLAS_FUNC(hpr)
(char *uplo, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *pap) {
  using functype = void (*)(Eigen::Index, Scalar *, const Scalar *, RealScalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::selfadjoint_packed_rank1_update<Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Upper, false,
                                                        Conj>::run),
      // array index: LO
      (Eigen::internal::selfadjoint_packed_rank1_update<Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Lower, false,
                                                        Conj>::run),
  };

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *ap = reinterpret_cast<Scalar *>(pap);
  RealScalar alpha = *palpha;

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HPR  ", &info, kBlasNameLength);

  if (*n == 0 || alpha == Scalar(0)) return;

  compact_vector<const Scalar> x_cpy(x, *n, *incx);

  int code = UPLO(*uplo);

  func[code](*n, ap, x_cpy.data(), alpha);
}

/**  ZHPR2  performs the hermitian rank 2 operation
 *
 *     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A,
 *
 *  where alpha is a scalar, x and y are n element vectors and A is an
 *  n by n hermitian matrix, supplied in packed form.
 */
EIGEN_BLAS_FUNC(hpr2)
(char *uplo, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py,
 EIGEN_BLAS_INT *incy, RealScalar *pap) {
  using functype = void (*)(Eigen::Index, Scalar *, const Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::packed_rank2_update_selector<Scalar, Eigen::Index, Eigen::Upper>::run),
      // array index: LO
      (Eigen::internal::packed_rank2_update_selector<Scalar, Eigen::Index, Eigen::Lower>::run),
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
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HPR2 ", &info, kBlasNameLength);

  if (*n == 0 || alpha == Scalar(0)) return;

  compact_vector<const Scalar> x_cpy(x, *n, *incx);
  compact_vector<const Scalar> y_cpy(y, *n, *incy);

  int code = UPLO(*uplo);

  func[code](*n, ap, x_cpy.data(), y_cpy.data(), alpha);
}

/**  ZHER   performs the hermitian rank 1 operation
 *
 *     A := alpha*x*conjg( x' ) + A,
 *
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n hermitian matrix.
 */
EIGEN_BLAS_FUNC(her)
(char *uplo, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *pa,
 EIGEN_BLAS_INT *lda) {
  using functype = void (*)(Eigen::Index, Scalar *, Eigen::Index, const Scalar *, const Scalar *, const Scalar &);
  static const functype func[2] = {
      // array index: UP
      (Eigen::selfadjoint_rank1_update<Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Upper, false, Conj>::run),
      // array index: LO
      (Eigen::selfadjoint_rank1_update<Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Lower, false, Conj>::run),
  };

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *a = reinterpret_cast<Scalar *>(pa);
  RealScalar alpha = *reinterpret_cast<RealScalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (*n < 0)
    info = 2;
  else if (*incx == 0)
    info = 5;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 7;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HER  ", &info, kBlasNameLength);

  if (*n == 0 || alpha == RealScalar(0)) return;

  compact_vector<const Scalar> x_cpy(x, *n, *incx);

  int code = UPLO(*uplo);

  func[code](*n, a, *lda, x_cpy.data(), x_cpy.data(), alpha);

  matrix(a, *n, *n, *lda).diagonal().imag().setZero();
}

/**  ZHER2  performs the hermitian rank 2 operation
 *
 *     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A,
 *
 *  where alpha is a scalar, x and y are n element vectors and A is an n
 *  by n hermitian matrix.
 */
EIGEN_BLAS_FUNC(her2)
(char *uplo, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py,
 EIGEN_BLAS_INT *incy, RealScalar *pa, EIGEN_BLAS_INT *lda) {
  using functype = void (*)(Eigen::Index, Scalar *, Eigen::Index, const Scalar *, const Scalar *, Scalar);
  static const functype func[2] = {
      // array index: UP
      (Eigen::internal::rank2_update_selector<Scalar, Eigen::Index, Eigen::Upper>::run),
      // array index: LO
      (Eigen::internal::rank2_update_selector<Scalar, Eigen::Index, Eigen::Lower>::run),
  };

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar *a = reinterpret_cast<Scalar *>(pa);
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
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 9;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HER2 ", &info, kBlasNameLength);

  if (*n == 0 || alpha == Scalar(0)) return;

  compact_vector<const Scalar> x_cpy(x, *n, *incx);
  compact_vector<const Scalar> y_cpy(y, *n, *incy);

  int code = UPLO(*uplo);

  func[code](*n, a, *lda, x_cpy.data(), y_cpy.data(), alpha);

  matrix(a, *n, *n, *lda).diagonal().imag().setZero();
}

/**  ZGERU  performs the rank 1 operation
 *
 *     A := alpha*x*y' + A,
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 */
EIGEN_BLAS_FUNC(geru)
(EIGEN_BLAS_INT *m, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py,
 EIGEN_BLAS_INT *incy, RealScalar *pa, EIGEN_BLAS_INT *lda) {
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
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GERU ", &info, kBlasNameLength);

  if (*m == 0 || *n == 0 || alpha == Scalar(0)) return;

  compact_vector<const Scalar> x_cpy(x, *m, *incx);
  compact_vector<const Scalar> y_cpy(y, *n, *incy);

  Eigen::internal::general_rank1_update<Scalar, Eigen::Index, Eigen::ColMajor, false, false>::run(
      *m, *n, a, *lda, x_cpy.data(), y_cpy.data(), alpha);
}

/**  ZGERC  performs the rank 1 operation
 *
 *     A := alpha*x*conjg( y' ) + A,
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 */
EIGEN_BLAS_FUNC(gerc)
(EIGEN_BLAS_INT *m, EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py,
 EIGEN_BLAS_INT *incy, RealScalar *pa, EIGEN_BLAS_INT *lda) {
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
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GERC ", &info, kBlasNameLength);

  if (*m == 0 || *n == 0 || alpha == Scalar(0)) return;

  compact_vector<const Scalar> x_cpy(x, *m, *incx);
  compact_vector<const Scalar> y_cpy(y, *n, *incy);

  Eigen::internal::general_rank1_update<Scalar, Eigen::Index, Eigen::ColMajor, false, Conj>::run(
      *m, *n, a, *lda, x_cpy.data(), y_cpy.data(), alpha);
}
