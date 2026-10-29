// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "../blas.h"

namespace {

// With a singleton outer dimension (n = 1 column or 1 vector element), BLAS permits arbitrarily
// large leading dimensions and vector increments up to INT_MAX while only accessing the first
// column or element on the stack. This exercises internal index widening (such as
// lhs.stride() * sizeof(Scalar) in GEMV) across all platforms and compilers without allocating
// large buffers.
void test_level1_large_stride() {
  EIGEN_BLAS_INT one = 1;
  EIGEN_BLAS_INT zero = 0;
  EIGEN_BLAS_INT large_inc = (std::numeric_limits<EIGEN_BLAS_INT>::max)();
  EIGEN_BLAS_INT neg_inc = -large_inc;

  // Broadcast and negative-stride copy.
  double x = 42.0, y = 0.0;
  BLASFUNC(dcopy)(&one, &x, &zero, &y, &neg_inc);
  VERIFY_IS_EQUAL(y, 42.0);

  y = 0.0;
  BLASFUNC(dcopy)(&one, &x, &large_inc, &y, &neg_inc);
  VERIFY_IS_EQUAL(y, 42.0);

  // Modified Givens rotation with extreme positive and negative increments.
  double param[5] = {-1.0, 0.0, 1.0, 1.0, 0.0};
  x = 3.0;
  y = 7.0;
  BLASFUNC(drotm)(&one, &x, &neg_inc, &y, &large_inc, param);
  VERIFY_IS_EQUAL(x, 7.0);
  VERIFY_IS_EQUAL(y, 3.0);

  // Dot products, axpy, scal, swap, reductions.
  VERIFY_IS_EQUAL(BLASFUNC(ddot)(&one, &x, &large_inc, &y, &neg_inc), 21.0);

  float sx = 3.0f, sy = 7.0f, sb = 1.0f;
  VERIFY_IS_EQUAL(BLASFUNC(dsdot)(&one, &sx, &large_inc, &sy, &neg_inc), 21.0);
  VERIFY_IS_EQUAL(BLASFUNC(sdsdot)(&one, &sb, &sx, &large_inc, &sy, &neg_inc), 22.0f);

  double alpha = 2.0;
  BLASFUNC(daxpy)(&one, &alpha, &x, &large_inc, &y, &neg_inc);
  VERIFY_IS_EQUAL(y, 17.0);

  BLASFUNC(dscal)(&one, &alpha, &x, &large_inc);
  VERIFY_IS_EQUAL(x, 14.0);

  BLASFUNC(dswap)(&one, &x, &large_inc, &y, &neg_inc);
  VERIFY_IS_EQUAL(x, 17.0);
  VERIFY_IS_EQUAL(y, 14.0);

  VERIFY_IS_EQUAL(BLASFUNC(dasum)(&one, &x, &large_inc), 17.0);
  VERIFY_IS_EQUAL(BLASFUNC(dnrm2)(&one, &x, &large_inc), 17.0);
  VERIFY_IS_EQUAL(BLASFUNC(idamax)(&one, &x, &large_inc), 1);
}

void test_level2_real_large_stride() {
  char uplo = 'U', trans = 'N', trans_t = 'T', diag = 'N';
  EIGEN_BLAS_INT one = 1, two = 2, eight = 8, k8 = 8;
  EIGEN_BLAS_INT large_lda = (std::numeric_limits<EIGEN_BLAS_INT>::max)();
  EIGEN_BLAS_INT stride_28 = EIGEN_BLAS_INT(1) << 28;  // stride_28 * sizeof(double) == 2^31 overflows 32-bit int.
  EIGEN_BLAS_INT neg_inc = -large_lda;
  double alpha = 2.0, beta = 0.0;

  double a[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  double x[8] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
  double y[8] = {};

  for (EIGEN_BLAS_INT lda : {stride_28, large_lda}) {
    // dgemv "N" and "T" for m = 2 (run_small_cols) and m = 8 (full packet path).
    BLASFUNC(dgemv)("N", &two, &one, &alpha, a, &lda, x, &large_lda, &beta, y, &one);
    VERIFY_IS_EQUAL(y[0], 2.0);
    VERIFY_IS_EQUAL(y[1], 4.0);

    BLASFUNC(dgemv)("T", &two, &one, &alpha, a, &lda, x, &one, &beta, y, &large_lda);
    VERIFY_IS_EQUAL(y[0], 10.0);

    BLASFUNC(dgemv)("T", &eight, &one, &alpha, a, &lda, x, &one, &beta, y, &large_lda);
    VERIFY_IS_EQUAL(y[0], 112.0);
  }

  // dtrmv & dtrsv (1x1 with large_lda and large_inc).
  double tx = 3.0;
  BLASFUNC(dtrmv)("U", "N", "N", &one, a, &large_lda, &tx, &large_lda);
  VERIFY_IS_EQUAL(tx, 3.0);
  a[0] = 2.0;
  tx = 6.0;
  BLASFUNC(dtrsv)("U", "N", "N", &one, a, &large_lda, &tx, &neg_inc);
  VERIFY_IS_EQUAL(tx, 3.0);

  // Banded routines: dgbmv, dtbmv, dtbsv, dsbmv (k = 1 scalar and k = 8 vectorized paths).
  a[0] = 2.0;
  a[1] = 3.0;
  BLASFUNC(dgbmv)(&trans, &two, &one, &one, &one, &alpha, a, &large_lda, x, &large_lda, &beta, y, &one);
  VERIFY_IS_EQUAL(y[0], 6.0);
  BLASFUNC(dgbmv)(&trans_t, &two, &one, &one, &one, &alpha, a, &large_lda, x, &one, &beta, y, &large_lda);
  VERIFY_IS_EQUAL(y[0], 18.0);

  for (EIGEN_BLAS_INT k : {one, k8}) {
    a[k] = 2.0;
    tx = 3.0;
    BLASFUNC(dtbmv)(&uplo, &trans, &diag, &one, &k, a, &large_lda, &tx, &large_lda);
    VERIFY_IS_EQUAL(tx, 6.0);
    BLASFUNC(dtbsv)(&uplo, &trans, &diag, &one, &k, a, &large_lda, &tx, &large_lda);
    VERIFY_IS_EQUAL(tx, 3.0);

    y[0] = 0.0;
    BLASFUNC(dsbmv)(&uplo, &one, &k, &alpha, a, &large_lda, &tx, &large_lda, &beta, y, &large_lda);
    VERIFY_IS_EQUAL(y[0], 12.0);
  }

  // Symmetric & rank updates: dsymv, dger, dsyr, dsyr2.
  a[0] = 3.0;
  y[0] = 0.0;
  BLASFUNC(dsymv)(&uplo, &one, &alpha, a, &large_lda, x, &large_lda, &beta, y, &large_lda);
  VERIFY_IS_EQUAL(y[0], 6.0);

  a[0] = 0.0;
  a[1] = 0.0;
  BLASFUNC(dger)(&two, &one, &alpha, x, &one, &tx, &large_lda, a, &large_lda);
  VERIFY_IS_EQUAL(a[0], 6.0);
  VERIFY_IS_EQUAL(a[1], 12.0);

  BLASFUNC(dsyr)(&uplo, &one, &alpha, &tx, &large_lda, a, &large_lda);
  VERIFY_IS_EQUAL(a[0], 24.0);

  BLASFUNC(dsyr2)(&uplo, &one, &alpha, &tx, &large_lda, x, &large_lda, a, &large_lda);
  VERIFY_IS_EQUAL(a[0], 36.0);

  // Packed routines: dspmv, dtpmv, dtpsv, dspr, dspr2.
  double ap = 2.0;
  tx = 3.0;
  y[0] = 0.0;
  BLASFUNC(dspmv)(&uplo, &one, &alpha, &ap, &tx, &large_lda, &beta, y, &large_lda);
  VERIFY_IS_EQUAL(y[0], 12.0);

  BLASFUNC(dtpmv)(&uplo, &trans, &diag, &one, &ap, &tx, &large_lda);
  VERIFY_IS_EQUAL(tx, 6.0);
  BLASFUNC(dtpsv)(&uplo, &trans, &diag, &one, &ap, &tx, &large_lda);
  VERIFY_IS_EQUAL(tx, 3.0);

  BLASFUNC(dspr)(&uplo, &one, &alpha, &tx, &large_lda, &ap);
  VERIFY_IS_EQUAL(ap, 20.0);
  BLASFUNC(dspr2)(&uplo, &one, &alpha, &tx, &large_lda, x, &large_lda, &ap);
  VERIFY_IS_EQUAL(ap, 32.0);
}

void test_level2_complex_large_stride() {
  char uplo = 'U';
  EIGEN_BLAS_INT one = 1, two = 2, eight = 8, k8 = 8;
  EIGEN_BLAS_INT large_lda = (std::numeric_limits<EIGEN_BLAS_INT>::max)();
  EIGEN_BLAS_INT stride_27 = EIGEN_BLAS_INT(1) << 27;  // stride_27 * sizeof(complex<double>) == 2^31.
  std::complex<double> alpha(2.0, 0.0), beta(0.0, 0.0);
  double ralpha = 2.0;

  std::complex<double> a[12] = {{1.0, 1.0}, {2.0, -1.0}, {3.0, 0.0}, {4.0, 2.0}, {1.0, 0.0}, {2.0, 0.0},
                                {3.0, 0.0}, {4.0, 0.0},  {5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}};
  std::complex<double> x[8] = {{1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
                               {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
  std::complex<double> y[8] = {};

  for (EIGEN_BLAS_INT lda : {stride_27, large_lda}) {
    BLASFUNC(zgemv)
    ("N", &two, &one, reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(a), &lda,
     reinterpret_cast<const double*>(x), &large_lda, reinterpret_cast<const double*>(&beta),
     reinterpret_cast<double*>(y), &one);
    VERIFY_IS_EQUAL(y[0], std::complex<double>(2.0, 2.0));
    VERIFY_IS_EQUAL(y[1], std::complex<double>(4.0, -2.0));

    BLASFUNC(zgemv)
    ("C", &two, &one, reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(a), &lda,
     reinterpret_cast<const double*>(x), &one, reinterpret_cast<const double*>(&beta), reinterpret_cast<double*>(y),
     &large_lda);
    VERIFY_IS_EQUAL(y[0], std::complex<double>(10.0, 2.0));

    BLASFUNC(zgemv)
    ("C", &eight, &one, reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(a), &lda,
     reinterpret_cast<const double*>(x), &one, reinterpret_cast<const double*>(&beta), reinterpret_cast<double*>(y),
     &large_lda);
    VERIFY_IS_EQUAL(y[0], std::complex<double>(44.0, -2.0));
  }

  a[0] = {3.0, 0.0};
  BLASFUNC(zhemv)
  (&uplo, &one, reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(a), &large_lda,
   reinterpret_cast<const double*>(x), &large_lda, reinterpret_cast<const double*>(&beta), reinterpret_cast<double*>(y),
   &large_lda);
  VERIFY_IS_EQUAL(y[0], std::complex<double>(6.0, 0.0));

  for (EIGEN_BLAS_INT k : {one, k8}) {
    a[k] = {3.0, 0.0};
    y[0] = {0.0, 0.0};
    BLASFUNC(zhbmv)
    (&uplo, &one, &k, reinterpret_cast<double*>(&alpha), reinterpret_cast<double*>(a), &large_lda,
     reinterpret_cast<double*>(x), &large_lda, reinterpret_cast<double*>(&beta), reinterpret_cast<double*>(y),
     &large_lda);
    VERIFY_IS_EQUAL(y[0], std::complex<double>(6.0, 0.0));
  }

  a[0] = {0.0, 0.0};
  std::complex<double> zx(2.0, 1.0), zy(3.0, -1.0);
  BLASFUNC(zher)
  (&uplo, &one, &ralpha, reinterpret_cast<double*>(&zx), &large_lda, reinterpret_cast<double*>(a), &large_lda);
  VERIFY_IS_EQUAL(a[0], std::complex<double>(10.0, 0.0));

  BLASFUNC(zher2)
  (&uplo, &one, reinterpret_cast<double*>(&alpha), reinterpret_cast<double*>(&zx), &large_lda,
   reinterpret_cast<double*>(&zy), &large_lda, reinterpret_cast<double*>(a), &large_lda);
  VERIFY_IS_EQUAL(a[0], std::complex<double>(30.0, 0.0));

  a[0] = {0.0, 0.0};
  a[1] = {0.0, 0.0};
  BLASFUNC(zgeru)
  (&two, &one, reinterpret_cast<double*>(&alpha), reinterpret_cast<double*>(x), &one, reinterpret_cast<double*>(&zy),
   &large_lda, reinterpret_cast<double*>(a), &large_lda);
  VERIFY_IS_EQUAL(a[0], std::complex<double>(6.0, -2.0));
  VERIFY_IS_EQUAL(a[1], std::complex<double>(12.0, -4.0));

  BLASFUNC(zgerc)
  (&two, &one, reinterpret_cast<double*>(&alpha), reinterpret_cast<double*>(x), &one, reinterpret_cast<double*>(&zy),
   &large_lda, reinterpret_cast<double*>(a), &large_lda);
  VERIFY_IS_EQUAL(a[0], std::complex<double>(12.0, 0.0));
  VERIFY_IS_EQUAL(a[1], std::complex<double>(24.0, 0.0));
}

}  // namespace

EIGEN_DECLARE_TEST(blas_large_stride) {
  CALL_SUBTEST(test_level1_large_stride());
  CALL_SUBTEST(test_level2_real_large_stride());
  CALL_SUBTEST(test_level2_complex_large_stride());
}
