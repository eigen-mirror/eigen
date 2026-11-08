// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"

#if !ISCOMPLEX

// Compute sqrt(x*x + y*y) taking care not to cause unnecessary overflow.
extern "C" EIGEN_LAPACK_API Scalar EIGEN_BLAS_FUNC_NAME(lapy2)(Scalar* x, Scalar* y) {
  return Eigen::internal::xlapy2(*x, *y);
}

// Compute sqrt(x*x + y*y + z*z) taking care not to cause unnecessary overflow.
extern "C" EIGEN_LAPACK_API Scalar EIGEN_BLAS_FUNC_NAME(lapy3)(Scalar* x, Scalar* y, Scalar* z) {
  return Eigen::internal::xlapy3(*x, *y, *z);
}

// Determines machine parameters according to LAPACK specification.
extern "C" EIGEN_LAPACK_API Scalar EIGEN_BLAS_FUNC_NAME(lamch)(const char* cmach) {
  return Eigen::internal::xlamch<Scalar>(*cmach);
}

// xLAMC3 is intended to force A and B to be stored prior to doing
// the addition of A and B, for use in situations where optimizers
// might hold one of these in a register.
extern "C" EIGEN_LAPACK_API Scalar EIGEN_BLAS_FUNC_NAME(lamc3)(Scalar* a, Scalar* b) {
  volatile Scalar retval = *a + *b;
  return retval;
}

// Performs complex division in real arithmetic, avoiding unnecessary overflow.
// Consistent with LAPACK SLADIV / DLADIV.
EIGEN_LAPACK_FUNC(ladiv)(const Scalar* a, const Scalar* b, const Scalar* c, const Scalar* d, Scalar* p, Scalar* q) {
  const std::complex<Scalar> res = Eigen::internal::xladiv(*a, *b, *c, *d);
  *p = Eigen::numext::real(res);
  *q = Eigen::numext::imag(res);
}

// Last non-zero column/row scanning for real matrices.
extern "C" EIGEN_LAPACK_API EIGEN_BLAS_INT EIGEN_CAT(ila, EIGEN_CAT(SCALAR_SUFFIX, lc_))(EIGEN_BLAS_INT* m,
                                                                                         EIGEN_BLAS_INT* n, Scalar* a,
                                                                                         EIGEN_BLAS_INT* lda) {
  return Eigen::internal::ilaxlc(*m, *n, a, *lda);
}

extern "C" EIGEN_LAPACK_API EIGEN_BLAS_INT EIGEN_CAT(ila, EIGEN_CAT(SCALAR_SUFFIX, lr_))(EIGEN_BLAS_INT* m,
                                                                                         EIGEN_BLAS_INT* n, Scalar* a,
                                                                                         EIGEN_BLAS_INT* lda) {
  return Eigen::internal::ilaxlr(*m, *n, a, *lda);
}

#else

// C layout of a Fortran COMPLEX or COMPLEX*16 function result.
using ComplexReturnType =
    std::conditional_t<std::is_same<RealScalar, float>::value, eigen_lapack_complex_float, eigen_lapack_complex_double>;

// Conjugates entries of vector x.
EIGEN_LAPACK_FUNC(lacgv)(EIGEN_BLAS_INT* n, RealScalar* px, EIGEN_BLAS_INT* incx) {
  Eigen::internal::xlacgv(*n, reinterpret_cast<Scalar*>(px), *incx);
}

// Performs complex division avoiding unnecessary overflow, consistent with LAPACK CLADIV / ZLADIV.
// Returning a two-member struct by value matches a Fortran COMPLEX function result under the SysV AMD64 ABI (gfortran
// on Linux/x86_64), as cdotc/zdotc in blas/complexdots.cpp do. Other conventions, such as MSVC's or f2c's, may return
// complex Fortran functions through a hidden pointer instead.
extern "C" EIGEN_LAPACK_API ComplexReturnType EIGEN_BLAS_FUNC_NAME(ladiv)(const RealScalar* px, const RealScalar* py) {
  const auto* x = reinterpret_cast<const Scalar*>(px);
  const auto* y = reinterpret_cast<const Scalar*>(py);
  const Scalar res = Eigen::internal::xladiv(*x, *y);
  return {Eigen::numext::real(res), Eigen::numext::imag(res)};
}

// Last non-zero column/row scanning for complex matrices.
extern "C" EIGEN_LAPACK_API EIGEN_BLAS_INT EIGEN_CAT(ila, EIGEN_CAT(SCALAR_SUFFIX,
                                                                    lc_))(EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n,
                                                                          RealScalar* a, EIGEN_BLAS_INT* lda) {
  return Eigen::internal::ilaxlc(*m, *n, reinterpret_cast<Scalar*>(a), *lda);
}

extern "C" EIGEN_LAPACK_API EIGEN_BLAS_INT EIGEN_CAT(ila, EIGEN_CAT(SCALAR_SUFFIX,
                                                                    lr_))(EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n,
                                                                          RealScalar* a, EIGEN_BLAS_INT* lda) {
  return Eigen::internal::ilaxlr(*m, *n, reinterpret_cast<Scalar*>(a), *lda);
}

#endif
