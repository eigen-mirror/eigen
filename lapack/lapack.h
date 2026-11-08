// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef LAPACK_H
#define LAPACK_H

#include "../blas/blas.h"

#if defined(_WIN32)
#if defined(EIGEN_LAPACK_BUILD_DLL)
#define EIGEN_LAPACK_API __declspec(dllexport)
#elif defined(EIGEN_LAPACK_LINK_DLL)
#define EIGEN_LAPACK_API __declspec(dllimport)
#else
#define EIGEN_LAPACK_API
#endif
#elif ((defined(__GNUC__) && __GNUC__ >= 4) || defined(__clang__)) && defined(EIGEN_LAPACK_BUILD_DLL)
#define EIGEN_LAPACK_API __attribute__((visibility("default")))
#else
#define EIGEN_LAPACK_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float r, i;
} eigen_lapack_complex_float;

typedef struct {
  double r, i;
} eigen_lapack_complex_double;

// Exactly the routines this library defines, each also listed in eigen_lapack.def. EIGEN_LAPACK_API on these
// declarations exports the definitions from the shared library.

// Cholesky.
EIGEN_LAPACK_API void BLASFUNC(spotrf)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dpotrf)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cpotrf)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zpotrf)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(spotrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dpotrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cpotrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zpotrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(spstrf)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dpstrf)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cpstrf)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zpstrf)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);

// Symmetric and Hermitian indefinite (Bunch-Kaufman).
EIGEN_LAPACK_API void BLASFUNC(ssytrf)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dsytrf)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(chetrf)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zhetrf)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(ssytrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dsytrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(chetrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zhetrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(spotri)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dpotri)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cpotri)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zpotri)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

// Eigenvalues.
EIGEN_LAPACK_API void BLASFUNC(ssyev)(const char *, const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                      float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dsyev)(const char *, const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                      double *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cheev)(const char *, const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                      float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zheev)(const char *, const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                      double *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

// LU.
EIGEN_LAPACK_API void BLASFUNC(sgetrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dgetrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cgetrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zgetrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(sgetrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dgetrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cgetrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zgetrs)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(sgetri)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dgetri)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cgetri)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zgetri)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

// Triangular.
EIGEN_LAPACK_API void BLASFUNC(strtri)(const char *, const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dtrtri)(const char *, const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(ctrtri)(const char *, const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(ztrtri)(const char *, const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(slauum)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dlauum)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(clauum)(const char *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zlauum)(const char *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(strtrs)(const char *, const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dtrtrs)(const char *, const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(ctrtrs)(const char *, const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(ztrtrs)(const char *, const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

// SVD.
EIGEN_LAPACK_API void BLASFUNC(sgesdd)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dgesdd)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       double *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cgesdd)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zgesdd)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       double *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(sgesvd)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dgesvd)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cgesvd)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zgesvd)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

// QR.
EIGEN_LAPACK_API void BLASFUNC(sgeqrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, float *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dgeqrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                       double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cgeqrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, float *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zgeqrf)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                       double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(sorgqr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dorgqr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       double *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cungqr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                       float *, float *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zungqr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                       double *, double *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(sormqr)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dormqr)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(cunmqr)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       float *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zunmqr)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                       double *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

// Time.
EIGEN_LAPACK_API float BLASFUNC(second)();
EIGEN_LAPACK_API double BLASFUNC(dsecnd)();

// Householder.
EIGEN_LAPACK_API void BLASFUNC(slarfg)(EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_LAPACK_API void BLASFUNC(dlarfg)(EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *);
EIGEN_LAPACK_API void BLASFUNC(clarfg)(EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_LAPACK_API void BLASFUNC(zlarfg)(EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *);

EIGEN_LAPACK_API void BLASFUNC(slarf)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                      float *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_LAPACK_API void BLASFUNC(dlarf)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                      double *, double *, EIGEN_BLAS_INT *, double *);
EIGEN_LAPACK_API void BLASFUNC(clarf)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *,
                                      float *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_LAPACK_API void BLASFUNC(zlarf)(const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                      double *, double *, EIGEN_BLAS_INT *, double *);

EIGEN_LAPACK_API void BLASFUNC(slarft)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dlarft)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(clarft)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zlarft)(const char *, const char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(slarfb)(const char *, const char *, const char *, const char *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(dlarfb)(const char *, const char *, const char *, const char *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(clarfb)(const char *, const char *, const char *, const char *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                       EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zlarfb)(const char *, const char *, const char *, const char *, EIGEN_BLAS_INT *,
                                       EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                       EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

// Utilities.
EIGEN_LAPACK_API float BLASFUNC(slapy2)(float *, float *);
EIGEN_LAPACK_API double BLASFUNC(dlapy2)(double *, double *);

EIGEN_LAPACK_API float BLASFUNC(slapy3)(float *, float *, float *);
EIGEN_LAPACK_API double BLASFUNC(dlapy3)(double *, double *, double *);

EIGEN_LAPACK_API void BLASFUNC(clacgv)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API void BLASFUNC(zlacgv)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API float BLASFUNC(slamch)(const char *);
EIGEN_LAPACK_API double BLASFUNC(dlamch)(const char *);
EIGEN_LAPACK_API float BLASFUNC(slamc3)(float *, float *);
EIGEN_LAPACK_API double BLASFUNC(dlamc3)(double *, double *);

EIGEN_LAPACK_API void BLASFUNC(sladiv)(const float *, const float *, const float *, const float *, float *, float *);
EIGEN_LAPACK_API void BLASFUNC(dladiv)(const double *, const double *, const double *, const double *, double *,
                                       double *);

// Note: Returning complex structs by value adheres to the SysV AMD64 ABI (gfortran
// on Linux/x86_64), consistent with the return-by-value convention used for cdotc/zdotc
// in blas/complexdots.cpp.
EIGEN_LAPACK_API eigen_lapack_complex_float BLASFUNC(cladiv)(const float *, const float *);
EIGEN_LAPACK_API eigen_lapack_complex_double BLASFUNC(zladiv)(const double *, const double *);

EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(ilaslc)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(iladlc)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(ilaclc)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(ilazlc)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(ilaslr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(iladlr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(ilaclr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_LAPACK_API EIGEN_BLAS_INT BLASFUNC(ilazlr)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_LAPACK_API void BLASFUNC(ilaver)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *);

#ifdef __cplusplus
}
#endif

#endif
