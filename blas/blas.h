// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef BLAS_H
#define BLAS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(EIGEN_BLAS_BUILD_DLL)
#define EIGEN_BLAS_API __declspec(dllexport)
#elif defined(EIGEN_BLAS_LINK_DLL)
#define EIGEN_BLAS_API __declspec(dllimport)
#else
#define EIGEN_BLAS_API
#endif
#elif ((defined(__GNUC__) && __GNUC__ >= 4) || defined(__clang__)) && defined(EIGEN_BLAS_BUILD_DLL)
#define EIGEN_BLAS_API __attribute__((visibility("default")))
#else
#define EIGEN_BLAS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BLAS_FUNC_SUFFIX
#define BLAS_FUNC_SUFFIX _
#endif

#define CONCAT_EXPAND(a, b) a##b
#define CONCAT(a, b) CONCAT_EXPAND(a, b)
#define BLASFUNC(FUNC) CONCAT(FUNC, BLAS_FUNC_SUFFIX)

#ifndef EIGEN_BLAS_INT
#if defined(EIGEN_64BIT_BLAS)
#define EIGEN_BLAS_INT int64_t
#else
#define EIGEN_BLAS_INT int
#endif
#endif

EIGEN_BLAS_API int BLASFUNC(lsame)(const char *, const char *);
EIGEN_BLAS_API void BLASFUNC(xerbla)(const char *, EIGEN_BLAS_INT *info, size_t len);

EIGEN_BLAS_API float BLASFUNC(sdot)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API float BLASFUNC(sdsdot)(EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API double BLASFUNC(dsdot)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API double BLASFUNC(ddot)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

// cdotc, cdotu, zdotc and zdotu return a complex value; complexdots.cpp defines them on top of these workers.
EIGEN_BLAS_API void BLASFUNC(cdotuw)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_BLAS_API void BLASFUNC(cdotcw)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_BLAS_API void BLASFUNC(zdotuw)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                     double *);
EIGEN_BLAS_API void BLASFUNC(zdotcw)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *,
                                     double *);

EIGEN_BLAS_API void BLASFUNC(saxpy)(const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(daxpy)(const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(caxpy)(const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zaxpy)(const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(saxpby)(const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                     const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(daxpby)(const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                     const double *, double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(caxpby)(const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                     const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zaxpby)(const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                     const double *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(scopy)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dcopy)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ccopy)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zcopy)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(sswap)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dswap)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cswap)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zswap)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API float BLASFUNC(sasum)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API float BLASFUNC(scasum)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API double BLASFUNC(dasum)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API double BLASFUNC(dzasum)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(isamax)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(idamax)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(icamax)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(izamax)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(isamin)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(idamin)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(icamin)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API EIGEN_BLAS_INT BLASFUNC(izamin)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(sscal)(EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dscal)(EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cscal)(EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zscal)(EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(csscal)(EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zdscal)(EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API float BLASFUNC(snrm2)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API float BLASFUNC(scnrm2)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API double BLASFUNC(dnrm2)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API double BLASFUNC(dznrm2)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(srot)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                   float *);
EIGEN_BLAS_API void BLASFUNC(drot)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                   double *);
EIGEN_BLAS_API void BLASFUNC(csrot)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *,
                                    float *);
EIGEN_BLAS_API void BLASFUNC(zdrot)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                    double *);

EIGEN_BLAS_API void BLASFUNC(srotg)(float *, float *, float *, float *);
EIGEN_BLAS_API void BLASFUNC(drotg)(double *, double *, double *, double *);
EIGEN_BLAS_API void BLASFUNC(crotg)(float *, float *, float *, float *);
EIGEN_BLAS_API void BLASFUNC(zrotg)(double *, double *, double *, double *);

EIGEN_BLAS_API void BLASFUNC(srotmg)(float *, float *, float *, float *, float *);
EIGEN_BLAS_API void BLASFUNC(drotmg)(double *, double *, double *, double *, double *);

EIGEN_BLAS_API void BLASFUNC(srotm)(EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_BLAS_API void BLASFUNC(drotm)(EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *);

/* Level 2 routines */

EIGEN_BLAS_API void BLASFUNC(sger)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                   EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dger)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                   EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cgeru)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cgerc)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zgeru)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zgerc)(EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(sgemv)(const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *, const float *,
                                    const float *, const EIGEN_BLAS_INT *, const float *, const EIGEN_BLAS_INT *,
                                    const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dgemv)(const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *, const double *,
                                    const double *, const EIGEN_BLAS_INT *, const double *, const EIGEN_BLAS_INT *,
                                    const double *, double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cgemv)(const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *, const float *,
                                    const float *, const EIGEN_BLAS_INT *, const float *, const EIGEN_BLAS_INT *,
                                    const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zgemv)(const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *, const double *,
                                    const double *, const EIGEN_BLAS_INT *, const double *, const EIGEN_BLAS_INT *,
                                    const double *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(strsv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const float *,
                                    const EIGEN_BLAS_INT *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtrsv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const double *,
                                    const EIGEN_BLAS_INT *, double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctrsv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const float *,
                                    const EIGEN_BLAS_INT *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztrsv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const double *,
                                    const EIGEN_BLAS_INT *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(stpsv)(char *, char *, char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtpsv)(char *, char *, char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctpsv)(char *, char *, char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztpsv)(char *, char *, char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(strmv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const float *,
                                    const EIGEN_BLAS_INT *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtrmv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const double *,
                                    const EIGEN_BLAS_INT *, double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctrmv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const float *,
                                    const EIGEN_BLAS_INT *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztrmv)(const char *, const char *, const char *, const EIGEN_BLAS_INT *, const double *,
                                    const EIGEN_BLAS_INT *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(stpmv)(char *, char *, char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtpmv)(char *, char *, char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctpmv)(char *, char *, char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztpmv)(char *, char *, char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(stbmv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtbmv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctbmv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztbmv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(stbsv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtbsv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctbsv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztbsv)(char *, char *, char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(ssymv)(const char *, const EIGEN_BLAS_INT *, const float *, const float *,
                                    const EIGEN_BLAS_INT *, const float *, const EIGEN_BLAS_INT *, const float *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dsymv)(const char *, const EIGEN_BLAS_INT *, const double *, const double *,
                                    const EIGEN_BLAS_INT *, const double *, const EIGEN_BLAS_INT *, const double *,
                                    double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(sspmv)(char *, EIGEN_BLAS_INT *, float *, float *, float *, EIGEN_BLAS_INT *, float *,
                                    float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dspmv)(char *, EIGEN_BLAS_INT *, double *, double *, double *, EIGEN_BLAS_INT *, double *,
                                    double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(ssyr)(const char *, const EIGEN_BLAS_INT *, const float *, const float *,
                                   const EIGEN_BLAS_INT *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dsyr)(const char *, const EIGEN_BLAS_INT *, const double *, const double *,
                                   const EIGEN_BLAS_INT *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(ssyr2)(const char *, const EIGEN_BLAS_INT *, const float *, const float *,
                                    const EIGEN_BLAS_INT *, const float *, const EIGEN_BLAS_INT *, float *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dsyr2)(const char *, const EIGEN_BLAS_INT *, const double *, const double *,
                                    const EIGEN_BLAS_INT *, const double *, const EIGEN_BLAS_INT *, double *,
                                    const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(sspr)(char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_BLAS_API void BLASFUNC(dspr)(char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *);

EIGEN_BLAS_API void BLASFUNC(sspr2)(char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *);
EIGEN_BLAS_API void BLASFUNC(dspr2)(char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *);

EIGEN_BLAS_API void BLASFUNC(cher)(char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                   EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zher)(char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                   EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(chpr)(char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *);
EIGEN_BLAS_API void BLASFUNC(zhpr)(char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *);

EIGEN_BLAS_API void BLASFUNC(cher2)(char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zher2)(char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(chpr2)(char *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *, float *,
                                    EIGEN_BLAS_INT *, float *);
EIGEN_BLAS_API void BLASFUNC(zhpr2)(char *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *, double *,
                                    EIGEN_BLAS_INT *, double *);

EIGEN_BLAS_API void BLASFUNC(chemv)(const char *, const EIGEN_BLAS_INT *, const float *, const float *,
                                    const EIGEN_BLAS_INT *, const float *, const EIGEN_BLAS_INT *, const float *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zhemv)(const char *, const EIGEN_BLAS_INT *, const double *, const double *,
                                    const EIGEN_BLAS_INT *, const double *, const EIGEN_BLAS_INT *, const double *,
                                    double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(chpmv)(char *, EIGEN_BLAS_INT *, float *, float *, float *, EIGEN_BLAS_INT *, float *,
                                    float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zhpmv)(char *, EIGEN_BLAS_INT *, double *, double *, double *, EIGEN_BLAS_INT *, double *,
                                    double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(sgbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                    float *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, float *,
                                    EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dgbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                    double *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                    double *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cgbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                    float *, float *, EIGEN_BLAS_INT *, float *, EIGEN_BLAS_INT *, float *, float *,
                                    EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zgbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *,
                                    double *, double *, EIGEN_BLAS_INT *, double *, EIGEN_BLAS_INT *, double *,
                                    double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(ssbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *,
                                    float *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dsbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *,
                                    double *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(chbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *,
                                    float *, EIGEN_BLAS_INT *, float *, float *, EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zhbmv)(char *, EIGEN_BLAS_INT *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *,
                                    double *, EIGEN_BLAS_INT *, double *, double *, EIGEN_BLAS_INT *);

/* Level 3 routines */

EIGEN_BLAS_API void BLASFUNC(sgemm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    const float *, const EIGEN_BLAS_INT *, const float *, float *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dgemm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    const double *, const EIGEN_BLAS_INT *, const double *, double *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cgemm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    const float *, const EIGEN_BLAS_INT *, const float *, float *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zgemm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    const double *, const EIGEN_BLAS_INT *, const double *, double *,
                                    const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(strsm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtrsm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctrsm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztrsm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(strmm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dtrmm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ctrmm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                    float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(ztrmm)(const char *, const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                    const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                    double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(ssymm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const float *, const float *, const EIGEN_BLAS_INT *, const float *,
                                    const EIGEN_BLAS_INT *, const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dsymm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const double *, const double *, const EIGEN_BLAS_INT *, const double *,
                                    const EIGEN_BLAS_INT *, const double *, double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(csymm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const float *, const float *, const EIGEN_BLAS_INT *, const float *,
                                    const EIGEN_BLAS_INT *, const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zsymm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const double *, const double *, const EIGEN_BLAS_INT *, const double *,
                                    const EIGEN_BLAS_INT *, const double *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(ssyrk)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const float *, const float *, const EIGEN_BLAS_INT *, const float *, float *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dsyrk)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const double *, const double *, const EIGEN_BLAS_INT *, const double *, double *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(csyrk)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const float *, const float *, const EIGEN_BLAS_INT *, const float *, float *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zsyrk)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const double *, const double *, const EIGEN_BLAS_INT *, const double *, double *,
                                    const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(ssyr2k)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                     const float *, const float *, const EIGEN_BLAS_INT *, const float *,
                                     const EIGEN_BLAS_INT *, const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dsyr2k)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                     const double *, const double *, const EIGEN_BLAS_INT *, const double *,
                                     const EIGEN_BLAS_INT *, const double *, double *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(csyr2k)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                     const float *, const float *, const EIGEN_BLAS_INT *, const float *,
                                     const EIGEN_BLAS_INT *, const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zsyr2k)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                     const double *, const double *, const EIGEN_BLAS_INT *, const double *,
                                     const EIGEN_BLAS_INT *, const double *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(chemm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const float *, const float *, const EIGEN_BLAS_INT *, const float *,
                                    const EIGEN_BLAS_INT *, const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zhemm)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const double *, const double *, const EIGEN_BLAS_INT *, const double *,
                                    const EIGEN_BLAS_INT *, const double *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(cherk)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const float *, const float *, const EIGEN_BLAS_INT *, const float *, float *,
                                    const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zherk)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                    const double *, const double *, const EIGEN_BLAS_INT *, const double *, double *,
                                    const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(cher2k)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                     const float *, const float *, const EIGEN_BLAS_INT *, const float *,
                                     const EIGEN_BLAS_INT *, const float *, float *, const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zher2k)(const char *, const char *, const EIGEN_BLAS_INT *, const EIGEN_BLAS_INT *,
                                     const double *, const double *, const EIGEN_BLAS_INT *, const double *,
                                     const EIGEN_BLAS_INT *, const double *, double *, const EIGEN_BLAS_INT *);

EIGEN_BLAS_API void BLASFUNC(sgemmtr)(const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                      const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                      const float *, const EIGEN_BLAS_INT *, const float *, float *,
                                      const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(dgemmtr)(const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                      const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                      const double *, const EIGEN_BLAS_INT *, const double *, double *,
                                      const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(cgemmtr)(const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                      const EIGEN_BLAS_INT *, const float *, const float *, const EIGEN_BLAS_INT *,
                                      const float *, const EIGEN_BLAS_INT *, const float *, float *,
                                      const EIGEN_BLAS_INT *);
EIGEN_BLAS_API void BLASFUNC(zgemmtr)(const char *, const char *, const char *, const EIGEN_BLAS_INT *,
                                      const EIGEN_BLAS_INT *, const double *, const double *, const EIGEN_BLAS_INT *,
                                      const double *, const EIGEN_BLAS_INT *, const double *, double *,
                                      const EIGEN_BLAS_INT *);

#ifdef __cplusplus
}
#endif

#endif
