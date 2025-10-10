#ifndef LAPACK_H
#define LAPACK_H

#include "../blas/blas.h"

#ifdef __cplusplus
extern "C" {
#endif

EIGEN_BLAS_API void BLASFUNC(csymv)(const char *, const int *, const float *, const float *, const int *, const float *,
                                    const int *, const float *, float *, const int *);
EIGEN_BLAS_API void BLASFUNC(zsymv)(const char *, const int *, const double *, const double *, const int *,
                                    const double *, const int *, const double *, double *, const int *);
EIGEN_BLAS_API void BLASFUNC(xsymv)(const char *, const int *, const double *, const double *, const int *,
                                    const double *, const int *, const double *, double *, const int *);

EIGEN_BLAS_API void BLASFUNC(cspmv)(char *, int *, float *, float *, float *, int *, float *, float *, int *);
EIGEN_BLAS_API void BLASFUNC(zspmv)(char *, int *, double *, double *, double *, int *, double *, double *, int *);
EIGEN_BLAS_API void BLASFUNC(xspmv)(char *, int *, double *, double *, double *, int *, double *, double *, int *);

EIGEN_BLAS_API void BLASFUNC(csyr)(char *, int *, float *, float *, int *, float *, int *);
EIGEN_BLAS_API void BLASFUNC(zsyr)(char *, int *, double *, double *, int *, double *, int *);
EIGEN_BLAS_API void BLASFUNC(xsyr)(char *, int *, double *, double *, int *, double *, int *);

EIGEN_BLAS_API void BLASFUNC(cspr)(char *, int *, float *, float *, int *, float *);
EIGEN_BLAS_API void BLASFUNC(zspr)(char *, int *, double *, double *, int *, double *);
EIGEN_BLAS_API void BLASFUNC(xspr)(char *, int *, double *, double *, int *, double *);

EIGEN_BLAS_API void BLASFUNC(sgemt)(char *, int *, int *, float *, float *, int *, float *, int *);
EIGEN_BLAS_API void BLASFUNC(dgemt)(char *, int *, int *, double *, double *, int *, double *, int *);
EIGEN_BLAS_API void BLASFUNC(cgemt)(char *, int *, int *, float *, float *, int *, float *, int *);
EIGEN_BLAS_API void BLASFUNC(zgemt)(char *, int *, int *, double *, double *, int *, double *, int *);

EIGEN_BLAS_API void BLASFUNC(sgema)(char *, char *, int *, int *, float *, float *, int *, float *, float *, int *,
                                    float *, int *);
EIGEN_BLAS_API void BLASFUNC(dgema)(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *,
                                    double *, int *);
EIGEN_BLAS_API void BLASFUNC(cgema)(char *, char *, int *, int *, float *, float *, int *, float *, float *, int *,
                                    float *, int *);
EIGEN_BLAS_API void BLASFUNC(zgema)(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *,
                                    double *, int *);

EIGEN_BLAS_API void BLASFUNC(sgems)(char *, char *, int *, int *, float *, float *, int *, float *, float *, int *,
                                    float *, int *);
EIGEN_BLAS_API void BLASFUNC(dgems)(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *,
                                    double *, int *);
EIGEN_BLAS_API void BLASFUNC(cgems)(char *, char *, int *, int *, float *, float *, int *, float *, float *, int *,
                                    float *, int *);
EIGEN_BLAS_API void BLASFUNC(zgems)(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *,
                                    double *, int *);

EIGEN_BLAS_API void BLASFUNC(sgetf2)(int *, int *, float *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dgetf2)(int *, int *, double *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qgetf2)(int *, int *, double *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(cgetf2)(int *, int *, float *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zgetf2)(int *, int *, double *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xgetf2)(int *, int *, double *, int *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(sgetrf)(int *, int *, float *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dgetrf)(int *, int *, double *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qgetrf)(int *, int *, double *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(cgetrf)(int *, int *, float *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zgetrf)(int *, int *, double *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xgetrf)(int *, int *, double *, int *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(slaswp)(int *, float *, int *, int *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dlaswp)(int *, double *, int *, int *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qlaswp)(int *, double *, int *, int *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(claswp)(int *, float *, int *, int *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zlaswp)(int *, double *, int *, int *, int *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xlaswp)(int *, double *, int *, int *, int *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(sgetrs)(char *, int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(cgetrs)(char *, int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(sgesv)(int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dgesv)(int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qgesv)(int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(cgesv)(int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zgesv)(int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xgesv)(int *, int *, double *, int *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(spotf2)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dpotf2)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qpotf2)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(cpotf2)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zpotf2)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xpotf2)(char *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(spotrf)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dpotrf)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qpotrf)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(cpotrf)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zpotrf)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xpotrf)(char *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(slauu2)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dlauu2)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qlauu2)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(clauu2)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zlauu2)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xlauu2)(char *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(slauum)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dlauum)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qlauum)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(clauum)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zlauum)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xlauum)(char *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(strti2)(char *, char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dtrti2)(char *, char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qtrti2)(char *, char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(ctrti2)(char *, char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(ztrti2)(char *, char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xtrti2)(char *, char *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(strtri)(char *, char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dtrtri)(char *, char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qtrtri)(char *, char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(ctrtri)(char *, char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(ztrtri)(char *, char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xtrtri)(char *, char *, int *, double *, int *, int *);

EIGEN_BLAS_API void BLASFUNC(spotri)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(dpotri)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(qpotri)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(cpotri)(char *, int *, float *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(zpotri)(char *, int *, double *, int *, int *);
EIGEN_BLAS_API void BLASFUNC(xpotri)(char *, int *, double *, int *, int *);

#ifdef __cplusplus
}
#endif

#endif
