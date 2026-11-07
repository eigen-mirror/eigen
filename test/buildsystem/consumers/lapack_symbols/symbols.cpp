// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Stand-ins for the libraries lapack/testing_symbols.cmake inspects. Each library compiles this file with one
// FAKE_* macro; symbols with a trailing underscore are the Fortran-callable ones.

#if defined(__GNUC__) && !defined(_WIN32)
#define FAKE_EXPORT __attribute__((visibility("default")))
#define FAKE_WEAK __attribute__((weak))
#else
#define FAKE_EXPORT
#define FAKE_WEAK
#endif

#if defined(FAKE_LAPACK)
// Neither a C++ function nor a C symbol without the trailing underscore is Fortran-callable.
void fake_lapack_helper() {}
extern "C" FAKE_EXPORT void eigen_lapack_helper() {}
extern "C" FAKE_EXPORT void dgesv_() {}
#if defined(FAKE_HIDE_DPOTRF)
extern "C" void dpotrf_() {}
#else
extern "C" FAKE_EXPORT void dpotrf_() {}
#endif
#elif defined(FAKE_BLAS)
extern "C" FAKE_EXPORT void daxpy_() {}
// Weak, like Eigen's, so that a program can replace it.
extern "C" FAKE_EXPORT FAKE_WEAK void xerbla_() {}
#elif defined(FAKE_REFERENCE)
// Referenced but not defined here.
extern "C" void dgesv_();
extern "C" void dgetrf_() { dgesv_(); }
#if defined(FAKE_OVERLAP)
extern "C" void dpotrf_() {}
#endif
#endif
