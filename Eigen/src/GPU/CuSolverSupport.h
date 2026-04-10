// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cuSOLVER-specific support types:
//   - cuSOLVER error-checking macro
//   - RAII wrapper for cusolverDnParams
//   - Scalar → cudaDataType_t mapping
//   - (UpLo, StorageOrder) → cublasFillMode_t mapping
//
// Generic CUDA runtime utilities (DeviceBuffer, EIGEN_CUDA_RUNTIME_CHECK)
// are in GpuSupport.h.

#ifndef EIGEN_GPU_CUSOLVER_SUPPORT_H
#define EIGEN_GPU_CUSOLVER_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSupport.h"
#include <cusolverDn.h>

namespace Eigen {
namespace gpu {
namespace internal {

// ---- Error-checking macros --------------------------------------------------

#define EIGEN_CUSOLVER_CHECK(expr)                                         \
  do {                                                                     \
    cusolverStatus_t _s = (expr);                                          \
    eigen_assert(_s == CUSOLVER_STATUS_SUCCESS && "cuSOLVER call failed"); \
  } while (0)

// ---- RAII: cusolverDnParams -------------------------------------------------

struct CusolverParams {
  cusolverDnParams_t p = nullptr;

  CusolverParams() { EIGEN_CUSOLVER_CHECK(cusolverDnCreateParams(&p)); }

  ~CusolverParams() {
    if (p) (void)cusolverDnDestroyParams(p);  // destructor: can't propagate
  }

  // Move-only.
  CusolverParams(CusolverParams&& o) noexcept : p(o.p) { o.p = nullptr; }
  CusolverParams& operator=(CusolverParams&& o) noexcept {
    if (this != &o) {
      if (p) (void)cusolverDnDestroyParams(p);
      p = o.p;
      o.p = nullptr;
    }
    return *this;
  }

  CusolverParams(const CusolverParams&) = delete;
  CusolverParams& operator=(const CusolverParams&) = delete;
};

// ---- Scalar → cudaDataType_t ------------------------------------------------
// Alias for backward compatibility. The canonical trait is cuda_data_type<> in GpuSupport.h.
template <typename Scalar>
using cusolver_data_type = cuda_data_type<Scalar>;

// ---- (UpLo, StorageOrder) → cublasFillMode_t --------------------------------
// cuSOLVER always interprets the matrix as column-major. A row-major matrix A
// appears as A^T to cuSOLVER, so the upper/lower triangle is swapped.

template <int UpLo, int StorageOrder>
struct cusolver_fill_mode;

template <>
struct cusolver_fill_mode<Lower, ColMajor> {
  static constexpr cublasFillMode_t value = CUBLAS_FILL_MODE_LOWER;
};
template <>
struct cusolver_fill_mode<Upper, ColMajor> {
  static constexpr cublasFillMode_t value = CUBLAS_FILL_MODE_UPPER;
};
template <>
struct cusolver_fill_mode<Lower, RowMajor> {
  static constexpr cublasFillMode_t value = CUBLAS_FILL_MODE_UPPER;
};
template <>
struct cusolver_fill_mode<Upper, RowMajor> {
  static constexpr cublasFillMode_t value = CUBLAS_FILL_MODE_LOWER;
};

// ---- Type-specific cuSOLVER wrappers ----------------------------------------
// cuSOLVER does not provide generic X variants for ormqr/unmqr. These overloaded
// wrappers dispatch to the correct type-specific function.
// For real types: ormqr (orthogonal Q). For complex types: unmqr (unitary Q).

inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const float* A, int lda, const float* tau, float* C, int ldc,
                                         float* work, int lwork, int* info) {
  return cusolverDnSormqr(h, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const double* A, int lda, const double* tau, double* C, int ldc,
                                         double* work, int lwork, int* info) {
  return cusolverDnDormqr(h, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const std::complex<float>* A, int lda,
                                         const std::complex<float>* tau, std::complex<float>* C, int ldc,
                                         std::complex<float>* work, int lwork, int* info) {
  return cusolverDnCunmqr(h, side, trans, m, n, k, reinterpret_cast<const cuComplex*>(A), lda,
                          reinterpret_cast<const cuComplex*>(tau), reinterpret_cast<cuComplex*>(C), ldc,
                          reinterpret_cast<cuComplex*>(work), lwork, info);
}
inline cusolverStatus_t cusolverDnXormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,
                                         int n, int k, const std::complex<double>* A, int lda,
                                         const std::complex<double>* tau, std::complex<double>* C, int ldc,
                                         std::complex<double>* work, int lwork, int* info) {
  return cusolverDnZunmqr(h, side, trans, m, n, k, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                          reinterpret_cast<const cuDoubleComplex*>(tau), reinterpret_cast<cuDoubleComplex*>(C), ldc,
                          reinterpret_cast<cuDoubleComplex*>(work), lwork, info);
}

// Buffer size wrappers for ormqr/unmqr.
inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k, const float* A,
                                                    int lda, const float* tau, const float* C, int ldc, int* lwork) {
  return cusolverDnSormqr_bufferSize(h, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}
inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k, const double* A,
                                                    int lda, const double* tau, const double* C, int ldc, int* lwork) {
  return cusolverDnDormqr_bufferSize(h, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}
inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k,
                                                    const std::complex<float>* A, int lda,
                                                    const std::complex<float>* tau, const std::complex<float>* C,
                                                    int ldc, int* lwork) {
  return cusolverDnCunmqr_bufferSize(h, side, trans, m, n, k, reinterpret_cast<const cuComplex*>(A), lda,
                                     reinterpret_cast<const cuComplex*>(tau), reinterpret_cast<const cuComplex*>(C),
                                     ldc, lwork);
}
inline cusolverStatus_t cusolverDnXormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side,
                                                    cublasOperation_t trans, int m, int n, int k,
                                                    const std::complex<double>* A, int lda,
                                                    const std::complex<double>* tau, const std::complex<double>* C,
                                                    int ldc, int* lwork) {
  return cusolverDnZunmqr_bufferSize(h, side, trans, m, n, k, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                                     reinterpret_cast<const cuDoubleComplex*>(tau),
                                     reinterpret_cast<const cuDoubleComplex*>(C), ldc, lwork);
}

}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_CUSOLVER_SUPPORT_H
