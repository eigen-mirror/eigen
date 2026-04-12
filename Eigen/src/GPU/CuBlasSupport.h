// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cuBLAS-specific support types:
//   - Error-checking macro
//   - Operation enum and mapping to cublasOperation_t
//
// Generic CUDA runtime utilities (DeviceBuffer, cuda_data_type) are in GpuSupport.h.

#ifndef EIGEN_GPU_CUBLAS_SUPPORT_H
#define EIGEN_GPU_CUBLAS_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSupport.h"
#include <cublas_v2.h>

namespace Eigen {
namespace internal {

// ---- Error-checking macro ---------------------------------------------------

#define EIGEN_CUBLAS_CHECK(expr)                                       \
  do {                                                                 \
    cublasStatus_t _s = (expr);                                        \
    eigen_assert(_s == CUBLAS_STATUS_SUCCESS && "cuBLAS call failed"); \
  } while (0)

// ---- Operation enum ---------------------------------------------------------
// Maps transpose/adjoint flags to cublasOperation_t.

enum class GpuOp { NoTrans, Trans, ConjTrans };

constexpr cublasOperation_t to_cublas_op(GpuOp op) {
  switch (op) {
    case GpuOp::Trans:
      return CUBLAS_OP_T;
    case GpuOp::ConjTrans:
      return CUBLAS_OP_C;
    default:
      return CUBLAS_OP_N;
  }
}

// ---- Type-specific cuBLAS wrappers ------------------------------------------
// cuBLAS uses separate functions per type (Sgemm, Dgemm, etc.).
// These overloaded wrappers allow calling cublasXgemm/cublasXtrsm/etc.
// with any supported scalar type.

// GEMM wrappers
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
                                  const float* beta, float* C, int ldc) {
  return cublasSgemm(h, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const double* alpha, const double* A, int lda, const double* B, int ldb,
                                  const double* beta, double* C, int ldc) {
  return cublasDgemm(h, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
// Complex wrappers must not be inlined: the reinterpret_cast from
// std::complex<T>* to cuComplex*/cuDoubleComplex* is a type-pun that
// causes clang/MSVC optimizers to elide the source memory when the
// wrapper is inlined into the caller (the compiler no longer sees a
// read through the original type).  Keeping these out-of-line ensures
// the caller materialises the scalars before the call.
EIGEN_DONT_INLINE cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB,
                                             int m, int n, int k, const std::complex<float>* alpha,
                                             const std::complex<float>* A, int lda, const std::complex<float>* B,
                                             int ldb, const std::complex<float>* beta, std::complex<float>* C,
                                             int ldc) {
  return cublasCgemm(h, transA, transB, m, n, k, reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb,
                     reinterpret_cast<const cuComplex*>(beta), reinterpret_cast<cuComplex*>(C), ldc);
}
EIGEN_DONT_INLINE cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB,
                                             int m, int n, int k, const std::complex<double>* alpha,
                                             const std::complex<double>* A, int lda, const std::complex<double>* B,
                                             int ldb, const std::complex<double>* beta, std::complex<double>* C,
                                             int ldc) {
  return cublasZgemm(h, transA, transB, m, n, k, reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                     reinterpret_cast<const cuDoubleComplex*>(beta), reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// TRSM wrappers
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha,
                                  const float* A, int lda, float* B, int ldb) {
  return cublasStrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha,
                                  const double* A, int lda, double* B, int ldb) {
  return cublasDtrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
EIGEN_DONT_INLINE cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                             cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                             const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                             std::complex<float>* B, int ldb) {
  return cublasCtrsm(h, side, uplo, trans, diag, m, n, reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
}
EIGEN_DONT_INLINE cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                             cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                             const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                             std::complex<double>* B, int ldb) {
  return cublasZtrsm(h, side, uplo, trans, diag, m, n, reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
}

// SYMM wrappers (real → symm, complex → hemm)
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const float* alpha, const float* A, int lda, const float* B, int ldb,
                                  const float* beta, float* C, int ldc) {
  return cublasSsymm(h, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const double* alpha, const double* A, int lda, const double* B, int ldb,
                                  const double* beta, double* C, int ldc) {
  return cublasDsymm(h, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
EIGEN_DONT_INLINE cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m,
                                             int n, const std::complex<float>* alpha, const std::complex<float>* A,
                                             int lda, const std::complex<float>* B, int ldb,
                                             const std::complex<float>* beta, std::complex<float>* C, int ldc) {
  return cublasChemm(h, side, uplo, m, n, reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb,
                     reinterpret_cast<const cuComplex*>(beta), reinterpret_cast<cuComplex*>(C), ldc);
}
EIGEN_DONT_INLINE cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m,
                                             int n, const std::complex<double>* alpha, const std::complex<double>* A,
                                             int lda, const std::complex<double>* B, int ldb,
                                             const std::complex<double>* beta, std::complex<double>* C, int ldc) {
  return cublasZhemm(h, side, uplo, m, n, reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                     reinterpret_cast<const cuDoubleComplex*>(beta), reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// SYRK wrappers (real → syrk, complex → herk)
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) {
  return cublasSsyrk(h, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const double* alpha, const double* A, int lda, const double* beta, double* C,
                                  int ldc) {
  return cublasDsyrk(h, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
EIGEN_DONT_INLINE cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n,
                                             int k, const float* alpha, const std::complex<float>* A, int lda,
                                             const float* beta, std::complex<float>* C, int ldc) {
  return cublasCherk(h, uplo, trans, n, k, alpha, reinterpret_cast<const cuComplex*>(A), lda, beta,
                     reinterpret_cast<cuComplex*>(C), ldc);
}
EIGEN_DONT_INLINE cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n,
                                             int k, const double* alpha, const std::complex<double>* A, int lda,
                                             const double* beta, std::complex<double>* C, int ldc) {
  return cublasZherk(h, uplo, trans, n, k, alpha, reinterpret_cast<const cuDoubleComplex*>(A), lda, beta,
                     reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_GPU_CUBLAS_SUPPORT_H
