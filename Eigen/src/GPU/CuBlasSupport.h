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

// ---- Scalar → cublasComputeType_t -------------------------------------------
// cublasGemmEx requires a compute type (separate from the data type).
//
// Precision policy:
//   - Default: tensor core algorithms enabled (CUBLAS_GEMM_DEFAULT_TENSOR_OP).
//     For double, cuBLAS may use Ozaki emulation on sm_80+ tensor cores.
//   - EIGEN_CUDA_TF32: opt-in to TF32 for float (~2x faster, 10-bit mantissa).
//   - EIGEN_NO_CUDA_TENSOR_OPS: disables all tensor core usage. Uses pedantic
//     compute types and CUBLAS_GEMM_DEFAULT algorithm. For bit-exact reproducibility.

template <typename Scalar>
struct cuda_compute_type;

template <>
struct cuda_compute_type<float> {
#if defined(EIGEN_NO_CUDA_TENSOR_OPS)
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F_PEDANTIC;
#elif defined(EIGEN_CUDA_TF32)
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F;
#endif
};
template <>
struct cuda_compute_type<double> {
#ifdef EIGEN_NO_CUDA_TENSOR_OPS
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_64F_PEDANTIC;
#else
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_64F;
#endif
};
template <>
struct cuda_compute_type<std::complex<float>> {
#if defined(EIGEN_NO_CUDA_TENSOR_OPS)
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F_PEDANTIC;
#elif defined(EIGEN_CUDA_TF32)
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F;
#endif
};
template <>
struct cuda_compute_type<std::complex<double>> {
#ifdef EIGEN_NO_CUDA_TENSOR_OPS
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_64F_PEDANTIC;
#else
  static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_64F;
#endif
};
// ---- GEMM algorithm hint ----------------------------------------------------

constexpr cublasGemmAlgo_t cuda_gemm_algo() {
#ifdef EIGEN_NO_CUDA_TENSOR_OPS
  return CUBLAS_GEMM_DEFAULT;
#else
  return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#endif
}

// ---- Alpha/beta scalar type for cublasGemmEx --------------------------------
// For standard types, alpha/beta match the scalar type.

template <typename Scalar>
struct cuda_gemm_scalar {
  using type = Scalar;
};

// ---- Type-specific cuBLAS wrappers ------------------------------------------
// cuBLAS uses separate functions per type (Strsm, Dtrsm, etc.).
// These overloaded wrappers allow calling cublasXtrsm/cublasXsymm/cublasXsyrk
// with any supported scalar type.

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
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  std::complex<float>* B, int ldb) {
  return cublasCtrsm(h, side, uplo, trans, diag, m, n, reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
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
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  const std::complex<float>* B, int ldb, const std::complex<float>* beta,
                                  std::complex<float>* C, int ldc) {
  return cublasChemm(h, side, uplo, m, n, reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb,
                     reinterpret_cast<const cuComplex*>(beta), reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* B, int ldb, const std::complex<double>* beta,
                                  std::complex<double>* C, int ldc) {
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
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const float* alpha, const std::complex<float>* A, int lda, const float* beta,
                                  std::complex<float>* C, int ldc) {
  return cublasCherk(h, uplo, trans, n, k, alpha, reinterpret_cast<const cuComplex*>(A), lda, beta,
                     reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXsyrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                                  const double* alpha, const std::complex<double>* A, int lda, const double* beta,
                                  std::complex<double>* C, int ldc) {
  return cublasZherk(h, uplo, trans, n, k, alpha, reinterpret_cast<const cuDoubleComplex*>(A), lda, beta,
                     reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// GEAM wrappers: C = alpha * op(A) + beta * op(B)
// Covers transpose, scale, matrix add/subtract in one call.
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                  const float* alpha, const float* A, int lda, const float* beta, const float* B,
                                  int ldb, float* C, int ldc) {
  return cublasSgeam(h, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                  const double* alpha, const double* A, int lda, const double* beta, const double* B,
                                  int ldb, double* C, int ldc) {
  return cublasDgeam(h, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  const std::complex<float>* beta, const std::complex<float>* B, int ldb,
                                  std::complex<float>* C, int ldc) {
  return cublasCgeam(h, transa, transb, m, n, reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(beta),
                     reinterpret_cast<const cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* beta, const std::complex<double>* B, int ldb,
                                  std::complex<double>* C, int ldc) {
  return cublasZgeam(h, transa, transb, m, n, reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(beta),
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb, reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_GPU_CUBLAS_SUPPORT_H
