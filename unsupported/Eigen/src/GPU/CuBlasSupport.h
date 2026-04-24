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
#include <cublasLt.h>
#include <cstring>

namespace Eigen {
namespace gpu {

// ---- Operation enum ---------------------------------------------------------
// Public flag for transpose/adjoint in BLAS- and solver-style calls.

enum class GpuOp { NoTrans, Trans, ConjTrans };

namespace internal {

// ---- Error-checking macro ---------------------------------------------------

#define EIGEN_CUBLAS_CHECK(expr)                                       \
  do {                                                                 \
    cublasStatus_t _s = (expr);                                        \
    eigen_assert(_s == CUBLAS_STATUS_SUCCESS && "cuBLAS call failed"); \
  } while (0)

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
static_assert(sizeof(cuComplex) == sizeof(std::complex<float>), "cuComplex and std::complex<float> layout mismatch");
static_assert(sizeof(cuDoubleComplex) == sizeof(std::complex<double>),
              "cuDoubleComplex and std::complex<double> layout mismatch");

// Complex scalar args (alpha, beta) are type-punned from std::complex<T>*
// to cuComplex*/cuDoubleComplex*.  A reinterpret_cast violates strict
// aliasing: when inlined, clang/MSVC can elide the caller's store (the
// compiler no longer sees a read through the original type), causing
// segfaults.  We use memcpy — the standard-blessed type-pun — for scalars.
// Device array pointers (A, B, C) are opaque to the host compiler, so
// reinterpret_cast is safe there.
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  const std::complex<float>* B, int ldb, const std::complex<float>* beta,
                                  std::complex<float>* C, int ldc) {
  cuComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasCgemm(h, transA, transB, m, n, k, &a, reinterpret_cast<const cuComplex*>(A), lda,
                     reinterpret_cast<const cuComplex*>(B), ldb, &b, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXgemm(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  int k, const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* B, int ldb, const std::complex<double>* beta,
                                  std::complex<double>* C, int ldc) {
  cuDoubleComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasZgemm(h, transA, transB, m, n, k, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb, &b, reinterpret_cast<cuDoubleComplex*>(C), ldc);
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
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  std::complex<float>* B, int ldb) {
  cuComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasCtrsm(h, side, uplo, trans, diag, m, n, &a, reinterpret_cast<const cuComplex*>(A), lda,
                     reinterpret_cast<cuComplex*>(B), ldb);
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  std::complex<double>* B, int ldb) {
  cuDoubleComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasZtrsm(h, side, uplo, trans, diag, m, n, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<cuDoubleComplex*>(B), ldb);
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
  cuComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasChemm(h, side, uplo, m, n, &a, reinterpret_cast<const cuComplex*>(A), lda,
                     reinterpret_cast<const cuComplex*>(B), ldb, &b, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXsymm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* B, int ldb, const std::complex<double>* beta,
                                  std::complex<double>* C, int ldc) {
  cuDoubleComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasZhemm(h, side, uplo, m, n, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb, &b, reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// GEAM wrappers: C = alpha * op(A) + beta * op(B)
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const float* alpha, const float* A, int lda, const float* beta, const float* B,
                                  int ldb, float* C, int ldc) {
  return cublasSgeam(h, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const double* alpha, const double* A, int lda, const double* beta, const double* B,
                                  int ldb, double* C, int ldc) {
  return cublasDgeam(h, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const std::complex<float>* alpha, const std::complex<float>* A, int lda,
                                  const std::complex<float>* beta, const std::complex<float>* B, int ldb,
                                  std::complex<float>* C, int ldc) {
  cuComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasCgeam(h, transA, transB, m, n, &a, reinterpret_cast<const cuComplex*>(A), lda, &b,
                     reinterpret_cast<const cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXgeam(cublasHandle_t h, cublasOperation_t transA, cublasOperation_t transB, int m, int n,
                                  const std::complex<double>* alpha, const std::complex<double>* A, int lda,
                                  const std::complex<double>* beta, const std::complex<double>* B, int ldb,
                                  std::complex<double>* C, int ldc) {
  cuDoubleComplex a, b;
  std::memcpy(&a, alpha, sizeof(a));
  std::memcpy(&b, beta, sizeof(b));
  return cublasZgeam(h, transA, transB, m, n, &a, reinterpret_cast<const cuDoubleComplex*>(A), lda, &b,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb, reinterpret_cast<cuDoubleComplex*>(C), ldc);
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

// DGMM wrappers: C = A * diag(x)  (side=RIGHT) or C = diag(x) * A  (side=LEFT).
// Useful for applying a diagonal scaling without materialising diag(x) as a
// dense matrix. cuBLAS docs guarantee in-place is safe when C == A.
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const float* A, int lda,
                                  const float* x, int incx, float* C, int ldc) {
  return cublasSdgmm(h, side, m, n, A, lda, x, incx, C, ldc);
}
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const double* A, int lda,
                                  const double* x, int incx, double* C, int ldc) {
  return cublasDdgmm(h, side, m, n, A, lda, x, incx, C, ldc);
}
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const std::complex<float>* A,
                                  int lda, const std::complex<float>* x, int incx, std::complex<float>* C, int ldc) {
  return cublasCdgmm(h, side, m, n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(x),
                     incx, reinterpret_cast<cuComplex*>(C), ldc);
}
inline cublasStatus_t cublasXdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const std::complex<double>* A,
                                  int lda, const std::complex<double>* x, int incx, std::complex<double>* C, int ldc) {
  return cublasZdgmm(h, side, m, n, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

// ---- cuBLAS Level-1 wrappers ------------------------------------------------
// Type-dispatched wrappers for BLAS-1 vector operations: dot, axpy, nrm2, scal, copy.
// These work with CUBLAS_POINTER_MODE_HOST or CUBLAS_POINTER_MODE_DEVICE depending
// on the caller's configuration. For device pointer mode, scalar result pointers
// (dot, nrm2) must point to device memory.

// dot: result = x^T * y (real) or x^H * y (complex conjugate dot)
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const float* x, int incx, const float* y, int incy,
                                 float* result) {
  return cublasSdot(h, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const double* x, int incx, const double* y, int incy,
                                 double* result) {
  return cublasDdot(h, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const std::complex<float>* x, int incx,
                                 const std::complex<float>* y, int incy, std::complex<float>* result) {
  return cublasCdotc(h, n, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<const cuComplex*>(y), incy,
                     reinterpret_cast<cuComplex*>(result));
}
inline cublasStatus_t cublasXdot(cublasHandle_t h, int n, const std::complex<double>* x, int incx,
                                 const std::complex<double>* y, int incy, std::complex<double>* result) {
  return cublasZdotc(h, n, reinterpret_cast<const cuDoubleComplex*>(x), incx,
                     reinterpret_cast<const cuDoubleComplex*>(y), incy, reinterpret_cast<cuDoubleComplex*>(result));
}

// nrm2: result = ||x||_2 (always returns real)
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const float* x, int incx, float* result) {
  return cublasSnrm2(h, n, x, incx, result);
}
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const double* x, int incx, double* result) {
  return cublasDnrm2(h, n, x, incx, result);
}
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const std::complex<float>* x, int incx, float* result) {
  return cublasScnrm2(h, n, reinterpret_cast<const cuComplex*>(x), incx, result);
}
inline cublasStatus_t cublasXnrm2(cublasHandle_t h, int n, const std::complex<double>* x, int incx, double* result) {
  return cublasDznrm2(h, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, result);
}

// axpy: y += alpha * x
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const float* alpha, const float* x, int incx, float* y,
                                  int incy) {
  return cublasSaxpy(h, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const double* alpha, const double* x, int incx, double* y,
                                  int incy) {
  return cublasDaxpy(h, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const std::complex<float>* alpha,
                                  const std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
  cuComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasCaxpy(h, n, &a, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<cuComplex*>(y), incy);
}
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const std::complex<double>* alpha,
                                  const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
  cuDoubleComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasZaxpy(h, n, &a, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(y),
                     incy);
}

// scal: x *= alpha
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const float* alpha, float* x, int incx) {
  return cublasSscal(h, n, alpha, x, incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const double* alpha, double* x, int incx) {
  return cublasDscal(h, n, alpha, x, incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const std::complex<float>* alpha, std::complex<float>* x,
                                  int incx) {
  cuComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasCscal(h, n, &a, reinterpret_cast<cuComplex*>(x), incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const std::complex<double>* alpha, std::complex<double>* x,
                                  int incx) {
  cuDoubleComplex a;
  std::memcpy(&a, alpha, sizeof(a));
  return cublasZscal(h, n, &a, reinterpret_cast<cuDoubleComplex*>(x), incx);
}

// copy: y = x
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const float* x, int incx, float* y, int incy) {
  return cublasScopy(h, n, x, incx, y, incy);
}
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const double* x, int incx, double* y, int incy) {
  return cublasDcopy(h, n, x, incx, y, incy);
}
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const std::complex<float>* x, int incx,
                                  std::complex<float>* y, int incy) {
  return cublasCcopy(h, n, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<cuComplex*>(y), incy);
}
inline cublasStatus_t cublasXcopy(cublasHandle_t h, int n, const std::complex<double>* x, int incx,
                                  std::complex<double>* y, int incy) {
  return cublasZcopy(h, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(y),
                     incy);
}

}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_CUBLAS_SUPPORT_H
