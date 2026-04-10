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
// cublasLtMatmul requires a compute type (separate from the data type).
//
// Precision policy:
//   - Default: tensor core algorithms enabled via cublasLtMatmul heuristics.
//     For double, cuBLAS may use Ozaki emulation on sm_80+ tensor cores.
//   - EIGEN_CUDA_TF32: opt-in to TF32 for float (~2x faster, 10-bit mantissa).
//   - EIGEN_NO_CUDA_TENSOR_OPS: disables all tensor core usage. Uses pedantic
//     compute types. For bit-exact reproducibility.

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
// ---- Alpha/beta scalar type for cublasLtMatmul ------------------------------
// For standard types, alpha/beta match the scalar type.

template <typename Scalar>
struct cuda_gemm_scalar {
  using type = Scalar;
};

// ---- cublasLt GEMM dispatch -------------------------------------------------
// Wraps cublasLtMatmul with descriptor setup, heuristic algorithm selection,
// and lazy workspace management. Supports 64-bit dimensions natively.
//
// The workspace buffer (DeviceBuffer*) is grown lazily to match the selected
// algorithm's actual requirement. The heuristic is queried with a generous
// 32 MB cap so that the best algorithm is never excluded. Growth is monotonic:
// the buffer only grows, never shrinks, so reallocation happens at most a few
// times during the lifetime of the owning GpuContext or solver.
//
// EIGEN_NO_CUDA_TENSOR_OPS: pedantic compute types (CUBLAS_COMPUTE_32F_PEDANTIC,
// CUBLAS_COMPUTE_64F_PEDANTIC) prevent cublasLt from selecting tensor core
// algorithms, matching the previous cublasGemmEx behavior.
//
// Thread safety: the workspace buffer is not thread-safe. All GEMM calls
// sharing a workspace must be on the same CUDA stream (guaranteed by GpuContext's
// single-stream design and by each GpuSVD owning its own stream).
//
// Future optimization: for hot loops (e.g., CG iteration), caching descriptors
// and the selected algorithm by (m, n, k, dtype, transA, transB) would avoid
// per-call descriptor creation and heuristic lookup overhead.

#define EIGEN_CUBLASLT_CHECK(expr)                                          \
  do {                                                                      \
    cublasStatus_t _s = (expr);                                             \
    eigen_assert(_s == CUBLAS_STATUS_SUCCESS && "cuBLASLt call failed");    \
  } while (0)

// Maximum workspace the heuristic is allowed to consider. This is a preference
// ceiling, not an allocation — actual allocation matches the selected algorithm.
static constexpr size_t kCublasLtMaxWorkspaceBytes = 32 * 1024 * 1024;  // 32 MB

// cublasGemmEx fallback algorithm hint (used when cublasLt heuristic returns no results).
constexpr cublasGemmAlgo_t cuda_gemm_algo() {
#ifdef EIGEN_NO_CUDA_TENSOR_OPS
  return CUBLAS_GEMM_DEFAULT;
#else
  return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#endif
}

template <typename Scalar>
void cublaslt_gemm(cublasLtHandle_t lt_handle, cublasHandle_t cublas_handle, cublasOperation_t transA,
                   cublasOperation_t transB, int64_t m, int64_t n, int64_t k,
                   const typename cuda_gemm_scalar<Scalar>::type* alpha, const Scalar* A, int64_t lda, const Scalar* B,
                   int64_t ldb, const typename cuda_gemm_scalar<Scalar>::type* beta, Scalar* C, int64_t ldc,
                   DeviceBuffer* workspace, cudaStream_t stream) {
  constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
  constexpr cublasComputeType_t compute = cuda_compute_type<Scalar>::value;
  using AlphaType = typename cuda_gemm_scalar<Scalar>::type;
  constexpr cudaDataType_t alpha_type = cuda_data_type<AlphaType>::value;

  // Matmul descriptor.
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  EIGEN_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc, compute, alpha_type));
  EIGEN_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA,
                                                      sizeof(transA)));
  EIGEN_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB,
                                                      sizeof(transB)));

  // Matrix layout descriptors (column-major).
  // Physical layout dimensions: rows × cols with leading dimension lda/ldb/ldc.
  const int64_t a_rows = (transA == CUBLAS_OP_N) ? m : k;
  const int64_t b_rows = (transB == CUBLAS_OP_N) ? k : n;

  cublasLtMatrixLayout_t layout_A = nullptr, layout_B = nullptr, layout_C = nullptr;
  EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_A, dtype, a_rows, (transA == CUBLAS_OP_N) ? k : m, lda));
  EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_B, dtype, b_rows, (transB == CUBLAS_OP_N) ? n : k, ldb));
  EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_C, dtype, m, n, ldc));

  // Heuristic selection: query with generous workspace cap, allocate only what's needed.
  cublasLtMatmulPreference_t preference = nullptr;
  EIGEN_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  size_t max_ws = kCublasLtMaxWorkspaceBytes;
  EIGEN_CUBLASLT_CHECK(
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)));

  cublasLtMatmulHeuristicResult_t result;
  int returned_results = 0;
  cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      lt_handle, matmul_desc, layout_A, layout_B, layout_C, layout_C, preference, 1, &result, &returned_results);

  if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_results > 0) {
    // cublasLt path: use the selected algorithm with lazy workspace.
    const size_t needed = result.workspaceSize;
    if (needed > workspace->size()) {
      // Sync only when freeing an existing buffer that may be in use.
      if (workspace->ptr) EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
      *workspace = DeviceBuffer(needed);
    }

    EIGEN_CUBLASLT_CHECK(cublasLtMatmul(lt_handle, matmul_desc, alpha, A, layout_A, B, layout_B, beta, C, layout_C, C,
                                        layout_C, &result.algo, workspace->ptr, needed, stream));
  } else {
    // Fallback: cublasGemmEx for shapes/types that cublasLt cannot handle.
    EIGEN_CUBLAS_CHECK(cublasGemmEx(cublas_handle, transA, transB, static_cast<int>(m), static_cast<int>(n),
                                    static_cast<int>(k), alpha, A, dtype, static_cast<int>(lda), B, dtype,
                                    static_cast<int>(ldb), beta, C, dtype, static_cast<int>(ldc), compute,
                                    cuda_gemm_algo()));
  }

  // Cleanup descriptors.
  EIGEN_CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_C));
  EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_B));
  EIGEN_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_A));
  EIGEN_CUBLASLT_CHECK(cublasLtMatmulDescDestroy(matmul_desc));
}

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
  return cublasCaxpy(h, n, reinterpret_cast<const cuComplex*>(alpha), reinterpret_cast<const cuComplex*>(x), incx,
                     reinterpret_cast<cuComplex*>(y), incy);
}
inline cublasStatus_t cublasXaxpy(cublasHandle_t h, int n, const std::complex<double>* alpha,
                                  const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
  return cublasZaxpy(h, n, reinterpret_cast<const cuDoubleComplex*>(alpha), reinterpret_cast<const cuDoubleComplex*>(x),
                     incx, reinterpret_cast<cuDoubleComplex*>(y), incy);
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
  return cublasCscal(h, n, reinterpret_cast<const cuComplex*>(alpha), reinterpret_cast<cuComplex*>(x), incx);
}
inline cublasStatus_t cublasXscal(cublasHandle_t h, int n, const std::complex<double>* alpha, std::complex<double>* x,
                                  int incx) {
  return cublasZscal(h, n, reinterpret_cast<const cuDoubleComplex*>(alpha), reinterpret_cast<cuDoubleComplex*>(x),
                     incx);
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
}  // namespace Eigen

#endif  // EIGEN_GPU_CUBLAS_SUPPORT_H
