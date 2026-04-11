// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Dispatch functions that map DeviceMatrix expressions to NVIDIA library calls.
//
// dispatch_gemm()  — GemmExpr → cublasXgemm
//
// Each function documents the exact library call and parameters.

#ifndef EIGEN_GPU_DEVICE_DISPATCH_H
#define EIGEN_GPU_DEVICE_DISPATCH_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <climits>

#include "./DeviceExpr.h"
#include "./DeviceBlasExpr.h"
#include "./DeviceSolverExpr.h"
#include "./GpuContext.h"
#include "./CuSolverSupport.h"

namespace Eigen {
namespace internal {

// ---- GEMM dispatch ----------------------------------------------------------
// GemmExpr<Lhs, Rhs> → cublasGemmEx(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
//
// The generic API cublasGemmEx handles all scalar types (float, double,
// complex<float>, complex<double>) via cudaDataType_t.

template <typename Lhs, typename Rhs>
void dispatch_gemm(
    GpuContext& ctx, DeviceMatrix<typename device_expr_traits<Lhs>::scalar_type>& dst, const GemmExpr<Lhs, Rhs>& expr,
    typename device_expr_traits<Lhs>::scalar_type beta_val,
    typename device_expr_traits<Lhs>::scalar_type alpha_scale = typename device_expr_traits<Lhs>::scalar_type(1)) {
  using Scalar = typename device_expr_traits<Lhs>::scalar_type;
  using traits_lhs = device_expr_traits<Lhs>;
  using traits_rhs = device_expr_traits<Rhs>;

  const DeviceMatrix<Scalar>& A = traits_lhs::matrix(expr.lhs());
  const DeviceMatrix<Scalar>& B = traits_rhs::matrix(expr.rhs());

  constexpr cublasOperation_t transA = to_cublas_op(traits_lhs::op);
  constexpr cublasOperation_t transB = to_cublas_op(traits_rhs::op);

  // GEMM dimensions: C(m,n) = op(A)(m,k) * op(B)(k,n)
  // op(A) has dimensions (A.rows, A.cols) if NoTrans, (A.cols, A.rows) if Trans/ConjTrans.
  const int64_t m = (traits_lhs::op == GpuOp::NoTrans) ? A.rows() : A.cols();
  const int64_t k = (traits_lhs::op == GpuOp::NoTrans) ? A.cols() : A.rows();
  const int64_t n = (traits_rhs::op == GpuOp::NoTrans) ? B.cols() : B.rows();
  const int64_t rhs_k = (traits_rhs::op == GpuOp::NoTrans) ? B.rows() : B.cols();

  eigen_assert(k == rhs_k && "DeviceMatrix GEMM dimension mismatch");

  const int64_t lda = A.outerStride();
  const int64_t ldb = B.outerStride();

  // Serialize all accesses to the destination buffer on this stream.
  if (!dst.empty()) {
    dst.waitReady(ctx.stream());
  }

  // Allocate or resize destination.
  const bool resized = dst.empty() || dst.rows() != m || dst.cols() != n;
  if (resized) {
    dst.resize(m, n);
  }
  const int64_t ldc = dst.outerStride();

  Scalar alpha_val = alpha_scale * traits_lhs::alpha(expr.lhs()) * traits_rhs::alpha(expr.rhs());

  // Wait for operands to be ready on this stream.
  A.waitReady(ctx.stream());
  B.waitReady(ctx.stream());

  // If there is no existing valid destination to accumulate into, treat it as
  // zero rather than reading uninitialized memory.
  if (resized && beta_val != Scalar(0) && dst.sizeInBytes() > 0) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(dst.data(), 0, dst.sizeInBytes(), ctx.stream()));
  }

  constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
  constexpr cublasComputeType_t compute = cuda_compute_type<Scalar>::value;

  eigen_assert(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX && lda <= INT_MAX && ldb <= INT_MAX && ldc <= INT_MAX &&
               "cublasGemmEx dimensions exceed int range");

  EIGEN_CUBLAS_CHECK(cublasGemmEx(ctx.cublasHandle(), transA, transB, static_cast<int>(m), static_cast<int>(n),
                                  static_cast<int>(k), &alpha_val, A.data(), dtype, static_cast<int>(lda), B.data(),
                                  dtype, static_cast<int>(ldb), &beta_val, dst.data(), dtype, static_cast<int>(ldc),
                                  compute, CUBLAS_GEMM_DEFAULT));

  dst.recordReady(ctx.stream());
}

// ---- LLT solve dispatch -----------------------------------------------------
// LltSolveExpr → cusolverDnXpotrf (factorize) + cusolverDnXpotrs (solve).
// No caching — factor and workspace are temporary. Syncs to check info.

template <typename Scalar, int UpLo>
void dispatch_llt_solve(GpuContext& ctx, DeviceMatrix<Scalar>& dst, const LltSolveExpr<Scalar, UpLo>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "LLT requires a square matrix");
  eigen_assert(B.rows() == A.rows() && "LLT solve: RHS rows must match matrix size");

  const Index n = A.rows();
  const int64_t nrhs = static_cast<int64_t>(B.cols());

  // Zero-size fast paths: no work, just resize dst.
  // Wait on dst before resize to avoid freeing memory another stream is using.
  if (n == 0 || nrhs == 0) {
    if (!dst.empty()) dst.waitReady(ctx.stream());
    dst.resize(n == 0 ? 0 : n, B.cols());
    return;
  }

  A.waitReady(ctx.stream());
  B.waitReady(ctx.stream());
  if (!dst.empty()) dst.waitReady(ctx.stream());

  constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
  constexpr cublasFillMode_t uplo = cusolver_fill_mode<UpLo, ColMajor>::value;
  const int64_t lda = static_cast<int64_t>(A.outerStride());
  const int64_t ldb = static_cast<int64_t>(B.outerStride());
  eigen_assert(ldb == static_cast<int64_t>(B.rows()) && "DeviceMatrix must be densely packed");
  const size_t mat_bytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(Scalar);
  const size_t rhs_bytes = static_cast<size_t>(ldb) * static_cast<size_t>(nrhs) * sizeof(Scalar);

  // D2D copy A → factor buffer (potrf is in-place).
  DeviceBuffer d_factor(mat_bytes);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_factor.ptr, A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  // Pinned host memory for async info download (avoids compute-sanitizer warnings).
  PinnedHostBuffer h_info(sizeof(int));
  int& info_word = *static_cast<int*>(h_info.ptr);

  // Query workspace and factorize.
  CusolverParams params;
  DeviceBuffer d_info(sizeof(int));
  size_t dev_ws = 0, host_ws = 0;
  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(ctx.cusolverHandle(), params.p, uplo, static_cast<int64_t>(n), dtype,
                                                   d_factor.ptr, lda, dtype, &dev_ws, &host_ws));

  DeviceBuffer d_workspace(dev_ws);
  std::vector<char> h_workspace(host_ws);

  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf(
      ctx.cusolverHandle(), params.p, uplo, static_cast<int64_t>(n), dtype, d_factor.ptr, lda, dtype, d_workspace.ptr,
      dev_ws, host_ws > 0 ? h_workspace.data() : nullptr, host_ws, static_cast<int*>(d_info.ptr)));

  // Async download to pinned memory, then sync and check.
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&info_word, d_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(info_word == 0 && "cuSOLVER LLT factorization failed (matrix not positive definite)");

  // D2D copy B → dst (potrs is in-place on the RHS).
  dst.resize(n, B.cols());
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  // Solve.
  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrs(ctx.cusolverHandle(), params.p, uplo, static_cast<int64_t>(n), nrhs, dtype,
                                        d_factor.ptr, lda, dtype, dst.data(), static_cast<int64_t>(dst.outerStride()),
                                        static_cast<int*>(d_info.ptr)));

  // Async download to pinned memory, sync, check. Workspace locals must outlive async kernels.
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&info_word, d_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(info_word == 0 && "cuSOLVER LLT solve failed");

  dst.recordReady(ctx.stream());
}

// ---- LU solve dispatch ------------------------------------------------------
// LuSolveExpr → cusolverDnXgetrf (factorize) + cusolverDnXgetrs (solve).

template <typename Scalar>
void dispatch_lu_solve(GpuContext& ctx, DeviceMatrix<Scalar>& dst, const LuSolveExpr<Scalar>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "LU requires a square matrix");
  eigen_assert(B.rows() == A.rows() && "LU solve: RHS rows must match matrix size");

  const Index n = A.rows();
  const int64_t nrhs = static_cast<int64_t>(B.cols());

  if (n == 0 || nrhs == 0) {
    if (!dst.empty()) dst.waitReady(ctx.stream());
    dst.resize(n == 0 ? 0 : n, B.cols());
    return;
  }

  A.waitReady(ctx.stream());
  B.waitReady(ctx.stream());
  if (!dst.empty()) dst.waitReady(ctx.stream());

  constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
  const int64_t lda = static_cast<int64_t>(A.outerStride());
  const int64_t ldb = static_cast<int64_t>(B.outerStride());
  eigen_assert(ldb == static_cast<int64_t>(B.rows()) && "DeviceMatrix must be densely packed");
  const size_t mat_bytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(Scalar);
  const size_t rhs_bytes = static_cast<size_t>(ldb) * static_cast<size_t>(nrhs) * sizeof(Scalar);
  const size_t ipiv_bytes = static_cast<size_t>(n) * sizeof(int64_t);

  // D2D copy A → LU buffer (getrf is in-place).
  DeviceBuffer d_lu(mat_bytes);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_lu.ptr, A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  DeviceBuffer d_ipiv(ipiv_bytes);

  // Pinned host memory for async info download.
  PinnedHostBuffer h_info(sizeof(int));
  int& info_word = *static_cast<int*>(h_info.ptr);

  // Query workspace and factorize.
  CusolverParams params;
  DeviceBuffer d_info(sizeof(int));
  size_t dev_ws = 0, host_ws = 0;
  EIGEN_CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(ctx.cusolverHandle(), params.p, static_cast<int64_t>(n),
                                                   static_cast<int64_t>(n), dtype, d_lu.ptr, lda, dtype, &dev_ws,
                                                   &host_ws));

  DeviceBuffer d_workspace(dev_ws);
  std::vector<char> h_workspace(host_ws);

  EIGEN_CUSOLVER_CHECK(
      cusolverDnXgetrf(ctx.cusolverHandle(), params.p, static_cast<int64_t>(n), static_cast<int64_t>(n), dtype,
                       d_lu.ptr, lda, static_cast<int64_t*>(d_ipiv.ptr), dtype, d_workspace.ptr, dev_ws,
                       host_ws > 0 ? h_workspace.data() : nullptr, host_ws, static_cast<int*>(d_info.ptr)));

  // Async download to pinned memory, then sync and check.
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&info_word, d_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(info_word == 0 && "cuSOLVER LU factorization failed (singular matrix)");

  // D2D copy B → dst (getrs is in-place on the RHS).
  dst.resize(n, B.cols());
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  // Solve (NoTranspose).
  EIGEN_CUSOLVER_CHECK(cusolverDnXgetrs(ctx.cusolverHandle(), params.p, CUBLAS_OP_N, static_cast<int64_t>(n), nrhs,
                                        dtype, d_lu.ptr, lda, static_cast<const int64_t*>(d_ipiv.ptr), dtype,
                                        dst.data(), static_cast<int64_t>(dst.outerStride()),
                                        static_cast<int*>(d_info.ptr)));

  // Async download to pinned memory, sync, check. Workspace locals must outlive async kernels.
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&info_word, d_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(info_word == 0 && "cuSOLVER LU solve failed");

  dst.recordReady(ctx.stream());
}

// ---- TRSM dispatch ----------------------------------------------------------
// TrsmExpr → cublasXtrsm: solve op(A) * X = B where A is triangular.
// Side=Left, Diag=NonUnit. A is square, B is n×nrhs.

template <typename Scalar, int UpLo>
void dispatch_trsm(GpuContext& ctx, DeviceMatrix<Scalar>& dst, const TrsmExpr<Scalar, UpLo>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "TRSM requires a square triangular matrix");
  eigen_assert(B.rows() == A.rows() && "TRSM: RHS rows must match matrix size");

  eigen_assert(A.rows() <= INT_MAX && B.cols() <= INT_MAX && A.outerStride() <= INT_MAX &&
               "cublasXtrsm dimensions exceed int range");

  const int n = static_cast<int>(A.rows());
  const int nrhs = static_cast<int>(B.cols());

  if (n == 0 || nrhs == 0) {
    if (!dst.empty()) dst.waitReady(ctx.stream());
    dst.resize(n == 0 ? 0 : n, B.cols());
    return;
  }

  A.waitReady(ctx.stream());
  B.waitReady(ctx.stream());
  if (!dst.empty()) dst.waitReady(ctx.stream());

  // D2D copy B → dst (trsm is in-place on the RHS).
  dst.resize(n, B.cols());
  const size_t rhs_bytes = static_cast<size_t>(dst.outerStride()) * static_cast<size_t>(nrhs) * sizeof(Scalar);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  constexpr cublasFillMode_t uplo = (UpLo == Lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  Scalar alpha(1);

  EIGEN_CUBLAS_CHECK(cublasXtrsm(ctx.cublasHandle(), CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs,
                                 &alpha, A.data(), static_cast<int>(A.outerStride()), dst.data(),
                                 static_cast<int>(dst.outerStride())));

  dst.recordReady(ctx.stream());
}

// ---- SYMM/HEMM dispatch -----------------------------------------------------
// SymmExpr → cublasXsymm (real) or cublasXhemm (complex).
// C = A * B where A is symmetric/Hermitian. Side=Left.

template <typename Scalar, int UpLo>
void dispatch_symm(GpuContext& ctx, DeviceMatrix<Scalar>& dst, const SymmExpr<Scalar, UpLo>& expr) {
  const DeviceMatrix<Scalar>& A = expr.matrix();
  const DeviceMatrix<Scalar>& B = expr.rhs();

  eigen_assert(A.rows() == A.cols() && "SYMM requires a square matrix");
  eigen_assert(B.rows() == A.rows() && "SYMM: RHS rows must match matrix size");
  eigen_assert(A.rows() <= INT_MAX && B.cols() <= INT_MAX && A.outerStride() <= INT_MAX && B.outerStride() <= INT_MAX &&
               "cublasXsymm dimensions exceed int range");

  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(B.cols());

  if (m == 0 || n == 0) {
    if (!dst.empty()) dst.waitReady(ctx.stream());
    dst.resize(m == 0 ? 0 : m, B.cols());
    return;
  }

  A.waitReady(ctx.stream());
  B.waitReady(ctx.stream());
  if (!dst.empty()) dst.waitReady(ctx.stream());

  dst.resize(m, n);

  constexpr cublasFillMode_t uplo = (UpLo == Lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  Scalar alpha(1), beta(0);

  EIGEN_CUBLAS_CHECK(cublasXsymm(ctx.cublasHandle(), CUBLAS_SIDE_LEFT, uplo, m, n, &alpha, A.data(),
                                 static_cast<int>(A.outerStride()), B.data(), static_cast<int>(B.outerStride()), &beta,
                                 dst.data(), static_cast<int>(dst.outerStride())));

  dst.recordReady(ctx.stream());
}

// ---- SYRK/HERK dispatch -----------------------------------------------------
// SyrkExpr → cublasXsyrk (real) or cublasXherk (complex).
// C = alpha * A * A^H + beta * C. UpLo specifies which triangle of C is stored.

template <typename Scalar, int UpLo>
void dispatch_syrk(GpuContext& ctx, DeviceMatrix<Scalar>& dst, const SyrkExpr<Scalar, UpLo>& expr,
                   typename NumTraits<Scalar>::Real alpha_val, typename NumTraits<Scalar>::Real beta_val) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  const DeviceMatrix<Scalar>& A = expr.matrix();

  eigen_assert(A.rows() <= INT_MAX && A.cols() <= INT_MAX && A.outerStride() <= INT_MAX &&
               "cublasXsyrk dimensions exceed int range");

  const int n = static_cast<int>(A.rows());
  const int k = static_cast<int>(A.cols());

  if (n == 0) {
    if (!dst.empty()) dst.waitReady(ctx.stream());
    dst.resize(0, 0);
    return;
  }

  A.waitReady(ctx.stream());
  if (!dst.empty()) dst.waitReady(ctx.stream());

  if (dst.empty() || dst.rows() != n || dst.cols() != n) {
    dst.resize(n, n);
    if (beta_val != RealScalar(0)) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(dst.data(), 0, dst.sizeInBytes(), ctx.stream()));
    }
  }

  constexpr cublasFillMode_t uplo = (UpLo == Lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  EIGEN_CUBLAS_CHECK(cublasXsyrk(ctx.cublasHandle(), uplo, CUBLAS_OP_N, n, k, &alpha_val, A.data(),
                                 static_cast<int>(A.outerStride()), &beta_val, dst.data(),
                                 static_cast<int>(dst.outerStride())));

  dst.recordReady(ctx.stream());
}

}  // namespace internal

// ---- DeviceAssignment: d_C.device(ctx) = expr ------------------------------
// Returned by DeviceMatrix::device(ctx). Dispatches expressions to library calls.

template <typename Scalar_>
class DeviceAssignment {
 public:
  using Scalar = Scalar_;

  DeviceAssignment(DeviceMatrix<Scalar>& dst, GpuContext& ctx) : dst_(dst), ctx_(ctx) {}

  // operator= dispatches GEMM with beta=0 (overwrite).
  template <typename Lhs, typename Rhs>
  DeviceMatrix<Scalar>& operator=(const GemmExpr<Lhs, Rhs>& expr) {
    internal::dispatch_gemm(ctx_, dst_, expr, Scalar(0));
    return dst_;
  }

  // operator+= dispatches GEMM with beta=1 (accumulate).
  template <typename Lhs, typename Rhs>
  DeviceMatrix<Scalar>& operator+=(const GemmExpr<Lhs, Rhs>& expr) {
    internal::dispatch_gemm(ctx_, dst_, expr, Scalar(1));
    return dst_;
  }

  // operator-= dispatches GEMM with negated alpha, beta=1: C = C - alpha*op(A)*op(B).
  template <typename Lhs, typename Rhs>
  DeviceMatrix<Scalar>& operator-=(const GemmExpr<Lhs, Rhs>& expr) {
    internal::dispatch_gemm(ctx_, dst_, expr, Scalar(1), Scalar(-1));
    return dst_;
  }

  // operator= dispatches LLT solve (potrf + potrs).
  template <int UpLo>
  DeviceMatrix<Scalar>& operator=(const LltSolveExpr<Scalar, UpLo>& expr) {
    internal::dispatch_llt_solve(ctx_, dst_, expr);
    return dst_;
  }

  // operator= dispatches LU solve (getrf + getrs).
  DeviceMatrix<Scalar>& operator=(const LuSolveExpr<Scalar>& expr) {
    internal::dispatch_lu_solve(ctx_, dst_, expr);
    return dst_;
  }

  // operator= dispatches TRSM (triangular solve).
  template <int UpLo>
  DeviceMatrix<Scalar>& operator=(const TrsmExpr<Scalar, UpLo>& expr) {
    internal::dispatch_trsm(ctx_, dst_, expr);
    return dst_;
  }

  // operator= dispatches SYMM/HEMM (symmetric/Hermitian multiply).
  template <int UpLo>
  DeviceMatrix<Scalar>& operator=(const SymmExpr<Scalar, UpLo>& expr) {
    internal::dispatch_symm(ctx_, dst_, expr);
    return dst_;
  }

  // Catch-all: static_assert for unsupported expressions.
  template <typename Expr>
  DeviceMatrix<Scalar>& operator=(const Expr&) {
    static_assert(sizeof(Expr) == 0,
                  "DeviceMatrix expression not supported: no cuBLAS/cuSOLVER mapping. "
                  "Supported: GEMM (A*B), TRSM (.triangularView().solve()), "
                  "SYMM (.selfadjointView()*B), LLT (.llt().solve()), LU (.lu().solve()).");
    return dst_;
  }

 private:
  DeviceMatrix<Scalar>& dst_;
  GpuContext& ctx_;
};

// ---- Out-of-line DeviceMatrix expression operator= definitions -------------
// These are declared in DeviceMatrix.h but defined here because they need
// GpuContext::threadLocal() which requires the full GpuContext definition.

template <typename Scalar_>
template <typename Lhs, typename Rhs>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const GemmExpr<Lhs, Rhs>& expr) {
  device(GpuContext::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
template <typename Lhs, typename Rhs>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const GemmExpr<Lhs, Rhs>& expr) {
  device(GpuContext::threadLocal()) += expr;
  return *this;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const LltSolveExpr<Scalar_, UpLo>& expr) {
  device(GpuContext::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const LuSolveExpr<Scalar_>& expr) {
  device(GpuContext::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const TrsmExpr<Scalar_, UpLo>& expr) {
  device(GpuContext::threadLocal()) = expr;
  return *this;
}

template <typename Scalar_>
template <int UpLo>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const SymmExpr<Scalar_, UpLo>& expr) {
  device(GpuContext::threadLocal()) = expr;
  return *this;
}

// DeviceSelfAdjointView::rankUpdate — defined here because it needs GpuContext.
template <typename Scalar_, int UpLo_>
void DeviceSelfAdjointView<Scalar_, UpLo_>::rankUpdate(const DeviceMatrix<Scalar_>& A, RealScalar alpha) {
  SyrkExpr<Scalar_, UpLo_> expr(A);
  RealScalar beta = matrix().empty() ? RealScalar(0) : RealScalar(1);
  internal::dispatch_syrk(GpuContext::threadLocal(), matrix(), expr, alpha, beta);
}

}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_DISPATCH_H
