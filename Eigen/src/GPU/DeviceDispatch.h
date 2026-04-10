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

#include "./DeviceExpr.h"
#include "./DeviceBlasExpr.h"
#include "./DeviceSolverExpr.h"
#include "./GpuContext.h"
#include "./CuSolverSupport.h"

namespace Eigen {
namespace internal {

// ---- GEMM dispatch ----------------------------------------------------------
// GemmExpr<Lhs, Rhs> → cublasLtMatmul via GpuContext.
//
// Uses cublasLtMatmul for 64-bit dimension support and heuristic algorithm
// selection. All scalar types (float, double, complex<float>, complex<double>)
// are handled via cudaDataType_t.

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

  // cuBLAS GEMM: C must not alias A or B (undefined behavior).
  eigen_assert(dst.data() != A.data() && "GEMM: output aliases left operand (use a temporary)");
  eigen_assert(dst.data() != B.data() && "GEMM: output aliases right operand (use a temporary)");

  constexpr cublasOperation_t transA = to_cublas_op(traits_lhs::op);
  constexpr cublasOperation_t transB = to_cublas_op(traits_rhs::op);

  // GEMM dimensions: C(m,n) = op(A)(m,k) * op(B)(k,n)
  // op(A) has dimensions (A.rows, A.cols) if NoTrans, (A.cols, A.rows) if Trans/ConjTrans.
  const int64_t m = (traits_lhs::op == GpuOp::NoTrans) ? A.rows() : A.cols();
  const int64_t k = (traits_lhs::op == GpuOp::NoTrans) ? A.cols() : A.rows();
  const int64_t n = (traits_rhs::op == GpuOp::NoTrans) ? B.cols() : B.rows();
  const int64_t rhs_k = (traits_rhs::op == GpuOp::NoTrans) ? B.rows() : B.cols();

  eigen_assert(k == rhs_k && "DeviceMatrix GEMM dimension mismatch");

  const int64_t lda = A.rows();
  const int64_t ldb = B.rows();

  // Serialize all accesses to the destination buffer on this stream.
  if (!dst.empty()) {
    dst.waitReady(ctx.stream());
  }

  // Allocate or resize destination.
  const bool resized = dst.empty() || dst.rows() != m || dst.cols() != n;
  if (resized) {
    dst.resize(m, n);
  }
  const int64_t ldc = dst.rows();

  // cuBLAS requires alpha/beta as float for half/bfloat16 inputs.
  using GemmScalar = typename cuda_gemm_scalar<Scalar>::type;
  GemmScalar alpha_gval =
      static_cast<GemmScalar>(alpha_scale * traits_lhs::alpha(expr.lhs()) * traits_rhs::alpha(expr.rhs()));
  GemmScalar beta_gval = static_cast<GemmScalar>(beta_val);

  // Wait for operands to be ready on this stream.
  A.waitReady(ctx.stream());
  B.waitReady(ctx.stream());

  // If there is no existing valid destination to accumulate into, treat it as
  // zero rather than reading uninitialized memory.
  if (resized && beta_gval != GemmScalar(0) && dst.sizeInBytes() > 0) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(dst.data(), 0, dst.sizeInBytes(), ctx.stream()));
  }

  cublaslt_gemm<Scalar>(ctx.cublasLtHandle(), ctx.cublasHandle(), transA, transB, m, n, k, &alpha_gval, A.data(), lda,
                         B.data(), ldb, &beta_gval, dst.data(), ldc, ctx.gemmWorkspace(), ctx.stream());

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
  const int64_t lda = static_cast<int64_t>(A.rows());
  const int64_t ldb = static_cast<int64_t>(B.rows());

  const size_t mat_bytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(Scalar);
  const size_t rhs_bytes = static_cast<size_t>(ldb) * static_cast<size_t>(nrhs) * sizeof(Scalar);

  // D2D copy A → factor buffer (potrf is in-place).
  DeviceBuffer d_factor(mat_bytes);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_factor.ptr, A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  // Query workspace and factorize.
  CusolverParams params;
  DeviceBuffer d_factorize_info(sizeof(int));
  size_t dev_ws = 0, host_ws = 0;
  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(ctx.cusolverHandle(), params.p, uplo, static_cast<int64_t>(n), dtype,
                                                   d_factor.ptr, lda, dtype, &dev_ws, &host_ws));

  DeviceBuffer d_workspace(dev_ws);
  std::vector<char> h_workspace(host_ws);

  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf(
      ctx.cusolverHandle(), params.p, uplo, static_cast<int64_t>(n), dtype, d_factor.ptr, lda, dtype, d_workspace.ptr,
      dev_ws, host_ws > 0 ? h_workspace.data() : nullptr, host_ws, static_cast<int*>(d_factorize_info.ptr)));

  // Check factorization info before proceeding to solve.
  int factorize_info = 0;
  EIGEN_CUDA_RUNTIME_CHECK(
      cudaMemcpyAsync(&factorize_info, d_factorize_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(factorize_info == 0 && "cuSOLVER LLT factorization failed (matrix not positive definite)");

  // D2D copy B → dst (potrs is in-place on the RHS).
  dst.resize(n, B.cols());
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  // Solve.
  DeviceBuffer d_solve_info(sizeof(int));
  EIGEN_CUSOLVER_CHECK(cusolverDnXpotrs(ctx.cusolverHandle(), params.p, uplo, static_cast<int64_t>(n), nrhs, dtype,
                                        d_factor.ptr, lda, dtype, dst.data(), static_cast<int64_t>(dst.rows()),
                                        static_cast<int*>(d_solve_info.ptr)));

  // Sync to ensure workspace locals can be freed safely.
  int solve_info = 0;
  EIGEN_CUDA_RUNTIME_CHECK(
      cudaMemcpyAsync(&solve_info, d_solve_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(solve_info == 0 && "cuSOLVER LLT solve failed");

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
  const int64_t lda = static_cast<int64_t>(A.rows());
  const int64_t ldb = static_cast<int64_t>(B.rows());

  const size_t mat_bytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(Scalar);
  const size_t rhs_bytes = static_cast<size_t>(ldb) * static_cast<size_t>(nrhs) * sizeof(Scalar);
  const size_t ipiv_bytes = static_cast<size_t>(n) * sizeof(int64_t);

  // D2D copy A → LU buffer (getrf is in-place).
  DeviceBuffer d_lu(mat_bytes);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_lu.ptr, A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  DeviceBuffer d_ipiv(ipiv_bytes);

  // Query workspace and factorize.
  CusolverParams params;
  DeviceBuffer d_factorize_info(sizeof(int));
  size_t dev_ws = 0, host_ws = 0;
  EIGEN_CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(ctx.cusolverHandle(), params.p, static_cast<int64_t>(n),
                                                   static_cast<int64_t>(n), dtype, d_lu.ptr, lda, dtype, &dev_ws,
                                                   &host_ws));

  DeviceBuffer d_workspace(dev_ws);
  std::vector<char> h_workspace(host_ws);

  EIGEN_CUSOLVER_CHECK(
      cusolverDnXgetrf(ctx.cusolverHandle(), params.p, static_cast<int64_t>(n), static_cast<int64_t>(n), dtype,
                       d_lu.ptr, lda, static_cast<int64_t*>(d_ipiv.ptr), dtype, d_workspace.ptr, dev_ws,
                       host_ws > 0 ? h_workspace.data() : nullptr, host_ws, static_cast<int*>(d_factorize_info.ptr)));

  // Check factorization info before proceeding to solve.
  int factorize_info = 0;
  EIGEN_CUDA_RUNTIME_CHECK(
      cudaMemcpyAsync(&factorize_info, d_factorize_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(factorize_info == 0 && "cuSOLVER LU factorization failed (singular matrix)");

  // D2D copy B → dst (getrs is in-place on the RHS).
  dst.resize(n, B.cols());
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  // Solve (NoTranspose).
  DeviceBuffer d_solve_info(sizeof(int));
  EIGEN_CUSOLVER_CHECK(cusolverDnXgetrs(ctx.cusolverHandle(), params.p, CUBLAS_OP_N, static_cast<int64_t>(n), nrhs,
                                        dtype, d_lu.ptr, lda, static_cast<const int64_t*>(d_ipiv.ptr), dtype,
                                        dst.data(), static_cast<int64_t>(dst.rows()),
                                        static_cast<int*>(d_solve_info.ptr)));

  // Sync to ensure workspace locals can be freed safely.
  int solve_info = 0;
  EIGEN_CUDA_RUNTIME_CHECK(
      cudaMemcpyAsync(&solve_info, d_solve_info.ptr, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream()));
  eigen_assert(solve_info == 0 && "cuSOLVER LU solve failed");

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
  const size_t rhs_bytes = static_cast<size_t>(dst.rows()) * static_cast<size_t>(nrhs) * sizeof(Scalar);
  EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst.data(), B.data(), rhs_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

  constexpr cublasFillMode_t uplo = (UpLo == Lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  Scalar alpha(1);

  EIGEN_CUBLAS_CHECK(cublasXtrsm(ctx.cublasHandle(), CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs,
                                 &alpha, A.data(), static_cast<int>(A.rows()), dst.data(),
                                 static_cast<int>(dst.rows())));

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
                                 static_cast<int>(A.rows()), B.data(), static_cast<int>(B.rows()), &beta, dst.data(),
                                 static_cast<int>(dst.rows())));

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
                                 static_cast<int>(A.rows()), &beta_val, dst.data(), static_cast<int>(dst.rows())));

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

// ---- DeviceMatrix BLAS-1 out-of-line definitions ----------------------------
// Defined here because they need the full GpuContext definition.
// All methods take an explicit GpuContext& so callers can ensure same-stream
// execution (zero event overhead when all operations share one context).
//
// Reduction methods (dot, norm, squaredNorm) use CUBLAS_POINTER_MODE_HOST:
// the scalar result is written to host memory and cuBLAS synchronizes
// internally before returning. This is necessary for Eigen template
// compatibility — CG does `Scalar alpha = absNew / p.dot(tmp)` which
// requires the host value immediately. A future GPU CG implementation
// that controls the iteration loop can use CUBLAS_POINTER_MODE_DEVICE
// to batch multiple reductions into a single sync point.

template <typename Scalar_>
DeviceScalar<typename DeviceMatrix<Scalar_>::Scalar> DeviceMatrix<Scalar_>::dot(GpuContext& ctx,
                                                                                const DeviceMatrix& other) const {
  const int n = static_cast<int>(rows_ * cols_);
  eigen_assert(n == static_cast<int>(other.rows_ * other.cols_));
  DeviceScalar<Scalar> result(Scalar(0), ctx.stream());
  if (n > 0) {
    waitReady(ctx.stream());
    other.waitReady(ctx.stream());
    cublasPointerMode_t prev;
    EIGEN_CUBLAS_CHECK(cublasGetPointerMode(ctx.cublasHandle(), &prev));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), CUBLAS_POINTER_MODE_DEVICE));
    EIGEN_CUBLAS_CHECK(internal::cublasXdot(ctx.cublasHandle(), n, data_, 1, other.data_, 1, result.devicePtr()));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), prev));
  }
  return result;
}

namespace internal {
// Real: dot(x,x) returns DeviceScalar<Scalar> which IS DeviceScalar<RealScalar>.
// Move-construct without any sync.
template <typename Scalar, typename RealScalar>
typename std::enable_if<std::is_same<Scalar, RealScalar>::value, DeviceScalar<RealScalar>>::type
squaredNorm_from_dot(DeviceScalar<Scalar>&& d, cudaStream_t) {
  return std::move(d);
}
// Complex: must sync to extract the real part (DeviceScalar arithmetic is real-only).
template <typename Scalar, typename RealScalar>
typename std::enable_if<!std::is_same<Scalar, RealScalar>::value, DeviceScalar<RealScalar>>::type
squaredNorm_from_dot(DeviceScalar<Scalar>&& d, cudaStream_t stream) {
  return DeviceScalar<RealScalar>(numext::real(Scalar(d)), stream);
}
}  // namespace internal

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::squaredNorm(GpuContext& ctx) const {
  // Use dot(x,x) instead of nrm2()^2: dot kernel is ~4.5x faster than nrm2
  // (nrm2 uses a numerically careful scaled-sum-of-squares algorithm that is
  // unnecessary for CG convergence checks).
  using RealScalar = typename NumTraits<Scalar_>::Real;
  return internal::squaredNorm_from_dot<Scalar_, RealScalar>(dot(ctx, *this), ctx.stream());
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::norm(GpuContext& ctx) const {
  using RealScalar = typename NumTraits<Scalar>::Real;
  const int n = static_cast<int>(rows_ * cols_);
  DeviceScalar<RealScalar> result(RealScalar(0), ctx.stream());
  if (n > 0) {
    waitReady(ctx.stream());
    cublasPointerMode_t prev;
    EIGEN_CUBLAS_CHECK(cublasGetPointerMode(ctx.cublasHandle(), &prev));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), CUBLAS_POINTER_MODE_DEVICE));
    EIGEN_CUBLAS_CHECK(internal::cublasXnrm2(ctx.cublasHandle(), n, data_, 1, result.devicePtr()));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), prev));
  }
  return result;
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::setZero(GpuContext& ctx) {
  if (sizeInBytes() > 0) {
    waitReady(ctx.stream());
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemsetAsync(data_, 0, sizeInBytes(), ctx.stream()));
    recordReady(ctx.stream());
  }
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::addScaled(GpuContext& ctx, Scalar alpha, const DeviceMatrix& x) {
  const int n = static_cast<int>(rows_ * cols_);
  eigen_assert(n == static_cast<int>(x.rows_ * x.cols_));
  if (n > 0) {
    waitReady(ctx.stream());
    x.waitReady(ctx.stream());
    EIGEN_CUBLAS_CHECK(internal::cublasXaxpy(ctx.cublasHandle(), n, &alpha, x.data_, 1, data_, 1));
    recordReady(ctx.stream());
  }
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::scale(GpuContext& ctx, Scalar alpha) {
  const int n = static_cast<int>(rows_ * cols_);
  if (n > 0) {
    waitReady(ctx.stream());
    EIGEN_CUBLAS_CHECK(internal::cublasXscal(ctx.cublasHandle(), n, &alpha, data_, 1));
    recordReady(ctx.stream());
  }
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::copyFrom(GpuContext& ctx, const DeviceMatrix& other) {
  // Wait on *this before resize — resize may free the old buffer while another
  // stream is still reading it.
  if (!empty()) waitReady(ctx.stream());
  resize(other.rows_, other.cols_);
  const int n = static_cast<int>(rows_ * cols_);
  if (n > 0) {
    other.waitReady(ctx.stream());
    EIGEN_CUBLAS_CHECK(internal::cublasXcopy(ctx.cublasHandle(), n, other.data_, 1, data_, 1));
    recordReady(ctx.stream());
  }
}

// ---- BLAS-1 operator overloads for CG compatibility -------------------------

// this += alpha * x  (axpy)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const DeviceScaled<DeviceMatrix>& expr) {
  addScaled(GpuContext::threadLocal(), expr.scalar(), internal::device_expr_traits<DeviceMatrix>::matrix(expr.inner()));
  return *this;
}

// this -= alpha * x  (axpy with negated alpha)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator-=(const DeviceScaled<DeviceMatrix>& expr) {
  addScaled(GpuContext::threadLocal(), -expr.scalar(),
            internal::device_expr_traits<DeviceMatrix>::matrix(expr.inner()));
  return *this;
}

// this += x  (axpy with alpha=1)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const DeviceMatrix& other) {
  Scalar one(1);
  addScaled(GpuContext::threadLocal(), one, other);
  return *this;
}

// this -= x  (axpy with alpha=-1)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator-=(const DeviceMatrix& other) {
  Scalar neg_one(-1);
  addScaled(GpuContext::threadLocal(), neg_one, other);
  return *this;
}

// this *= alpha  (scal, host pointer)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator*=(Scalar alpha) {
  scale(GpuContext::threadLocal(), alpha);
  return *this;
}

// this *= alpha  (scal, device pointer — avoids host sync)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator*=(const DeviceScalar<Scalar>& alpha) {
  const int n = static_cast<int>(rows_ * cols_);
  if (n > 0) {
    auto& ctx = GpuContext::threadLocal();
    waitReady(ctx.stream());
    cublasPointerMode_t prev;
    EIGEN_CUBLAS_CHECK(cublasGetPointerMode(ctx.cublasHandle(), &prev));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), CUBLAS_POINTER_MODE_DEVICE));
    EIGEN_CUBLAS_CHECK(internal::cublasXscal(ctx.cublasHandle(), n, alpha.devicePtr(), data_, 1));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), prev));
    recordReady(ctx.stream());
  }
  return *this;
}

// this += DeviceScalar * x  (axpy with CUBLAS_POINTER_MODE_DEVICE)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator+=(const DeviceScaledDevice<Scalar_>& expr) {
  const int n = static_cast<int>(rows_ * cols_);
  const auto& x = expr.matrix();
  eigen_assert(n == static_cast<int>(x.rows_ * x.cols_));
  if (n > 0) {
    auto& ctx = GpuContext::threadLocal();
    waitReady(ctx.stream());
    x.waitReady(ctx.stream());
    cublasPointerMode_t prev;
    EIGEN_CUBLAS_CHECK(cublasGetPointerMode(ctx.cublasHandle(), &prev));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), CUBLAS_POINTER_MODE_DEVICE));
    EIGEN_CUBLAS_CHECK(internal::cublasXaxpy(ctx.cublasHandle(), n, expr.alpha().devicePtr(), x.data_, 1, data_, 1));
    EIGEN_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasHandle(), prev));
    recordReady(ctx.stream());
  }
  return *this;
}

// this -= DeviceScalar * x  (axpy with negated device scalar)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator-=(const DeviceScaledDevice<Scalar_>& expr) {
  auto neg_alpha = -expr.alpha();
  DeviceScaledDevice<Scalar_> neg_expr(neg_alpha, expr.matrix());
  return operator+=(neg_expr);
}

// this = alpha * A + beta * B  (cuBLAS geam)
template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const DeviceAddExpr<Scalar_>& expr) {
  auto& ctx = GpuContext::threadLocal();
  const auto& A = expr.A();
  const auto& B = expr.B();
  eigen_assert(A.rows() == B.rows() && A.cols() == B.cols());
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());
  // Wait on *this before resize — resize may free the old buffer while another
  // stream is still reading it.
  if (!empty()) waitReady(ctx.stream());
  resize(A.rows(), A.cols());
  if (m > 0 && n > 0) {
    A.waitReady(ctx.stream());
    B.waitReady(ctx.stream());
    Scalar_ alpha = expr.alpha();
    Scalar_ beta = expr.beta();
    EIGEN_CUBLAS_CHECK(internal::cublasXgeam(ctx.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, A.data(), m,
                                             &beta, B.data(), m, data_, m));
    recordReady(ctx.stream());
  }
  return *this;
}

// cwiseProduct via NPP nppsMul (allocating).
template <typename Scalar_>
DeviceMatrix<Scalar_> DeviceMatrix<Scalar_>::cwiseProduct(GpuContext& ctx, const DeviceMatrix& other) const {
  const int n = static_cast<int>(rows_ * cols_);
  eigen_assert(n == static_cast<int>(other.rows_ * other.cols_));
  DeviceMatrix result(rows_, cols_);
  if (n > 0) {
    waitReady(ctx.stream());
    other.waitReady(ctx.stream());
    internal::device_cwiseProduct(data_, other.data_, result.data_, n, ctx.stream());
    result.recordReady(ctx.stream());
  }
  return result;
}

// In-place cwiseProduct: this = a .* b (reuses this buffer, no allocation).
template <typename Scalar_>
void DeviceMatrix<Scalar_>::cwiseProduct(GpuContext& ctx, const DeviceMatrix& a, const DeviceMatrix& b) {
  const int n = static_cast<int>(a.rows_ * a.cols_);
  eigen_assert(n == static_cast<int>(b.rows_ * b.cols_));
  if (!empty()) waitReady(ctx.stream());
  resize(a.rows_, a.cols_);
  if (n > 0) {
    a.waitReady(ctx.stream());
    b.waitReady(ctx.stream());
    internal::device_cwiseProduct(a.data_, b.data_, data_, n, ctx.stream());
    recordReady(ctx.stream());
  }
}

// Convenience overloads using thread-local default GpuContext.
template <typename Scalar_>
DeviceScalar<typename DeviceMatrix<Scalar_>::Scalar> DeviceMatrix<Scalar_>::dot(const DeviceMatrix& other) const {
  return dot(GpuContext::threadLocal(), other);
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::squaredNorm() const {
  return squaredNorm(GpuContext::threadLocal());
}

template <typename Scalar_>
DeviceScalar<typename NumTraits<Scalar_>::Real> DeviceMatrix<Scalar_>::norm() const {
  return norm(GpuContext::threadLocal());
}

template <typename Scalar_>
void DeviceMatrix<Scalar_>::setZero() {
  setZero(GpuContext::threadLocal());
}

}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_DISPATCH_H
