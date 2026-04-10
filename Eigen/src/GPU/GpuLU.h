// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// GPU partial-pivoting LU decomposition using cuSOLVER.
//
// Wraps cusolverDnXgetrf (factorization) and cusolverDnXgetrs (solve).
// The factored LU matrix and pivot array are kept in device memory for the
// lifetime of the object, so repeated solves only transfer the RHS/solution.
//
// Requires CUDA 11.0+ (cusolverDnX generic API).
//
// Usage:
//   LU<double> lu(A);              // upload A, getrf, LU+ipiv on device
//   if (lu.info() != Success) { ... }
//   MatrixXd x = lu.solve(b);         // getrs NoTrans, only b transferred
//   MatrixXd xt = lu.solve(b, LU<double>::Transpose);   // A^T x = b

#ifndef EIGEN_GPU_LU_H
#define EIGEN_GPU_LU_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSolverContext.h"

namespace Eigen {
namespace gpu {

/** \ingroup GPU_Module
 * \class LU
 * \brief GPU LU decomposition with partial pivoting via cuSOLVER
 *
 * \tparam Scalar_  Element type: float, double, complex<float>, complex<double>
 *
 * Decomposes a square matrix A = P L U on the GPU and retains the factored
 * matrix and pivot array in device memory. Solves A*X=B, A^T*X=B, or
 * A^H*X=B by passing the appropriate TransposeMode.
 *
 * Each LU object owns a dedicated CUDA stream and cuSOLVER handle.
 */
template <typename Scalar_>
class LU {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  /** Controls which system is solved in solve(). */
  enum TransposeMode {
    NoTranspose,        ///< Solve A   * X = B
    Transpose,          ///< Solve A^T * X = B
    ConjugateTranspose  ///< Solve A^H * X = B (same as Transpose for real types)
  };

  // ---- Construction / destruction ------------------------------------------

  LU() = default;

  template <typename InputType>
  explicit LU(const EigenBase<InputType>& A) {
    compute(A);
  }

  ~LU() = default;

  LU(const LU&) = delete;
  LU& operator=(const LU&) = delete;

  LU(LU&& o) noexcept
      : ctx_(std::move(o.ctx_)),
        d_lu_(std::move(o.d_lu_)),
        lu_alloc_size_(o.lu_alloc_size_),
        d_ipiv_(std::move(o.d_ipiv_)),
        n_(o.n_),
        lda_(o.lda_) {
    o.lu_alloc_size_ = 0;
    o.n_ = 0;
    o.lda_ = 0;
  }

  LU& operator=(LU&& o) noexcept {
    if (this != &o) {
      ctx_ = std::move(o.ctx_);
      d_lu_ = std::move(o.d_lu_);
      lu_alloc_size_ = o.lu_alloc_size_;
      d_ipiv_ = std::move(o.d_ipiv_);
      n_ = o.n_;
      lda_ = o.lda_;
      o.lu_alloc_size_ = 0;
      o.n_ = 0;
      o.lda_ = 0;
    }
    return *this;
  }

  // ---- Factorization -------------------------------------------------------

  /** Compute the LU factorization of A (host matrix, must be square). */
  template <typename InputType>
  LU& compute(const EigenBase<InputType>& A) {
    eigen_assert(A.rows() == A.cols() && "LU requires a square matrix");
    if (!begin_compute(A.rows())) return *this;

    const PlainMatrix mat(A.derived());
    lda_ = static_cast<int64_t>(mat.rows());
    allocate_lu_storage();
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_lu_.ptr, mat.data(), matrixBytes(), cudaMemcpyHostToDevice, ctx_.stream_));

    factorize();
    return *this;
  }

  /** Compute the LU factorization from a device-resident matrix (D2D copy). */
  LU& compute(const DeviceMatrix<Scalar>& d_A) {
    eigen_assert(d_A.rows() == d_A.cols() && "LU requires a square matrix");
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.rows());
    d_A.waitReady(ctx_.stream_);
    allocate_lu_storage();
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_lu_.ptr, d_A.data(), matrixBytes(), cudaMemcpyDeviceToDevice, ctx_.stream_));

    factorize();
    return *this;
  }

  /** Compute the LU factorization from a device matrix (move, no copy). */
  LU& compute(DeviceMatrix<Scalar>&& d_A) {
    eigen_assert(d_A.rows() == d_A.cols() && "LU requires a square matrix");
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.rows());
    d_A.waitReady(ctx_.stream_);
    d_lu_ = internal::DeviceBuffer::adopt(static_cast<void*>(d_A.release()));

    factorize();
    return *this;
  }

  // ---- Solve ---------------------------------------------------------------

  /** Solve op(A) * X = B using the cached LU factorization (host → host). */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B, TransposeMode mode = NoTranspose) const {
    ctx_.sync_info();
    eigen_assert(ctx_.info_ == Success && "LU::solve called on a failed or uninitialized factorization");
    eigen_assert(B.rows() == n_);

    const PlainMatrix rhs(B);
    const int64_t nrhs = static_cast<int64_t>(rhs.cols());
    const int64_t ldb = static_cast<int64_t>(rhs.rows());
    DeviceMatrix<Scalar> d_X = solve_impl(nrhs, ldb, mode, [&](Scalar* d_x_ptr) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_x_ptr, rhs.data(), matrixBytes(nrhs, ldb), cudaMemcpyHostToDevice, ctx_.stream_));
    });

    PlainMatrix X(n_, B.cols());
    int solve_info = 0;
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(X.data(), d_X.data(), matrixBytes(nrhs, ldb), cudaMemcpyDeviceToHost, ctx_.stream_));
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&solve_info, ctx_.scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, ctx_.stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx_.stream_));

    eigen_assert(solve_info == 0 && "cusolverDnXgetrs reported an error");
    return X;
  }

  /** Solve op(A) * X = B with device-resident RHS. Fully async. */
  DeviceMatrix<Scalar> solve(const DeviceMatrix<Scalar>& d_B, TransposeMode mode = NoTranspose) const {
    eigen_assert(d_B.rows() == n_);
    d_B.waitReady(ctx_.stream_);
    const int64_t nrhs = static_cast<int64_t>(d_B.cols());
    const int64_t ldb = static_cast<int64_t>(d_B.rows());
    return solve_impl(nrhs, ldb, mode, [&](Scalar* d_x_ptr) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_x_ptr, d_B.data(), matrixBytes(nrhs, ldb), cudaMemcpyDeviceToDevice, ctx_.stream_));
    });
  }

  // ---- Accessors -----------------------------------------------------------

  ComputationInfo info() const { return ctx_.info(); }
  Index rows() const { return n_; }
  Index cols() const { return n_; }
  cudaStream_t stream() const { return ctx_.stream_; }

 private:
  mutable internal::GpuSolverContext ctx_;
  internal::DeviceBuffer d_lu_;
  size_t lu_alloc_size_ = 0;
  internal::DeviceBuffer d_ipiv_;
  Index n_ = 0;
  int64_t lda_ = 0;

  bool begin_compute(Index rows) {
    n_ = rows;
    ctx_.info_ = InvalidInput;
    if (n_ == 0) {
      ctx_.info_ = Success;
      return false;
    }
    return true;
  }

  size_t matrixBytes() const { return matrixBytes(static_cast<int64_t>(n_), lda_); }

  static size_t matrixBytes(int64_t cols, int64_t ld) {
    return static_cast<size_t>(ld) * static_cast<size_t>(cols) * sizeof(Scalar);
  }

  void allocate_lu_storage() {
    size_t needed = matrixBytes();
    if (needed > lu_alloc_size_) {
      d_lu_ = internal::DeviceBuffer(needed);
      lu_alloc_size_ = needed;
    }
  }

  template <typename CopyRhs>
  DeviceMatrix<Scalar> solve_impl(int64_t nrhs, int64_t ldb, TransposeMode mode, CopyRhs&& copy_rhs) const {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    const cublasOperation_t trans = to_cublas_op(mode);

    internal::DeviceBuffer d_x(matrixBytes(nrhs, ldb));
    Scalar* d_x_ptr = static_cast<Scalar*>(d_x.ptr);
    copy_rhs(d_x_ptr);

    EIGEN_CUSOLVER_CHECK(cusolverDnXgetrs(ctx_.cusolver_, ctx_.params_.p, trans, static_cast<int64_t>(n_), nrhs, dtype,
                                          d_lu_.ptr, lda_, static_cast<const int64_t*>(d_ipiv_.ptr), dtype, d_x_ptr,
                                          ldb, ctx_.scratch_info()));

    DeviceMatrix<Scalar> result =
        DeviceMatrix<Scalar>::adopt(static_cast<Scalar*>(d_x.ptr), n_, static_cast<Index>(nrhs));
    d_x.ptr = nullptr;
    result.recordReady(ctx_.stream_);
    return result;
  }

  void factorize() {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    const size_t ipiv_bytes = static_cast<size_t>(n_) * sizeof(int64_t);

    ctx_.mark_pending();

    d_ipiv_ = internal::DeviceBuffer(ipiv_bytes);

    size_t dev_ws_bytes = 0, host_ws_bytes = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(ctx_.cusolver_, ctx_.params_.p, static_cast<int64_t>(n_),
                                                     static_cast<int64_t>(n_), dtype, d_lu_.ptr, lda_, dtype,
                                                     &dev_ws_bytes, &host_ws_bytes));

    ctx_.ensure_scratch(dev_ws_bytes);
    ctx_.h_workspace_.resize(host_ws_bytes);

    EIGEN_CUSOLVER_CHECK(cusolverDnXgetrf(
        ctx_.cusolver_, ctx_.params_.p, static_cast<int64_t>(n_), static_cast<int64_t>(n_), dtype, d_lu_.ptr, lda_,
        static_cast<int64_t*>(d_ipiv_.ptr), dtype, ctx_.scratch_workspace(), dev_ws_bytes,
        host_ws_bytes > 0 ? ctx_.h_workspace_.data() : nullptr, host_ws_bytes, ctx_.scratch_info()));

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&ctx_.info_word_, ctx_.scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, ctx_.stream_));
  }

  static cublasOperation_t to_cublas_op(TransposeMode mode) {
    switch (mode) {
      case Transpose:
        return CUBLAS_OP_T;
      case ConjugateTranspose:
        return CUBLAS_OP_C;
      default:
        return CUBLAS_OP_N;
    }
  }
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_LU_H
