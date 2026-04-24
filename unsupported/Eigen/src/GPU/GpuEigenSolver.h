// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// GPU self-adjoint eigenvalue decomposition using cuSOLVER.
//
// Wraps cusolverDnXsyevd (symmetric/Hermitian divide-and-conquer).
// Stores eigenvalues and eigenvectors on device.
//
// Usage:
//   SelfAdjointEigenSolver<double> es(A);
//   VectorXd eigenvals = es.eigenvalues();
//   MatrixXd eigenvecs = es.eigenvectors();

#ifndef EIGEN_GPU_EIGENSOLVER_H
#define EIGEN_GPU_EIGENSOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSolverContext.h"

namespace Eigen {
namespace gpu {

template <typename Scalar_>
class SelfAdjointEigenSolver {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RealVector = Eigen::Matrix<RealScalar, Dynamic, 1>;

  /** Eigenvalue-only or eigenvalues + eigenvectors. */
  enum ComputeMode { EigenvaluesOnly, ComputeEigenvectors };

  SelfAdjointEigenSolver() = default;

  template <typename InputType>
  explicit SelfAdjointEigenSolver(const EigenBase<InputType>& A, ComputeMode mode = ComputeEigenvectors) {
    compute(A, mode);
  }

  ~SelfAdjointEigenSolver() = default;

  SelfAdjointEigenSolver(const SelfAdjointEigenSolver&) = delete;
  SelfAdjointEigenSolver& operator=(const SelfAdjointEigenSolver&) = delete;

  // ---- Factorization -------------------------------------------------------

  template <typename InputType>
  SelfAdjointEigenSolver& compute(const EigenBase<InputType>& A, ComputeMode mode = ComputeEigenvectors) {
    eigen_assert(A.rows() == A.cols() && "SelfAdjointEigenSolver requires a square matrix");
    mode_ = mode;
    n_ = A.rows();
    ctx_.mark_pending();

    if (n_ == 0) {
      ctx_.info_ = Success;
      ctx_.info_synced_ = true;
      return *this;
    }

    const PlainMatrix mat(A.derived());
    lda_ = n_;
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);

    // syevd overwrites A with eigenvectors (if requested).
    d_A_ = internal::DeviceBuffer(mat_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_A_.get(), mat.data(), mat_bytes, cudaMemcpyHostToDevice, ctx_.stream_));
    factorize();
    return *this;
  }

  SelfAdjointEigenSolver& compute(const DeviceMatrix<Scalar>& d_A, ComputeMode mode = ComputeEigenvectors) {
    eigen_assert(d_A.rows() == d_A.cols() && "SelfAdjointEigenSolver requires a square matrix");
    mode_ = mode;
    n_ = d_A.rows();
    ctx_.mark_pending();

    if (n_ == 0) {
      ctx_.info_ = Success;
      ctx_.info_synced_ = true;
      return *this;
    }

    d_A.waitReady(ctx_.stream_);
    lda_ = n_;
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);

    d_A_ = internal::DeviceBuffer(mat_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_A_.get(), d_A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx_.stream_));

    factorize();
    return *this;
  }

  // ---- Accessors -----------------------------------------------------------

  ComputationInfo info() const { return ctx_.info(); }

  Index cols() const { return n_; }
  Index rows() const { return n_; }

  /** Eigenvalues in ascending order. Downloads from device. */
  RealVector eigenvalues() const {
    ctx_.sync_info();
    eigen_assert(ctx_.info_ == Success);
    RealVector W(n_);
    if (n_ > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpy(W.data(), d_W_.get(), static_cast<size_t>(n_) * sizeof(RealScalar), cudaMemcpyDeviceToHost));
    }
    return W;
  }

  /** Eigenvectors (columns). Downloads from device.
   * Requires ComputeEigenvectors mode. */
  PlainMatrix eigenvectors() const {
    ctx_.sync_info();
    eigen_assert(ctx_.info_ == Success);
    eigen_assert(mode_ == ComputeEigenvectors && "eigenvectors() requires ComputeEigenvectors mode");
    PlainMatrix V(n_, n_);
    if (n_ > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(V.data(), d_A_.get(),
                                          static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
    }
    return V;
  }

  cudaStream_t stream() const { return ctx_.stream_; }

 private:
  mutable internal::GpuSolverContext ctx_;
  internal::DeviceBuffer d_A_;  // overwritten with eigenvectors by syevd
  internal::DeviceBuffer d_W_;  // eigenvalues (RealScalar, length n)
  ComputeMode mode_ = ComputeEigenvectors;
  int64_t n_ = 0;
  int64_t lda_ = 0;

  void factorize() {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    constexpr cudaDataType_t rtype = internal::cuda_data_type<RealScalar>::value;

    ctx_.mark_pending();

    d_W_ = internal::DeviceBuffer(static_cast<size_t>(n_) * sizeof(RealScalar));

    const cusolverEigMode_t jobz =
        (mode_ == ComputeEigenvectors) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    // Use lower triangle (standard convention).
    constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    size_t dev_ws = 0, host_ws = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(ctx_.cusolver_, ctx_.params_.p, jobz, uplo, n_, dtype, d_A_.get(),
                                                     lda_, rtype, d_W_.get(), dtype, &dev_ws, &host_ws));

    ctx_.ensure_scratch(dev_ws);
    ctx_.h_workspace_.resize(host_ws);

    EIGEN_CUSOLVER_CHECK(cusolverDnXsyevd(ctx_.cusolver_, ctx_.params_.p, jobz, uplo, n_, dtype, d_A_.get(), lda_,
                                          rtype, d_W_.get(), dtype, ctx_.scratch_workspace(), dev_ws,
                                          host_ws > 0 ? ctx_.h_workspace_.data() : nullptr, host_ws,
                                          ctx_.scratch_info()));

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&ctx_.info_word_, ctx_.scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, ctx_.stream_));
  }
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_EIGENSOLVER_H
