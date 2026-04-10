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
//   GpuSelfAdjointEigenSolver<double> es(A);
//   VectorXd eigenvals = es.eigenvalues();
//   MatrixXd eigenvecs = es.eigenvectors();

#ifndef EIGEN_GPU_EIGENSOLVER_H
#define EIGEN_GPU_EIGENSOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuSolverSupport.h"
#include <vector>

namespace Eigen {

template <typename Scalar_>
class GpuSelfAdjointEigenSolver {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;

  /** Eigenvalue-only or eigenvalues + eigenvectors. */
  enum ComputeMode { EigenvaluesOnly, ComputeEigenvectors };

  GpuSelfAdjointEigenSolver() { init_context(); }

  template <typename InputType>
  explicit GpuSelfAdjointEigenSolver(const EigenBase<InputType>& A, ComputeMode mode = ComputeEigenvectors) {
    init_context();
    compute(A, mode);
  }

  ~GpuSelfAdjointEigenSolver() {
    if (handle_) (void)cusolverDnDestroy(handle_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  GpuSelfAdjointEigenSolver(const GpuSelfAdjointEigenSolver&) = delete;
  GpuSelfAdjointEigenSolver& operator=(const GpuSelfAdjointEigenSolver&) = delete;

  // ---- Factorization -------------------------------------------------------

  template <typename InputType>
  GpuSelfAdjointEigenSolver& compute(const EigenBase<InputType>& A, ComputeMode mode = ComputeEigenvectors) {
    eigen_assert(A.rows() == A.cols() && "GpuSelfAdjointEigenSolver requires a square matrix");
    mode_ = mode;
    n_ = A.rows();
    info_ = InvalidInput;
    info_synced_ = false;

    if (n_ == 0) {
      info_ = Success;
      info_synced_ = true;
      return *this;
    }

    const PlainMatrix mat(A.derived());
    lda_ = static_cast<int64_t>(n_);
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);

    // syevd overwrites A with eigenvectors (if requested).
    d_A_ = internal::DeviceBuffer(mat_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_A_.ptr, mat.data(), mat_bytes, cudaMemcpyHostToDevice, stream_));
    factorize();
    return *this;
  }

  GpuSelfAdjointEigenSolver& compute(const DeviceMatrix<Scalar>& d_A, ComputeMode mode = ComputeEigenvectors) {
    eigen_assert(d_A.rows() == d_A.cols() && "GpuSelfAdjointEigenSolver requires a square matrix");
    mode_ = mode;
    n_ = d_A.rows();
    info_ = InvalidInput;
    info_synced_ = false;

    if (n_ == 0) {
      info_ = Success;
      info_synced_ = true;
      return *this;
    }

    d_A.waitReady(stream_);
    lda_ = static_cast<int64_t>(n_);
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);

    d_A_ = internal::DeviceBuffer(mat_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_A_.ptr, d_A.data(), mat_bytes, cudaMemcpyDeviceToDevice, stream_));

    factorize();
    return *this;
  }

  // ---- Accessors -----------------------------------------------------------

  ComputationInfo info() const {
    sync_info();
    return info_;
  }

  Index cols() const { return n_; }
  Index rows() const { return n_; }

  /** Eigenvalues in ascending order. Downloads from device. */
  RealVector eigenvalues() const {
    sync_info();
    eigen_assert(info_ == Success);
    RealVector W(n_);
    if (n_ > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpy(W.data(), d_W_.ptr, static_cast<size_t>(n_) * sizeof(RealScalar), cudaMemcpyDeviceToHost));
    }
    return W;
  }

  /** Eigenvectors (columns). Downloads from device.
   * Requires ComputeEigenvectors mode. */
  PlainMatrix eigenvectors() const {
    sync_info();
    eigen_assert(info_ == Success);
    eigen_assert(mode_ == ComputeEigenvectors && "eigenvectors() requires ComputeEigenvectors mode");
    PlainMatrix V(n_, n_);
    if (n_ > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(V.data(), d_A_.ptr,
                                          static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
    }
    return V;
  }

  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t handle_ = nullptr;
  internal::CusolverParams params_;
  internal::DeviceBuffer d_A_;        // overwritten with eigenvectors by syevd
  internal::DeviceBuffer d_W_;        // eigenvalues (RealScalar, length n)
  internal::DeviceBuffer d_scratch_;  // workspace + info
  size_t scratch_size_ = 0;
  std::vector<char> h_workspace_;
  ComputeMode mode_ = ComputeEigenvectors;
  Index n_ = 0;
  int64_t lda_ = 0;
  ComputationInfo info_ = InvalidInput;
  int info_word_ = 0;
  bool info_synced_ = true;

  void init_context() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    ensure_scratch(0);
  }

  void ensure_scratch(size_t workspace_bytes) {
    constexpr size_t kAlign = 16;
    workspace_bytes = (workspace_bytes + kAlign - 1) & ~(kAlign - 1);
    size_t needed = workspace_bytes + sizeof(int);
    if (needed > scratch_size_) {
      if (d_scratch_.ptr) EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      d_scratch_ = internal::DeviceBuffer(needed);
      scratch_size_ = needed;
    }
  }

  void* scratch_workspace() const { return d_scratch_.ptr; }
  int* scratch_info() const {
    return reinterpret_cast<int*>(static_cast<char*>(d_scratch_.ptr) + scratch_size_ - sizeof(int));
  }

  void sync_info() const {
    if (!info_synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      const_cast<GpuSelfAdjointEigenSolver*>(this)->info_ = (info_word_ == 0) ? Success : NumericalIssue;
      const_cast<GpuSelfAdjointEigenSolver*>(this)->info_synced_ = true;
    }
  }

  void factorize() {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    constexpr cudaDataType_t rtype = internal::cuda_data_type<RealScalar>::value;

    info_synced_ = false;
    info_ = InvalidInput;

    d_W_ = internal::DeviceBuffer(static_cast<size_t>(n_) * sizeof(RealScalar));

    const cusolverEigMode_t jobz =
        (mode_ == ComputeEigenvectors) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    // Use lower triangle (standard convention).
    constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    size_t dev_ws = 0, host_ws = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(handle_, params_.p, jobz, uplo, static_cast<int64_t>(n_), dtype,
                                                     d_A_.ptr, lda_, rtype, d_W_.ptr, dtype, &dev_ws, &host_ws));

    ensure_scratch(dev_ws);
    h_workspace_.resize(host_ws);

    EIGEN_CUSOLVER_CHECK(cusolverDnXsyevd(handle_, params_.p, jobz, uplo, static_cast<int64_t>(n_), dtype, d_A_.ptr,
                                          lda_, rtype, d_W_.ptr, dtype, scratch_workspace(), dev_ws,
                                          host_ws > 0 ? h_workspace_.data() : nullptr, host_ws, scratch_info()));

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&info_word_, scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
  }
};

}  // namespace Eigen

#endif  // EIGEN_GPU_EIGENSOLVER_H
