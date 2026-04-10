// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// GPU QR decomposition using cuSOLVER.
//
// Wraps cusolverDnXgeqrf (factorization), cusolverDnXormqr (apply Q),
// cusolverDnXorgqr (form Q), and cublasXtrsm (triangular solve on R).
//
// The factored matrix (reflectors + R) and tau stay in device memory.
// Solve uses ormqr + trsm without forming Q explicitly.
//
// Usage:
//   GpuQR<double> qr(A);              // upload A, geqrf
//   if (qr.info() != Success) { ... }
//   MatrixXd X = qr.solve(B);         // Q^H * B via ormqr, then trsm on R
//
// Expression syntax:
//   d_X = d_A.qr().solve(d_B);        // temporary, no caching

#ifndef EIGEN_GPU_QR_H
#define EIGEN_GPU_QR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuSolverSupport.h"
#include "./CuBlasSupport.h"
#include <vector>

namespace Eigen {

template <typename Scalar_>
class GpuQR {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  GpuQR() { init_context(); }

  template <typename InputType>
  explicit GpuQR(const EigenBase<InputType>& A) {
    init_context();
    compute(A);
  }

  ~GpuQR() {
    if (handle_) (void)cusolverDnDestroy(handle_);
    if (cublas_) (void)cublasDestroy(cublas_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  GpuQR(const GpuQR&) = delete;
  GpuQR& operator=(const GpuQR&) = delete;

  GpuQR(GpuQR&& o) noexcept
      : stream_(o.stream_),
        handle_(o.handle_),
        cublas_(o.cublas_),
        params_(std::move(o.params_)),
        d_qr_(std::move(o.d_qr_)),
        d_tau_(std::move(o.d_tau_)),
        d_scratch_(std::move(o.d_scratch_)),
        scratch_size_(o.scratch_size_),
        h_workspace_(std::move(o.h_workspace_)),
        m_(o.m_),
        n_(o.n_),
        lda_(o.lda_),
        info_(o.info_),
        info_word_(o.info_word_),
        info_synced_(o.info_synced_) {
    o.stream_ = nullptr;
    o.handle_ = nullptr;
    o.cublas_ = nullptr;
    o.scratch_size_ = 0;
    o.m_ = 0;
    o.n_ = 0;
    o.lda_ = 0;
    o.info_ = InvalidInput;
    o.info_word_ = 0;
    o.info_synced_ = true;
  }

  GpuQR& operator=(GpuQR&& o) noexcept {
    if (this != &o) {
      if (handle_) (void)cusolverDnDestroy(handle_);
      if (cublas_) (void)cublasDestroy(cublas_);
      if (stream_) (void)cudaStreamDestroy(stream_);
      stream_ = o.stream_;
      handle_ = o.handle_;
      cublas_ = o.cublas_;
      params_ = std::move(o.params_);
      d_qr_ = std::move(o.d_qr_);
      d_tau_ = std::move(o.d_tau_);
      d_scratch_ = std::move(o.d_scratch_);
      scratch_size_ = o.scratch_size_;
      h_workspace_ = std::move(o.h_workspace_);
      m_ = o.m_;
      n_ = o.n_;
      lda_ = o.lda_;
      info_ = o.info_;
      info_word_ = o.info_word_;
      info_synced_ = o.info_synced_;
      o.stream_ = nullptr;
      o.handle_ = nullptr;
      o.cublas_ = nullptr;
      o.scratch_size_ = 0;
      o.m_ = 0;
      o.n_ = 0;
      o.lda_ = 0;
      o.info_ = InvalidInput;
      o.info_word_ = 0;
      o.info_synced_ = true;
    }
    return *this;
  }

  // ---- Factorization -------------------------------------------------------

  template <typename InputType>
  GpuQR& compute(const EigenBase<InputType>& A) {
    m_ = A.rows();
    n_ = A.cols();
    info_ = InvalidInput;
    info_synced_ = false;

    if (m_ == 0 || n_ == 0) {
      info_ = Success;
      info_synced_ = true;
      return *this;
    }

    const PlainMatrix mat(A.derived());
    lda_ = static_cast<int64_t>(mat.rows());
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);
    const size_t tau_bytes = static_cast<size_t>((std::min)(m_, n_)) * sizeof(Scalar);

    d_qr_ = internal::DeviceBuffer(mat_bytes);
    d_tau_ = internal::DeviceBuffer(tau_bytes);

    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_qr_.ptr, mat.data(), mat_bytes, cudaMemcpyHostToDevice, stream_));

    factorize();
    return *this;
  }

  GpuQR& compute(const DeviceMatrix<Scalar>& d_A) {
    m_ = d_A.rows();
    n_ = d_A.cols();
    info_ = InvalidInput;
    info_synced_ = false;

    if (m_ == 0 || n_ == 0) {
      info_ = Success;
      info_synced_ = true;
      return *this;
    }

    lda_ = static_cast<int64_t>(d_A.rows());
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);
    const size_t tau_bytes = static_cast<size_t>((std::min)(m_, n_)) * sizeof(Scalar);

    d_A.waitReady(stream_);
    d_qr_ = internal::DeviceBuffer(mat_bytes);
    d_tau_ = internal::DeviceBuffer(tau_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_qr_.ptr, d_A.data(), mat_bytes, cudaMemcpyDeviceToDevice, stream_));

    factorize();
    return *this;
  }

  // ---- Solve ---------------------------------------------------------------

  /** Solve A * X = B via QR: X = R^{-1} * Q^H * B (least-squares for m >= n).
   * Uses ormqr (apply Q^H) + trsm (solve R), without forming Q explicitly.
   * Requires m >= n (overdetermined or square). Underdetermined not supported. */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B) const {
    sync_info();
    eigen_assert(info_ == Success && "GpuQR::solve called on a failed or uninitialized factorization");
    eigen_assert(B.rows() == m_);
    eigen_assert(m_ >= n_ && "GpuQR::solve requires m >= n (use SVD for underdetermined systems)");

    const PlainMatrix rhs(B);
    const int64_t nrhs = static_cast<int64_t>(rhs.cols());
    const int64_t ldb = static_cast<int64_t>(rhs.rows());  // = m_
    const size_t b_bytes = static_cast<size_t>(ldb) * static_cast<size_t>(nrhs) * sizeof(Scalar);

    // Upload B to device (m × nrhs buffer).
    internal::DeviceBuffer d_B(b_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_B.ptr, rhs.data(), b_bytes, cudaMemcpyHostToDevice, stream_));

    // Apply Q^H to B in-place: d_B becomes m × nrhs, first n rows hold Q^H * B relevant part.
    apply_QH(d_B.ptr, ldb, nrhs);

    // Solve R * X = (Q^H * B)[0:n,:] via trsm on the first n rows.
    Scalar alpha(1);
    EIGEN_CUBLAS_CHECK(internal::cublasXtrsm(cublas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                             CUBLAS_DIAG_NON_UNIT, static_cast<int>(n_), static_cast<int>(nrhs), &alpha,
                                             static_cast<const Scalar*>(d_qr_.ptr), static_cast<int>(lda_),
                                             static_cast<Scalar*>(d_B.ptr), static_cast<int>(ldb)));

    // Download the first n rows of each column (stride = ldb = m, width = n).
    PlainMatrix X(n_, rhs.cols());
    if (m_ == n_) {
      // Square: dense copy, no stride mismatch.
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(X.data(), d_B.ptr,
                                               static_cast<size_t>(n_) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                               cudaMemcpyDeviceToHost, stream_));
    } else {
      // Overdetermined: 2D copy to extract first n rows from each column.
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy2DAsync(
          X.data(), static_cast<size_t>(n_) * sizeof(Scalar), d_B.ptr, static_cast<size_t>(ldb) * sizeof(Scalar),
          static_cast<size_t>(n_) * sizeof(Scalar), static_cast<size_t>(nrhs), cudaMemcpyDeviceToHost, stream_));
    }
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
    return X;
  }

  /** Solve with device-resident RHS. Returns n × nrhs DeviceMatrix. */
  DeviceMatrix<Scalar> solve(const DeviceMatrix<Scalar>& d_B) const {
    sync_info();
    eigen_assert(info_ == Success && "GpuQR::solve called on a failed or uninitialized factorization");
    eigen_assert(d_B.rows() == m_);
    eigen_assert(m_ >= n_ && "GpuQR::solve requires m >= n (use SVD for underdetermined systems)");
    d_B.waitReady(stream_);

    const int64_t nrhs = static_cast<int64_t>(d_B.cols());
    const int64_t ldb = static_cast<int64_t>(d_B.rows());  // = m_
    const size_t b_bytes = static_cast<size_t>(ldb) * static_cast<size_t>(nrhs) * sizeof(Scalar);

    // D2D copy B into working buffer (ormqr and trsm are in-place).
    internal::DeviceBuffer d_work(b_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_work.ptr, d_B.data(), b_bytes, cudaMemcpyDeviceToDevice, stream_));

    apply_QH(d_work.ptr, ldb, nrhs);

    // trsm on the first n rows.
    Scalar alpha(1);
    EIGEN_CUBLAS_CHECK(internal::cublasXtrsm(cublas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                             CUBLAS_DIAG_NON_UNIT, static_cast<int>(n_), static_cast<int>(nrhs), &alpha,
                                             static_cast<const Scalar*>(d_qr_.ptr), static_cast<int>(lda_),
                                             static_cast<Scalar*>(d_work.ptr), static_cast<int>(ldb)));

    if (m_ == n_) {
      // Square: result is the whole buffer, dense.
      DeviceMatrix<Scalar> result(static_cast<Scalar*>(d_work.ptr), n_, static_cast<Index>(nrhs));
      d_work.ptr = nullptr;  // transfer ownership
      result.recordReady(stream_);
      return result;
    } else {
      // Overdetermined: copy first n rows of each column into a dense n × nrhs result.
      DeviceMatrix<Scalar> result(n_, static_cast<Index>(nrhs));
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy2DAsync(result.data(), static_cast<size_t>(n_) * sizeof(Scalar), d_work.ptr,
                                                 static_cast<size_t>(ldb) * sizeof(Scalar),
                                                 static_cast<size_t>(n_) * sizeof(Scalar), static_cast<size_t>(nrhs),
                                                 cudaMemcpyDeviceToDevice, stream_));
      result.recordReady(stream_);
      return result;
      // d_work freed here via RAII — safe because stream is ordered.
    }
  }

  // ---- Accessors -----------------------------------------------------------

  ComputationInfo info() const {
    sync_info();
    return info_;
  }

  Index rows() const { return m_; }
  Index cols() const { return n_; }
  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t handle_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  internal::CusolverParams params_;
  internal::DeviceBuffer d_qr_;       // QR factors (reflectors in lower, R in upper)
  internal::DeviceBuffer d_tau_;      // Householder scalars (min(m,n))
  internal::DeviceBuffer d_scratch_;  // workspace + info word
  size_t scratch_size_ = 0;
  std::vector<char> h_workspace_;
  Index m_ = 0;
  Index n_ = 0;
  int64_t lda_ = 0;
  ComputationInfo info_ = InvalidInput;
  int info_word_ = 0;
  bool info_synced_ = true;

  void init_context() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    EIGEN_CUBLAS_CHECK(cublasCreate(&cublas_));
    EIGEN_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
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
      const_cast<GpuQR*>(this)->info_ = (info_word_ == 0) ? Success : NumericalIssue;
      const_cast<GpuQR*>(this)->info_synced_ = true;
    }
  }

  void factorize() {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;

    info_synced_ = false;
    info_ = InvalidInput;

    size_t dev_ws = 0, host_ws = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(handle_, params_.p, static_cast<int64_t>(m_),
                                                     static_cast<int64_t>(n_), dtype, d_qr_.ptr, lda_, dtype,
                                                     d_tau_.ptr, dtype, &dev_ws, &host_ws));

    ensure_scratch(dev_ws);
    h_workspace_.resize(host_ws);

    EIGEN_CUSOLVER_CHECK(cusolverDnXgeqrf(handle_, params_.p, static_cast<int64_t>(m_), static_cast<int64_t>(n_), dtype,
                                          d_qr_.ptr, lda_, dtype, d_tau_.ptr, dtype, scratch_workspace(), dev_ws,
                                          host_ws > 0 ? h_workspace_.data() : nullptr, host_ws, scratch_info()));

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&info_word_, scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
  }

  // Apply Q^H to a device buffer in-place: d_B = Q^H * d_B.
  // Uses type-specific ormqr (real) or unmqr (complex) wrappers from CuSolverSupport.h.
  // For real types: Q^H = Q^T, use CUBLAS_OP_T. For complex: use CUBLAS_OP_C.
  void apply_QH(void* d_B, int64_t ldb, int64_t nrhs) const {
    const int im = static_cast<int>(m_);
    const int in = static_cast<int>(nrhs);
    const int ik = static_cast<int>((std::min)(m_, n_));
    const int ilda = static_cast<int>(lda_);
    const int ildb = static_cast<int>(ldb);
    constexpr cublasOperation_t trans = NumTraits<Scalar>::IsComplex ? CUBLAS_OP_C : CUBLAS_OP_T;

    int lwork = 0;
    EIGEN_CUSOLVER_CHECK(internal::cusolverDnXormqr_bufferSize(
        handle_, CUBLAS_SIDE_LEFT, trans, im, in, ik, static_cast<const Scalar*>(d_qr_.ptr), ilda,
        static_cast<const Scalar*>(d_tau_.ptr), static_cast<const Scalar*>(d_B), ildb, &lwork));

    internal::DeviceBuffer d_work(static_cast<size_t>(lwork) * sizeof(Scalar));

    EIGEN_CUSOLVER_CHECK(internal::cusolverDnXormqr(handle_, CUBLAS_SIDE_LEFT, trans, im, in, ik,
                                                    static_cast<const Scalar*>(d_qr_.ptr), ilda,
                                                    static_cast<const Scalar*>(d_tau_.ptr), static_cast<Scalar*>(d_B),
                                                    ildb, static_cast<Scalar*>(d_work.ptr), lwork, scratch_info()));

    // Sync to ensure workspace can be freed safely, and check ormqr info.
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
    int ormqr_info = 0;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(&ormqr_info, scratch_info(), sizeof(int), cudaMemcpyDeviceToHost));
    eigen_assert(ormqr_info == 0 && "cusolverDnXormqr reported an error");
  }
};

}  // namespace Eigen

#endif  // EIGEN_GPU_QR_H
