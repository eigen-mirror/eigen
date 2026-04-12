// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// GPU Cholesky (LLT) decomposition using cuSOLVER.
//
// Unlike Eigen's CPU LLT<MatrixType>, gpu::LLT keeps the factored Cholesky
// factor in device memory for the lifetime of the object. Multiple solves
// against the same factor therefore only transfer the RHS and solution
// vectors, not the factor itself.
//
// Requires CUDA 11.0+ (cusolverDnXpotrf / cusolverDnXpotrs generic API).
// Requires CUDA 11.4+ (cusolverDnX generic API + cudaMallocAsync).
//
// Usage:
//   gpu::LLT<double> llt(A);            // upload A, potrf, L stays on device
//   if (llt.info() != Success) { ... }
//   MatrixXd x1 = llt.solve(b1);        // potrs, only b1 transferred
//   MatrixXd x2 = llt.solve(b2);        // L already on device

#ifndef EIGEN_GPU_LLT_H
#define EIGEN_GPU_LLT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuSolverSupport.h"
#include <vector>

namespace Eigen {
namespace gpu {

/** \ingroup GPU_Module
 * \class LLT
 * \brief GPU Cholesky (LL^T) decomposition via cuSOLVER
 *
 * \tparam Scalar_  Element type: float, double, complex<float>, complex<double>
 * \tparam UpLo_    Triangle used: Lower (default) or Upper
 *
 * Factorizes a symmetric positive-definite matrix A = LL^H on the GPU and
 * caches the factor L in device memory. Each subsequent solve(B) uploads only
 * B, calls cusolverDnXpotrs, and downloads the result — the factor is not
 * re-transferred.
 *
 * Each LLT object owns a dedicated CUDA stream and cuSOLVER handle,
 * enabling concurrent factorizations from multiple objects on the same host
 * thread.
 */
template <typename Scalar_, int UpLo_ = Lower>
class LLT {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  enum { UpLo = UpLo_ };

  // ---- Construction / destruction ------------------------------------------

  /** Default constructor. Does not factorize; call compute() before solve(). */
  LLT() { init_context(); }

  /** Factor A immediately. Equivalent to LLT llt; llt.compute(A). */
  template <typename InputType>
  explicit LLT(const EigenBase<InputType>& A) {
    init_context();
    compute(A);
  }

  ~LLT() {
    // Ignore errors in destructors — cannot propagate.
    if (handle_) (void)cusolverDnDestroy(handle_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  // Non-copyable (owns device memory and library handles).
  LLT(const LLT&) = delete;
  LLT& operator=(const LLT&) = delete;

  // Movable.
  LLT(LLT&& o) noexcept
      : stream_(o.stream_),
        handle_(o.handle_),
        params_(std::move(o.params_)),
        d_factor_(std::move(o.d_factor_)),
        factor_alloc_size_(o.factor_alloc_size_),
        d_scratch_(std::move(o.d_scratch_)),
        scratch_size_(o.scratch_size_),
        h_workspace_(std::move(o.h_workspace_)),
        n_(o.n_),
        lda_(o.lda_),
        info_(o.info_),
        pinned_info_(std::move(o.pinned_info_)),
        info_synced_(o.info_synced_) {
    o.stream_ = nullptr;
    o.handle_ = nullptr;
    o.factor_alloc_size_ = 0;
    o.scratch_size_ = 0;
    o.n_ = 0;
    o.lda_ = 0;
    o.info_ = InvalidInput;
    o.info_synced_ = true;
  }

  LLT& operator=(LLT&& o) noexcept {
    if (this != &o) {
      if (handle_) (void)cusolverDnDestroy(handle_);
      if (stream_) (void)cudaStreamDestroy(stream_);
      stream_ = o.stream_;
      handle_ = o.handle_;
      params_ = std::move(o.params_);
      d_factor_ = std::move(o.d_factor_);
      factor_alloc_size_ = o.factor_alloc_size_;
      d_scratch_ = std::move(o.d_scratch_);
      scratch_size_ = o.scratch_size_;
      h_workspace_ = std::move(o.h_workspace_);
      n_ = o.n_;
      lda_ = o.lda_;
      info_ = o.info_;
      pinned_info_ = std::move(o.pinned_info_);
      info_synced_ = o.info_synced_;
      o.stream_ = nullptr;
      o.handle_ = nullptr;
      o.factor_alloc_size_ = 0;
      o.scratch_size_ = 0;
      o.n_ = 0;
      o.lda_ = 0;
      o.info_ = InvalidInput;
      o.info_synced_ = true;
    }
    return *this;
  }

  // ---- Factorization -------------------------------------------------------

  /** Compute the Cholesky factorization of A (host matrix).
   *
   * Uploads A to device memory, calls cusolverDnXpotrf, and retains the
   * factored matrix on device. Any previous factorization is overwritten.
   */
  template <typename InputType>
  LLT& compute(const EigenBase<InputType>& A) {
    eigen_assert(A.rows() == A.cols());
    if (!begin_compute(A.rows())) return *this;

    // Evaluate A into a contiguous ColMajor matrix (handles arbitrary expressions).
    const PlainMatrix mat(A.derived());
    lda_ = static_cast<int64_t>(mat.outerStride());
    allocate_factor_storage();
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_factor_.ptr, mat.data(), factorBytes(), cudaMemcpyHostToDevice, stream_));

    factorize();
    return *this;
  }

  /** Compute the Cholesky factorization from a device-resident matrix (D2D copy). */
  LLT& compute(const DeviceMatrix<Scalar>& d_A) {
    eigen_assert(d_A.rows() == d_A.cols());
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.outerStride());
    d_A.waitReady(stream_);
    allocate_factor_storage();
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_factor_.ptr, d_A.data(), factorBytes(), cudaMemcpyDeviceToDevice, stream_));

    factorize();
    return *this;
  }

  /** Compute the Cholesky factorization from a device matrix (move, no copy). */
  LLT& compute(DeviceMatrix<Scalar>&& d_A) {
    eigen_assert(d_A.rows() == d_A.cols());
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.outerStride());
    d_A.waitReady(stream_);
    d_factor_ = internal::DeviceBuffer::adopt(static_cast<void*>(d_A.release()));

    factorize();
    return *this;
  }

  // ---- Solve ---------------------------------------------------------------

  /** Solve A * X = B using the cached Cholesky factor (host → host).
   *
   * Uploads B to device memory, calls cusolverDnXpotrs using the factor
   * retained from compute(), and returns the solution X on the host.
   * The factor is not re-transferred; only B goes up and X comes down.
   *
   * \pre compute() must have been called and info() == Success.
   * \returns X such that A * X ≈ B
   */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B) const {
    const_cast<LLT*>(this)->sync_info();
    eigen_assert(info_ == Success && "LLT::solve called on a failed or uninitialized factorization");
    eigen_assert(B.rows() == n_);

    const PlainMatrix rhs(B);
    const int64_t nrhs = static_cast<int64_t>(rhs.cols());
    const int64_t ldb = static_cast<int64_t>(rhs.outerStride());
    DeviceMatrix<Scalar> d_X = solve_impl(nrhs, ldb, [&](Scalar* d_x_ptr) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_x_ptr, rhs.data(), rhsBytes(nrhs, ldb), cudaMemcpyHostToDevice, stream_));
    });

    PlainMatrix X(n_, B.cols());
    int solve_info = 0;
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(X.data(), d_X.data(), rhsBytes(nrhs, ldb), cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&solve_info, scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));

    eigen_assert(solve_info == 0 && "cusolverDnXpotrs reported an error");
    return X;
  }

  /** Solve A * X = B with device-resident RHS. Fully async.
   *
   * All work is enqueued on this solver's stream. Returns a Matrix
   * with a recorded ready event — no host synchronization occurs.
   * The caller should check info() after compute() to verify the
   * factorization succeeded; this method does not check.
   */
  DeviceMatrix<Scalar> solve(const DeviceMatrix<Scalar>& d_B) const {
    eigen_assert(d_B.rows() == n_);
    d_B.waitReady(stream_);
    const int64_t nrhs = static_cast<int64_t>(d_B.cols());
    const int64_t ldb = static_cast<int64_t>(d_B.outerStride());
    return solve_impl(nrhs, ldb, [&](Scalar* d_x_ptr) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_x_ptr, d_B.data(), rhsBytes(nrhs, ldb), cudaMemcpyDeviceToDevice, stream_));
    });
  }

  // ---- Accessors -----------------------------------------------------------

  /** Returns Success if the last compute() succeeded, NumericalIssue otherwise.
   * Lazily synchronizes the stream on first call after compute(). */
  ComputationInfo info() const {
    const_cast<LLT*>(this)->sync_info();
    return info_;
  }

  Index rows() const { return n_; }
  Index cols() const { return n_; }

  /** Returns the CUDA stream owned by this object.
   *  Advanced users may submit additional GPU work on this stream
   *  to overlap with or chain after LLT operations. */
  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t handle_ = nullptr;
  internal::CusolverParams params_;   // cuSOLVER params (created once, reused)
  internal::DeviceBuffer d_factor_;   // factored L (or U) on device (grows, never shrinks)
  size_t factor_alloc_size_ = 0;      // current d_factor_ allocation size
  internal::DeviceBuffer d_scratch_;  // combined workspace + info word (grows, never shrinks)
  size_t scratch_size_ = 0;           // current scratch allocation size
  std::vector<char> h_workspace_;     // host workspace (kept alive until next compute)
  Index n_ = 0;
  int64_t lda_ = 0;
  ComputationInfo info_ = InvalidInput;
  internal::PinnedHostBuffer pinned_info_{sizeof(int)};  // pinned host memory for async D2H
  bool info_synced_ = true;                              // has the stream been synced for info?

  int& info_word() { return *static_cast<int*>(pinned_info_.ptr); }
  int info_word() const { return *static_cast<const int*>(pinned_info_.ptr); }

  bool begin_compute(Index rows) {
    n_ = rows;
    info_ = InvalidInput;
    if (n_ == 0) {
      info_ = Success;
      return false;
    }
    return true;
  }

  size_t factorBytes() const { return rhsBytes(static_cast<int64_t>(n_), lda_); }

  static size_t rhsBytes(int64_t cols, int64_t outer_stride) {
    return static_cast<size_t>(outer_stride) * static_cast<size_t>(cols) * sizeof(Scalar);
  }

  void allocate_factor_storage() {
    size_t needed = factorBytes();
    if (needed > factor_alloc_size_) {
      d_factor_ = internal::DeviceBuffer(needed);
      factor_alloc_size_ = needed;
    }
  }

  // Ensure d_scratch_ is at least `workspace_bytes + sizeof(int)`.
  // Layout: [workspace (workspace_bytes) | info_word (sizeof(int))].
  // Ensure d_scratch_ can hold workspace_bytes + an aligned info word.
  // Grows but never shrinks. Syncs the stream before reallocating to
  // avoid freeing memory that async kernels may still be using.
  void ensure_scratch(size_t workspace_bytes) {
    // Round up so the info word is naturally aligned.
    // 16-byte alignment for optimal GPU memory access.
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

  template <typename CopyRhs>
  DeviceMatrix<Scalar> solve_impl(int64_t nrhs, int64_t ldb, CopyRhs&& copy_rhs) const {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    constexpr cublasFillMode_t uplo = internal::cusolver_fill_mode<UpLo_, ColMajor>::value;

    internal::DeviceBuffer d_x(rhsBytes(nrhs, ldb));
    Scalar* d_x_ptr = static_cast<Scalar*>(d_x.ptr);
    copy_rhs(d_x_ptr);

    EIGEN_CUSOLVER_CHECK(cusolverDnXpotrs(handle_, params_.p, uplo, static_cast<int64_t>(n_), nrhs, dtype,
                                          d_factor_.ptr, lda_, dtype, d_x_ptr, ldb, scratch_info()));

    DeviceMatrix<Scalar> result = DeviceMatrix<Scalar>::adopt(static_cast<Scalar*>(d_x.ptr), n_, static_cast<Index>(nrhs));
    d_x.ptr = nullptr;  // ownership transferred to result
    result.recordReady(stream_);
    return result;
  }

  void init_context() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    ensure_scratch(0);  // allocate at least the info word
  }

  // Synchronize stream and interpret the info word. No-op if already synced.
  void sync_info() {
    if (!info_synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      info_ = (info_word() == 0) ? Success : NumericalIssue;
      info_synced_ = true;
    }
  }

  // Run cusolverDnXpotrf on d_factor_ (already on device).
  // Enqueues factorization + async info download. Does NOT sync.
  // Workspaces are stored as members to ensure they outlive the async kernels.
  void factorize() {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    constexpr cublasFillMode_t uplo = internal::cusolver_fill_mode<UpLo_, ColMajor>::value;

    info_synced_ = false;
    info_ = InvalidInput;

    size_t dev_ws_bytes = 0, host_ws_bytes = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(handle_, params_.p, uplo, static_cast<int64_t>(n_), dtype,
                                                     d_factor_.ptr, lda_, dtype, &dev_ws_bytes, &host_ws_bytes));

    ensure_scratch(dev_ws_bytes);
    h_workspace_.resize(host_ws_bytes);

    EIGEN_CUSOLVER_CHECK(cusolverDnXpotrf(
        handle_, params_.p, uplo, static_cast<int64_t>(n_), dtype, d_factor_.ptr, lda_, dtype, scratch_workspace(),
        dev_ws_bytes, host_ws_bytes > 0 ? h_workspace_.data() : nullptr, host_ws_bytes, scratch_info()));

    // Enqueue async download of info word — sync deferred to info() or solve().
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&info_word(), scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
  }
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_LLT_H
