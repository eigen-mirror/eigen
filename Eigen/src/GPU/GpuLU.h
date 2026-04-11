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
//   GpuLU<double> lu(A);              // upload A, getrf, LU+ipiv on device
//   if (lu.info() != Success) { ... }
//   MatrixXd x = lu.solve(b);         // getrs NoTrans, only b transferred
//   MatrixXd xt = lu.solve(b, GpuLU<double>::Transpose);   // A^T x = b

#ifndef EIGEN_GPU_LU_H
#define EIGEN_GPU_LU_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuSolverSupport.h"
#include <vector>

namespace Eigen {

/** \ingroup GPU_Module
 * \class GpuLU
 * \brief GPU LU decomposition with partial pivoting via cuSOLVER
 *
 * \tparam Scalar_  Element type: float, double, complex<float>, complex<double>
 *
 * Decomposes a square matrix A = P L U on the GPU and retains the factored
 * matrix and pivot array in device memory. Solves A*X=B, A^T*X=B, or
 * A^H*X=B by passing the appropriate TransposeMode.
 *
 * Each GpuLU object owns a dedicated CUDA stream and cuSOLVER handle.
 */
template <typename Scalar_>
class GpuLU {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  /** Controls which system is solved in solve(). */
  enum TransposeMode {
    NoTranspose,        ///< Solve A   * X = B
    Transpose,          ///< Solve A^T * X = B
    ConjugateTranspose  ///< Solve A^H * X = B (same as Transpose for real types)
  };

  // ---- Construction / destruction ------------------------------------------

  GpuLU() { init_context(); }

  template <typename InputType>
  explicit GpuLU(const EigenBase<InputType>& A) {
    init_context();
    compute(A);
  }

  ~GpuLU() {
    if (handle_) (void)cusolverDnDestroy(handle_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  GpuLU(const GpuLU&) = delete;
  GpuLU& operator=(const GpuLU&) = delete;

  GpuLU(GpuLU&& o) noexcept
      : stream_(o.stream_),
        handle_(o.handle_),
        params_(std::move(o.params_)),
        d_lu_(std::move(o.d_lu_)),
        lu_alloc_size_(o.lu_alloc_size_),
        d_ipiv_(std::move(o.d_ipiv_)),
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
    o.lu_alloc_size_ = 0;
    o.scratch_size_ = 0;
    o.n_ = 0;
    o.lda_ = 0;
    o.info_ = InvalidInput;
    o.info_synced_ = true;
  }

  GpuLU& operator=(GpuLU&& o) noexcept {
    if (this != &o) {
      if (handle_) (void)cusolverDnDestroy(handle_);
      if (stream_) (void)cudaStreamDestroy(stream_);
      stream_ = o.stream_;
      handle_ = o.handle_;
      params_ = std::move(o.params_);
      d_lu_ = std::move(o.d_lu_);
      lu_alloc_size_ = o.lu_alloc_size_;
      d_ipiv_ = std::move(o.d_ipiv_);
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
      o.lu_alloc_size_ = 0;
      o.scratch_size_ = 0;
      o.n_ = 0;
      o.lda_ = 0;
      o.info_ = InvalidInput;
      o.info_synced_ = true;
    }
    return *this;
  }

  // ---- Factorization -------------------------------------------------------

  /** Compute the LU factorization of A (host matrix, must be square). */
  template <typename InputType>
  GpuLU& compute(const EigenBase<InputType>& A) {
    eigen_assert(A.rows() == A.cols() && "GpuLU requires a square matrix");
    if (!begin_compute(A.rows())) return *this;

    const PlainMatrix mat(A.derived());
    lda_ = static_cast<int64_t>(mat.outerStride());
    allocate_lu_storage();
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_lu_.ptr, mat.data(), matrixBytes(), cudaMemcpyHostToDevice, stream_));

    factorize();
    return *this;
  }

  /** Compute the LU factorization from a device-resident matrix (D2D copy). */
  GpuLU& compute(const DeviceMatrix<Scalar>& d_A) {
    eigen_assert(d_A.rows() == d_A.cols() && "GpuLU requires a square matrix");
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.outerStride());
    d_A.waitReady(stream_);
    allocate_lu_storage();
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_lu_.ptr, d_A.data(), matrixBytes(), cudaMemcpyDeviceToDevice, stream_));

    factorize();
    return *this;
  }

  /** Compute the LU factorization from a device matrix (move, no copy). */
  GpuLU& compute(DeviceMatrix<Scalar>&& d_A) {
    eigen_assert(d_A.rows() == d_A.cols() && "GpuLU requires a square matrix");
    if (!begin_compute(d_A.rows())) return *this;

    lda_ = static_cast<int64_t>(d_A.outerStride());
    d_A.waitReady(stream_);
    d_lu_ = internal::DeviceBuffer::adopt(static_cast<void*>(d_A.release()));

    factorize();
    return *this;
  }

  // ---- Solve ---------------------------------------------------------------

  /** Solve op(A) * X = B using the cached LU factorization (host → host).
   *
   * \param B    Right-hand side (n x nrhs host matrix).
   * \param mode NoTranspose (default), Transpose, or ConjugateTranspose.
   */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B, TransposeMode mode = NoTranspose) const {
    const_cast<GpuLU*>(this)->sync_info();
    eigen_assert(info_ == Success && "GpuLU::solve called on a failed or uninitialized factorization");
    eigen_assert(B.rows() == n_);

    const PlainMatrix rhs(B);
    const int64_t nrhs = static_cast<int64_t>(rhs.cols());
    const int64_t ldb = static_cast<int64_t>(rhs.outerStride());
    DeviceMatrix<Scalar> d_X = solve_impl(nrhs, ldb, mode, [&](Scalar* d_x_ptr) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_x_ptr, rhs.data(), matrixBytes(nrhs, ldb), cudaMemcpyHostToDevice, stream_));
    });

    PlainMatrix X(n_, B.cols());
    int solve_info = 0;
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(X.data(), d_X.data(), matrixBytes(nrhs, ldb), cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&solve_info, scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));

    eigen_assert(solve_info == 0 && "cusolverDnXgetrs reported an error");
    return X;
  }

  /** Solve op(A) * X = B with device-resident RHS. Fully async. */
  DeviceMatrix<Scalar> solve(const DeviceMatrix<Scalar>& d_B, TransposeMode mode = NoTranspose) const {
    eigen_assert(d_B.rows() == n_);
    d_B.waitReady(stream_);
    const int64_t nrhs = static_cast<int64_t>(d_B.cols());
    const int64_t ldb = static_cast<int64_t>(d_B.outerStride());
    return solve_impl(nrhs, ldb, mode, [&](Scalar* d_x_ptr) {
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_x_ptr, d_B.data(), matrixBytes(nrhs, ldb), cudaMemcpyDeviceToDevice, stream_));
    });
  }

  // ---- Accessors -----------------------------------------------------------

  /** Lazily synchronizes the stream on first call after compute(). */
  ComputationInfo info() const {
    const_cast<GpuLU*>(this)->sync_info();
    return info_;
  }
  Index rows() const { return n_; }
  Index cols() const { return n_; }
  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t handle_ = nullptr;
  internal::CusolverParams params_;   // cuSOLVER params (created once, reused)
  internal::DeviceBuffer d_lu_;       // LU factors on device (grows, never shrinks)
  size_t lu_alloc_size_ = 0;          // current d_lu_ allocation size
  internal::DeviceBuffer d_ipiv_;     // pivot indices (int64_t) on device
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

  size_t matrixBytes() const { return matrixBytes(static_cast<int64_t>(n_), lda_); }

  static size_t matrixBytes(int64_t cols, int64_t outer_stride) {
    return static_cast<size_t>(outer_stride) * static_cast<size_t>(cols) * sizeof(Scalar);
  }

  void allocate_lu_storage() {
    size_t needed = matrixBytes();
    if (needed > lu_alloc_size_) {
      d_lu_ = internal::DeviceBuffer(needed);
      lu_alloc_size_ = needed;
    }
  }

  // Ensure d_scratch_ is at least `workspace_bytes + sizeof(int)`.
  // Layout: [workspace (workspace_bytes) | info_word (sizeof(int))].
  // Ensure d_scratch_ can hold workspace_bytes + an aligned info word.
  // Grows but never shrinks. Syncs the stream before reallocating to
  // avoid freeing memory that async kernels may still be using.
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

  template <typename CopyRhs>
  DeviceMatrix<Scalar> solve_impl(int64_t nrhs, int64_t ldb, TransposeMode mode, CopyRhs&& copy_rhs) const {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    const cublasOperation_t trans = to_cublas_op(mode);

    internal::DeviceBuffer d_x(matrixBytes(nrhs, ldb));
    Scalar* d_x_ptr = static_cast<Scalar*>(d_x.ptr);
    copy_rhs(d_x_ptr);

    EIGEN_CUSOLVER_CHECK(cusolverDnXgetrs(handle_, params_.p, trans, static_cast<int64_t>(n_), nrhs, dtype, d_lu_.ptr,
                                          lda_, static_cast<const int64_t*>(d_ipiv_.ptr), dtype, d_x_ptr, ldb,
                                          scratch_info()));

    DeviceMatrix<Scalar> result(static_cast<Scalar*>(d_x.ptr), n_, static_cast<Index>(nrhs), static_cast<Index>(ldb));
    d_x.ptr = nullptr;  // transfer ownership to result
    result.recordReady(stream_);
    return result;
  }

  void init_context() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    ensure_scratch(0);  // allocate at least the info word
  }

  void sync_info() {
    if (!info_synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      info_ = (info_word() == 0) ? Success : NumericalIssue;
      info_synced_ = true;
    }
  }

  // Run cusolverDnXgetrf on d_lu_ (already on device). Allocates d_ipiv_.
  // Enqueues factorization + async info download. Does NOT sync.
  // Workspaces are stored as members to ensure they outlive the async kernels.
  void factorize() {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    const size_t ipiv_bytes = static_cast<size_t>(n_) * sizeof(int64_t);

    info_synced_ = false;
    info_ = InvalidInput;

    d_ipiv_ = internal::DeviceBuffer(ipiv_bytes);

    size_t dev_ws_bytes = 0, host_ws_bytes = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(handle_, params_.p, static_cast<int64_t>(n_),
                                                     static_cast<int64_t>(n_), dtype, d_lu_.ptr, lda_, dtype,
                                                     &dev_ws_bytes, &host_ws_bytes));

    ensure_scratch(dev_ws_bytes);
    h_workspace_.resize(host_ws_bytes);

    EIGEN_CUSOLVER_CHECK(
        cusolverDnXgetrf(handle_, params_.p, static_cast<int64_t>(n_), static_cast<int64_t>(n_), dtype, d_lu_.ptr, lda_,
                         static_cast<int64_t*>(d_ipiv_.ptr), dtype, scratch_workspace(), dev_ws_bytes,
                         host_ws_bytes > 0 ? h_workspace_.data() : nullptr, host_ws_bytes, scratch_info()));

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&info_word(), scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
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

}  // namespace Eigen

#endif  // EIGEN_GPU_LU_H
