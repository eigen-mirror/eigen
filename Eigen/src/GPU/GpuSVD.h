// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// GPU SVD decomposition using cuSOLVER (divide-and-conquer).
//
// Wraps cusolverDnXgesvd. Stores U, S, VT on device. Solve uses
// cuBLAS GEMM: X = VT^H * diag(D) * U^H * B.
//
// cuSOLVER returns VT (not V). We store and expose VT directly.
//
// Usage:
//   SVD<double> svd(A, ComputeThinU | ComputeThinV);
//   VectorXd S = svd.singularValues();
//   MatrixXd U = svd.matrixU();       // m×k or m×m
//   MatrixXd VT = svd.matrixVT();      // k×n or n×n (this is V^T)
//   MatrixXd X = svd.solve(B);        // pseudoinverse
//   MatrixXd X = svd.solve(B, k);     // truncated (top k triplets)
//   MatrixXd X = svd.solve(B, 0.1);   // Tikhonov regularized

#ifndef EIGEN_GPU_SVD_H
#define EIGEN_GPU_SVD_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSolverContext.h"

namespace Eigen {
namespace gpu {

template <typename Scalar_>
class SVD {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RealVector = Eigen::Matrix<RealScalar, Dynamic, 1>;

  SVD() = default;

  template <typename InputType>
  explicit SVD(const EigenBase<InputType>& A, unsigned int options = ComputeThinU | ComputeThinV) {
    compute(A, options);
  }

  ~SVD() = default;

  SVD(const SVD&) = delete;
  SVD& operator=(const SVD&) = delete;

  SVD(SVD&& o) noexcept
      : ctx_(std::move(o.ctx_)),
        d_A_(std::move(o.d_A_)),
        d_U_(std::move(o.d_U_)),
        d_S_(std::move(o.d_S_)),
        d_VT_(std::move(o.d_VT_)),
        options_(o.options_),
        m_(o.m_),
        n_(o.n_),
        lda_(o.lda_),
        transposed_(o.transposed_) {
    o.options_ = 0;
    o.m_ = 0;
    o.n_ = 0;
    o.lda_ = 0;
    o.transposed_ = false;
  }

  SVD& operator=(SVD&& o) noexcept {
    if (this != &o) {
      ctx_ = std::move(o.ctx_);
      d_A_ = std::move(o.d_A_);
      d_U_ = std::move(o.d_U_);
      d_S_ = std::move(o.d_S_);
      d_VT_ = std::move(o.d_VT_);
      options_ = o.options_;
      m_ = o.m_;
      n_ = o.n_;
      lda_ = o.lda_;
      transposed_ = o.transposed_;
      o.options_ = 0;
      o.m_ = 0;
      o.n_ = 0;
      o.lda_ = 0;
      o.transposed_ = false;
    }
    return *this;
  }

  // ---- Factorization -------------------------------------------------------

  template <typename InputType>
  SVD& compute(const EigenBase<InputType>& A, unsigned int options = ComputeThinU | ComputeThinV) {
    options_ = options;
    m_ = A.rows();
    n_ = A.cols();

    if (m_ == 0 || n_ == 0) {
      ctx_.info_ = Success;
      ctx_.info_synced_ = true;
      return *this;
    }

    // cuSOLVER gesvd requires m >= n. For wide matrices, transpose internally.
    transposed_ = (m_ < n_);
    const PlainMatrix mat = transposed_ ? PlainMatrix(A.derived().adjoint()) : PlainMatrix(A.derived());
    if (transposed_) std::swap(m_, n_);

    lda_ = static_cast<int64_t>(mat.rows());
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);

    d_A_ = internal::DeviceBuffer(mat_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_A_.ptr, mat.data(), mat_bytes, cudaMemcpyHostToDevice, ctx_.stream_));

    factorize();
    return *this;
  }

  SVD& compute(const DeviceMatrix<Scalar>& d_A, unsigned int options = ComputeThinU | ComputeThinV) {
    options_ = options;
    m_ = d_A.rows();
    n_ = d_A.cols();

    if (m_ == 0 || n_ == 0) {
      ctx_.info_ = Success;
      ctx_.info_synced_ = true;
      return *this;
    }

    transposed_ = (m_ < n_);
    d_A.waitReady(ctx_.stream_);

    if (transposed_) {
      // Transpose on device via cuBLAS geam: d_A_ = A^H.
      std::swap(m_, n_);
      lda_ = m_;
      const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);
      d_A_ = internal::DeviceBuffer(mat_bytes);
      // geam: C(m×n) = alpha * op(A) + beta * op(B). beta=0, B=nullptr.
      Scalar alpha_one(1), beta_zero(0);
      EIGEN_CUBLAS_CHECK(internal::cublasXgeam(
          ctx_.cublas_, CUBLAS_OP_C, CUBLAS_OP_N, static_cast<int>(m_), static_cast<int>(n_), &alpha_one, d_A.data(),
          static_cast<int>(d_A.rows()), &beta_zero, static_cast<const Scalar*>(nullptr), static_cast<int>(m_),
          static_cast<Scalar*>(d_A_.ptr), static_cast<int>(m_)));
    } else {
      lda_ = static_cast<int64_t>(d_A.rows());
      const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);
      d_A_ = internal::DeviceBuffer(mat_bytes);
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_A_.ptr, d_A.data(), mat_bytes, cudaMemcpyDeviceToDevice, ctx_.stream_));
    }

    factorize();
    return *this;
  }

  // ---- Accessors -----------------------------------------------------------

  ComputationInfo info() const { return ctx_.info(); }

  Index rows() const { return transposed_ ? n_ : m_; }
  Index cols() const { return transposed_ ? m_ : n_; }

  // TODO: Add device-side accessors (deviceU(), deviceVT(), deviceSingularValues())
  // returning DeviceMatrix views of the internal buffers, so users can chain
  // GPU operations without round-tripping through host memory.

  /** Singular values (always available). Downloads from device on each call. */
  RealVector singularValues() const {
    ctx_.sync_info();
    eigen_assert(ctx_.info_ == Success);
    const Index k = (std::min)(m_, n_);
    RealVector S(k);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpy(S.data(), d_S_.ptr, static_cast<size_t>(k) * sizeof(RealScalar), cudaMemcpyDeviceToHost));
    return S;
  }

  /** Left singular vectors U. */
  PlainMatrix matrixU() const {
    ctx_.sync_info();
    eigen_assert(ctx_.info_ == Success);
    eigen_assert((options_ & (ComputeThinU | ComputeFullU)) && "matrixU() requires ComputeThinU or ComputeFullU");
    const Index m_orig = transposed_ ? n_ : m_;
    const Index n_orig = transposed_ ? m_ : n_;
    const Index k = (std::min)(m_orig, n_orig);
    if (!transposed_) {
      const Index ucols = (options_ & ComputeFullU) ? m_ : k;
      PlainMatrix U(m_, ucols);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(U.data(), d_U_.ptr,
                                          static_cast<size_t>(m_) * static_cast<size_t>(ucols) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
      return U;
    } else {
      const Index vtrows = (options_ & ComputeFullU) ? m_orig : k;
      PlainMatrix VT_stored(vtrows, n_);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(VT_stored.data(), d_VT_.ptr,
                                          static_cast<size_t>(vtrows) * static_cast<size_t>(n_) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
      return VT_stored.adjoint();
    }
  }

  /** Right singular vectors transposed V^T. */
  PlainMatrix matrixVT() const {
    ctx_.sync_info();
    eigen_assert(ctx_.info_ == Success);
    eigen_assert((options_ & (ComputeThinV | ComputeFullV)) && "matrixVT() requires ComputeThinV or ComputeFullV");
    const Index m_orig = transposed_ ? n_ : m_;
    const Index n_orig = transposed_ ? m_ : n_;
    const Index k = (std::min)(m_orig, n_orig);
    if (!transposed_) {
      const Index vtrows = (options_ & ComputeFullV) ? n_ : k;
      PlainMatrix VT(vtrows, n_);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(VT.data(), d_VT_.ptr,
                                          static_cast<size_t>(vtrows) * static_cast<size_t>(n_) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
      return VT;
    } else {
      const Index ucols = (options_ & ComputeFullV) ? n_orig : k;
      PlainMatrix U_stored(m_, ucols);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(U_stored.data(), d_U_.ptr,
                                          static_cast<size_t>(m_) * static_cast<size_t>(ucols) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
      return U_stored.adjoint();
    }
  }

  /** Number of singular values above threshold. */
  Index rank(RealScalar threshold = RealScalar(-1)) const {
    RealVector S = singularValues();
    if (S.size() == 0) return 0;
    if (threshold < 0) {
      threshold = (std::max)(m_, n_) * S(0) * NumTraits<RealScalar>::epsilon();
    }
    Index r = 0;
    for (Index i = 0; i < S.size(); ++i) {
      if (S(i) > threshold) ++r;
    }
    return r;
  }

  // ---- Solve ---------------------------------------------------------------

  /** Pseudoinverse solve: X = V * diag(1/S) * U^H * B. */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B) const {
    return solve_impl(B, (std::min)(m_, n_), RealScalar(0));
  }

  /** Truncated solve: use only top trunc singular triplets. */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B, Index trunc) const {
    eigen_assert(trunc > 0 && trunc <= (std::min)(m_, n_));
    return solve_impl(B, trunc, RealScalar(0));
  }

  /** Tikhonov-regularized solve: D_ii = S_i / (S_i^2 + lambda^2). */
  template <typename Rhs>
  PlainMatrix solve(const MatrixBase<Rhs>& B, RealScalar lambda) const {
    eigen_assert(lambda > 0);
    return solve_impl(B, (std::min)(m_, n_), lambda);
  }

  cudaStream_t stream() const { return ctx_.stream_; }

 private:
  mutable internal::GpuSolverContext ctx_;
  internal::DeviceBuffer d_A_;
  internal::DeviceBuffer d_U_;
  internal::DeviceBuffer d_S_;
  internal::DeviceBuffer d_VT_;
  unsigned int options_ = 0;
  Index m_ = 0;
  Index n_ = 0;
  int64_t lda_ = 0;
  bool transposed_ = false;

  // Swap U↔V flags for the transposed case.
  static unsigned int swap_uv_options(unsigned int opts) {
    unsigned int result = 0;
    if (opts & ComputeThinU) result |= ComputeThinV;
    if (opts & ComputeFullU) result |= ComputeFullV;
    if (opts & ComputeThinV) result |= ComputeThinU;
    if (opts & ComputeFullV) result |= ComputeFullU;
    return result;
  }

  static signed char jobu(unsigned int opts) {
    if (opts & ComputeFullU) return 'A';
    if (opts & ComputeThinU) return 'S';
    return 'N';
  }

  static signed char jobvt(unsigned int opts) {
    if (opts & ComputeFullV) return 'A';
    if (opts & ComputeThinV) return 'S';
    return 'N';
  }

  void factorize() {
    constexpr cudaDataType_t dtype = internal::cusolver_data_type<Scalar>::value;
    constexpr cudaDataType_t rtype = internal::cuda_data_type<RealScalar>::value;
    const Index k = (std::min)(m_, n_);

    ctx_.mark_pending();

    d_S_ = internal::DeviceBuffer(static_cast<size_t>(k) * sizeof(RealScalar));

    const unsigned int int_opts = transposed_ ? swap_uv_options(options_) : options_;

    const Index ucols = (int_opts & ComputeFullU) ? m_ : ((int_opts & ComputeThinU) ? k : 0);
    const Index vtrows = (int_opts & ComputeFullV) ? n_ : ((int_opts & ComputeThinV) ? k : 0);
    const int64_t ldu = m_;
    const int64_t ldvt = vtrows > 0 ? vtrows : 1;

    if (ucols > 0) d_U_ = internal::DeviceBuffer(static_cast<size_t>(m_) * static_cast<size_t>(ucols) * sizeof(Scalar));
    if (vtrows > 0)
      d_VT_ = internal::DeviceBuffer(static_cast<size_t>(vtrows) * static_cast<size_t>(n_) * sizeof(Scalar));

    eigen_assert(m_ >= n_ && "Internal error: m_ < n_ should have been handled by transpose in compute()");
    size_t dev_ws = 0, host_ws = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
        ctx_.cusolver_, ctx_.params_.p, jobu(int_opts), jobvt(int_opts), static_cast<int64_t>(m_),
        static_cast<int64_t>(n_), dtype, d_A_.ptr, lda_, rtype, d_S_.ptr, dtype, ucols > 0 ? d_U_.ptr : nullptr, ldu,
        dtype, vtrows > 0 ? d_VT_.ptr : nullptr, ldvt, dtype, &dev_ws, &host_ws));

    ctx_.ensure_scratch(dev_ws);
    ctx_.h_workspace_.resize(host_ws);

    EIGEN_CUSOLVER_CHECK(cusolverDnXgesvd(
        ctx_.cusolver_, ctx_.params_.p, jobu(int_opts), jobvt(int_opts), static_cast<int64_t>(m_),
        static_cast<int64_t>(n_), dtype, d_A_.ptr, lda_, rtype, d_S_.ptr, dtype, ucols > 0 ? d_U_.ptr : nullptr, ldu,
        dtype, vtrows > 0 ? d_VT_.ptr : nullptr, ldvt, dtype, ctx_.scratch_workspace(), dev_ws,
        host_ws > 0 ? ctx_.h_workspace_.data() : nullptr, host_ws, ctx_.scratch_info()));

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&ctx_.info_word_, ctx_.scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, ctx_.stream_));
  }

  template <typename Rhs>
  PlainMatrix solve_impl(const MatrixBase<Rhs>& B, Index trunc, RealScalar lambda) const {
    ctx_.sync_info();
    eigen_assert(ctx_.info_ == Success && "SVD::solve called on a failed or uninitialized decomposition");
    eigen_assert((options_ & (ComputeThinU | ComputeFullU)) && "solve requires U");
    eigen_assert((options_ & (ComputeThinV | ComputeFullV)) && "solve requires V");

    const Index m_orig = transposed_ ? n_ : m_;
    const Index n_orig = transposed_ ? m_ : n_;
    eigen_assert(B.rows() == m_orig);

    const Index k = (std::min)(m_, n_);
    const Index kk = (std::min)(trunc, k);
    const Index nrhs = B.cols();

    // Download S to host to build the diagonal scaling.
    RealVector S(k);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpy(S.data(), d_S_.ptr, static_cast<size_t>(k) * sizeof(RealScalar), cudaMemcpyDeviceToHost));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx_.stream_));

    // Upload B (m_orig × nrhs).
    const PlainMatrix rhs(B);
    internal::DeviceBuffer d_B(static_cast<size_t>(m_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_B.ptr, rhs.data(),
                                             static_cast<size_t>(m_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                             cudaMemcpyHostToDevice, ctx_.stream_));

    // Typed pointers for device buffers.
    auto* U_dev = static_cast<const Scalar*>(d_U_.ptr);
    auto* VT_dev = static_cast<const Scalar*>(d_VT_.ptr);
    auto* B_dev = static_cast<const Scalar*>(d_B.ptr);

    // Step 1: tmp = U_orig^H * B  (kk × nrhs).
    internal::DeviceBuffer d_tmp(static_cast<size_t>(kk) * static_cast<size_t>(nrhs) * sizeof(Scalar));
    auto* tmp_dev = static_cast<Scalar*>(d_tmp.ptr);
    {
      Scalar scalars[2] = {Scalar(1), Scalar(0)};

      if (!transposed_) {
        EIGEN_CUBLAS_CHECK(internal::cublasXgemm(ctx_.cublas_, CUBLAS_OP_C, CUBLAS_OP_N, static_cast<int>(kk),
                                                 static_cast<int>(nrhs), static_cast<int>(m_), &scalars[0], U_dev,
                                                 static_cast<int>(m_), B_dev, static_cast<int>(m_orig), &scalars[1],
                                                 tmp_dev, static_cast<int>(kk)));
      } else {
        const Index vtrows_stored = (swap_uv_options(options_) & ComputeFullV) ? n_ : k;
        EIGEN_CUBLAS_CHECK(internal::cublasXgemm(ctx_.cublas_, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(kk),
                                                 static_cast<int>(nrhs), static_cast<int>(m_orig), &scalars[0], VT_dev,
                                                 static_cast<int>(vtrows_stored), B_dev, static_cast<int>(m_orig),
                                                 &scalars[1], tmp_dev, static_cast<int>(kk)));
      }
    }

    // Step 2: Scale row i of tmp by D_ii (host round-trip).
    {
      PlainMatrix tmp(kk, nrhs);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(tmp.data(), d_tmp.ptr,
                                               static_cast<size_t>(kk) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                               cudaMemcpyDeviceToHost, ctx_.stream_));
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx_.stream_));

      for (Index i = 0; i < kk; ++i) {
        RealScalar si = S(i);
        RealScalar di = (lambda == RealScalar(0)) ? (si > 0 ? RealScalar(1) / si : RealScalar(0))
                                                  : si / (si * si + lambda * lambda);
        tmp.row(i) *= Scalar(di);
      }

      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_tmp.ptr, tmp.data(),
                                               static_cast<size_t>(kk) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                               cudaMemcpyHostToDevice, ctx_.stream_));
    }

    // Step 3: X = V_orig * tmp  (n_orig × nrhs).
    PlainMatrix X(n_orig, nrhs);
    {
      internal::DeviceBuffer d_X(static_cast<size_t>(n_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar));
      auto* X_dev = static_cast<Scalar*>(d_X.ptr);
      Scalar scalars[2] = {Scalar(1), Scalar(0)};

      if (!transposed_) {
        const Index vtrows = (options_ & ComputeFullV) ? n_ : k;
        EIGEN_CUBLAS_CHECK(internal::cublasXgemm(ctx_.cublas_, CUBLAS_OP_C, CUBLAS_OP_N, static_cast<int>(n_orig),
                                                 static_cast<int>(nrhs), static_cast<int>(kk), &scalars[0], VT_dev,
                                                 static_cast<int>(vtrows), tmp_dev, static_cast<int>(kk), &scalars[1],
                                                 X_dev, static_cast<int>(n_orig)));
      } else {
        EIGEN_CUBLAS_CHECK(internal::cublasXgemm(ctx_.cublas_, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n_orig),
                                                 static_cast<int>(nrhs), static_cast<int>(kk), &scalars[0], U_dev,
                                                 static_cast<int>(m_), tmp_dev, static_cast<int>(kk), &scalars[1],
                                                 X_dev, static_cast<int>(n_orig)));
      }

      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(X.data(), d_X.ptr,
                                               static_cast<size_t>(n_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                               cudaMemcpyDeviceToHost, ctx_.stream_));
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(ctx_.stream_));
    }

    return X;
  }
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SVD_H
