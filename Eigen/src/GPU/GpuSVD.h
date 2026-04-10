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
//   GpuSVD<double> svd(A, ComputeThinU | ComputeThinV);
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

#include "./CuSolverSupport.h"
#include "./CuBlasSupport.h"
#include <vector>

namespace Eigen {

template <typename Scalar_>
class GpuSVD {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;

  GpuSVD() { init_context(); }

  template <typename InputType>
  explicit GpuSVD(const EigenBase<InputType>& A, unsigned int options = ComputeThinU | ComputeThinV) {
    init_context();
    compute(A, options);
  }

  ~GpuSVD() {
    if (handle_) (void)cusolverDnDestroy(handle_);
    if (cublas_) (void)cublasDestroy(cublas_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  GpuSVD(const GpuSVD&) = delete;
  GpuSVD& operator=(const GpuSVD&) = delete;
  // Move constructors omitted for brevity — follow GpuQR pattern.

  // ---- Factorization -------------------------------------------------------

  template <typename InputType>
  GpuSVD& compute(const EigenBase<InputType>& A, unsigned int options = ComputeThinU | ComputeThinV) {
    options_ = options;
    m_ = A.rows();
    n_ = A.cols();
    info_ = InvalidInput;
    info_synced_ = false;

    if (m_ == 0 || n_ == 0) {
      info_ = Success;
      info_synced_ = true;
      return *this;
    }

    // cuSOLVER gesvd requires m >= n. For wide matrices, transpose internally.
    transposed_ = (m_ < n_);
    const PlainMatrix mat = transposed_ ? PlainMatrix(A.derived().adjoint()) : PlainMatrix(A.derived());
    if (transposed_) std::swap(m_, n_);

    lda_ = static_cast<int64_t>(mat.rows());
    const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);

    // Copy (possibly transposed) A to device (gesvd overwrites it).
    d_A_ = internal::DeviceBuffer(mat_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_A_.ptr, mat.data(), mat_bytes, cudaMemcpyHostToDevice, stream_));

    factorize();
    return *this;
  }

  GpuSVD& compute(const DeviceMatrix<Scalar>& d_A, unsigned int options = ComputeThinU | ComputeThinV) {
    options_ = options;
    m_ = d_A.rows();
    n_ = d_A.cols();
    info_ = InvalidInput;
    info_synced_ = false;

    if (m_ == 0 || n_ == 0) {
      info_ = Success;
      info_synced_ = true;
      return *this;
    }

    transposed_ = (m_ < n_);
    d_A.waitReady(stream_);

    if (transposed_) {
      // Transpose on device via cuBLAS geam: d_A_ = A^H.
      std::swap(m_, n_);
      lda_ = m_;
      const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);
      d_A_ = internal::DeviceBuffer(mat_bytes);
      Scalar alpha_one(1), beta_zero(0);
      // geam: C(m×n) = alpha * op(A) + beta * op(B). Use B = nullptr trick: beta=0.
      // A is the original d_A (n_orig × m_orig = n × m after swap), transposed → m × n.
      EIGEN_CUBLAS_CHECK(internal::cublasXgeam(
          cublas_, CUBLAS_OP_C, CUBLAS_OP_N, static_cast<int>(m_), static_cast<int>(n_), &alpha_one, d_A.data(),
          static_cast<int>(d_A.rows()), &beta_zero, static_cast<const Scalar*>(nullptr), static_cast<int>(m_),
          static_cast<Scalar*>(d_A_.ptr), static_cast<int>(m_)));
    } else {
      lda_ = static_cast<int64_t>(d_A.rows());
      const size_t mat_bytes = static_cast<size_t>(lda_) * static_cast<size_t>(n_) * sizeof(Scalar);
      d_A_ = internal::DeviceBuffer(mat_bytes);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_A_.ptr, d_A.data(), mat_bytes, cudaMemcpyDeviceToDevice, stream_));
    }

    factorize();
    return *this;
  }

  // ---- Accessors -----------------------------------------------------------

  ComputationInfo info() const {
    sync_info();
    return info_;
  }

  Index rows() const { return transposed_ ? n_ : m_; }
  Index cols() const { return transposed_ ? m_ : n_; }

  // TODO: Add device-side accessors (deviceU(), deviceVT(), deviceSingularValues())
  // returning DeviceMatrix views of the internal buffers, so users can chain
  // GPU operations without round-tripping through host memory.

  /** Singular values (always available). Downloads from device on each call. */
  RealVector singularValues() const {
    sync_info();
    eigen_assert(info_ == Success);
    const Index k = (std::min)(m_, n_);
    RealVector S(k);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpy(S.data(), d_S_.ptr, static_cast<size_t>(k) * sizeof(RealScalar), cudaMemcpyDeviceToHost));
    return S;
  }

  /** Left singular vectors U. Returns m_orig × k or m_orig × m_orig.
   * For transposed case (m_orig < n_orig), U comes from cuSOLVER's VT. */
  PlainMatrix matrixU() const {
    sync_info();
    eigen_assert(info_ == Success);
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
      // Transposed: U_orig = VT_stored^H. VT_stored is vtrows × n_ (= vtrows × m_orig).
      const Index vtrows = (options_ & ComputeFullU) ? m_orig : k;  // Note: FullU maps to FullV of A^H
      PlainMatrix VT_stored(vtrows, n_);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(VT_stored.data(), d_VT_.ptr,
                                          static_cast<size_t>(vtrows) * static_cast<size_t>(n_) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
      return VT_stored.adjoint();  // m_orig × vtrows
    }
  }

  /** Right singular vectors transposed V^T. Returns k × n_orig or n_orig × n_orig.
   * For transposed case, VT comes from cuSOLVER's U. */
  PlainMatrix matrixVT() const {
    sync_info();
    eigen_assert(info_ == Success);
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
      // Transposed: VT_orig = U_stored^H. U_stored is m_ × ucols (= n_orig × ucols).
      const Index ucols = (options_ & ComputeFullV) ? n_orig : k;  // FullV maps to FullU of A^H
      PlainMatrix U_stored(m_, ucols);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpy(U_stored.data(), d_U_.ptr,
                                          static_cast<size_t>(m_) * static_cast<size_t>(ucols) * sizeof(Scalar),
                                          cudaMemcpyDeviceToHost));
      return U_stored.adjoint();  // ucols × n_orig
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

  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t handle_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  internal::CusolverParams params_;
  internal::DeviceBuffer d_A_;        // working copy of A (overwritten by gesvd)
  internal::DeviceBuffer d_U_;        // left singular vectors
  internal::DeviceBuffer d_S_;        // singular values (RealScalar)
  internal::DeviceBuffer d_VT_;       // right singular vectors transposed
  internal::DeviceBuffer d_scratch_;  // workspace + info
  size_t scratch_size_ = 0;
  std::vector<char> h_workspace_;
  unsigned int options_ = 0;
  Index m_ = 0;
  Index n_ = 0;
  int64_t lda_ = 0;
  bool transposed_ = false;  // true if m < n (we compute SVD of A^T internally)
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
      const_cast<GpuSVD*>(this)->info_ = (info_word_ == 0) ? Success : NumericalIssue;
      const_cast<GpuSVD*>(this)->info_synced_ = true;
    }
  }

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

    info_synced_ = false;
    info_ = InvalidInput;

    // Allocate output buffers. When transposed, swap U/V roles for cuSOLVER.
    d_S_ = internal::DeviceBuffer(static_cast<size_t>(k) * sizeof(RealScalar));

    // Internal options: for transposed case, what user wants as U we compute as VT of A^H.
    const unsigned int int_opts = transposed_ ? swap_uv_options(options_) : options_;

    const Index ucols = (int_opts & ComputeFullU) ? m_ : ((int_opts & ComputeThinU) ? k : 0);
    const Index vtrows = (int_opts & ComputeFullV) ? n_ : ((int_opts & ComputeThinV) ? k : 0);
    const int64_t ldu = m_;
    const int64_t ldvt = vtrows > 0 ? vtrows : 1;

    if (ucols > 0) d_U_ = internal::DeviceBuffer(static_cast<size_t>(m_) * static_cast<size_t>(ucols) * sizeof(Scalar));
    if (vtrows > 0)
      d_VT_ = internal::DeviceBuffer(static_cast<size_t>(vtrows) * static_cast<size_t>(n_) * sizeof(Scalar));

    // computeType must match the matrix data type (dtype), not the singular value type (rtype).
    eigen_assert(m_ >= n_ && "Internal error: m_ < n_ should have been handled by transpose in compute()");
    size_t dev_ws = 0, host_ws = 0;
    EIGEN_CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
        handle_, params_.p, jobu(int_opts), jobvt(int_opts), static_cast<int64_t>(m_), static_cast<int64_t>(n_), dtype,
        d_A_.ptr, lda_, rtype, d_S_.ptr, dtype, ucols > 0 ? d_U_.ptr : nullptr, ldu, dtype,
        vtrows > 0 ? d_VT_.ptr : nullptr, ldvt, dtype, &dev_ws, &host_ws));

    ensure_scratch(dev_ws);
    h_workspace_.resize(host_ws);

    // Compute SVD.
    EIGEN_CUSOLVER_CHECK(cusolverDnXgesvd(handle_, params_.p, jobu(int_opts), jobvt(int_opts), static_cast<int64_t>(m_),
                                          static_cast<int64_t>(n_), dtype, d_A_.ptr, lda_, rtype, d_S_.ptr, dtype,
                                          ucols > 0 ? d_U_.ptr : nullptr, ldu, dtype, vtrows > 0 ? d_VT_.ptr : nullptr,
                                          ldvt, dtype, scratch_workspace(), dev_ws,
                                          host_ws > 0 ? h_workspace_.data() : nullptr, host_ws, scratch_info()));

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&info_word_, scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
  }

  // Internal solve: X = V * diag(D) * U^H * B, using top `trunc` triplets.
  // D_ii = 1/S_i (if lambda==0) or S_i/(S_i^2+lambda^2).
  //
  // For non-transposed: stored U, VT. X = VT^H * D * U^H * B.
  // For transposed (SVD of A^H): stored U', VT'. X = U' * D * VT' * B.
  template <typename Rhs>
  PlainMatrix solve_impl(const MatrixBase<Rhs>& B, Index trunc, RealScalar lambda) const {
    sync_info();
    eigen_assert(info_ == Success && "GpuSVD::solve called on a failed or uninitialized decomposition");
    eigen_assert((options_ & (ComputeThinU | ComputeFullU)) && "solve requires U");
    eigen_assert((options_ & (ComputeThinV | ComputeFullV)) && "solve requires V");

    const Index m_orig = transposed_ ? n_ : m_;
    const Index n_orig = transposed_ ? m_ : n_;
    eigen_assert(B.rows() == m_orig);

    const Index k = (std::min)(m_, n_);  // = min(m_orig, n_orig)
    const Index kk = (std::min)(trunc, k);
    const Index nrhs = B.cols();

    // Download S to host to build the diagonal scaling.
    RealVector S(k);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpy(S.data(), d_S_.ptr, static_cast<size_t>(k) * sizeof(RealScalar), cudaMemcpyDeviceToHost));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));

    // Upload B (m_orig × nrhs).
    const PlainMatrix rhs(B);
    internal::DeviceBuffer d_B(static_cast<size_t>(m_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_B.ptr, rhs.data(),
                                             static_cast<size_t>(m_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                             cudaMemcpyHostToDevice, stream_));

    // Step 1: tmp = U_orig^H * B  (kk × nrhs).
    // Non-transposed: U_stored is m_×ucols, U_orig = U_stored. Use U_stored^H * B.
    // Transposed: U_orig = VT_stored^H, so U_orig^H = VT_stored. Use VT_stored * B (no transpose!).
    internal::DeviceBuffer d_tmp(static_cast<size_t>(kk) * static_cast<size_t>(nrhs) * sizeof(Scalar));
    {
      Scalar alpha_one(1), beta_zero(0);
      constexpr cudaDataType_t dtype = internal::cuda_data_type<Scalar>::value;
      constexpr cublasComputeType_t compute = internal::cuda_compute_type<Scalar>::value;

      if (!transposed_) {
        // U_stored^H * B: (m_×kk)^H × (m_×nrhs) → kk×nrhs.
        EIGEN_CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_C, CUBLAS_OP_N, static_cast<int>(kk), static_cast<int>(nrhs),
                                        static_cast<int>(m_), &alpha_one, d_U_.ptr, dtype, static_cast<int>(m_),
                                        d_B.ptr, dtype, static_cast<int>(m_orig), &beta_zero, d_tmp.ptr, dtype,
                                        static_cast<int>(kk), compute, internal::cuda_gemm_algo()));
      } else {
        // VT_stored * B: VT_stored is vtrows×n_ = kk×m_orig (thin), NoTrans.
        // vtrows×m_orig times m_orig×nrhs → vtrows×nrhs. Use first kk rows.
        const Index vtrows_stored = (swap_uv_options(options_) & ComputeFullV) ? n_ : k;
        EIGEN_CUBLAS_CHECK(cublasGemmEx(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(kk), static_cast<int>(nrhs), static_cast<int>(m_orig),
            &alpha_one, d_VT_.ptr, dtype, static_cast<int>(vtrows_stored), d_B.ptr, dtype, static_cast<int>(m_orig),
            &beta_zero, d_tmp.ptr, dtype, static_cast<int>(kk), compute, internal::cuda_gemm_algo()));
      }
    }

    // Step 2: Scale row i of tmp by D_ii.
    // Download tmp to host, scale, re-upload. (Simple and correct; a device kernel would be faster.)
    {
      PlainMatrix tmp(kk, nrhs);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(tmp.data(), d_tmp.ptr,
                                               static_cast<size_t>(kk) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                               cudaMemcpyDeviceToHost, stream_));
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));

      for (Index i = 0; i < kk; ++i) {
        RealScalar si = S(i);
        RealScalar di = (lambda == RealScalar(0)) ? (si > 0 ? RealScalar(1) / si : RealScalar(0))
                                                  : si / (si * si + lambda * lambda);
        tmp.row(i) *= Scalar(di);
      }

      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_tmp.ptr, tmp.data(),
                                               static_cast<size_t>(kk) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                               cudaMemcpyHostToDevice, stream_));
    }

    // Step 3: X = V_orig * tmp  (n_orig × nrhs).
    // Non-transposed: V_orig = VT_stored^H. VT_stored[:kk,:]^H * tmp → n_orig × nrhs.
    // Transposed: V_orig = U_stored[:,:kk]. U_stored * tmp → n_orig × nrhs (NoTrans).
    PlainMatrix X(n_orig, nrhs);
    {
      internal::DeviceBuffer d_X(static_cast<size_t>(n_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar));
      Scalar alpha_one(1), beta_zero(0);
      constexpr cudaDataType_t dtype = internal::cuda_data_type<Scalar>::value;
      constexpr cublasComputeType_t compute = internal::cuda_compute_type<Scalar>::value;

      if (!transposed_) {
        const Index vtrows = (options_ & ComputeFullV) ? n_ : k;
        EIGEN_CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_C, CUBLAS_OP_N, static_cast<int>(n_orig),
                                        static_cast<int>(nrhs), static_cast<int>(kk), &alpha_one, d_VT_.ptr, dtype,
                                        static_cast<int>(vtrows), d_tmp.ptr, dtype, static_cast<int>(kk), &beta_zero,
                                        d_X.ptr, dtype, static_cast<int>(n_orig), compute, internal::cuda_gemm_algo()));
      } else {
        // U_stored is m_×ucols. V_orig = U_stored[:,:kk]. NoTrans × tmp.
        EIGEN_CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n_orig),
                                        static_cast<int>(nrhs), static_cast<int>(kk), &alpha_one, d_U_.ptr, dtype,
                                        static_cast<int>(m_), d_tmp.ptr, dtype, static_cast<int>(kk), &beta_zero,
                                        d_X.ptr, dtype, static_cast<int>(n_orig), compute, internal::cuda_gemm_algo()));
      }

      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(X.data(), d_X.ptr,
                                               static_cast<size_t>(n_orig) * static_cast<size_t>(nrhs) * sizeof(Scalar),
                                               cudaMemcpyDeviceToHost, stream_));
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
    }

    return X;
  }
};

}  // namespace Eigen

#endif  // EIGEN_GPU_SVD_H
