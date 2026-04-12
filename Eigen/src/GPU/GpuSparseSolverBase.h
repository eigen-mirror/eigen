// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Common base for GPU sparse direct solvers (LLT, LDLT, LU) via cuDSS.
//
// All three solver types share the same three-phase workflow
// (analyzePattern → factorize → solve) and differ only in the
// cudssMatrixType_t and cudssMatrixViewType_t passed to cuDSS.
// This CRTP base implements the entire workflow; derived classes
// provide the matrix type/view via static constexpr members.

#ifndef EIGEN_GPU_SPARSE_SOLVER_BASE_H
#define EIGEN_GPU_SPARSE_SOLVER_BASE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuDssSupport.h"

namespace Eigen {
namespace gpu {
namespace internal {

/** CRTP base for GPU sparse direct solvers.
 *
 * \tparam Scalar_  Element type (passed explicitly to avoid incomplete-type issues with CRTP).
 * \tparam Derived  The concrete solver class (SparseLLT, SparseLDLT, SparseLU).
 *                  Must provide:
 *                  - `static constexpr cudssMatrixType_t cudss_matrix_type()`
 *                  - `static constexpr cudssMatrixViewType_t cudss_matrix_view()`
 */
template <typename Scalar_, typename Derived>
class SparseSolverBase {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using CsrMat = SparseMatrix<Scalar, RowMajor, StorageIndex>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  SparseSolverBase() { init_context(); }

  ~SparseSolverBase() {
    destroy_cudss_objects();
    if (handle_) (void)cudssDestroy(handle_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  SparseSolverBase(const SparseSolverBase&) = delete;
  SparseSolverBase& operator=(const SparseSolverBase&) = delete;

  // ---- Configuration --------------------------------------------------------

  /** Set the fill-reducing ordering algorithm. Must be called before compute/analyzePattern. */
  void setOrdering(GpuSparseOrdering ordering) { ordering_ = ordering; }

  // ---- Factorization --------------------------------------------------------

  /** Symbolic analysis + numeric factorization. */
  template <typename InputType>
  Derived& compute(const SparseMatrixBase<InputType>& A) {
    analyzePattern(A);
    if (info_ == Success) {
      factorize(A);
    }
    return derived();
  }

  /** Symbolic analysis only. Uploads sparsity structure to device.
   * This phase is synchronous (blocks until complete). */
  template <typename InputType>
  Derived& analyzePattern(const SparseMatrixBase<InputType>& A) {
    const SpMat csc(A.derived());
    eigen_assert(csc.rows() == csc.cols() && "GpuSparseSolver requires a square matrix");
    eigen_assert(csc.isCompressed() && "GpuSparseSolver requires a compressed sparse matrix");

    n_ = csc.rows();
    info_ = InvalidInput;
    analysis_done_ = false;

    if (n_ == 0) {
      nnz_ = 0;
      info_ = Success;
      analysis_done_ = true;
      return derived();
    }

    // For symmetric solvers, ColMajor CSC can be reinterpreted as CSR with
    // swapped triangle view (zero copy). For general solvers, we must convert
    // to actual RowMajor CSR so cuDSS sees the correct matrix, not A^T.
    if (Derived::needs_csr_conversion()) {
      const CsrMat csr(csc);
      nnz_ = csr.nonZeros();
      upload_csr(csr);
    } else {
      nnz_ = csc.nonZeros();
      upload_csr_from_csc(csc);
    }
    create_cudss_matrix();
    apply_ordering_config();

    if (data_) EIGEN_CUDSS_CHECK(cudssDataDestroy(handle_, data_));
    EIGEN_CUDSS_CHECK(cudssDataCreate(handle_, &data_));

    create_placeholder_dense();

    EIGEN_CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, data_, d_A_cudss_, d_x_cudss_, d_b_cudss_));

    analysis_done_ = true;
    info_ = Success;
    return derived();
  }

  /** Numeric factorization using the symbolic analysis from analyzePattern.
   *
   * \warning The sparsity pattern (outerIndexPtr, innerIndexPtr) must be
   * identical to the one passed to analyzePattern(). Only the numerical
   * values may change. Passing a different pattern is undefined behavior.
   * This matches the contract of CHOLMOD, UMFPACK, and cuDSS's own API.
   *
   * This phase is asynchronous — info() lazily synchronizes. */
  template <typename InputType>
  Derived& factorize(const SparseMatrixBase<InputType>& A) {
    eigen_assert(analysis_done_ && "factorize() requires analyzePattern() first");

    if (n_ == 0) {
      info_ = Success;
      return derived();
    }

    // Convert to the same format used in analyzePattern.
    // Both temporaries must outlive the async memcpy (pageable H2D is actually
    // synchronous w.r.t. the host, but keep them alive for clarity).
    const SpMat csc(A.derived());
    eigen_assert(csc.rows() == n_ && csc.cols() == n_);

    const Scalar* value_ptr;
    Index value_nnz;
    CsrMat csr_tmp;
    if (Derived::needs_csr_conversion()) {
      csr_tmp = CsrMat(csc);
      value_ptr = csr_tmp.valuePtr();
      value_nnz = csr_tmp.nonZeros();
    } else {
      value_ptr = csc.valuePtr();
      value_nnz = csc.nonZeros();
    }
    eigen_assert(value_nnz == nnz_);

    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_values_.ptr, value_ptr, static_cast<size_t>(nnz_) * sizeof(Scalar),
                                             cudaMemcpyHostToDevice, stream_));

    EIGEN_CUDSS_CHECK(cudssMatrixSetValues(d_A_cudss_, d_values_.ptr));

    info_ = InvalidInput;
    info_synced_ = false;
    EIGEN_CUDSS_CHECK(
        cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, data_, d_A_cudss_, d_x_cudss_, d_b_cudss_));

    return derived();
  }

  // ---- Solve ----------------------------------------------------------------

  /** Solve A * X = B. Returns X as a dense matrix.
   * Supports single or multiple right-hand sides. */
  template <typename Rhs>
  DenseMatrix solve(const MatrixBase<Rhs>& B) const {
    sync_info();
    eigen_assert(info_ == Success && "GpuSparseSolver::solve requires a successful factorization");
    eigen_assert(B.rows() == n_);

    const DenseMatrix rhs(B);
    const int64_t nrhs = static_cast<int64_t>(rhs.cols());

    if (n_ == 0) return DenseMatrix(0, rhs.cols());

    const size_t rhs_bytes = static_cast<size_t>(n_) * static_cast<size_t>(nrhs) * sizeof(Scalar);
    DeviceBuffer d_b(rhs_bytes);
    DeviceBuffer d_x(rhs_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_b.ptr, rhs.data(), rhs_bytes, cudaMemcpyHostToDevice, stream_));

    constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
    cudssMatrix_t b_cudss = nullptr, x_cudss = nullptr;
    EIGEN_CUDSS_CHECK(cudssMatrixCreateDn(&b_cudss, static_cast<int64_t>(n_), nrhs, static_cast<int64_t>(n_), d_b.ptr,
                                          dtype, CUDSS_LAYOUT_COL_MAJOR));
    EIGEN_CUDSS_CHECK(cudssMatrixCreateDn(&x_cudss, static_cast<int64_t>(n_), nrhs, static_cast<int64_t>(n_), d_x.ptr,
                                          dtype, CUDSS_LAYOUT_COL_MAJOR));

    EIGEN_CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, d_A_cudss_, x_cudss, b_cudss));

    DenseMatrix X(n_, rhs.cols());
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(X.data(), d_x.ptr, rhs_bytes, cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));

    (void)cudssMatrixDestroy(b_cudss);
    (void)cudssMatrixDestroy(x_cudss);

    return X;
  }

  // ---- Accessors ------------------------------------------------------------

  ComputationInfo info() const {
    sync_info();
    return info_;
  }
  Index rows() const { return n_; }
  Index cols() const { return n_; }

  cudaStream_t stream() const { return stream_; }

 protected:
  // ---- CUDA / cuDSS handles -------------------------------------------------
  cudaStream_t stream_ = nullptr;
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t d_A_cudss_ = nullptr;
  cudssMatrix_t d_x_cudss_ = nullptr;
  cudssMatrix_t d_b_cudss_ = nullptr;

  // ---- Device buffers for CSR arrays ----------------------------------------
  DeviceBuffer d_rowPtr_;
  DeviceBuffer d_colIdx_;
  DeviceBuffer d_values_;

  // ---- State ----------------------------------------------------------------
  Index n_ = 0;
  Index nnz_ = 0;
  ComputationInfo info_ = InvalidInput;
  bool info_synced_ = true;
  bool analysis_done_ = false;
  GpuSparseOrdering ordering_ = GpuSparseOrdering::AMD;

 private:
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  void init_context() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUDSS_CHECK(cudssCreate(&handle_));
    EIGEN_CUDSS_CHECK(cudssSetStream(handle_, stream_));
    EIGEN_CUDSS_CHECK(cudssConfigCreate(&config_));
  }

  void sync_info() const {
    if (!info_synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      int cudss_info = 0;
      EIGEN_CUDSS_CHECK(cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &cudss_info, sizeof(cudss_info), nullptr));
      auto* self = const_cast<SparseSolverBase*>(this);
      self->info_ = (cudss_info == 0) ? Success : NumericalIssue;
      self->info_synced_ = true;
    }
  }

  void destroy_cudss_objects() {
    if (d_A_cudss_) {
      (void)cudssMatrixDestroy(d_A_cudss_);
      d_A_cudss_ = nullptr;
    }
    if (d_x_cudss_) {
      (void)cudssMatrixDestroy(d_x_cudss_);
      d_x_cudss_ = nullptr;
    }
    if (d_b_cudss_) {
      (void)cudssMatrixDestroy(d_b_cudss_);
      d_b_cudss_ = nullptr;
    }
    if (data_) {
      (void)cudssDataDestroy(handle_, data_);
      data_ = nullptr;
    }
    if (config_) {
      (void)cudssConfigDestroy(config_);
      config_ = nullptr;
    }
  }

  // Upload CSR from a RowMajor sparse matrix (native CSR).
  void upload_csr(const CsrMat& csr) { upload_compressed(csr.outerIndexPtr(), csr.innerIndexPtr(), csr.valuePtr()); }

  // Upload CSC arrays reinterpreted as CSR (for symmetric matrices: CSC(A) = CSR(A^T) = CSR(A)).
  void upload_csr_from_csc(const SpMat& csc) {
    upload_compressed(csc.outerIndexPtr(), csc.innerIndexPtr(), csc.valuePtr());
  }

  void upload_compressed(const StorageIndex* outer, const StorageIndex* inner, const Scalar* values) {
    const size_t rowptr_bytes = static_cast<size_t>(n_ + 1) * sizeof(StorageIndex);
    const size_t colidx_bytes = static_cast<size_t>(nnz_) * sizeof(StorageIndex);
    const size_t values_bytes = static_cast<size_t>(nnz_) * sizeof(Scalar);

    d_rowPtr_ = DeviceBuffer(rowptr_bytes);
    d_colIdx_ = DeviceBuffer(colidx_bytes);
    d_values_ = DeviceBuffer(values_bytes);

    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_rowPtr_.ptr, outer, rowptr_bytes, cudaMemcpyHostToDevice, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_colIdx_.ptr, inner, colidx_bytes, cudaMemcpyHostToDevice, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_values_.ptr, values, values_bytes, cudaMemcpyHostToDevice, stream_));
  }

  void create_cudss_matrix() {
    if (d_A_cudss_) (void)cudssMatrixDestroy(d_A_cudss_);

    constexpr cudaDataType_t idx_type = cudss_index_type<StorageIndex>::value;
    constexpr cudaDataType_t val_type = cuda_data_type<Scalar>::value;
    constexpr cudssMatrixType_t mtype = Derived::cudss_matrix_type();
    constexpr cudssMatrixViewType_t mview = Derived::cudss_matrix_view();

    EIGEN_CUDSS_CHECK(cudssMatrixCreateCsr(
        &d_A_cudss_, static_cast<int64_t>(n_), static_cast<int64_t>(n_), static_cast<int64_t>(nnz_), d_rowPtr_.ptr,
        /*rowEnd=*/nullptr, d_colIdx_.ptr, d_values_.ptr, idx_type, val_type, mtype, mview, CUDSS_BASE_ZERO));
  }

  void apply_ordering_config() {
    cudssAlgType_t alg;
    switch (ordering_) {
      case GpuSparseOrdering::AMD:
        alg = CUDSS_ALG_DEFAULT;
        break;
      case GpuSparseOrdering::METIS:
        alg = CUDSS_ALG_2;
        break;
      case GpuSparseOrdering::RCM:
        alg = CUDSS_ALG_3;
        break;
      default:
        alg = CUDSS_ALG_DEFAULT;
        break;
    }
    EIGEN_CUDSS_CHECK(cudssConfigSet(config_, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)));
  }

  void create_placeholder_dense() {
    if (d_x_cudss_) (void)cudssMatrixDestroy(d_x_cudss_);
    if (d_b_cudss_) (void)cudssMatrixDestroy(d_b_cudss_);
    constexpr cudaDataType_t dtype = cuda_data_type<Scalar>::value;
    EIGEN_CUDSS_CHECK(cudssMatrixCreateDn(&d_x_cudss_, static_cast<int64_t>(n_), 1, static_cast<int64_t>(n_), nullptr,
                                          dtype, CUDSS_LAYOUT_COL_MAJOR));
    EIGEN_CUDSS_CHECK(cudssMatrixCreateDn(&d_b_cudss_, static_cast<int64_t>(n_), 1, static_cast<int64_t>(n_), nullptr,
                                          dtype, CUDSS_LAYOUT_COL_MAJOR));
  }
};

}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SPARSE_SOLVER_BASE_H
