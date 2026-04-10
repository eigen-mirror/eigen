// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// GPU sparse matrix-vector multiply (SpMV) and sparse matrix-dense matrix
// multiply (SpMM) via cuSPARSE.
//
// GpuSparseContext manages a cuSPARSE handle and device buffers. It accepts
// Eigen SparseMatrix<Scalar, ColMajor> (CSC) and performs SpMV/SpMM on the
// GPU. RowMajor input is implicitly converted to ColMajor.
//
// Usage:
//   GpuSparseContext<double> ctx;
//   VectorXd y = ctx.multiply(A, x);           // y = A * x
//   ctx.multiply(A, x, y, 2.0, 1.0);           // y = 2*A*x + y
//   VectorXd z = ctx.multiplyT(A, x);          // z = A^T * x
//   MatrixXd Y = ctx.multiplyMat(A, X);        // Y = A * X (multiple RHS)

#ifndef EIGEN_GPU_SPARSE_CONTEXT_H
#define EIGEN_GPU_SPARSE_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuSparseSupport.h"

namespace Eigen {

template <typename Scalar_>
class GpuSparseContext {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  GpuSparseContext() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUSPARSE_CHECK(cusparseCreate(&handle_));
    EIGEN_CUSPARSE_CHECK(cusparseSetStream(handle_, stream_));
  }

  ~GpuSparseContext() {
    destroy_descriptors();
    if (handle_) (void)cusparseDestroy(handle_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  GpuSparseContext(const GpuSparseContext&) = delete;
  GpuSparseContext& operator=(const GpuSparseContext&) = delete;

  // ---- SpMV: y = A * x -----------------------------------------------------

  /** Compute y = A * x. Returns y as a new dense vector. */
  template <typename InputType, typename Rhs>
  DenseVector multiply(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x) {
    const SpMat mat(A.derived());
    DenseVector y(mat.rows());
    y.setZero();
    multiply_impl(mat, x.derived(), y, Scalar(1), Scalar(0), CUSPARSE_OPERATION_NON_TRANSPOSE);
    return y;
  }

  /** Compute y = alpha * op(A) * x + beta * y (in-place). */
  template <typename InputType, typename Rhs, typename Dest>
  void multiply(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x, MatrixBase<Dest>& y,
                Scalar alpha = Scalar(1), Scalar beta = Scalar(0),
                cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE) {
    const SpMat mat(A.derived());
    multiply_impl(mat, x.derived(), y.derived(), alpha, beta, op);
  }

  // ---- SpMV transpose: y = A^T * x -----------------------------------------

  /** Compute y = A^T * x. Returns y as a new dense vector. */
  template <typename InputType, typename Rhs>
  DenseVector multiplyT(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x) {
    const SpMat mat(A.derived());
    DenseVector y(mat.cols());
    y.setZero();
    multiply_impl(mat, x.derived(), y, Scalar(1), Scalar(0), CUSPARSE_OPERATION_TRANSPOSE);
    return y;
  }

  // ---- SpMM: Y = A * X (multiple RHS) --------------------------------------

  /** Compute Y = A * X where X is a dense matrix (multiple RHS). Returns Y. */
  template <typename InputType, typename Rhs>
  DenseMatrix multiplyMat(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& X) {
    const SpMat mat(A.derived());
    const DenseMatrix rhs(X.derived());
    eigen_assert(mat.cols() == rhs.rows());

    const Index m = mat.rows();
    const Index n = rhs.cols();
    if (m == 0 || n == 0 || mat.nonZeros() == 0) return DenseMatrix::Zero(m, n);

    DenseMatrix Y = DenseMatrix::Zero(m, n);
    spmm_impl(mat, rhs, Y, Scalar(1), Scalar(0), CUSPARSE_OPERATION_NON_TRANSPOSE);
    return Y;
  }

  // ---- Accessors ------------------------------------------------------------

  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusparseHandle_t handle_ = nullptr;

  // Cached device buffers (grow-only).
  internal::DeviceBuffer d_outerPtr_;
  internal::DeviceBuffer d_innerIdx_;
  internal::DeviceBuffer d_values_;
  internal::DeviceBuffer d_x_;
  internal::DeviceBuffer d_y_;
  internal::DeviceBuffer d_workspace_;
  size_t d_outerPtr_size_ = 0;
  size_t d_innerIdx_size_ = 0;
  size_t d_values_size_ = 0;
  size_t d_x_size_ = 0;
  size_t d_y_size_ = 0;
  size_t d_workspace_size_ = 0;

  // Cached cuSPARSE descriptors.
  cusparseSpMatDescr_t spmat_desc_ = nullptr;
  Index cached_rows_ = -1;
  Index cached_cols_ = -1;
  Index cached_nnz_ = -1;

  // ---- SpMV implementation --------------------------------------------------

  template <typename RhsDerived, typename DestDerived>
  void multiply_impl(const SpMat& A, const RhsDerived& x, DestDerived& y, Scalar alpha, Scalar beta,
                     cusparseOperation_t op) {
    eigen_assert(A.isCompressed());

    const Index m = A.rows();
    const Index n = A.cols();
    const Index nnz = A.nonZeros();
    const Index x_size = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? n : m;
    const Index y_size = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? m : n;

    eigen_assert(x.size() == x_size);
    eigen_assert(y.size() == y_size);

    if (m == 0 || n == 0 || nnz == 0) {
      if (beta == Scalar(0))
        y.setZero();
      else
        y *= beta;
      return;
    }

    // Upload sparse matrix to device.
    upload_sparse(A);

    // Upload x to device.
    ensure_buffer(d_x_, d_x_size_, static_cast<size_t>(x_size) * sizeof(Scalar));
    const DenseVector x_tmp(x);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_x_.ptr, x_tmp.data(), x_size * sizeof(Scalar), cudaMemcpyHostToDevice, stream_));

    // Upload y to device (for beta != 0).
    ensure_buffer(d_y_, d_y_size_, static_cast<size_t>(y_size) * sizeof(Scalar));
    if (beta != Scalar(0)) {
      const DenseVector y_tmp(y);
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_y_.ptr, y_tmp.data(), y_size * sizeof(Scalar), cudaMemcpyHostToDevice, stream_));
    }

    // Create dense vector descriptors.
    constexpr cudaDataType_t dtype = internal::cuda_data_type<Scalar>::value;
    cusparseDnVecDescr_t x_desc = nullptr, y_desc = nullptr;
    EIGEN_CUSPARSE_CHECK(cusparseCreateDnVec(&x_desc, x_size, d_x_.ptr, dtype));
    EIGEN_CUSPARSE_CHECK(cusparseCreateDnVec(&y_desc, y_size, d_y_.ptr, dtype));

    // Query workspace size.
    size_t ws_size = 0;
    EIGEN_CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle_, op, &alpha, spmat_desc_, x_desc, &beta, y_desc, dtype,
                                                 CUSPARSE_SPMV_ALG_DEFAULT, &ws_size));
    ensure_buffer(d_workspace_, d_workspace_size_, ws_size);

    // Execute SpMV.
    EIGEN_CUSPARSE_CHECK(cusparseSpMV(handle_, op, &alpha, spmat_desc_, x_desc, &beta, y_desc, dtype,
                                      CUSPARSE_SPMV_ALG_DEFAULT, d_workspace_.ptr));

    // Download result.
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(y.data(), d_y_.ptr, y_size * sizeof(Scalar), cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));

    (void)cusparseDestroyDnVec(x_desc);
    (void)cusparseDestroyDnVec(y_desc);
  }

  // ---- SpMM implementation --------------------------------------------------

  void spmm_impl(const SpMat& A, const DenseMatrix& X, DenseMatrix& Y, Scalar alpha, Scalar beta,
                 cusparseOperation_t op) {
    eigen_assert(A.isCompressed());

    const Index m = A.rows();
    const Index n = X.cols();
    const Index k = A.cols();
    const Index nnz = A.nonZeros();

    if (m == 0 || n == 0 || k == 0 || nnz == 0) {
      if (beta == Scalar(0))
        Y.setZero();
      else
        Y *= beta;
      return;
    }

    upload_sparse(A);

    // Upload X to device.
    const size_t x_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(Scalar);
    const size_t y_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(Scalar);
    ensure_buffer(d_x_, d_x_size_, x_bytes);
    ensure_buffer(d_y_, d_y_size_, y_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_x_.ptr, X.data(), x_bytes, cudaMemcpyHostToDevice, stream_));
    if (beta != Scalar(0)) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_y_.ptr, Y.data(), y_bytes, cudaMemcpyHostToDevice, stream_));
    }

    // Create dense matrix descriptors.
    constexpr cudaDataType_t dtype = internal::cuda_data_type<Scalar>::value;
    cusparseDnMatDescr_t x_desc = nullptr, y_desc = nullptr;
    // Eigen is column-major, so ld = rows.
    EIGEN_CUSPARSE_CHECK(cusparseCreateDnMat(&x_desc, k, n, k, d_x_.ptr, dtype, CUSPARSE_ORDER_COL));
    EIGEN_CUSPARSE_CHECK(cusparseCreateDnMat(&y_desc, m, n, m, d_y_.ptr, dtype, CUSPARSE_ORDER_COL));

    // Query workspace.
    size_t ws_size = 0;
    EIGEN_CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle_, op, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmat_desc_,
                                                 x_desc, &beta, y_desc, dtype, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size));
    ensure_buffer(d_workspace_, d_workspace_size_, ws_size);

    // Execute SpMM.
    EIGEN_CUSPARSE_CHECK(cusparseSpMM(handle_, op, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmat_desc_, x_desc, &beta,
                                      y_desc, dtype, CUSPARSE_SPMM_ALG_DEFAULT, d_workspace_.ptr));

    // Download result.
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(Y.data(), d_y_.ptr, y_bytes, cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));

    (void)cusparseDestroyDnMat(x_desc);
    (void)cusparseDestroyDnMat(y_desc);
  }

  // ---- Helpers --------------------------------------------------------------

  void upload_sparse(const SpMat& A) {
    const Index m = A.rows();
    const Index n = A.cols();
    const Index nnz = A.nonZeros();

    const size_t outer_bytes = static_cast<size_t>(n + 1) * sizeof(StorageIndex);
    const size_t inner_bytes = static_cast<size_t>(nnz) * sizeof(StorageIndex);
    const size_t val_bytes = static_cast<size_t>(nnz) * sizeof(Scalar);

    ensure_buffer(d_outerPtr_, d_outerPtr_size_, outer_bytes);
    ensure_buffer(d_innerIdx_, d_innerIdx_size_, inner_bytes);
    ensure_buffer(d_values_, d_values_size_, val_bytes);

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_outerPtr_.ptr, A.outerIndexPtr(), outer_bytes, cudaMemcpyHostToDevice, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_innerIdx_.ptr, A.innerIndexPtr(), inner_bytes, cudaMemcpyHostToDevice, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_values_.ptr, A.valuePtr(), val_bytes, cudaMemcpyHostToDevice, stream_));

    // Recreate descriptor if shape changed.
    if (m != cached_rows_ || n != cached_cols_ || nnz != cached_nnz_) {
      destroy_descriptors();

      constexpr cusparseIndexType_t idx_type = (sizeof(StorageIndex) == 4) ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
      constexpr cudaDataType_t val_type = internal::cuda_data_type<Scalar>::value;

      // ColMajor → CSC. outerIndexPtr = col offsets, innerIndexPtr = row indices.
      EIGEN_CUSPARSE_CHECK(cusparseCreateCsc(&spmat_desc_, m, n, nnz, d_outerPtr_.ptr, d_innerIdx_.ptr, d_values_.ptr,
                                             idx_type, idx_type, CUSPARSE_INDEX_BASE_ZERO, val_type));
      cached_rows_ = m;
      cached_cols_ = n;
      cached_nnz_ = nnz;
    } else {
      // Same shape — just update pointers.
      EIGEN_CUSPARSE_CHECK(cusparseCscSetPointers(spmat_desc_, d_outerPtr_.ptr, d_innerIdx_.ptr, d_values_.ptr));
    }
  }

  void destroy_descriptors() {
    if (spmat_desc_) {
      (void)cusparseDestroySpMat(spmat_desc_);
      spmat_desc_ = nullptr;
    }
    cached_rows_ = -1;
    cached_cols_ = -1;
    cached_nnz_ = -1;
  }

  void ensure_buffer(internal::DeviceBuffer& buf, size_t& current_size, size_t needed) {
    if (needed > current_size) {
      if (buf.ptr) EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      buf = internal::DeviceBuffer(needed);
      current_size = needed;
    }
  }
};

}  // namespace Eigen

#endif  // EIGEN_GPU_SPARSE_CONTEXT_H
