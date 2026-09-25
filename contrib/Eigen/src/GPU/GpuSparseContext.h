// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// GPU sparse matrix-vector (SpMV) and sparse matrix-dense matrix (SpMM) multiply
// via cuSPARSE.
//
// SparseContext owns the cuSPARSE descriptors and device buffers. It takes
// SparseMatrix<Scalar, ColMajor> (CSC), implicitly converting RowMajor input, and
// can borrow a gpu::Context so that sparse products share a stream with BLAS-1
// operations — which removes the cross-stream event waits in solvers like CG.
// It also takes BlockSparseMatrix: as BSR (internal::BsrBinding) where
// internal::use_cusparse_bsr holds, as the scalar-level toSparse() copy otherwise.
//
// Caching: host-input calls re-upload the values *and* index arrays on every
// call. Host pointer identity cannot detect a sparsity pattern rewritten in
// place or assigned into the same allocations (SparseMatrix reuses them for
// same-shape assignments), so the structure is never assumed unchanged. What
// is cached are the cuSPARSE descriptors (sparse descriptor keyed on
// dimensions + nonzero count + block size, dense descriptors keyed on shape)
// and the SpMV/SpMM workspace-size queries. For repeated products with no
// re-upload at all, use deviceView().
//
// Not thread-safe: concurrent multiply* calls on one instance race on the
// cuSPARSE handle, the bound stream, and the cached buffers. Use one per thread.

#ifndef EIGEN_GPU_SPARSE_CONTEXT_H
#define EIGEN_GPU_SPARSE_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <cstdint>

#include "./CuSparseSupport.h"
#include "./FwdDecl.h"

namespace Eigen {
namespace gpu {

namespace internal {
/** Whether a BlockSparseMatrix with this block shape uploads as BSR. cuSPARSE
 * runs BSR products only on square blocks of size at least 2 (cusparseCreateBsr
 * rejects rectangular blocks; SpMV on 1 x 1 blocks returns
 * CUSPARSE_STATUS_NOT_SUPPORTED) and only from cuSPARSE 12.6.3; every other
 * case takes the CSC path. */
template <int BlockRows, int BlockCols>
struct use_cusparse_bsr
    : Eigen::internal::bool_constant<(EIGEN_HAS_CUSPARSE_BSR != 0 && BlockRows == BlockCols && BlockRows >= 2)> {};
}  // namespace internal

#if EIGEN_HAS_CUSPARSE_BSR
namespace internal {

/** The arrays cusparseCreateBsr takes: block-row offsets (brows + 1),
 * block-column indices (bnnz), and bnnz row-major blockSize x blockSize
 * value blocks. Sizes are in blocks; rows()/cols()/nonZeros() in scalars. */
template <typename Scalar, typename StorageIndex>
struct BsrArrays {
  Index brows;
  Index bcols;
  Index bnnz;
  Index blockSize;
  const StorageIndex* outer;
  const StorageIndex* inner;
  const Scalar* values;

  Index rows() const { return brows * blockSize; }
  Index cols() const { return bcols * blockSize; }
  Index nonZeros() const { return bnnz * blockSize * blockSize; }
};

/** Reads a square-block BlockSparseMatrix's buffers as BSR without copying.
 *
 * A RowMajor M is BSR of M: block rows are its outer vectors and its blocks are
 * stored row-major. A ColMajor M is BSR of M^T: its outer vectors are the block
 * rows of M^T, and a column-major B x B block read row-major is the transposed
 * block. */
template <typename Bsm>
BsrArrays<typename Bsm::Scalar, typename Bsm::StorageIndex> bsr_of(const Bsm& M) {
  return BsrArrays<typename Bsm::Scalar, typename Bsm::StorageIndex>{
      /*brows=*/M.blockOuterSize(), /*bcols=*/M.blockInnerSize(),
      /*bnnz=*/M.nonZeroBlocks(),   /*blockSize=*/Index(Bsm::BlockRows),
      /*outer=*/M.outerIndexPtr(),  /*inner=*/M.innerIndexPtr(),
      /*values=*/M.valuePtr()};
}

/** Binds op(A) as BSR arrays of op(A) itself, the only form cuSPARSE
 * multiplies. Per bsr_of(), one of {NoTrans, Trans} reads A's buffers
 * directly — NoTrans for RowMajor, Trans for ColMajor — and the other needs
 * the transposed copy; ConjTrans on a complex matrix always copies for the
 * conjugation. The copy lives as long as the binding. */
template <typename Bsm>
class BsrBinding {
  EIGEN_STATIC_ASSERT((use_cusparse_bsr<int(Bsm::BlockRows), int(Bsm::BlockCols)>::value),
                      CUSPARSE_BSR_REQUIRES_SQUARE_BLOCKS_OF_SIZE_AT_LEAST_2)

 public:
  using Scalar = typename Bsm::Scalar;
  using Arrays = BsrArrays<Scalar, typename Bsm::StorageIndex>;

  BsrBinding(const Bsm& A, GpuOp op) {
    if (op == GpuOp::ConjTrans && !NumTraits<Scalar>::IsComplex) op = GpuOp::Trans;
    const bool direct = Bsm::IsRowMajor ? (op == GpuOp::NoTrans) : (op == GpuOp::Trans);
    if (direct) {
      arrays_ = bsr_of(A);
      return;
    }
    if (op == GpuOp::ConjTrans && !Bsm::IsRowMajor) {
      // ColMajor conj(A) reads as BSR of conj(A)^T = A^H.
      copy_ = A.unaryExpr(Eigen::internal::scalar_conjugate_op<Scalar>());
    } else if (op == GpuOp::ConjTrans) {
      copy_ = A.adjoint();
    } else {
      copy_ = A.transpose();
    }
    arrays_ = bsr_of(copy_);
  }

  BsrBinding(const BsrBinding&) = delete;
  BsrBinding& operator=(const BsrBinding&) = delete;

  const Arrays& arrays() const { return arrays_; }

 private:
  // transpose()/adjoint() swap the block dimensions, which are equal here, so
  // the copy has A's own type in either storage order.
  Bsm copy_;
  Arrays arrays_{};
};

}  // namespace internal
#endif  // EIGEN_HAS_CUSPARSE_BSR

/** Sparse product expression: DeviceSparseView * DeviceMatrix → SpMVExpr.
 * Evaluated by DeviceMatrix::operator=(SpMVExpr): dispatches to cusparseSpMV
 * when the dense operand has one column, cusparseSpMM otherwise. Adding or
 * subtracting a DeviceMatrix yields an SpMVAffineExpr. */
template <typename Scalar_>
class SpMVExpr {
 public:
  using Scalar = Scalar_;
  SpMVExpr(const DeviceSparseView<Scalar>& view, const DeviceMatrix<Scalar>& x) : view_(view), x_(x) {}
  const DeviceSparseView<Scalar>& view() const { return view_; }
  const DeviceMatrix<Scalar>& x() const { return x_; }

 private:
  const DeviceSparseView<Scalar>& view_;
  const DeviceMatrix<Scalar>& x_;
};

/** alpha * (DeviceSparseView * DeviceMatrix) + beta * addend with alpha, beta = ±1: the
 * result of `d_b - d_A * d_x`, `d_b + d_A * d_x`, `d_A * d_x + d_b` and `d_A * d_x - d_b`,
 * so that Eigen's iterative solver templates (`residual = rhs - mat * x`) compile.
 * Evaluated by DeviceMatrix::operator=(SpMVAffineExpr): the addend is copied into the
 * destination unless it already is the destination, then one cusparseSpMV (one column)
 * or cusparseSpMM call runs with the given alpha and beta. No operator takes an
 * SpMVAffineExpr, so an expression with a second addend does not compile. */
template <typename Scalar_>
class SpMVAffineExpr {
 public:
  using Scalar = Scalar_;
  SpMVAffineExpr(const SpMVExpr<Scalar>& product, Scalar alpha, Scalar beta, const DeviceMatrix<Scalar>& addend)
      : view_(product.view()), x_(product.x()), alpha_(alpha), beta_(beta), addend_(addend) {}
  const DeviceSparseView<Scalar>& view() const { return view_; }
  const DeviceMatrix<Scalar>& x() const { return x_; }
  Scalar alpha() const { return alpha_; }
  Scalar beta() const { return beta_; }
  const DeviceMatrix<Scalar>& addend() const { return addend_; }

 private:
  const DeviceSparseView<Scalar>& view_;
  const DeviceMatrix<Scalar>& x_;
  Scalar alpha_;
  Scalar beta_;
  const DeviceMatrix<Scalar>& addend_;
};

template <typename S>
SpMVAffineExpr<S> operator-(const DeviceMatrix<S>& b, const SpMVExpr<S>& p) {
  return SpMVAffineExpr<S>(p, S(-1), S(1), b);
}

template <typename S>
SpMVAffineExpr<S> operator+(const DeviceMatrix<S>& b, const SpMVExpr<S>& p) {
  return SpMVAffineExpr<S>(p, S(1), S(1), b);
}

template <typename S>
SpMVAffineExpr<S> operator+(const SpMVExpr<S>& p, const DeviceMatrix<S>& b) {
  return b + p;
}

template <typename S>
SpMVAffineExpr<S> operator-(const SpMVExpr<S>& p, const DeviceMatrix<S>& b) {
  return SpMVAffineExpr<S>(p, S(1), S(-1), b);
}

}  // namespace gpu

namespace internal {
// DeviceSparseView is a matrix-free matrix type for Eigen's iterative solvers: it looks like a
// SparseMatrix to the traits machinery, and IterativeSolverBase stores it by pointer because
// Ref<> cannot bind to it (the same mechanism as the matrix-free example in the documentation).
template <typename Scalar_>
struct traits<gpu::DeviceSparseView<Scalar_>> : traits<SparseMatrix<Scalar_, ColMajor, int>> {};
}  // namespace internal

namespace gpu {

/** Device-resident sparse matrix view. Returned by SparseContext::deviceView()
 * for a SparseMatrix (CSC) or a BlockSparseMatrix (BSR). Lightweight handle
 * referencing the context's cached device data.
 *
 * It is also a matrix-free matrix type for Eigen's iterative solvers:
 * `ConjugateGradient<DeviceSparseView<Scalar>, Lower | Upper>` runs Eigen's own
 * algorithm on device vectors through `solveWithGuessInPlace()`; see the README.
 *
 * \warning One SparseContext caches one sparse matrix at a time. Any later
 * upload through the same context — a second deviceView() or any host-input
 * multiply — replaces the cached data and invalidates earlier views; a stale
 * view is caught by an assert at evaluation time via a generation counter.
 * For multiple simultaneous sparse matrices, use separate SparseContext
 * instances (they can share a Context for same-stream execution).
 *
 * Supports `d_y = d_A * d_x` (SpMV) and `d_Y = d_A * d_X` (SpMM). */
template <typename Scalar_>
class DeviceSparseView : public EigenBase<DeviceSparseView<Scalar_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  // What IterativeSolverBase and ConjugateGradient read from a matrix type.
  static constexpr int ColsAtCompileTime = Dynamic;
  static constexpr int MaxColsAtCompileTime = Dynamic;
  static constexpr bool IsRowMajor = false;

  DeviceSparseView(SparseContext<Scalar>& ctx, Index rows, Index cols, uint64_t generation)
      : ctx_(ctx), rows_(rows), cols_(cols), generation_(generation) {}

  /** Sparse product expression: d_A * d_x. Evaluated by DeviceMatrix::operator=. */
  SpMVExpr<Scalar> operator*(const DeviceMatrix<Scalar>& x) const { return SpMVExpr<Scalar>(*this, x); }

  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  /** Number of stored entries of the cached matrix. */
  Index nonZeros() const { return ctx_.nonZeros(); }
  const SparseContext<Scalar>& context() const { return ctx_; }

  /** Upload generation this view was created against. Used to detect stale
   * views after the context has cached a different matrix. */
  uint64_t generation() const { return generation_; }

 private:
  SparseContext<Scalar>& ctx_;
  Index rows_;
  Index cols_;
  uint64_t generation_;
};

template <typename Scalar_>
class SparseContext {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  /** BlockSparseMatrix types the block-sparse overloads accept: `int` indices,
   * either storage order, any block shape. internal::use_cusparse_bsr picks the
   * upload: BSR — no host copy for `op == NoTrans` on a RowMajor matrix or
   * `op == Trans` on a ColMajor one, a transposing (and conjugating) copy
   * otherwise — or the scalar-level toSparse() copy on the CSC path. */
  template <int Options, int BlockRows, int BlockCols>
  using BlockSpMat = BlockSparseMatrix<Scalar, Options, BlockRows, BlockCols, StorageIndex>;

  /** Standalone: creates own stream and cuSPARSE handle. */
  SparseContext() : owns_handle_(true) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    owns_stream_ = true;
    EIGEN_CUSPARSE_CHECK(cusparseCreate(&handle_));
    EIGEN_CUSPARSE_CHECK(cusparseSetStream(handle_, stream_));
  }

  /** Borrow a Context: shares stream and cuSPARSE handle.
   * The Context must outlive this SparseContext. */
  explicit SparseContext(Context& ctx)
      : stream_(ctx.stream()), handle_(ctx.cusparseHandle()), owns_stream_(false), owns_handle_(false) {}

  ~SparseContext() {
    destroy_spmat_descriptor(/*checked=*/false);
    destroy_dense_descriptors();
    if (owns_handle_ && handle_) (void)cusparseDestroy(handle_);
    if (owns_stream_ && stream_) (void)cudaStreamDestroy(stream_);
  }

  SparseContext(const SparseContext&) = delete;
  SparseContext& operator=(const SparseContext&) = delete;

  /** Upload a sparse matrix to device and return a lightweight view.
   * The sparse data is uploaded immediately and cached in this context.
   * The returned view can be used for repeated SpMV/SpMM without re-uploading.
   * If the matrix values change, call deviceView() again to re-upload.
   *
   * \warning One context caches one matrix. Any later upload — another
   * deviceView() or any host-input multiply — overwrites the previous upload
   * and invalidates earlier views (asserted at evaluation time). For multiple
   * simultaneous matrices, use separate SparseContext instances sharing the
   * same Context.
   *
   * Supports `d_y = d_A * d_x` (SpMV) and `d_Y = d_A * d_X` (SpMM). */
  DeviceSparseView<Scalar> deviceView(const SpMat& A) {
    eigen_assert(A.isCompressed());
    upload_sparse(A);
    return DeviceSparseView<Scalar>(*this, A.rows(), A.cols(), generation_);
  }

  /** Generation counter of the currently cached sparse matrix. Bumped on
   * every sparse upload (deviceView() or a host-input multiply). */
  uint64_t uploadGeneration() const { return generation_; }

  /** Number of stored entries of the cached sparse matrix, -1 before the first upload. */
  Index nonZeros() const { return cached_nnz_; }

  /** Compute y = A * x. Returns y as a new dense vector. */
  template <typename InputType, typename Rhs>
  DenseVector multiply(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x) {
    return multiply_host_return(A, x, GpuOp::NoTrans);
  }

  /** Compute y = alpha * op(A) * x + beta * y (in-place, host vectors). */
  template <typename InputType, typename Rhs, typename Dest>
  void multiply(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x, MatrixBase<Dest>& y,
                Scalar alpha = Scalar(1), Scalar beta = Scalar(0), GpuOp op = GpuOp::NoTrans) {
    const InputType& input = A.derived();
    internal::check_storage_index_bounds<StorageIndex>(input.rows(), input.cols(), input.nonZeros());
    SpMat storage;
    const SpMat& mat = internal::bind_sparse<SpMat>(input, storage);
    multiply_host_impl(mat, x.derived(), y.derived(), alpha, beta, internal::to_cusparse_op<Scalar>(op));
  }

  /** Compute d_y = A * d_x. Device-resident dense vectors, no host transfer
   * for x/y. The sparse matrix (values and index arrays) is re-uploaded on
   * each call; for a fully device-resident sparse matrix use deviceView(). */
  template <typename InputType>
  void multiply(const SparseMatrixBase<InputType>& A, const DeviceMatrix<Scalar>& d_x, DeviceMatrix<Scalar>& d_y) {
    multiply(A, d_x, d_y, Scalar(1), Scalar(0), GpuOp::NoTrans);
  }

  /** Compute d_y = alpha * op(A) * d_x + beta * d_y (DeviceMatrix, in-place). */
  template <typename InputType>
  void multiply(const SparseMatrixBase<InputType>& A, const DeviceMatrix<Scalar>& d_x, DeviceMatrix<Scalar>& d_y,
                Scalar alpha, Scalar beta, GpuOp op = GpuOp::NoTrans) {
    const InputType& input = A.derived();
    internal::check_storage_index_bounds<StorageIndex>(input.rows(), input.cols(), input.nonZeros());
    SpMat storage;
    const SpMat& mat = internal::bind_sparse<SpMat>(input, storage);
    upload_sparse(mat);
    spmv_device_exec(d_x, d_y, alpha, beta, op);
  }

  /** Compute y = A^T * x (host vectors). */
  template <typename InputType, typename Rhs>
  DenseVector multiplyT(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x) {
    return multiply_host_return(A, x, GpuOp::Trans);
  }

  /** Compute y = A^H * x (conjugate transpose). For real Scalar this is equivalent to multiplyT. */
  template <typename InputType, typename Rhs>
  DenseVector multiplyAdjoint(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x) {
    return multiply_host_return(A, x, GpuOp::ConjTrans);
  }

  /** Compute Y = op(A) * X where X is a dense matrix (multiple RHS). Returns Y. */
  template <typename InputType, typename Rhs>
  DenseMatrix multiplyMat(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& X, GpuOp op = GpuOp::NoTrans) {
    const InputType& input = A.derived();
    internal::check_storage_index_bounds<StorageIndex>(input.rows(), input.cols(), input.nonZeros());
    SpMat storage;
    const SpMat& mat = internal::bind_sparse<SpMat>(input, storage);
    const DenseMatrix rhs(X.derived());

    const cusparseOperation_t cu_op = internal::to_cusparse_op<Scalar>(op);
    const Index m = (op == GpuOp::NoTrans) ? mat.rows() : mat.cols();
    const Index k = (op == GpuOp::NoTrans) ? mat.cols() : mat.rows();
    eigen_assert(k == rhs.rows());

    const Index n = rhs.cols();
    if (m == 0 || n == 0 || mat.nonZeros() == 0) return DenseMatrix::Zero(m, n);

    DenseMatrix Y = DenseMatrix::Zero(m, n);
    spmm_impl(mat, rhs, Y, Scalar(1), Scalar(0), cu_op);
    return Y;
  }

  // BlockSparseMatrix overloads: same contracts as the SparseMatrix ones,
  // dispatched on internal::use_cusparse_bsr<BlockRows, BlockCols> to a BSR
  // upload of op(A) or to the scalar-level CSC path.

  /** Upload a block-sparse matrix and return a view; see the SparseMatrix
   * overload for the caching contract. A ColMajor matrix bound as BSR is
   * transposed on the host once, at upload. */
  template <int Options, int BlockRows, int BlockCols>
  DeviceSparseView<Scalar> deviceView(const BlockSpMat<Options, BlockRows, BlockCols>& A) {
    return device_view_block(A, internal::use_cusparse_bsr<BlockRows, BlockCols>());
  }

  /** Compute y = A * x for a block-sparse A. Returns y as a new dense vector. */
  template <int Options, int BlockRows, int BlockCols, typename Rhs>
  DenseVector multiply(const BlockSpMat<Options, BlockRows, BlockCols>& A, const MatrixBase<Rhs>& x) {
    return multiply_host_return_block(A, x, GpuOp::NoTrans);
  }

  /** Compute y = alpha * op(A) * x + beta * y (in-place, host vectors) for a block-sparse A. */
  template <int Options, int BlockRows, int BlockCols, typename Rhs, typename Dest>
  void multiply(const BlockSpMat<Options, BlockRows, BlockCols>& A, const MatrixBase<Rhs>& x, MatrixBase<Dest>& y,
                Scalar alpha = Scalar(1), Scalar beta = Scalar(0), GpuOp op = GpuOp::NoTrans) {
    multiply_host_block(A, x.derived(), y.derived(), alpha, beta, op,
                        internal::use_cusparse_bsr<BlockRows, BlockCols>());
  }

  /** Compute d_y = A * d_x for a block-sparse A; the matrix is re-uploaded on
   * each call, use deviceView() to upload once. */
  template <int Options, int BlockRows, int BlockCols>
  void multiply(const BlockSpMat<Options, BlockRows, BlockCols>& A, const DeviceMatrix<Scalar>& d_x,
                DeviceMatrix<Scalar>& d_y) {
    multiply(A, d_x, d_y, Scalar(1), Scalar(0), GpuOp::NoTrans);
  }

  /** Compute d_y = alpha * op(A) * d_x + beta * d_y (DeviceMatrix, in-place) for a block-sparse A. */
  template <int Options, int BlockRows, int BlockCols>
  void multiply(const BlockSpMat<Options, BlockRows, BlockCols>& A, const DeviceMatrix<Scalar>& d_x,
                DeviceMatrix<Scalar>& d_y, Scalar alpha, Scalar beta, GpuOp op = GpuOp::NoTrans) {
    multiply_device_block(A, d_x, d_y, alpha, beta, op, internal::use_cusparse_bsr<BlockRows, BlockCols>());
  }

  /** Compute y = A^T * x (host vectors) for a block-sparse A. */
  template <int Options, int BlockRows, int BlockCols, typename Rhs>
  DenseVector multiplyT(const BlockSpMat<Options, BlockRows, BlockCols>& A, const MatrixBase<Rhs>& x) {
    return multiply_host_return_block(A, x, GpuOp::Trans);
  }

  /** Compute y = A^H * x for a block-sparse A. For real Scalar this is equivalent to multiplyT. */
  template <int Options, int BlockRows, int BlockCols, typename Rhs>
  DenseVector multiplyAdjoint(const BlockSpMat<Options, BlockRows, BlockCols>& A, const MatrixBase<Rhs>& x) {
    return multiply_host_return_block(A, x, GpuOp::ConjTrans);
  }

  /** Compute Y = op(A) * X for a block-sparse A and a dense X (multiple RHS). Returns Y. */
  template <int Options, int BlockRows, int BlockCols, typename Rhs>
  DenseMatrix multiplyMat(const BlockSpMat<Options, BlockRows, BlockCols>& A, const MatrixBase<Rhs>& X,
                          GpuOp op = GpuOp::NoTrans) {
    return multiply_mat_block(A, X, op, internal::use_cusparse_bsr<BlockRows, BlockCols>());
  }

  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusparseHandle_t handle_ = nullptr;
  bool owns_stream_ = false;
  bool owns_handle_ = false;

  // Cached device buffers for sparse matrix (grow-only).
  internal::DeviceBuffer d_outerPtr_;
  internal::DeviceBuffer d_innerIdx_;
  internal::DeviceBuffer d_values_;

  // Cached device buffers for host-API dense vectors (grow-only).
  internal::DeviceBuffer d_x_;
  internal::DeviceBuffer d_y_;

  mutable internal::DeviceBuffer d_workspace_;

  // Cached cuSPARSE sparse matrix descriptor, keyed on the scalar-level shape
  // and nonzero count plus the block size (0 for CSC/CSR, B for BSR).
  cusparseSpMatDescr_t spmat_desc_ = nullptr;
  Index cached_rows_ = -1;
  Index cached_cols_ = -1;
  Index cached_nnz_ = -1;
  Index cached_block_size_ = 0;

  // Bumped on every sparse upload; DeviceSparseViews record it at creation so
  // a stale view (its data replaced by a later upload) asserts at evaluation.
  uint64_t generation_ = 0;

  // Cached dense-vector/matrix descriptors, re-pointed per call and recreated
  // only when the shape changes.
  mutable cusparseDnVecDescr_t x_vec_desc_ = nullptr;
  mutable cusparseDnVecDescr_t y_vec_desc_ = nullptr;
  mutable int64_t x_vec_size_ = -1;
  mutable int64_t y_vec_size_ = -1;
  mutable cusparseDnMatDescr_t x_mat_desc_ = nullptr;
  mutable cusparseDnMatDescr_t y_mat_desc_ = nullptr;
  mutable int64_t x_mat_rows_ = -1, x_mat_cols_ = -1;
  mutable int64_t y_mat_rows_ = -1, y_mat_cols_ = -1;

  // Cached workspace-size query results, indexed by cusparseOperation_t.
  // Invalidated when the sparse descriptor or a dense-descriptor shape changes.
  static constexpr size_t kWsUnknown = static_cast<size_t>(-1);
  mutable size_t spmv_ws_size_[3] = {kWsUnknown, kWsUnknown, kWsUnknown};
  mutable size_t spmm_ws_size_[3] = {kWsUnknown, kWsUnknown, kWsUnknown};

  static constexpr cusparseIndexType_t kIndexType =
      (sizeof(StorageIndex) == 4) ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
  static constexpr cudaDataType_t kValueType = internal::cuda_data_type<Scalar>::value;

  // Empty-operand result on the host: y <- beta * y, no device work.
  template <typename Dest>
  static void scale_host(Dest& y, Scalar beta) {
    if (beta == Scalar(0))
      y.setZero();
    else
      y *= beta;
  }

  // Shared host-input SpMV entry: y = op(A) * x into a fresh vector.
  template <typename InputType, typename Rhs>
  DenseVector multiply_host_return(const SparseMatrixBase<InputType>& A, const MatrixBase<Rhs>& x, GpuOp op) {
    const InputType& input = A.derived();
    internal::check_storage_index_bounds<StorageIndex>(input.rows(), input.cols(), input.nonZeros());
    SpMat storage;
    const SpMat& mat = internal::bind_sparse<SpMat>(input, storage);
    DenseVector y((op == GpuOp::NoTrans) ? mat.rows() : mat.cols());
    y.setZero();
    multiply_host_impl(mat, x.derived(), y, Scalar(1), Scalar(0), internal::to_cusparse_op<Scalar>(op));
    return y;
  }

  template <typename RhsDerived, typename DestDerived>
  void multiply_host_impl(const SpMat& A, const RhsDerived& x, DestDerived& y, Scalar alpha, Scalar beta,
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
      scale_host(y, beta);
      return;
    }

    upload_sparse(A);
    spmv_host_cached(x, y, x_size, y_size, alpha, beta, op);
  }

  // BlockSparseMatrix primitives, dispatched on use_cusparse_bsr: the true_type
  // overloads bind op(A) as BSR (internal::BsrBinding) and run cuSPARSE with
  // NON_TRANSPOSE, the false_type ones forward the scalar-level toSparse() copy
  // to the CSC overloads.
  template <typename Bsm, typename Rhs>
  DenseVector multiply_host_return_block(const Bsm& A, const MatrixBase<Rhs>& x, GpuOp op) {
    DenseVector y((op == GpuOp::NoTrans) ? A.rows() : A.cols());
    y.setZero();
    multiply_host_block(A, x.derived(), y, Scalar(1), Scalar(0), op,
                        internal::use_cusparse_bsr<int(Bsm::BlockRows), int(Bsm::BlockCols)>());
    return y;
  }

  template <typename Bsm>
  DeviceSparseView<Scalar> device_view_block(const Bsm& A, std::false_type) {
    return deviceView(SpMat(A.toSparse()));
  }

  template <typename Bsm, typename RhsDerived, typename DestDerived>
  void multiply_host_block(const Bsm& A, const RhsDerived& x, DestDerived& y, Scalar alpha, Scalar beta, GpuOp op,
                           std::false_type) {
    multiply(A.toSparse(), x, y, alpha, beta, op);
  }

  template <typename Bsm>
  void multiply_device_block(const Bsm& A, const DeviceMatrix<Scalar>& d_x, DeviceMatrix<Scalar>& d_y, Scalar alpha,
                             Scalar beta, GpuOp op, std::false_type) {
    multiply(A.toSparse(), d_x, d_y, alpha, beta, op);
  }

  template <typename Bsm, typename Rhs>
  DenseMatrix multiply_mat_block(const Bsm& A, const MatrixBase<Rhs>& X, GpuOp op, std::false_type) {
    return multiplyMat(A.toSparse(), X, op);
  }

#if EIGEN_HAS_CUSPARSE_BSR
  template <typename Bsm>
  DeviceSparseView<Scalar> device_view_block(const Bsm& A, std::true_type) {
    internal::check_storage_index_bounds<StorageIndex>(A.rows(), A.cols(), A.nonZeros());
    const internal::BsrBinding<Bsm> bound(A, GpuOp::NoTrans);
    upload_bsr(bound.arrays());
    return DeviceSparseView<Scalar>(*this, A.rows(), A.cols(), generation_);
  }

  template <typename Bsm, typename RhsDerived, typename DestDerived>
  void multiply_host_block(const Bsm& A, const RhsDerived& x, DestDerived& y, Scalar alpha, Scalar beta, GpuOp op,
                           std::true_type) {
    internal::check_storage_index_bounds<StorageIndex>(A.rows(), A.cols(), A.nonZeros());
    const internal::BsrBinding<Bsm> bound(A, op);
    const internal::BsrArrays<Scalar, StorageIndex>& opA = bound.arrays();

    eigen_assert(x.size() == opA.cols());
    eigen_assert(y.size() == opA.rows());

    if (opA.rows() == 0 || opA.cols() == 0 || opA.bnnz == 0) {
      scale_host(y, beta);
      return;
    }

    upload_bsr(opA);
    spmv_host_cached(x, y, opA.cols(), opA.rows(), alpha, beta, CUSPARSE_OPERATION_NON_TRANSPOSE);
  }

  template <typename Bsm>
  void multiply_device_block(const Bsm& A, const DeviceMatrix<Scalar>& d_x, DeviceMatrix<Scalar>& d_y, Scalar alpha,
                             Scalar beta, GpuOp op, std::true_type) {
    internal::check_storage_index_bounds<StorageIndex>(A.rows(), A.cols(), A.nonZeros());
    const internal::BsrBinding<Bsm> bound(A, op);
    upload_bsr(bound.arrays());
    spmv_device_exec(d_x, d_y, alpha, beta, GpuOp::NoTrans);
  }

  template <typename Bsm, typename Rhs>
  DenseMatrix multiply_mat_block(const Bsm& A, const MatrixBase<Rhs>& X, GpuOp op, std::true_type) {
    internal::check_storage_index_bounds<StorageIndex>(A.rows(), A.cols(), A.nonZeros());
    const internal::BsrBinding<Bsm> bound(A, op);
    const internal::BsrArrays<Scalar, StorageIndex>& opA = bound.arrays();
    const DenseMatrix rhs(X.derived());
    eigen_assert(opA.cols() == rhs.rows());

    const Index n = rhs.cols();
    if (opA.rows() == 0 || opA.cols() == 0 || n == 0 || opA.bnnz == 0) return DenseMatrix::Zero(opA.rows(), n);

    DenseMatrix Y = DenseMatrix::Zero(opA.rows(), n);
    upload_bsr(opA);
    spmm_host_cached(rhs, Y, opA.rows(), opA.cols(), Scalar(1), Scalar(0), CUSPARSE_OPERATION_NON_TRANSPOSE);
    return Y;
  }
#endif  // EIGEN_HAS_CUSPARSE_BSR

  // y = alpha * op(D) * x + beta * y against the cached upload D, staging the
  // host vectors through d_x_ / d_y_. x_size and y_size are the column and row
  // counts of op(D).
  template <typename RhsDerived, typename DestDerived>
  void spmv_host_cached(const RhsDerived& x, DestDerived& y, Index x_size, Index y_size, Scalar alpha, Scalar beta,
                        cusparseOperation_t op) {
    ensure_buffer(d_x_, static_cast<size_t>(x_size) * sizeof(Scalar));
    // Ref binds in place when x is already a contiguous vector; only genuine
    // expressions are evaluated into the Ref's internal temporary.
    const Ref<const DenseVector> x_ref(x);
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_x_.get(), x_ref.data(), x_size * sizeof(Scalar), cudaMemcpyHostToDevice, stream_));

    ensure_buffer(d_y_, static_cast<size_t>(y_size) * sizeof(Scalar));
    if (beta != Scalar(0)) {
      const Ref<const DenseVector> y_ref(y);
      EIGEN_CUDA_RUNTIME_CHECK(
          cudaMemcpyAsync(d_y_.get(), y_ref.data(), y_size * sizeof(Scalar), cudaMemcpyHostToDevice, stream_));
    }

    exec_spmv(x_size, y_size, d_x_.get(), d_y_.get(), alpha, beta, op);

    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(y.data(), d_y_.get(), y_size * sizeof(Scalar), cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
  }

 public:
  /** Execute SpMV using the already-uploaded sparse matrix (no re-upload).
   * Used by SpMVExpr (d_y = d_A * d_x) for cached deviceView() paths.
   * The sparse matrix must have been uploaded via deviceView() or multiply().
   * A BSR upload describes op(A) as bound at upload time and cuSPARSE runs no
   * transposed BSR product, so \p op must then be GpuOp::NoTrans. */
  void spmv_device_exec(const DeviceMatrix<Scalar>& d_x, DeviceMatrix<Scalar>& d_y, Scalar alpha = Scalar(1),
                        Scalar beta = Scalar(0), GpuOp op = GpuOp::NoTrans) const {
    eigen_assert(spmat_desc_ && "sparse matrix not uploaded — call deviceView() or multiply() first");
    check_op_against_upload(op);
    // cuSPARSE SpMV: y must not alias x (undefined behavior).
    eigen_assert(d_x.data() != d_y.data() && "SpMV: output aliases input vector");

    const cusparseOperation_t cu_op = internal::to_cusparse_op<Scalar>(op);
    const Index m = cached_rows_;
    const Index n = cached_cols_;
    const Index x_size = (cu_op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? n : m;
    const Index y_size = (cu_op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? m : n;

    eigen_assert(d_x.rows() * d_x.cols() == x_size);

    if (m == 0 || n == 0 || cached_nnz_ == 0) {
      // Empty A reduces SpMV to y <- beta*y; SparseContext owns no cuBLAS
      // handle for the scale, so the beta != 0 case must be handled by the caller.
      eigen_assert(beta == Scalar(0) && "SpMV with empty A and beta != 0 is unsupported; scale d_y externally");
      if (d_y.rows() * d_y.cols() != y_size) d_y.resize(y_size, 1);
      d_y.setZero(stream_);
      return;
    }

    // Ensure d_y is allocated.
    if (d_y.rows() * d_y.cols() != y_size) {
      d_y.resize(y_size, 1);
    }

    // Wait for input data to be ready on this stream.
    d_x.waitReady(stream_);
    d_y.waitReady(stream_);

    exec_spmv(x_size, y_size, const_cast<void*>(static_cast<const void*>(d_x.data())), static_cast<void*>(d_y.data()),
              alpha, beta, cu_op);

    d_y.recordReady(stream_);
  }

  /** Execute SpMM (d_Y = alpha * op(A) * d_X + beta * d_Y) using the
   * already-uploaded sparse matrix. d_X / d_Y are device-resident dense
   * column-major matrices. Used by SpMVExpr when the RHS has > 1 column.
   * As for spmv_device_exec(), a BSR upload requires \p op == GpuOp::NoTrans. */
  void spmm_device_exec(const DeviceMatrix<Scalar>& d_X, DeviceMatrix<Scalar>& d_Y, Scalar alpha = Scalar(1),
                        Scalar beta = Scalar(0), GpuOp op = GpuOp::NoTrans) const {
    eigen_assert(spmat_desc_ && "sparse matrix not uploaded — call deviceView() or multiply() first");
    check_op_against_upload(op);
    eigen_assert(d_X.data() != d_Y.data() && "SpMM: output aliases input matrix");

    const cusparseOperation_t cu_op = internal::to_cusparse_op<Scalar>(op);
    const bool transposed = (cu_op != CUSPARSE_OPERATION_NON_TRANSPOSE);
    const Index m_op = transposed ? cached_cols_ : cached_rows_;
    const Index k_op = transposed ? cached_rows_ : cached_cols_;
    const Index n = d_X.cols();

    eigen_assert(d_X.rows() == k_op);

    if (m_op == 0 || n == 0 || cached_nnz_ == 0) {
      eigen_assert(beta == Scalar(0) && "SpMM with empty A and beta != 0 is unsupported; scale d_Y externally");
      if (d_Y.rows() != m_op || d_Y.cols() != n) d_Y.resize(m_op, n);
      d_Y.setZero(stream_);
      return;
    }

    if (d_Y.rows() != m_op || d_Y.cols() != n) {
      d_Y.resize(m_op, n);
    }

    d_X.waitReady(stream_);
    d_Y.waitReady(stream_);

    exec_spmm(m_op, k_op, n, const_cast<void*>(static_cast<const void*>(d_X.data())), static_cast<void*>(d_Y.data()),
              alpha, beta, cu_op);

    d_Y.recordReady(stream_);
  }

 private:
  // cuSPARSE 11.x's cusparseSpMM rejects CSC for matA (CSC support landed in
  // CUDA 12.0). On 11.x we register the same buffers as CSR-of-A^T (dims
  // swapped) and invert the user-facing op before each cuSPARSE call. On 12+
  // we keep the natural CSC path so users pay no extra cost.
#if !defined(CUSPARSE_VERSION) || CUSPARSE_VERSION < 12000
  static constexpr bool kUseCsrOfTranspose = true;
  static constexpr cusparseSpMMAlg_t kSpMMAlg = CUSPARSE_SPMM_CSR_ALG2;
#else
  static constexpr bool kUseCsrOfTranspose = false;
  static constexpr cusparseSpMMAlg_t kSpMMAlg = CUSPARSE_SPMM_ALG_DEFAULT;
#endif

  // Map a user-facing op on A to the cuSPARSE op on the cached descriptor.
  // Identity on cuSPARSE 12+ (descriptor is CSC of A, or BSR of op(A));
  // inverted on 11.x (descriptor is CSR of A^T).
  static cusparseOperation_t descriptor_op(cusparseOperation_t user_op) {
    EIGEN_IF_CONSTEXPR (!kUseCsrOfTranspose) return user_op;
    switch (user_op) {
      case CUSPARSE_OPERATION_NON_TRANSPOSE:
        return CUSPARSE_OPERATION_TRANSPOSE;
      case CUSPARSE_OPERATION_TRANSPOSE:
        return CUSPARSE_OPERATION_NON_TRANSPOSE;
      default:
        // CONJUGATE_TRANSPOSE on the CSR-of-A^T descriptor would compute
        // conj(A) * x, not A^H * x — not supported via this representation.
        eigen_assert(false && "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not supported on cuSPARSE < 12.0");
        return user_op;
    }
  }

  static int op_index(cusparseOperation_t op) {
    switch (op) {
      case CUSPARSE_OPERATION_TRANSPOSE:
        return 1;
      case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return 2;
      default:
        return 0;
    }
  }

  void invalidate_ws_caches() const {
    for (int i = 0; i < 3; ++i) {
      spmv_ws_size_[i] = kWsUnknown;
      spmm_ws_size_[i] = kWsUnknown;
    }
  }

  // Recreate the descriptor when the size changes; otherwise just re-point it
  // at the new device buffer (cusparseDnVecSetValues is a host-side pointer
  // update, no GPU work).
  void update_dnvec(cusparseDnVecDescr_t& desc, int64_t& cur_size, int64_t size, void* ptr) const {
    if (!desc || cur_size != size) {
      if (desc) EIGEN_CUSPARSE_CHECK(cusparseDestroyDnVec(desc));
      EIGEN_CUSPARSE_CHECK(cusparseCreateDnVec(&desc, size, ptr, kValueType));
      cur_size = size;
      invalidate_ws_caches();
    } else {
      EIGEN_CUSPARSE_CHECK(cusparseDnVecSetValues(desc, ptr));
    }
  }

  void update_dnmat(cusparseDnMatDescr_t& desc, int64_t& cur_rows, int64_t& cur_cols, int64_t rows, int64_t cols,
                    void* ptr) const {
    if (!desc || cur_rows != rows || cur_cols != cols) {
      if (desc) EIGEN_CUSPARSE_CHECK(cusparseDestroyDnMat(desc));
      // Column-major with ld = rows.
      EIGEN_CUSPARSE_CHECK(cusparseCreateDnMat(&desc, rows, cols, rows, ptr, kValueType, CUSPARSE_ORDER_COL));
      cur_rows = rows;
      cur_cols = cols;
      invalidate_ws_caches();
    } else {
      EIGEN_CUSPARSE_CHECK(cusparseDnMatSetValues(desc, ptr));
    }
  }

  void destroy_dense_descriptors() {
    if (x_vec_desc_) (void)cusparseDestroyDnVec(x_vec_desc_);
    if (y_vec_desc_) (void)cusparseDestroyDnVec(y_vec_desc_);
    if (x_mat_desc_) (void)cusparseDestroyDnMat(x_mat_desc_);
    if (y_mat_desc_) (void)cusparseDestroyDnMat(y_mat_desc_);
    x_vec_desc_ = y_vec_desc_ = nullptr;
    x_mat_desc_ = y_mat_desc_ = nullptr;
  }

  void exec_spmv(Index x_size, Index y_size, void* d_x_ptr, void* d_y_ptr, Scalar alpha, Scalar beta,
                 cusparseOperation_t op) const {
    const cusparseOperation_t cu_op = descriptor_op(op);
    update_dnvec(x_vec_desc_, x_vec_size_, x_size, d_x_ptr);
    update_dnvec(y_vec_desc_, y_vec_size_, y_size, d_y_ptr);

    size_t& ws_size = spmv_ws_size_[op_index(cu_op)];
    if (ws_size == kWsUnknown) {
      EIGEN_CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle_, cu_op, &alpha, spmat_desc_, x_vec_desc_, &beta, y_vec_desc_,
                                                   kValueType, CUSPARSE_SPMV_ALG_DEFAULT, &ws_size));
    }
    ensure_buffer(d_workspace_, ws_size);

    EIGEN_CUSPARSE_CHECK(cusparseSpMV(handle_, cu_op, &alpha, spmat_desc_, x_vec_desc_, &beta, y_vec_desc_, kValueType,
                                      CUSPARSE_SPMV_ALG_DEFAULT, d_workspace_.get()));
  }

  void exec_spmm(Index m_op, Index k_op, Index n, void* d_x_ptr, void* d_y_ptr, Scalar alpha, Scalar beta,
                 cusparseOperation_t op) const {
    const cusparseOperation_t cu_op = descriptor_op(op);
    // X is k_op x n, Y is m_op x n (column-major, post-op shapes).
    update_dnmat(x_mat_desc_, x_mat_rows_, x_mat_cols_, k_op, n, d_x_ptr);
    update_dnmat(y_mat_desc_, y_mat_rows_, y_mat_cols_, m_op, n, d_y_ptr);

    size_t& ws_size = spmm_ws_size_[op_index(cu_op)];
    if (ws_size == kWsUnknown) {
      EIGEN_CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle_, cu_op, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                                   spmat_desc_, x_mat_desc_, &beta, y_mat_desc_, kValueType, kSpMMAlg,
                                                   &ws_size));
    }
    ensure_buffer(d_workspace_, ws_size);

    EIGEN_CUSPARSE_CHECK(cusparseSpMM(handle_, cu_op, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmat_desc_,
                                      x_mat_desc_, &beta, y_mat_desc_, kValueType, kSpMMAlg, d_workspace_.get()));
  }

  void spmm_impl(const SpMat& A, const DenseMatrix& X, DenseMatrix& Y, Scalar alpha, Scalar beta,
                 cusparseOperation_t op) {
    eigen_assert(A.isCompressed());

    // For op != NON_TRANSPOSE, Y = op(A) * X. The dense X / Y descriptors must
    // describe the *post-op* shapes: X has k_op rows (= input dim of op(A)),
    // Y has m_op rows (= output dim of op(A)).
    const bool transposed = (op != CUSPARSE_OPERATION_NON_TRANSPOSE);
    const Index m_op = transposed ? A.cols() : A.rows();
    const Index k_op = transposed ? A.rows() : A.cols();
    const Index n = X.cols();
    const Index nnz = A.nonZeros();

    if (m_op == 0 || n == 0 || k_op == 0 || nnz == 0) {
      scale_host(Y, beta);
      return;
    }

    upload_sparse(A);
    spmm_host_cached(X, Y, m_op, k_op, alpha, beta, op);
  }

  // Y = alpha * op(D) * X + beta * Y against the cached upload D, staging the
  // host matrices through d_x_ / d_y_. X is k_op x n, Y is m_op x n.
  void spmm_host_cached(const DenseMatrix& X, DenseMatrix& Y, Index m_op, Index k_op, Scalar alpha, Scalar beta,
                        cusparseOperation_t op) {
    const Index n = X.cols();
    const size_t x_bytes = static_cast<size_t>(k_op) * static_cast<size_t>(n) * sizeof(Scalar);
    const size_t y_bytes = static_cast<size_t>(m_op) * static_cast<size_t>(n) * sizeof(Scalar);
    ensure_buffer(d_x_, x_bytes);
    ensure_buffer(d_y_, y_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_x_.get(), X.data(), x_bytes, cudaMemcpyHostToDevice, stream_));
    if (beta != Scalar(0)) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_y_.get(), Y.data(), y_bytes, cudaMemcpyHostToDevice, stream_));
    }

    exec_spmm(m_op, k_op, n, d_x_.get(), d_y_.get(), alpha, beta, op);

    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(Y.data(), d_y_.get(), y_bytes, cudaMemcpyDeviceToHost, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
  }

  void upload_sparse(const SpMat& A) {
    // cuSPARSE 12.0+ accepts CSC directly. On cuSPARSE 11.x, cusparseSpMM
    // rejects CSC and CONJUGATE_TRANSPOSE on CSC+complex SpMV silently
    // demotes to TRANSPOSE. We register the same CSC buffers as CSR-of-A^T
    // (dims swapped) on 11.x and invert the op at exec time via
    // descriptor_op() — no transpose-copy required.
    const Index m = A.rows();
    const Index n = A.cols();
    const Index nnz = A.nonZeros();
    upload_arrays(/*outer_count=*/n + 1, /*inner_count=*/nnz, /*value_count=*/nnz, A.outerIndexPtr(), A.innerIndexPtr(),
                  A.valuePtr());
    if (descriptor_key_matches(m, n, nnz, /*block_size=*/0)) return;

    destroy_spmat_descriptor(/*checked=*/true);
    EIGEN_IF_CONSTEXPR (kUseCsrOfTranspose) {
      // cuSPARSE 11.x: cusparseSpMM rejects CSC for matA. CSC of A and CSR of
      // A^T share the same buffers, so register the data as CSR-of-A^T (dims
      // swapped) and invert the op in exec_spmv / spmm_impl via descriptor_op.
      EIGEN_CUSPARSE_CHECK(cusparseCreateCsr(&spmat_desc_, n, m, nnz, d_outerPtr_.get(), d_innerIdx_.get(),
                                             d_values_.get(), kIndexType, kIndexType, CUSPARSE_INDEX_BASE_ZERO,
                                             kValueType));
    } else {
      EIGEN_CUSPARSE_CHECK(cusparseCreateCsc(&spmat_desc_, m, n, nnz, d_outerPtr_.get(), d_innerIdx_.get(),
                                             d_values_.get(), kIndexType, kIndexType, CUSPARSE_INDEX_BASE_ZERO,
                                             kValueType));
    }
    set_descriptor_key(m, n, nnz, /*block_size=*/0);
  }

#if EIGEN_HAS_CUSPARSE_BSR
  void upload_bsr(const internal::BsrArrays<Scalar, StorageIndex>& opA) {
    upload_arrays(/*outer_count=*/opA.brows + 1, /*inner_count=*/opA.bnnz, /*value_count=*/opA.nonZeros(), opA.outer,
                  opA.inner, opA.values);
    if (descriptor_key_matches(opA.rows(), opA.cols(), opA.nonZeros(), opA.blockSize)) return;

    destroy_spmat_descriptor(/*checked=*/true);
    // Row-major blocks: cusparseSpMM accepts no other block layout for BSR.
    EIGEN_CUSPARSE_CHECK(cusparseCreateBsr(&spmat_desc_, opA.brows, opA.bcols, opA.bnnz, opA.blockSize, opA.blockSize,
                                           d_outerPtr_.get(), d_innerIdx_.get(), d_values_.get(), kIndexType,
                                           kIndexType, CUSPARSE_INDEX_BASE_ZERO, kValueType, CUSPARSE_ORDER_ROW));
    set_descriptor_key(opA.rows(), opA.cols(), opA.nonZeros(), opA.blockSize);
  }
#endif  // EIGEN_HAS_CUSPARSE_BSR

  void upload_arrays(Index outer_count, Index inner_count, Index value_count, const StorageIndex* host_outer,
                     const StorageIndex* host_inner, const Scalar* host_values) {
    const size_t outer_bytes = static_cast<size_t>(outer_count) * sizeof(StorageIndex);
    const size_t inner_bytes = static_cast<size_t>(inner_count) * sizeof(StorageIndex);
    const size_t val_bytes = static_cast<size_t>(value_count) * sizeof(Scalar);

    // Values *and* index arrays are re-uploaded unconditionally: host pointer
    // identity cannot detect a same-shape/same-nnz pattern rewritten in place
    // or assigned into the same allocations (SparseMatrix reuses them), so a
    // structure cache keyed on pointers would silently serve stale indices.
    // Only the cuSPARSE descriptor and the workspace-size queries are cached,
    // keyed on (rows, cols, nnz, block size). Every upload invalidates
    // outstanding DeviceSparseViews via the generation counter.
    ++generation_;
    ensure_buffer(d_values_, val_bytes);
    ensure_buffer(d_outerPtr_, outer_bytes);
    ensure_buffer(d_innerIdx_, inner_bytes);
    EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(d_values_.get(), host_values, val_bytes, cudaMemcpyHostToDevice, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_outerPtr_.get(), host_outer, outer_bytes, cudaMemcpyHostToDevice, stream_));
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(d_innerIdx_.get(), host_inner, inner_bytes, cudaMemcpyHostToDevice, stream_));
  }

  // Same shape, nnz and format: the grow-only device buffers cannot have been
  // reallocated, so the existing descriptor still points at the freshly
  // written data.
  bool descriptor_key_matches(Index m, Index n, Index nnz, Index block_size) const {
    return m == cached_rows_ && n == cached_cols_ && nnz == cached_nnz_ && block_size == cached_block_size_;
  }

  void set_descriptor_key(Index m, Index n, Index nnz, Index block_size) {
    cached_rows_ = m;
    cached_cols_ = n;
    cached_nnz_ = nnz;
    cached_block_size_ = block_size;
  }

  // A BSR descriptor holds op(A) as bound at upload time, and cuSPARSE runs no
  // transposed BSR product.
  void check_op_against_upload(GpuOp op) const {
    eigen_assert((cached_block_size_ == 0 || op == GpuOp::NoTrans) &&
                 "cuSPARSE runs BSR products only with op == NoTrans; pass op to multiply(A, d_x, d_y, ...) instead");
    EIGEN_UNUSED_VARIABLE(op);
  }

  // Destroy the sparse-matrix descriptor and reset the cache identity.
  // `checked` selects assert-on-failure (mid-lifetime rebuilds) vs swallow
  // (noexcept destructor).
  void destroy_spmat_descriptor(bool checked) {
    if (spmat_desc_) {
      cusparseStatus_t s = cusparseDestroySpMat(spmat_desc_);
      eigen_assert((!checked || s == CUSPARSE_STATUS_SUCCESS) && "cusparseDestroySpMat failed");
      EIGEN_UNUSED_VARIABLE(s);
      EIGEN_UNUSED_VARIABLE(checked);
      spmat_desc_ = nullptr;
    }
    set_descriptor_key(-1, -1, -1, 0);
    invalidate_ws_caches();
  }

  void ensure_buffer(internal::DeviceBuffer& buf, size_t needed) const {
    if (needed > buf.size()) {
      if (buf) EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      buf = internal::DeviceBuffer(needed);
    }
  }
};

// Defined here because it needs the full SparseContext definition.

template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const SpMVExpr<Scalar_>& expr) {
  // Uses the sparse matrix already uploaded by deviceView() — no re-upload on
  // repeated products with the same view. A stale view (the context has since
  // uploaded again, replacing the cached data) is caught here.
  eigen_assert(expr.view().generation() == expr.view().context().uploadGeneration() &&
               "DeviceSparseView is stale: its SparseContext has since uploaded another sparse matrix");
  if (expr.x().cols() <= 1) {
    expr.view().context().spmv_device_exec(expr.x(), *this, Scalar_(1), Scalar_(0), GpuOp::NoTrans);
  } else {
    expr.view().context().spmm_device_exec(expr.x(), *this, Scalar_(1), Scalar_(0), GpuOp::NoTrans);
  }
  return *this;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>& DeviceMatrix<Scalar_>::operator=(const SpMVAffineExpr<Scalar_>& expr) {
  const DeviceSparseView<Scalar_>& view = expr.view();
  const DeviceMatrix& addend = expr.addend();
  eigen_assert(view.generation() == view.context().uploadGeneration() &&
               "DeviceSparseView is stale: its SparseContext has since uploaded another sparse matrix");
  eigen_assert(addend.rows() == view.rows() && addend.cols() == expr.x().cols() &&
               "SpMVAffineExpr: the addend must have the shape of the product");
  eigen_assert(&expr.x() != this && "SpMVAffineExpr: the destination aliases the dense operand");
  // The product accumulates into a copy of the addend; when the addend is the destination
  // itself (d_r = d_r - d_A * d_x) cuSPARSE's beta accumulates in place.
  if (&addend != this) copyFrom(Context::threadLocal(), addend);
  // With no stored entries the expression is beta * addend, which spmv_device_exec cannot
  // form: SparseContext owns no cuBLAS handle to scale d_y by beta.
  if (view.nonZeros() == 0 || view.rows() == 0 || view.cols() == 0) {
    if (expr.beta() != Scalar_(1)) scale(Context::threadLocal(), expr.beta());
    return *this;
  }
  if (expr.x().cols() <= 1) {
    view.context().spmv_device_exec(expr.x(), *this, expr.alpha(), expr.beta(), GpuOp::NoTrans);
  } else {
    view.context().spmm_device_exec(expr.x(), *this, expr.alpha(), expr.beta(), GpuOp::NoTrans);
  }
  return *this;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(const SpMVExpr<Scalar_>& expr) : DeviceMatrix() {
  *this = expr;
}

template <typename Scalar_>
DeviceMatrix<Scalar_>::DeviceMatrix(const SpMVAffineExpr<Scalar_>& expr) : DeviceMatrix() {
  *this = expr;
}
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SPARSE_CONTEXT_H
