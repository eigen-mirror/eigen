// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Typed RAII wrapper for a dense matrix in GPU device memory.
//
// DeviceMatrix<Scalar> holds a column-major matrix on the GPU with tracked
// dimensions. Always dense (leading dimension = rows). It can be passed to GPU solvers
// (GpuLLT, GpuLU, future cuBLAS/cuDSS) without host round-trips.
//
// Cross-stream safety is automatic: an internal CUDA event tracks when the
// last write completed. Consumers on a different stream wait on that event
// before reading.
//
// Usage:
//   auto d_A = DeviceMatrix<double>::fromHost(A);   // upload (sync)
//   GpuLLT<double> llt;
//   llt.compute(d_A);                                // factor on device
//   auto d_X = llt.solve(d_B);                       // async, no sync
//   MatrixXd X = d_X.toHost();                       // download + block
//
// Async variants:
//   auto d_A = DeviceMatrix<double>::fromHostAsync(A.data(), n, n, stream);
//   auto transfer = d_X.toHostAsync(stream);         // enqueue D2H
//   // ... overlap with other work ...
//   MatrixXd X = transfer.get();                     // block + retrieve

#ifndef EIGEN_GPU_DEVICE_MATRIX_H
#define EIGEN_GPU_DEVICE_MATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSupport.h"

namespace Eigen {

// Forward declarations.
template <typename, int>
class GpuLLT;
template <typename>
class GpuLU;
template <typename>
class DeviceAdjointView;
template <typename>
class DeviceTransposeView;
template <typename>
class DeviceAssignment;
template <typename, typename>
class GemmExpr;
template <typename>
class DeviceScaled;
template <typename>
class SpMVExpr;
template <typename>
class DeviceAddExpr;
template <typename>
class DeviceScaledDevice;
template <typename>
class DeviceScalar;
template <typename, int>
class LltSolveExpr;
template <typename>
class LuSolveExpr;
template <typename, int>
class DeviceLLTView;
template <typename>
class DeviceLUView;
template <typename, int>
class DeviceTriangularView;
template <typename, int>
class DeviceSelfAdjointView;
template <typename, int>
class ConstDeviceSelfAdjointView;
template <typename, int>
class TrsmExpr;
template <typename, int>
class SymmExpr;
template <typename, int>
class SyrkExpr;
class GpuContext;

// --------------------------------------------------------------------------
// HostTransfer — future-like wrapper for an async device-to-host transfer.
// --------------------------------------------------------------------------

/** \ingroup GPU_Module
 * \class HostTransfer
 * \brief Future for an asynchronous device-to-host matrix transfer.
 *
 * Returned by DeviceMatrix::toHostAsync(). The transfer runs asynchronously
 * on the given CUDA stream. Call get() to block until complete and retrieve
 * the host matrix, or ready() to poll without blocking.
 */
template <typename Scalar_>
class HostTransfer {
 public:
  using Scalar = Scalar_;
  using PlainMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  /** Block until the transfer completes and return the host matrix.
   * Idempotent: subsequent calls return the same matrix without re-syncing. */
  PlainMatrix& get() {
    if (!synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaEventSynchronize(event_));
      synced_ = true;
    }
    return host_buf_;
  }

  /** Non-blocking check: has the transfer completed? */
  bool ready() const {
    if (synced_) return true;
    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) return true;
    eigen_assert(err == cudaErrorNotReady && "cudaEventQuery failed");
    return false;
  }

  ~HostTransfer() {
    if (event_) (void)cudaEventDestroy(event_);
  }

  HostTransfer(HostTransfer&& o) noexcept : host_buf_(std::move(o.host_buf_)), event_(o.event_), synced_(o.synced_) {
    o.event_ = nullptr;
    o.synced_ = true;
  }

  HostTransfer& operator=(HostTransfer&& o) noexcept {
    if (this != &o) {
      if (event_) (void)cudaEventDestroy(event_);
      host_buf_ = std::move(o.host_buf_);
      event_ = o.event_;
      synced_ = o.synced_;
      o.event_ = nullptr;
      o.synced_ = true;
    }
    return *this;
  }

  HostTransfer(const HostTransfer&) = delete;
  HostTransfer& operator=(const HostTransfer&) = delete;

 private:
  template <typename>
  friend class DeviceMatrix;

  HostTransfer(PlainMatrix&& buf, cudaEvent_t event) : host_buf_(std::move(buf)), event_(event), synced_(false) {}

  PlainMatrix host_buf_;
  cudaEvent_t event_ = nullptr;
  bool synced_ = false;
};

// --------------------------------------------------------------------------
// DeviceMatrix — typed RAII wrapper for a dense matrix in device memory.
// --------------------------------------------------------------------------

/** \ingroup GPU_Module
 * \class DeviceMatrix
 * \brief RAII wrapper for a dense column-major matrix in GPU device memory.
 *
 * \tparam Scalar_  Element type: float, double, complex<float>, complex<double>
 *
 * Owns a device allocation with tracked dimensions. Always dense
 * (leading dimension = rows; no stride padding).
 * An internal CUDA event records when the data was last written, enabling
 * safe cross-stream consumption without user-visible synchronization.
 *
 * Each method has a synchronous and an asynchronous variant:
 *  - fromHost() / fromHostAsync(): upload from host
 *  - toHost() / toHostAsync(): download to host
 */
template <typename Scalar_>
class DeviceMatrix {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using PlainObject = DeviceMatrix;  // owning type (for CG template compatibility)
  using PlainMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  // ---- Construction / destruction ------------------------------------------

  /** Default: empty (0x0, no allocation). */
  DeviceMatrix() = default;

  /** Allocate uninitialized column vector of given size.
   * Matches Matrix<Scalar,Dynamic,1>(n) for CG template compatibility. */
  explicit DeviceMatrix(Index n) : rows_(n), cols_(1) {
    eigen_assert(n >= 0);
    size_t bytes = sizeInBytes();
    if (bytes > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
    }
  }

  /** Allocate uninitialized device memory for a rows x cols matrix. */
  DeviceMatrix(Index rows, Index cols) : rows_(rows), cols_(cols) {
    eigen_assert(rows >= 0 && cols >= 0);
    size_t bytes = sizeInBytes();
    if (bytes > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
    }
  }

  ~DeviceMatrix() {
    if (data_) (void)cudaFree(data_);
    if (ready_event_) (void)cudaEventDestroy(ready_event_);
  }

  // ---- Move-only -----------------------------------------------------------

  DeviceMatrix(DeviceMatrix&& o) noexcept
      : data_(o.data_),
        rows_(o.rows_),
        cols_(o.cols_),
        ready_event_(o.ready_event_),
        ready_stream_(o.ready_stream_),
        retained_buffer_(std::move(o.retained_buffer_)) {
    o.data_ = nullptr;
    o.rows_ = 0;
    o.cols_ = 0;
    o.ready_event_ = nullptr;
    o.ready_stream_ = nullptr;
  }

  DeviceMatrix& operator=(DeviceMatrix&& o) noexcept {
    if (this != &o) {
      if (data_) (void)cudaFree(data_);
      if (ready_event_) (void)cudaEventDestroy(ready_event_);
      data_ = o.data_;
      rows_ = o.rows_;
      cols_ = o.cols_;
      ready_event_ = o.ready_event_;
      ready_stream_ = o.ready_stream_;
      retained_buffer_ = std::move(o.retained_buffer_);
      o.data_ = nullptr;
      o.rows_ = 0;
      o.cols_ = 0;
      o.ready_event_ = nullptr;
      o.ready_stream_ = nullptr;
    }
    return *this;
  }

  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;

  // ---- Upload from host ----------------------------------------------------

  /** Upload a host Eigen matrix to device memory (synchronous).
   *
   * Evaluates the expression into a contiguous ColMajor temporary, copies to
   * device via cudaMemcpyAsync on \p stream, and synchronizes before returning.
   *
   * \param host   Any Eigen matrix expression.
   * \param stream CUDA stream for the transfer (default: stream 0).
   */
  template <typename Derived>
  static DeviceMatrix fromHost(const MatrixBase<Derived>& host, cudaStream_t stream = nullptr) {
    const PlainMatrix mat(host.derived());
    DeviceMatrix dm(mat.rows(), mat.cols());
    if (dm.sizeInBytes() > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dm.data_, mat.data(), dm.sizeInBytes(), cudaMemcpyHostToDevice, stream));
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    }
    return dm;
  }

  /** Upload from a raw host pointer to device memory (asynchronous).
   *
   * Enqueues an async H2D copy on \p stream and records an internal event.
   * The caller must keep \p host_data alive until the transfer completes
   * (check via the internal event or synchronize the stream).
   *
   * \param host_data  Pointer to contiguous column-major host data.
   * \param rows       Number of rows.
   * \param cols       Number of columns.
   * \param stream     CUDA stream for the transfer.
   */
  static DeviceMatrix fromHostAsync(const Scalar* host_data, Index rows, Index cols, cudaStream_t stream) {
    eigen_assert(rows >= 0 && cols >= 0);
    eigen_assert(host_data != nullptr || (rows == 0 || cols == 0));
    DeviceMatrix dm(rows, cols);
    if (dm.sizeInBytes() > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dm.data_, host_data, dm.sizeInBytes(), cudaMemcpyHostToDevice, stream));
      dm.recordReady(stream);
    }
    return dm;
  }

  // ---- Download to host ----------------------------------------------------

  /** Download device matrix to host memory (synchronous).
   *
   * Waits on the internal ready event, enqueues a D2H copy on \p stream,
   * synchronizes, and returns the host matrix directly.
   *
   * \param stream CUDA stream for the transfer (default: stream 0).
   */
  PlainMatrix toHost(cudaStream_t stream = nullptr) const {
    PlainMatrix host_buf(rows_, cols_);
    if (sizeInBytes() > 0) {
      waitReady(stream);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(host_buf.data(), data_, sizeInBytes(), cudaMemcpyDeviceToHost, stream));
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    }
    return host_buf;
  }

  /** Enqueue an async device-to-host transfer and return a future.
   *
   * Waits on the internal ready event (if any) to ensure the device data is
   * valid, then enqueues the D2H copy on \p stream. Returns a HostTransfer
   * future; call .get() to block and retrieve the host matrix.
   *
   * \param stream CUDA stream for the transfer (default: stream 0).
   */
  HostTransfer<Scalar> toHostAsync(cudaStream_t stream = nullptr) const {
    PlainMatrix host_buf(rows_, cols_);
    if (sizeInBytes() > 0) {
      waitReady(stream);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(host_buf.data(), data_, sizeInBytes(), cudaMemcpyDeviceToHost, stream));
    }
    // Record a transfer-complete event.
    cudaEvent_t transfer_event;
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&transfer_event, cudaEventDisableTiming));
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(transfer_event, stream));
    return HostTransfer<Scalar>(std::move(host_buf), transfer_event);
  }

  // ---- Device-to-device copy -----------------------------------------------

  /** Deep copy on device. Fully async — records event on the result, no sync.
   *
   * \param stream CUDA stream for the D2D copy (default: stream 0).
   */
  DeviceMatrix clone(cudaStream_t stream = nullptr) const {
    DeviceMatrix result(rows_, cols_);
    if (sizeInBytes() > 0) {
      waitReady(stream);
      EIGEN_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(result.data_, data_, sizeInBytes(), cudaMemcpyDeviceToDevice, stream));
      result.recordReady(stream);
    }
    return result;
  }

  // ---- Resize (destructive) ------------------------------------------------

  /** Discard contents and reallocate to (rows x cols). Clears the ready event. */
  void resize(Index rows, Index cols) {
    if (rows == rows_ && cols == cols_) return;
    if (data_) {
      (void)cudaFree(data_);
      data_ = nullptr;
    }
    if (ready_event_) {
      (void)cudaEventDestroy(ready_event_);
      ready_event_ = nullptr;
    }
    ready_stream_ = nullptr;
    retained_buffer_ = internal::DeviceBuffer();
    rows_ = rows;
    cols_ = cols;
    size_t bytes = sizeInBytes();
    if (bytes > 0) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
    }
  }

  // ---- Accessors -----------------------------------------------------------

  Scalar* data() { return data_; }
  const Scalar* data() const { return data_; }
  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  bool empty() const { return rows_ == 0 || cols_ == 0; }

  /** Size of the device allocation in bytes. */
  size_t sizeInBytes() const { return static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * sizeof(Scalar); }

  // ---- Event synchronization (public for library dispatch interop) ---------

  /** Record that device data is ready after work on \p stream. */
  void recordReady(cudaStream_t stream) {
    ensureEvent();
    EIGEN_CUDA_RUNTIME_CHECK(cudaEventRecord(ready_event_, stream));
    ready_stream_ = stream;
  }

  /** Make \p stream wait until the device data is ready.
   * No-op if no event recorded, or if the consumer stream is the same as the
   * producer stream (CUDA guarantees in-order execution within a stream). */
  void waitReady(cudaStream_t stream) const {
    if (ready_event_ && stream != ready_stream_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(stream, ready_event_, 0));
    }
  }

  // ---- Expression methods (dispatch to cuBLAS/cuSOLVER) --------------------

  /** Adjoint view for GEMM dispatch. Maps to cublasXgemm with ConjTrans. */
  DeviceAdjointView<Scalar> adjoint() const { return DeviceAdjointView<Scalar>(*this); }

  /** Transpose view for GEMM dispatch. Maps to cublasXgemm with Trans. */
  DeviceTransposeView<Scalar> transpose() const { return DeviceTransposeView<Scalar>(*this); }

  /** Bind this matrix to a GpuContext for expression assignment.
   * Returns a DeviceAssignment proxy: `d_C.device(ctx) = d_A * d_B;` */
  DeviceAssignment<Scalar> device(GpuContext& ctx) { return DeviceAssignment<Scalar>(*this, ctx); }

  /** Assign from a GEMM expression using the thread-local default GpuContext.
   * Defined out-of-line after GpuContext is fully declared (see DeviceDispatch.h). */
  template <typename Lhs, typename Rhs>
  DeviceMatrix& operator=(const GemmExpr<Lhs, Rhs>& expr);

  /** Accumulate from a GEMM expression using the thread-local default GpuContext. */
  template <typename Lhs, typename Rhs>
  DeviceMatrix& operator+=(const GemmExpr<Lhs, Rhs>& expr);

  /** Cholesky view: d_A.llt().solve(d_B) → LltSolveExpr. */
  DeviceLLTView<Scalar, Lower> llt() const { return DeviceLLTView<Scalar, Lower>(*this); }

  /** Cholesky view with explicit triangle: d_A.llt<Upper>().solve(d_B). */
  template <int UpLo>
  DeviceLLTView<Scalar, UpLo> llt() const {
    return DeviceLLTView<Scalar, UpLo>(*this);
  }

  /** LU view: d_A.lu().solve(d_B) → LuSolveExpr. */
  DeviceLUView<Scalar> lu() const { return DeviceLUView<Scalar>(*this); }

  /** Assign from an LLT solve expression (thread-local default context). */
  template <int UpLo>
  DeviceMatrix& operator=(const LltSolveExpr<Scalar, UpLo>& expr);

  /** Assign from an LU solve expression (thread-local default context). */
  DeviceMatrix& operator=(const LuSolveExpr<Scalar>& expr);

  /** Triangular view: d_A.triangularView<Lower>().solve(d_B) → TrsmExpr. */
  template <int UpLo>
  DeviceTriangularView<Scalar, UpLo> triangularView() const {
    return DeviceTriangularView<Scalar, UpLo>(*this);
  }

  /** Self-adjoint view (mutable): d_C.selfadjointView<Lower>().rankUpdate(d_A). */
  template <int UpLo>
  DeviceSelfAdjointView<Scalar, UpLo> selfadjointView() {
    return DeviceSelfAdjointView<Scalar, UpLo>(*this);
  }

  /** Self-adjoint view (const): d_A.selfadjointView<Lower>() * d_B → SymmExpr. */
  template <int UpLo>
  ConstDeviceSelfAdjointView<Scalar, UpLo> selfadjointView() const {
    return ConstDeviceSelfAdjointView<Scalar, UpLo>(*this);
  }

  /** Assign from a TRSM expression (thread-local default context). */
  template <int UpLo>
  DeviceMatrix& operator=(const TrsmExpr<Scalar, UpLo>& expr);

  /** Assign from a SYMM expression (thread-local default context). */
  template <int UpLo>
  DeviceMatrix& operator=(const SymmExpr<Scalar, UpLo>& expr);

  // ---- BLAS Level-1 operations ----------------------------------------------
  // DeviceMatrix is always dense (lda == rows), so a vector is simply a
  // DeviceMatrix with cols == 1. These BLAS-1 methods operate on the flat
  // rows*cols element array, making them work for both vectors and matrices.
  //
  // All methods take an explicit GpuContext& for stream/handle control.
  // When everything uses the same context, event waits are skipped (same-stream).
  // Defined out-of-line in DeviceDispatch.h (needs GpuContext).

  /** Dot product: this^H * other. Returns DeviceScalar — the result stays
   * on device until read via implicit conversion to Scalar (which syncs).
   * When used with `auto`, no sync occurs until the value is needed. */
  DeviceScalar<Scalar> dot(GpuContext& ctx, const DeviceMatrix& other) const;

  /** Squared L2 norm via dot(x, x). Returns DeviceScalar (no sync until read).
   * For real types, the result stays on device. For complex types, falls back
   * to host sync (DeviceScalar arithmetic is real-only). */
  DeviceScalar<typename NumTraits<Scalar>::Real> squaredNorm(GpuContext& ctx) const;

  /** L2 norm. Returns DeviceScalar (no host sync). */
  DeviceScalar<typename NumTraits<Scalar>::Real> norm(GpuContext& ctx) const;

  /** Set all elements to zero. */
  void setZero(GpuContext& ctx);

  /** this += alpha * x (cuBLAS axpy). Requires same total size. */
  void addScaled(GpuContext& ctx, Scalar alpha, const DeviceMatrix& x);

  /** this *= alpha (cuBLAS scal). */
  void scale(GpuContext& ctx, Scalar alpha);

  /** Deep copy: this = other (cuBLAS copy). Resizes if needed. */
  void copyFrom(GpuContext& ctx, const DeviceMatrix& other);

  // Convenience overloads using the thread-local default GpuContext.
  DeviceScalar<Scalar> dot(const DeviceMatrix& other) const;
  DeviceScalar<typename NumTraits<Scalar>::Real> squaredNorm() const;
  DeviceScalar<typename NumTraits<Scalar>::Real> norm() const;
  void setZero();

  // ---- BLAS-1 operator overloads for CG/iterative solver compatibility ------
  // These allow CG code like `x += alpha * p` to work with DeviceMatrix.
  // `alpha * DeviceMatrix` already returns `DeviceScaled<DeviceMatrix<Scalar>>`
  // (defined in DeviceExpr.h). These operators dispatch to cuBLAS axpy/scal.
  // Defined out-of-line in DeviceDispatch.h.

  /** this += alpha * x (cuBLAS axpy). For `x += alpha * p`. */
  DeviceMatrix& operator+=(const DeviceScaled<DeviceMatrix>& expr);

  /** this -= alpha * x (cuBLAS axpy with negated alpha). For `r -= alpha * tmp`. */
  DeviceMatrix& operator-=(const DeviceScaled<DeviceMatrix>& expr);

  /** this += x (cuBLAS axpy with alpha=1). */
  DeviceMatrix& operator+=(const DeviceMatrix& other);

  /** this -= x (cuBLAS axpy with alpha=-1). */
  DeviceMatrix& operator-=(const DeviceMatrix& other);

  /** this *= alpha (cuBLAS scal, host pointer mode). For `p *= beta`. */
  DeviceMatrix& operator*=(Scalar alpha);

  /** this *= alpha (cuBLAS scal, device pointer mode). Avoids host sync. */
  DeviceMatrix& operator*=(const DeviceScalar<Scalar>& alpha);

  /** Element-wise product: result[i] = this[i] * other[i] (NPP nppsMul).
   * Returns a new DeviceMatrix. Defined out-of-line in DeviceDispatch.h. */
  DeviceMatrix cwiseProduct(GpuContext& ctx, const DeviceMatrix& other) const;

  /** In-place element-wise product: this[i] = a[i] * b[i] (NPP nppsMul).
   * Reuses this matrix's buffer when sizes match, avoiding cudaMalloc. */
  void cwiseProduct(GpuContext& ctx, const DeviceMatrix& a, const DeviceMatrix& b);

  /** this += DeviceScalar * x (cuBLAS axpy with POINTER_MODE_DEVICE). */
  DeviceMatrix& operator+=(const DeviceScaledDevice<Scalar>& expr);

  /** this -= DeviceScalar * x (cuBLAS axpy with negated device scalar). */
  DeviceMatrix& operator-=(const DeviceScaledDevice<Scalar>& expr);

  /** Assign from an SpMV expression: d_y = d_A * d_x. */
  DeviceMatrix& operator=(const SpMVExpr<Scalar>& expr);

  /** Assign from an add expression: d_C = alpha * d_A + beta * d_B (cuBLAS geam). */
  DeviceMatrix& operator=(const DeviceAddExpr<Scalar>& expr);

  /** No-op — all DeviceMatrix operations are implicitly noalias.
   *
   * Unlike Eigen's Matrix, where omitting .noalias() triggers a copy to a
   * temporary for safety, DeviceMatrix dispatches directly to NVIDIA library
   * calls which have no built-in aliasing protection. Every assignment
   * (`d_C = d_A * d_B`, `d_y = d_A * d_x`, etc.) behaves as if .noalias()
   * were specified. The caller must ensure operands don't alias the
   * destination for GEMM and SpMV. geam (`d_C = d_A + alpha * d_B`) is
   * safe with aliasing. Debug asserts catch violations.
   *
   * This method exists so that `tmp.noalias() = mat * p` compiles for both
   * Matrix and DeviceMatrix. */
  DeviceMatrix& noalias() { return *this; }

 private:
  // ---- Private: adopt a raw device pointer (used by friend solvers) --------

  DeviceMatrix(Scalar* device_ptr, Index rows, Index cols) : data_(device_ptr), rows_(rows), cols_(cols) {}

  /** Transfer ownership of the device pointer out. Zeros internal state. */
  Scalar* release() {
    Scalar* p = data_;
    data_ = nullptr;
    rows_ = 0;
    cols_ = 0;
    if (ready_event_) {
      (void)cudaEventDestroy(ready_event_);
      ready_event_ = nullptr;
    }
    ready_stream_ = nullptr;
    return p;
  }

  // ---- Private helpers -------------------------------------------------------

  void ensureEvent() {
    if (!ready_event_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming));
    }
  }

  void retainBuffer(internal::DeviceBuffer&& buffer) { retained_buffer_ = std::move(buffer); }

  // ---- Friend declarations ------------------------------------------------

  template <typename, int>
  friend class GpuLLT;
  template <typename>
  friend class GpuLU;
  template <typename>
  friend class GpuQR;
  template <typename>
  friend class GpuSVD;
  template <typename>
  friend class GpuSelfAdjointEigenSolver;

  // ---- Data members --------------------------------------------------------

  Scalar* data_ = nullptr;
  Index rows_ = 0;
  Index cols_ = 0;
  cudaEvent_t ready_event_ = nullptr;       // internal: tracks last write completion
  cudaStream_t ready_stream_ = nullptr;     // stream that recorded ready_event_ (for same-stream skip)
  internal::DeviceBuffer retained_buffer_;  // internal: keeps async aux buffers alive
};

}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_MATRIX_H
