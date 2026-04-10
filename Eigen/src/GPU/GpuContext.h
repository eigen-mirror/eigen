// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unified GPU execution context.
//
// GpuContext owns a CUDA stream and all NVIDIA library handles (cuBLAS,
// cuSOLVER, future cuDSS/cuSPARSE). It is the entry point for all GPU
// operations on DeviceMatrix.
//
// Usage:
//   GpuContext ctx;                        // explicit context
//   d_C.device(ctx) = d_A * d_B;          // GEMM on ctx's stream
//
//   d_C = d_A * d_B;                      // thread-local default context
//   GpuContext& ctx = GpuContext::threadLocal();

#ifndef EIGEN_GPU_CONTEXT_H
#define EIGEN_GPU_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuBlasSupport.h"
#include "./CuSolverSupport.h"
#include <cusparse.h>
#include <cufft.h>

namespace Eigen {

/** \ingroup GPU_Module
 * \class GpuContext
 * \brief Unified GPU execution context owning a CUDA stream and library handles.
 *
 * Each GpuContext instance creates a dedicated CUDA stream, a cuBLAS handle,
 * and a cuSOLVER handle, all bound to that stream. Multiple contexts enable
 * concurrent execution on independent streams.
 *
 * A lazily-created thread-local default is available via threadLocal() for
 * simple single-stream usage.
 */
class GpuContext {
 public:
  /** Create a new context with a dedicated CUDA stream. */
  GpuContext() : owns_stream_(true) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    init_handles();
  }

  /** Create a context on an existing stream (e.g., stream 0 = nullptr).
   * The caller retains ownership of the stream — this context will not destroy it. */
  explicit GpuContext(cudaStream_t stream) : stream_(stream), owns_stream_(false) { init_handles(); }

  ~GpuContext() {
    if (cusparse_) (void)cusparseDestroy(cusparse_);
    if (cusolver_) (void)cusolverDnDestroy(cusolver_);
    if (cublas_lt_) (void)cublasLtDestroy(cublas_lt_);
    if (cublas_) (void)cublasDestroy(cublas_);
    if (owns_stream_ && stream_) (void)cudaStreamDestroy(stream_);
  }

  // Non-copyable, non-movable (owns library handles).
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

  /** Get the thread-local default context.
   * If setThreadLocal() has been called, returns that context.
   * Otherwise lazily creates a new context with a dedicated stream. */
  static GpuContext& threadLocal() {
    GpuContext* override = tl_override_ptr();
    if (override) return *override;
    thread_local GpuContext ctx;
    return ctx;
  }

  /** Override the thread-local default context for this thread.
   * The caller retains ownership of \p ctx — it must outlive all uses.
   * Pass nullptr to restore the lazily-created default. */
  static void setThreadLocal(GpuContext* ctx) { tl_override_ptr() = ctx; }

  cudaStream_t stream() const { return stream_; }
  cublasHandle_t cublasHandle() const { return cublas_; }
  cusolverDnHandle_t cusolverHandle() const { return cusolver_; }

  /** cuBLASLt handle (lazy-initialized on first GEMM call). */
  cublasLtHandle_t cublasLtHandle() const {
    if (!cublas_lt_) {
      EIGEN_CUBLAS_CHECK(cublasLtCreate(&cublas_lt_));
    }
    return cublas_lt_;
  }

  /** Workspace buffer for cublasLtMatmul (grown lazily by cublaslt_gemm).
   * Not thread-safe — all GEMM calls must be on this context's stream. */
  internal::DeviceBuffer* gemmWorkspace() const { return &gemm_workspace_; }

  /** cuSPARSE handle (lazy-initialized on first call). */
  cusparseHandle_t cusparseHandle() const {
    if (!cusparse_) {
      cusparseStatus_t s1 = cusparseCreate(&cusparse_);
      eigen_assert(s1 == CUSPARSE_STATUS_SUCCESS && "cusparseCreate failed");
      EIGEN_UNUSED_VARIABLE(s1);
      cusparseStatus_t s2 = cusparseSetStream(cusparse_, stream_);
      eigen_assert(s2 == CUSPARSE_STATUS_SUCCESS && "cusparseSetStream failed");
      EIGEN_UNUSED_VARIABLE(s2);
    }
    return cusparse_;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  cusolverDnHandle_t cusolver_ = nullptr;
  mutable cublasLtHandle_t cublas_lt_ = nullptr;                    // lazy
  mutable cusparseHandle_t cusparse_ = nullptr;                     // lazy
  mutable internal::DeviceBuffer gemm_workspace_;                   // lazy
  bool owns_stream_ = true;

  static GpuContext*& tl_override_ptr() {
    thread_local GpuContext* ptr = nullptr;
    return ptr;
  }

  void init_handles() {
    EIGEN_CUBLAS_CHECK(cublasCreate(&cublas_));
    EIGEN_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
    EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, stream_));
  }
};

}  // namespace Eigen

#endif  // EIGEN_GPU_CONTEXT_H
