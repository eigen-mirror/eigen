// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Shared context for GPU solvers (GpuLLT, GpuLU, GpuQR, GpuSVD, etc.).
//
// Owns a CUDA stream, cuSOLVER handle, cuBLAS handle, scratch buffer,
// and info word. Each solver holds a GpuSolverContext by composition
// and delegates lifecycle/scratch management to it.

#ifndef EIGEN_GPU_SOLVER_CONTEXT_H
#define EIGEN_GPU_SOLVER_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuSolverSupport.h"
#include "./CuBlasSupport.h"
#include "./GpuContext.h"
#include <vector>

namespace Eigen {
namespace gpu {
namespace internal {

struct GpuSolverContext {
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t cusolver_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  cublasLtHandle_t cublas_lt_ = nullptr;  // lazy: created on first GEMM-via-cublasLt call (standalone mode only)
  CusolverParams params_;
  DeviceBuffer d_scratch_;
  std::vector<char> h_workspace_;
  DeviceBuffer gemm_workspace_;  // grown lazily by cublaslt_gemm
  CublasLtPlanCache gemm_plan_cache_{kCublasLtPlanCacheCapacity};
  // Workspace ceiling fed to the cublasLtMatmul heuristic at plan-creation time.
  // See gpu::Context::setCublasLtMaxWorkspaceBytes() for semantics.
  std::size_t cublaslt_max_workspace_bytes_ = kCublasLtMaxWorkspaceBytes;
  ComputationInfo info_ = InvalidInput;
  PinnedHostBuffer pinned_info_{sizeof(int)};  // pinned host memory for async D2H of info word
  bool info_synced_ = true;
  // Non-null when this solver context borrows from a gpu::Context: stream and
  // cuSOLVER/cuBLAS handles are the Context's, and the cuBLASLt handle, GEMM
  // plan cache, and GEMM workspace are shared with it rather than duplicated.
  // Null in standalone mode, where all of the above are owned.
  Context* bound_ctx_ = nullptr;

  int& info_word() { return *static_cast<int*>(pinned_info_.get()); }
  int info_word() const { return *static_cast<const int*>(pinned_info_.get()); }

  GpuSolverContext() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
    EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, stream_));
    EIGEN_CUBLAS_CHECK(cublasCreate(&cublas_));
    EIGEN_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    ensure_scratch(0);
  }

  /** Borrow stream and cuSOLVER/cuBLAS handles from a gpu::Context, so solver
   * work runs on the same stream as the caller's other GPU operations (no
   * cross-stream event waits, and the solver creates no stream/handles of its
   * own). The cuBLASLt handle, GEMM plan cache, and GEMM workspace are shared
   * with the Context as well. The Context must outlive this solver context. */
  explicit GpuSolverContext(Context& ctx)
      : stream_(ctx.stream()), cusolver_(ctx.cusolverHandle()), cublas_(ctx.cublasHandle()), bound_ctx_(&ctx) {
    ensure_scratch(0);
  }

  ~GpuSolverContext() {
    // Ignore errors here: dtors are noexcept, and EIGEN_CU{BLAS,SOLVER,DA_RUNTIME}_CHECK
    // are eigen_assert-based — firing one from a noexcept dtor terminates the program.
    // The trailing free of the device buffers (via DeviceBuffer::~DeviceBuffer) is
    // stream-ordered (or synchronous on the cudaMalloc fallback), so it waits
    // for any in-flight kernel touching the buffer.
    // Destroy plan cache before its cublasLt handle (entries hold descriptors).
    gemm_plan_cache_.clear();
    if (cublas_lt_) (void)cublasLtDestroy(cublas_lt_);
    if (!bound_ctx_) {
      if (cublas_) (void)cublasDestroy(cublas_);
      if (cusolver_) (void)cusolverDnDestroy(cusolver_);
      if (stream_) (void)cudaStreamDestroy(stream_);
    }
  }

  GpuSolverContext(GpuSolverContext&& o) noexcept
      : stream_(o.stream_),
        cusolver_(o.cusolver_),
        cublas_(o.cublas_),
        cublas_lt_(o.cublas_lt_),
        params_(std::move(o.params_)),
        d_scratch_(std::move(o.d_scratch_)),
        h_workspace_(std::move(o.h_workspace_)),
        gemm_workspace_(std::move(o.gemm_workspace_)),
        gemm_plan_cache_(std::move(o.gemm_plan_cache_)),
        cublaslt_max_workspace_bytes_(o.cublaslt_max_workspace_bytes_),
        info_(o.info_),
        pinned_info_(std::move(o.pinned_info_)),
        info_synced_(o.info_synced_),
        bound_ctx_(o.bound_ctx_) {
    o.stream_ = nullptr;
    o.cusolver_ = nullptr;
    o.cublas_ = nullptr;
    o.cublas_lt_ = nullptr;
    o.info_ = InvalidInput;
    o.info_synced_ = true;
    o.bound_ctx_ = nullptr;
  }

  GpuSolverContext& operator=(GpuSolverContext&& o) noexcept {
    if (this != &o) {
      // Mirror the dtor: noexcept context, can't propagate. Drain the old stream
      // first so the upcoming move of d_scratch_ doesn't free buffers an in-flight
      // kernel is still touching; then swallow destroy errors (the EIGEN_CU*_CHECK
      // macros are eigen_assert-based and would terminate from a noexcept body).
      if (stream_) (void)cudaStreamSynchronize(stream_);
      gemm_plan_cache_.clear();
      if (cublas_lt_) (void)cublasLtDestroy(cublas_lt_);
      if (!bound_ctx_) {
        if (cublas_) (void)cublasDestroy(cublas_);
        if (cusolver_) (void)cusolverDnDestroy(cusolver_);
        if (stream_) (void)cudaStreamDestroy(stream_);
      }
      stream_ = o.stream_;
      cusolver_ = o.cusolver_;
      cublas_ = o.cublas_;
      cublas_lt_ = o.cublas_lt_;
      params_ = std::move(o.params_);
      d_scratch_ = std::move(o.d_scratch_);
      h_workspace_ = std::move(o.h_workspace_);
      gemm_workspace_ = std::move(o.gemm_workspace_);
      gemm_plan_cache_ = std::move(o.gemm_plan_cache_);
      cublaslt_max_workspace_bytes_ = o.cublaslt_max_workspace_bytes_;
      info_ = o.info_;
      pinned_info_ = std::move(o.pinned_info_);
      info_synced_ = o.info_synced_;
      bound_ctx_ = o.bound_ctx_;
      o.stream_ = nullptr;
      o.cusolver_ = nullptr;
      o.cublas_ = nullptr;
      o.cublas_lt_ = nullptr;
      o.info_ = InvalidInput;
      o.info_synced_ = true;
      o.bound_ctx_ = nullptr;
    }
    return *this;
  }

  /** cuBLASLt handle: the bound Context's when borrowing, otherwise an owned
   * handle lazy-initialized on first GEMM-via-cublasLt call. */
  cublasLtHandle_t cublasLtHandle() {
    if (bound_ctx_) return bound_ctx_->cublasLtHandle();
    if (!cublas_lt_) {
      EIGEN_CUBLAS_CHECK(cublasLtCreate(&cublas_lt_));
    }
    return cublas_lt_;
  }

  /** GEMM plan cache / workspace / workspace ceiling for cublaslt_gemm —
   * shared with the bound Context when borrowing, owned otherwise. */
  CublasLtPlanCache* gemmPlanCache() { return bound_ctx_ ? bound_ctx_->gemmPlanCache() : &gemm_plan_cache_; }
  DeviceBuffer* gemmWorkspace() { return bound_ctx_ ? bound_ctx_->gemmWorkspace() : &gemm_workspace_; }
  std::size_t cublasLtMaxWorkspaceBytes() const {
    return bound_ctx_ ? bound_ctx_->cublasLtMaxWorkspaceBytes() : cublaslt_max_workspace_bytes_;
  }

  GpuSolverContext(const GpuSolverContext&) = delete;
  GpuSolverContext& operator=(const GpuSolverContext&) = delete;

  // Scratch layout: [ workspace (aligned) | info_word (sizeof(int)) ].
  // Workspace size is rounded up to 16 bytes so the info word lands aligned.
  static constexpr size_t kInfoBytes = sizeof(int);
  static constexpr size_t kScratchAlign = 16;

  static size_t scratchBytesFor(size_t workspace_bytes) {
    workspace_bytes = (workspace_bytes + kScratchAlign - 1) & ~(kScratchAlign - 1);
    return workspace_bytes + kInfoBytes;
  }

  // Ensure d_scratch_ holds at least `workspace_bytes` of scratch plus the trailing
  // info word. Grows but never shrinks. Syncs the stream before reallocating to
  // avoid freeing memory that async kernels may still be using.
  void ensure_scratch(size_t workspace_bytes) {
    size_t needed = scratchBytesFor(workspace_bytes);
    if (needed > d_scratch_.size()) {
      if (d_scratch_) EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      d_scratch_ = DeviceBuffer(needed);
    }
  }

  void* scratch_workspace() const { return d_scratch_.get(); }

  int* scratch_info() const {
    eigen_assert(d_scratch_ && d_scratch_.size() >= kInfoBytes);
    return reinterpret_cast<int*>(static_cast<char*>(d_scratch_.get()) + d_scratch_.size() - kInfoBytes);
  }

  // Mark a factorization as pending (info not yet available).
  void mark_pending() {
    info_synced_ = false;
    info_ = InvalidInput;
  }

  // Common compute() prologue: reset info state. Returns false for the empty
  // (n == 0) case, which is trivially successful — the caller returns early.
  bool begin_compute(bool nonempty) {
    info_ = InvalidInput;
    if (!nonempty) {
      info_ = Success;
      info_synced_ = true;
      return false;
    }
    return true;
  }

  // Common factorize() epilogue: enqueue the async D2H copy of the info word
  // into pinned host memory. Read later by the lazy sync_info().
  void enqueue_info_copy() {
    EIGEN_CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(&info_word(), scratch_info(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
  }

  // Synchronize the stream and interpret the info word. No-op if already synced.
  void sync_info() {
    if (!info_synced_) {
      EIGEN_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream_));
      info_ = (info_word() == 0) ? Success : NumericalIssue;
      info_synced_ = true;
    }
  }

  ComputationInfo info() {
    sync_info();
    return info_;
  }
};

}  // namespace internal
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SOLVER_CONTEXT_H
