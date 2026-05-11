// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Unified GPU execution context.
//
// gpu::Context owns a CUDA stream and NVIDIA library handles (cuBLAS
// eagerly, cuSOLVER lazily on first use). It is the entry point for all
// GPU operations on gpu::DeviceMatrix.
//
// The cuSOLVER handle is created on the first call to cusolverHandle()
// so that translation units which only use cuFFT or cuBLAS paths (e.g.
// the cufft test) do not pull cusolverDn* symbols into the link.
//
// Usage:
//   gpu::Context ctx;                        // explicit context
//   d_C.device(ctx) = d_A * d_B;            // GEMM on ctx's stream
//
//   d_C = d_A * d_B;                        // thread-local default context
//   gpu::Context& ctx = gpu::Context::threadLocal();

#ifndef EIGEN_GPU_CONTEXT_H
#define EIGEN_GPU_CONTEXT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuBlasSupport.h"
#include "./CuSolverSupport.h"

namespace Eigen {
namespace gpu {

/** \ingroup GPU_Module
 * \class Context
 * \brief Unified GPU execution context owning a CUDA stream and library handles.
 *
 * Each Context instance creates a dedicated CUDA stream and a cuBLAS handle
 * bound to that stream. The cuSOLVER handle is created on first use via
 * cusolverHandle(); translation units that never call it do not require
 * cuSOLVER at link time. Multiple contexts enable concurrent execution on
 * independent streams.
 *
 * A lazily-created thread-local default is available via threadLocal() for
 * simple single-stream usage. A single Context is not thread-safe — use one
 * per thread, or external synchronization, since cuBLAS / cuSOLVER handles
 * are not thread-safe per handle and cusolverHandle() lazy-init is racy.
 */
class Context {
 public:
  Context() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUBLAS_CHECK(cublasCreate(&cublas_));
    EIGEN_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  }

  ~Context() {
    // Indirect call keeps cusolverDnDestroy out of TUs that never call cusolverHandle().
    if (cusolver_destroyer_) (void)cusolver_destroyer_(cusolver_);
    if (cublas_) (void)cublasDestroy(cublas_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  // Non-copyable, non-movable (owns library handles).
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  /** Lazily-created thread-local default context.
   *
   * \note The thread-local instance is destroyed when the thread exits (or at
   * static destruction time for the main thread). On some CUDA driver
   * configurations this may print "CUDA_ERROR_DEINITIALIZED" to stderr if the
   * CUDA context has already been torn down. These errors are harmless and are
   * suppressed in the destructor, but they can produce noise in test output.
   * To avoid this, call cudaDeviceReset() only after all Context instances
   * (including thread-local ones) have been destroyed. */
  static Context& threadLocal() {
    thread_local Context ctx;
    return ctx;
  }

  cudaStream_t stream() const { return stream_; }
  cublasHandle_t cublasHandle() const { return cublas_; }

  /** Returns the cuSOLVER handle, creating it on first call. */
  cusolverDnHandle_t cusolverHandle() {
    if (!cusolver_) {
      EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
      EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, stream_));
      cusolver_destroyer_ = &destroyCusolver;
    }
    return cusolver_;
  }

 private:
  static cusolverStatus_t destroyCusolver(cusolverDnHandle_t h) { return cusolverDnDestroy(h); }

  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  cusolverDnHandle_t cusolver_ = nullptr;
  cusolverStatus_t (*cusolver_destroyer_)(cusolverDnHandle_t) = nullptr;
};

}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_CONTEXT_H
