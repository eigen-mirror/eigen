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
  GpuContext() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUBLAS_CHECK(cublasCreate(&cublas_));
    EIGEN_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    EIGEN_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
    EIGEN_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, stream_));
  }

  ~GpuContext() {
    if (cusolver_) (void)cusolverDnDestroy(cusolver_);
    if (cublas_) (void)cublasDestroy(cublas_);
    if (stream_) (void)cudaStreamDestroy(stream_);
  }

  // Non-copyable, non-movable (owns library handles).
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

  /** Lazily-created thread-local default context.
   *
   * \note The thread-local instance is destroyed when the thread exits (or at
   * static destruction time for the main thread). On some CUDA driver
   * configurations this may print "CUDA_ERROR_DEINITIALIZED" to stderr if the
   * CUDA context has already been torn down. These errors are harmless and are
   * suppressed in the destructor, but they can produce noise in test output.
   * To avoid this, call cudaDeviceReset() only after all GpuContext instances
   * (including thread-local ones) have been destroyed. */
  static GpuContext& threadLocal() {
    thread_local GpuContext ctx;
    return ctx;
  }

  cudaStream_t stream() const { return stream_; }
  cublasHandle_t cublasHandle() const { return cublas_; }
  cusolverDnHandle_t cusolverHandle() const { return cusolver_; }

 private:
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  cusolverDnHandle_t cusolver_ = nullptr;
};

}  // namespace Eigen

#endif  // EIGEN_GPU_CONTEXT_H
