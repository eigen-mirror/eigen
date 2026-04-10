// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_GPU_CONTEXT_H
#define EIGEN_TEST_GPU_CONTEXT_H

// RAII context for GPU tests that use NVIDIA library APIs (cuBLAS, cuSOLVER, etc.).
// Owns a non-default CUDA stream. Library handles (cuBLAS, cuSOLVER, etc.) are added
// here by each integration phase as needed; each handle is bound to the owned stream.
//
// Usage:
//   GpuContext ctx;
//   auto buf = gpu_copy_to_device(ctx.stream, A);
//   // ... call NVIDIA library APIs using ctx.stream / ctx.cusolver ...
//   ctx.synchronize();

#include "gpu_test_helper.h"

#ifdef EIGEN_USE_GPU
#include <cusolverDn.h>

// Checks cuSOLVER return codes, aborts on failure.
#define CUSOLVER_CHECK(expr)                                                                 \
  do {                                                                                       \
    cusolverStatus_t _status = (expr);                                                       \
    if (_status != CUSOLVER_STATUS_SUCCESS) {                                                \
      printf("cuSOLVER error %d at %s:%d\n", static_cast<int>(_status), __FILE__, __LINE__); \
      gpu_assert(false);                                                                     \
    }                                                                                        \
  } while (0)

struct GpuContext {
  cudaStream_t stream = nullptr;
  cusolverDnHandle_t cusolver = nullptr;

  GpuContext() {
    GPU_CHECK(gpuGetDevice(&device_));
    GPU_CHECK(gpuGetDeviceProperties(&device_props_, device_));
    GPU_CHECK(cudaStreamCreate(&stream));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
  }

  ~GpuContext() {
    if (cusolver) CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    if (stream) GPU_CHECK(cudaStreamDestroy(stream));
  }

  int device() const { return device_; }
  const gpuDeviceProp_t& deviceProperties() const { return device_props_; }

  // Wait for all work submitted on this context's stream to complete.
  void synchronize() { GPU_CHECK(cudaStreamSynchronize(stream)); }

  // Non-copyable, non-movable.
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  int device_ = 0;
  gpuDeviceProp_t device_props_;
};

#endif  // EIGEN_USE_GPU

#endif  // EIGEN_TEST_GPU_CONTEXT_H
