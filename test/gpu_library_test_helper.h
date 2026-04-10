// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_GPU_LIBRARY_TEST_HELPER_H
#define EIGEN_TEST_GPU_LIBRARY_TEST_HELPER_H

// Helpers for GPU tests that call NVIDIA library APIs (cuBLAS, cuSOLVER, etc.)
// from the host side. Provides RAII GPU memory management and async matrix transfer.
//
// This is separate from gpu_common.h (element-parallel device kernels) and
// gpu_test_helper.h (serialization-based device kernels). Those patterns run
// user functors inside GPU kernels. This helper is for host-orchestrated tests
// that call library APIs which launch their own kernels internally.
//
// All transfers use an explicit stream and cudaMemcpyAsync. Callers must
// synchronize (ctx.synchronize() or cudaStreamSynchronize) before reading
// results back on the host.

#include "gpu_test_helper.h"

namespace Eigen {
namespace test {

// RAII wrapper for GPU device memory. Prevents leaks when VERIFY macros abort.
template <typename Scalar>
struct GpuBuffer {
  Scalar* data = nullptr;
  Index size = 0;

  GpuBuffer() = default;

  explicit GpuBuffer(Index n) : size(n) { GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&data), n * sizeof(Scalar))); }

  ~GpuBuffer() {
    if (data) GPU_CHECK(gpuFree(data));
  }

  // Move-only.
  GpuBuffer(GpuBuffer&& other) noexcept : data(other.data), size(other.size) {
    other.data = nullptr;
    other.size = 0;
  }
  GpuBuffer& operator=(GpuBuffer&& other) noexcept {
    if (this != &other) {
      if (data) GPU_CHECK(gpuFree(data));
      data = other.data;
      size = other.size;
      other.data = nullptr;
      other.size = 0;
    }
    return *this;
  }

  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;

  // Async zero the buffer on the given stream.
  void setZeroAsync(cudaStream_t stream) { GPU_CHECK(cudaMemsetAsync(data, 0, size * sizeof(Scalar), stream)); }
};

// Copy a dense Eigen matrix to a new GPU buffer, async on the given stream.
// Caller must synchronize before the host matrix is freed or modified.
template <typename Derived>
GpuBuffer<typename Derived::Scalar> gpu_copy_to_device(cudaStream_t stream, const MatrixBase<Derived>& host_mat) {
  using Scalar = typename Derived::Scalar;
  const auto& mat = host_mat.derived();
  GpuBuffer<Scalar> buf(mat.size());
  GPU_CHECK(cudaMemcpyAsync(buf.data, mat.data(), mat.size() * sizeof(Scalar), cudaMemcpyHostToDevice, stream));
  return buf;
}

// Copy GPU buffer contents back to a dense Eigen matrix, async on the given stream.
// Caller must synchronize before reading from host_mat.
template <typename Scalar, typename Derived>
void gpu_copy_to_host(cudaStream_t stream, const GpuBuffer<Scalar>& buf, MatrixBase<Derived>& host_mat) {
  auto& mat = host_mat.derived();
  eigen_assert(buf.size == mat.size());
  GPU_CHECK(cudaMemcpyAsync(mat.data(), buf.data, mat.size() * sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
}

}  // namespace test
}  // namespace Eigen

#endif  // EIGEN_TEST_GPU_LIBRARY_TEST_HELPER_H
