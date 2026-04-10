// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Generic CUDA runtime support shared across all GPU library integrations
// (cuSOLVER, cuBLAS, cuDSS, etc.):
//   - Error-checking macros
//   - RAII device buffer
//
// Only depends on <cuda_runtime.h>. No NVIDIA library headers.

#ifndef EIGEN_GPU_SUPPORT_H
#define EIGEN_GPU_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <cuda_runtime.h>

namespace Eigen {
namespace internal {

// ---- Error-checking macros --------------------------------------------------
// These abort (via eigen_assert) on failure. Not for use in destructors.

#define EIGEN_CUDA_RUNTIME_CHECK(expr)                             \
  do {                                                             \
    cudaError_t _e = (expr);                                       \
    eigen_assert(_e == cudaSuccess && "CUDA runtime call failed"); \
  } while (0)

// ---- RAII: device buffer ----------------------------------------------------

struct DeviceBuffer {
  void* ptr = nullptr;

  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t bytes) {
    if (bytes > 0) EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, bytes));
  }

  ~DeviceBuffer() {
    if (ptr) (void)cudaFree(ptr);  // destructor: ignore errors
  }

  // Move-only.
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr) (void)cudaFree(ptr);
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Adopt an existing device pointer. Caller relinquishes ownership.
  static DeviceBuffer adopt(void* p) {
    DeviceBuffer b;
    b.ptr = p;
    return b;
  }
};

// ---- Scalar → cudaDataType_t ------------------------------------------------
// Shared by cuBLAS and cuSOLVER. cudaDataType_t is defined in library_types.h
// which is included transitively by cuda_runtime.h.

template <typename Scalar>
struct cuda_data_type;

template <>
struct cuda_data_type<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};
template <>
struct cuda_data_type<double> {
  static constexpr cudaDataType_t value = CUDA_R_64F;
};
template <>
struct cuda_data_type<std::complex<float>> {
  static constexpr cudaDataType_t value = CUDA_C_32F;
};
template <>
struct cuda_data_type<std::complex<double>> {
  static constexpr cudaDataType_t value = CUDA_C_64F;
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_GPU_SUPPORT_H
