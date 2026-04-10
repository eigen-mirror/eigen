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
#include <vector>

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

// Thread-local pool of small device buffers to avoid cudaMalloc/cudaFree
// overhead for tiny allocations (e.g., DeviceScalar). Buffers up to
// kSmallBufferThreshold bytes are recycled; larger allocations bypass the pool.
template <size_t SmallBufferThreshold = 256, size_t MaxPoolSize = 64>
struct DeviceBufferPool {
  static constexpr size_t kSmallBufferThreshold = SmallBufferThreshold;
  static constexpr size_t kMaxPoolSize = MaxPoolSize;

  struct Entry {
    void* ptr;
    size_t bytes;
  };

  ~DeviceBufferPool() {
    for (auto& e : free_list_) (void)cudaFree(e.ptr);
  }

  void* allocate(size_t bytes) {
    // Search for a buffer of sufficient size.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      if (free_list_[i].bytes >= bytes) {
        void* p = free_list_[i].ptr;
        free_list_[i] = free_list_.back();
        free_list_.pop_back();
        return p;
      }
    }
    // No suitable buffer found — allocate new.
    void* p = nullptr;
    EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
    return p;
  }

  void deallocate(void* p, size_t bytes) {
    if (free_list_.size() < kMaxPoolSize) {
      free_list_.push_back({p, bytes});
    } else {
      (void)cudaFree(p);
    }
  }

  static DeviceBufferPool& threadLocal() {
    thread_local DeviceBufferPool pool;
    return pool;
  }

 private:
  std::vector<Entry> free_list_;
};

struct DeviceBuffer {
  void* ptr = nullptr;

  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t bytes) : size_(bytes) {
    if (bytes > 0) {
      if (bytes <= DeviceBufferPool<>::kSmallBufferThreshold) {
        ptr = DeviceBufferPool<>::threadLocal().allocate(bytes);
      } else {
        EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, bytes));
      }
    }
  }

  ~DeviceBuffer() {
    if (ptr) {
      if (size_ <= DeviceBufferPool<>::kSmallBufferThreshold) {
        DeviceBufferPool<>::threadLocal().deallocate(ptr, size_);
      } else {
        (void)cudaFree(ptr);
      }
    }
  }

  // Move-only.
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), size_(o.size_) {
    o.ptr = nullptr;
    o.size_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr) {
        if (size_ <= DeviceBufferPool<>::kSmallBufferThreshold) {
          DeviceBufferPool<>::threadLocal().deallocate(ptr, size_);
        } else {
          (void)cudaFree(ptr);
        }
      }
      ptr = o.ptr;
      size_ = o.size_;
      o.ptr = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  size_t size() const { return size_; }

  // Adopt an existing device pointer. Caller relinquishes ownership.
  // Adopted buffers bypass the pool on destruction.
  static DeviceBuffer adopt(void* p) {
    DeviceBuffer b;
    b.ptr = p;
    b.size_ = DeviceBufferPool<>::kSmallBufferThreshold + 1;  // force cudaFree
    return b;
  }

 private:
  size_t size_ = 0;
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
