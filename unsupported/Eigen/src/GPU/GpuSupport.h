// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Generic CUDA runtime support shared across all GPU library integrations
// (cuSOLVER and cuBLAS):
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

#include <limits>
#include <memory>

namespace Eigen {
namespace gpu {

// ---- Generic operation flag -------------------------------------------------
// Public flag for transpose/adjoint in BLAS-, solver-, and sparse-style calls.
// Each library's support header maps this to its own enum (cublasOperation_t,
// cusparseOperation_t, etc.) via a small to_<lib>_op() helper.

enum class GpuOp { NoTrans, Trans, ConjTrans };

namespace internal {

// ---- Error-checking macros --------------------------------------------------
// These abort (via eigen_assert) on failure. Not for use in destructors.

#define EIGEN_CUDA_RUNTIME_CHECK(expr)                             \
  do {                                                             \
    cudaError_t _e = (expr);                                       \
    eigen_assert(_e == cudaSuccess && "CUDA runtime call failed"); \
  } while (0)

// ---- Bounds-checked narrowing for cuBLAS/cuSOLVER int parameters ------------
// cuBLAS and the legacy cuSOLVER APIs take dimensions and leading dimensions as
// `int` (32-bit signed). Modern GPUs can host allocations whose dimensions
// exceed INT_MAX, and Eigen's Index is 64-bit by default. Use this helper at
// every narrowing call site so an out-of-range value triggers an assert
// instead of silently overflowing the BLAS argument.

inline int to_blas_int(int64_t v) {
  eigen_assert(v >= 0 && v <= static_cast<int64_t>((std::numeric_limits<int>::max)()) &&
               "dimension exceeds the int range supported by cuBLAS / cuSOLVER");
  return static_cast<int>(v);
}

// ---- Stream-ordered device allocation ----------------------------------------
// cudaMallocAsync / cudaFreeAsync (CUDA 11.2+) allocate from a stream-ordered
// memory pool: both are cheap enqueues instead of the device-wide
// synchronization performed by cudaMalloc / cudaFree. All module allocations
// go through device_malloc / device_free on the *legacy default stream*:
// legacy-stream ordering guarantees that work enqueued later on any blocking
// stream observes the allocation, and that a free waits for all previously
// enqueued work on blocking streams — the same lifetime guarantees callers
// got from cudaMalloc / cudaFree, minus the host stalls.
//
// Caveat: streams created with cudaStreamNonBlocking do not synchronize with
// the legacy stream. When borrowing such a stream (gpu::Context(stream)),
// define EIGEN_GPU_NO_STREAM_ORDERED_ALLOC to fall back to cudaMalloc/cudaFree.
//
// Support is detected once per process (attribute of the device current at
// first use); unsupported configurations fall back to cudaMalloc/cudaFree.

inline bool device_supports_memory_pools() {
#ifdef EIGEN_GPU_NO_STREAM_ORDERED_ALLOC
  return false;
#else
  static const bool supported = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMemoryPoolsSupported, device) != cudaSuccess) return false;
    if (v == 0) return false;
    // Keep freed memory in the pool instead of trimming at every stream
    // synchronize — repeated alloc/free cycles (temporaries in loops) then
    // recycle at user-space speed.
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
      // The attribute value type is cuuint64_t; use a same-size stand-in to
      // avoid requiring the driver-API header.
      unsigned long long threshold = ~0ULL;
      (void)cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }
    return true;
  }();
  return supported;
#endif
}

inline void* device_malloc(size_t bytes) {
  void* p = nullptr;
  if (device_supports_memory_pools()) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMallocAsync(&p, bytes, /*legacy default stream*/ nullptr));
  } else {
    EIGEN_CUDA_RUNTIME_CHECK(cudaMalloc(&p, bytes));
  }
  return p;
}

inline void device_free(void* p) noexcept {
  if (!p) return;
  if (device_supports_memory_pools()) {
    (void)cudaFreeAsync(p, /*legacy default stream*/ nullptr);
  } else {
    (void)cudaFree(p);
  }
}

// ---- Custom deleters for CUDA-allocated memory ------------------------------
// Used with std::unique_ptr to give CUDA allocations RAII semantics with no
// hand-rolled move/dtor boilerplate.

struct CudaFreeDeleter {
  // When `borrow == true`, the unique_ptr does not free the pointer. Used by
  // DeviceMatrix::view() to wrap a non-owning device pointer with the same
  // smart-pointer machinery as owning storage, without changing the type.
  bool borrow = false;
  void operator()(void* p) const noexcept {
    if (p && !borrow) device_free(p);
  }
};

struct CudaFreeHostDeleter {
  void operator()(void* p) const noexcept {
    if (p) (void)cudaFreeHost(p);
  }
};

// ---- Thread-local pool of small device buffers ------------------------------
// Recycles allocations up to kSmallBufferThreshold bytes (e.g., DeviceScalar)
// to avoid cudaMalloc/cudaFree overhead. Larger allocations bypass the pool.

template <size_t SmallBufferThreshold = 256, size_t MaxPoolSize = 64>
struct DeviceBufferPool {
  static constexpr size_t kSmallBufferThreshold = SmallBufferThreshold;
  static constexpr size_t kMaxPoolSize = MaxPoolSize;

  struct Entry {
    void* ptr;
    size_t bytes;
  };

  // Lifetime marker for the thread-local pool. thread_local destruction runs
  // in reverse construction order, so a long-lived object holding pooled
  // buffers (e.g. the thread-local gpu::Context, or a static) can be
  // destroyed *after* the pool. The marker is trivially destructible — it
  // stays readable during TLS teardown — letting the deleter fall back to a
  // direct device_free once the pool is gone instead of touching a destroyed
  // vector.
  enum class State : signed char { kNotConstructed = 0, kAlive = 1, kDestroyed = 2 };

  static State& threadState() {
    thread_local State state = State::kNotConstructed;
    return state;
  }

  DeviceBufferPool() { threadState() = State::kAlive; }

  ~DeviceBufferPool() {
    for (auto& e : free_list_) device_free(e.ptr);
    threadState() = State::kDestroyed;
  }

  void* allocate(size_t bytes) {
    for (size_t i = 0; i < free_list_.size(); ++i) {
      if (free_list_[i].bytes >= bytes) {
        void* p = free_list_[i].ptr;
        free_list_[i] = free_list_.back();
        free_list_.pop_back();
        return p;
      }
    }
    return device_malloc(bytes);
  }

  void deallocate(void* p, size_t bytes) {
    if (free_list_.size() < kMaxPoolSize) {
      free_list_.push_back({p, bytes});
    } else {
      device_free(p);
    }
  }

  static DeviceBufferPool& threadLocal() {
    thread_local DeviceBufferPool pool;
    return pool;
  }

 private:
  std::vector<Entry> free_list_;
};

// Stateful deleter that returns small buffers to the thread-local pool and
// device_free's larger ones. size==0 means "always device_free" (adopted ptrs).
struct PooledCudaFreeDeleter {
  size_t size = 0;

  void operator()(void* p) const noexcept {
    if (!p) return;
    if (size > 0 && size <= DeviceBufferPool<>::kSmallBufferThreshold &&
        DeviceBufferPool<>::threadState() == DeviceBufferPool<>::State::kAlive) {
      DeviceBufferPool<>::threadLocal().deallocate(p, size);
    } else {
      device_free(p);
    }
  }
};

// ---- RAII: device buffer ----------------------------------------------------

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes > 0) {
      void* p = nullptr;
      // Bypass the pool once its thread_local has been destroyed (allocation
      // from a static/TLS destructor); the matching deleter then also takes
      // the direct device_free path.
      if (bytes <= DeviceBufferPool<>::kSmallBufferThreshold &&
          DeviceBufferPool<>::threadState() != DeviceBufferPool<>::State::kDestroyed) {
        p = DeviceBufferPool<>::threadLocal().allocate(bytes);
      } else {
        p = device_malloc(bytes);
      }
      ptr_ = std::unique_ptr<void, PooledCudaFreeDeleter>(p, PooledCudaFreeDeleter{bytes});
    }
  }

  // Explicit moves so a moved-from buffer reports size() == 0 (callers use
  // size() for grow-only reuse decisions; a stale size on a null buffer would
  // suppress the reallocation).
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::move(o.ptr_)), bytes_(o.bytes_) { o.bytes_ = 0; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      ptr_ = std::move(o.ptr_);
      bytes_ = o.bytes_;
      o.bytes_ = 0;
    }
    return *this;
  }

  void* get() const noexcept { return ptr_.get(); }
  void* release() noexcept {
    bytes_ = 0;
    return ptr_.release();
  }
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

  /** Logical allocation size in bytes, tracked for adopted pointers as well. */
  size_t size() const noexcept { return bytes_; }

  // Adopt an existing device pointer of `bytes` usable bytes. Caller
  // relinquishes ownership. Adopted buffers bypass the pool on destruction
  // (deleter size == 0).
  static DeviceBuffer adopt(void* p, size_t bytes) noexcept {
    DeviceBuffer b;
    b.ptr_ = std::unique_ptr<void, PooledCudaFreeDeleter>(p, PooledCudaFreeDeleter{});
    b.bytes_ = p ? bytes : 0;
    return b;
  }

 private:
  std::unique_ptr<void, PooledCudaFreeDeleter> ptr_;
  size_t bytes_ = 0;
};

// ---- RAII: pinned host buffer -----------------------------------------------
// For async D2H copies (cudaMemcpyAsync requires pinned host memory for true
// asynchrony and to avoid compute-sanitizer warnings).

class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;

  explicit PinnedHostBuffer(size_t bytes) {
    if (bytes > 0) {
      void* p = nullptr;
      EIGEN_CUDA_RUNTIME_CHECK(cudaMallocHost(&p, bytes));
      ptr_.reset(p);
    }
  }

  void* get() const noexcept { return ptr_.get(); }
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

 private:
  std::unique_ptr<void, CudaFreeHostDeleter> ptr_;
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
}  // namespace gpu
}  // namespace Eigen

#endif  // EIGEN_GPU_SUPPORT_H
