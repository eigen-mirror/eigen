// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#if defined(EIGEN_USE_GPU) && !defined(EIGEN_TENSOR_TENSOR_DEVICE_GPU_H)
#define EIGEN_TENSOR_TENSOR_DEVICE_GPU_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "../../../../Eigen/src/Core/util/GpuHipCudaDefines.inc"
#include "../../../../Eigen/src/Core/util/GpuRuntime.h"

namespace Eigen {

static const int kGpuScratchSize = 1024;

// The device facts Eigen consults, read one at a time. The reason is portability, not speed: the opt-in
// shared-memory limit and memory-pool support are not fields a gpuDeviceProp_t carries under both backends, while
// the attribute enumerators exist in each. (Measured on CUDA 13.3, a warm gpuGetDeviceProperties costs 0.3-0.5 us
// against 1.4-1.5 us for these nine queries, and a cold call of either is dominated by runtime initialization.)
struct GpuDeviceAttributes {
  int multiProcessorCount;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor;
  int sharedMemPerBlock;
  int sharedMemPerBlockOptin;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int warpSize;
  int memoryPoolsSupported;
};

// Filled once for every visible device at first use, which is thread-safe by the initialization rules and needs no
// lock. A few microseconds per device, so there is nothing to gain from filling it per device on demand.
// Templated on the attribute enum: CUDA spells it cudaDeviceAttr and HIP hipDeviceAttribute_t, and the .inc pair
// aliases the enumerators but not the type.
template <typename GpuDeviceAttr>
inline int GetGpuDeviceAttribute(GpuDeviceAttr attribute, int device) {
  int value = 0;
  EIGEN_GPU_RUNTIME_CHECK(gpuDeviceGetAttribute(&value, attribute, device));
  return value;
}

inline const std::vector<GpuDeviceAttributes>& GetGpuDeviceAttributes() {
  static const std::vector<GpuDeviceAttributes>* kAttributes = [] {
    int num_devices = 0;
    EIGEN_GPU_RUNTIME_CHECK(gpuGetDeviceCount(&num_devices));
    auto* attributes = new std::vector<GpuDeviceAttributes>(num_devices);
    for (int device = 0; device < num_devices; ++device) {
      GpuDeviceAttributes& attribute = (*attributes)[device];
      attribute.multiProcessorCount = GetGpuDeviceAttribute(gpuDevAttrMultiProcessorCount, device);
      attribute.maxThreadsPerBlock = GetGpuDeviceAttribute(gpuDevAttrMaxThreadsPerBlock, device);
      attribute.maxThreadsPerMultiProcessor = GetGpuDeviceAttribute(gpuDevAttrMaxThreadsPerMultiProcessor, device);
      attribute.sharedMemPerBlock = GetGpuDeviceAttribute(gpuDevAttrMaxSharedMemoryPerBlock, device);
      attribute.sharedMemPerBlockOptin = GetGpuDeviceAttribute(gpuDevAttrMaxSharedMemoryPerBlockOptin, device);
      attribute.computeCapabilityMajor = GetGpuDeviceAttribute(gpuDevAttrComputeCapabilityMajor, device);
      attribute.computeCapabilityMinor = GetGpuDeviceAttribute(gpuDevAttrComputeCapabilityMinor, device);
      attribute.warpSize = GetGpuDeviceAttribute(gpuDevAttrWarpSize, device);
      attribute.memoryPoolsSupported = GetGpuDeviceAttribute(gpuDevAttrMemoryPoolsSupported, device);
    }
    return attributes;
  }();
  return *kAttributes;
}

inline const GpuDeviceAttributes& GetGpuDeviceAttributes(int device) {
  const std::vector<GpuDeviceAttributes>& attributes = GetGpuDeviceAttributes();
  eigen_assert(device >= 0 && device < static_cast<int>(attributes.size()) && "no such GPU device");
  return attributes[device];
}

// Attributes for StreamInterface implementations that follow the calling thread's current device.
// GpuStreamDevice instead queries the device that owns its stream.
inline const GpuDeviceAttributes& GetCurrentGpuDeviceAttributes() {
  int device = 0;
  EIGEN_GPU_RUNTIME_CHECK(gpuGetDevice(&device));
  return GetGpuDeviceAttributes(device);
}

// This defines an interface that GPUDevice can take to use
// HIP / CUDA streams underneath.
class StreamInterface {
 public:
  virtual ~StreamInterface() = default;

  virtual const gpuStream_t& stream() const = 0;
  virtual const gpuDeviceProp_t& deviceProperties() const = 0;

  // The attributes of the device this interface's stream runs on. The default reports the device the calling
  // thread is bound to, which is what an implementation that follows the current device wants; one that owns a
  // device index overrides it, as GpuStreamDevice does, so that the answer follows the stream rather than
  // whatever device the caller happens to have selected.
  virtual const GpuDeviceAttributes& deviceAttributes() const { return GetCurrentGpuDeviceAttributes(); }

  // Allocate memory on the actual device where the computation will run
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;

  // Return a scratchpad buffer of size 1k
  virtual void* scratchpad() const = 0;

  // Return a semaphore. The semaphore is initially initialized to 0, and
  // each kernel using it is responsible for resetting to 0 upon completion
  // to maintain the invariant that the semaphore is always equal to 0 upon
  // each kernel start.
  virtual unsigned int* semaphore() const = 0;
};

class GpuDeviceProperties {
 public:
  static const GpuDeviceProperties& instance() {
    static const GpuDeviceProperties* kInstance = new GpuDeviceProperties();

    return *kInstance;
  }

  EIGEN_STRONG_INLINE const gpuDeviceProp_t& get(int device) const {
    eigen_assert(device >= 0 && device < static_cast<int>(device_properties_.size()) && "no such GPU device");
    return device_properties_[device];
  }

 private:
  GpuDeviceProperties() = default;

  static std::vector<gpuDeviceProp_t> GetDeviceProperties() {
    int num_devices = 0;
    EIGEN_GPU_RUNTIME_CHECK(gpuGetDeviceCount(&num_devices));
    std::vector<gpuDeviceProp_t> device_properties(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      EIGEN_GPU_RUNTIME_CHECK(gpuGetDeviceProperties(&device_properties[i], i));
    }

    return device_properties;
  }

  std::vector<gpuDeviceProp_t> device_properties_ = GetDeviceProperties();
};

EIGEN_ALWAYS_INLINE const GpuDeviceProperties& GetGpuDeviceProperties() { return GpuDeviceProperties::instance(); }

EIGEN_ALWAYS_INLINE const gpuDeviceProp_t& GetGpuDeviceProperties(int device) {
  return GetGpuDeviceProperties().get(device);
}

static const gpuStream_t default_stream = gpuStreamDefault;

class GpuStreamDevice : public StreamInterface {
 public:
  // Use the default stream on the current device
  GpuStreamDevice() : stream_(&default_stream), scratch_(nullptr), semaphore_(nullptr) {
    EIGEN_GPU_RUNTIME_CHECK(gpuGetDevice(&device_));
  }
  // Use the default stream on the specified device
  GpuStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(nullptr), semaphore_(nullptr) {}
  // Use the specified stream. Note that it's the
  // caller's responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  GpuStreamDevice(const gpuStream_t* stream, int device = -1)
      : stream_(stream), device_(device), scratch_(nullptr), semaphore_(nullptr) {
    if (device < 0) {
      EIGEN_GPU_RUNTIME_CHECK(gpuGetDevice(&device_));
    } else {
      int num_devices = 0;
      EIGEN_GPU_RUNTIME_CHECK(gpuGetDeviceCount(&num_devices));
      EIGEN_UNUSED_VARIABLE(num_devices);
      gpu_assert(device < num_devices);
      device_ = device;
    }
  }

  virtual ~GpuStreamDevice() {
    if (scratch_) {
      deallocate(scratch_);
    }
  }

  const gpuStream_t& stream() const { return *stream_; }
  const gpuDeviceProp_t& deviceProperties() const { return GetGpuDeviceProperties(device_); }
  const GpuDeviceAttributes& deviceAttributes() const override { return GetGpuDeviceAttributes(device_); }
  virtual void* allocate(size_t num_bytes) const {
    EIGEN_GPU_RUNTIME_CHECK(gpuSetDevice(device_));
    void* result = nullptr;
    EIGEN_GPU_RUNTIME_CHECK(gpuMalloc(&result, num_bytes));
    gpu_assert(result != nullptr);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    EIGEN_GPU_RUNTIME_CHECK(gpuSetDevice(device_));
    gpu_assert(buffer != nullptr);
    EIGEN_GPU_RUNTIME_CHECK(gpuFree(buffer));
  }

  virtual void* scratchpad() const {
    if (scratch_ == nullptr) {
      scratch_ = allocate(kGpuScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  virtual unsigned int* semaphore() const {
    if (semaphore_ == nullptr) {
      char* scratch = static_cast<char*>(scratchpad()) + kGpuScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      EIGEN_GPU_RUNTIME_CHECK(gpuMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
    }
    return semaphore_;
  }

 private:
  const gpuStream_t* stream_;
  int device_;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

struct GpuDevice {
  // The StreamInterface is not owned: the caller is
  // responsible for its initialization and eventual destruction.
  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) { eigen_assert(stream); }
  // Nothing reads max_blocks_ any more; the executor sizes its grid from the device's own limits.
  EIGEN_DEPRECATED explicit GpuDevice(const StreamInterface* stream, int num_blocks)
      : stream_(stream), max_blocks_(num_blocks) {
    eigen_assert(stream);
  }
  // TODO(bsteiner): This is an internal API, we should not expose it.
  EIGEN_STRONG_INLINE const gpuStream_t& stream() const { return stream_->stream(); }

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const { return stream_->allocate(num_bytes); }

  EIGEN_STRONG_INLINE void deallocate(void* buffer) const { stream_->deallocate(buffer); }

  EIGEN_STRONG_INLINE void* allocate_temp(size_t num_bytes) const { return stream_->allocate(num_bytes); }

  EIGEN_STRONG_INLINE void deallocate_temp(void* buffer) const { stream_->deallocate(buffer); }

  template <typename Type>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Type get(Type data) const {
    return data;
  }

  EIGEN_STRONG_INLINE void* scratchpad() const { return stream_->scratchpad(); }

  EIGEN_STRONG_INLINE unsigned int* semaphore() const { return stream_->semaphore(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifndef EIGEN_GPU_COMPILE_PHASE
    EIGEN_GPU_RUNTIME_CHECK(gpuMemcpyAsync(dst, src, n, gpuMemcpyDeviceToDevice, stream_->stream()));
#else
    EIGEN_UNUSED_VARIABLE(dst);
    EIGEN_UNUSED_VARIABLE(src);
    EIGEN_UNUSED_VARIABLE(n);
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    EIGEN_GPU_RUNTIME_CHECK(gpuMemcpyAsync(dst, src, n, gpuMemcpyHostToDevice, stream_->stream()));
  }

  EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    EIGEN_GPU_RUNTIME_CHECK(gpuMemcpyAsync(dst, src, n, gpuMemcpyDeviceToHost, stream_->stream()));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifndef EIGEN_GPU_COMPILE_PHASE
    EIGEN_GPU_RUNTIME_CHECK(gpuMemsetAsync(buffer, c, n, stream_->stream()));
#else
    EIGEN_UNUSED_VARIABLE(buffer);
    EIGEN_UNUSED_VARIABLE(c);
    EIGEN_UNUSED_VARIABLE(n);
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  template <typename T>
  EIGEN_STRONG_INLINE void fill(T* begin, T* end, const T& value) const {
#ifndef EIGEN_GPU_COMPILE_PHASE
    const size_t count = end - begin;
    // Split value into bytes and run memset with stride.
    const int value_size = sizeof(value);
    char* buffer = (char*)begin;
    char* value_bytes = (char*)(&value);
    // If all value bytes are equal, then a single memset can be much faster.
    bool use_single_memset = true;
    for (int i = 1; i < value_size; ++i) {
      if (value_bytes[i] != value_bytes[0]) {
        use_single_memset = false;
      }
    }

    if (use_single_memset) {
      EIGEN_GPU_RUNTIME_CHECK(gpuMemsetAsync(buffer, value_bytes[0], count * sizeof(T), stream_->stream()));
    } else {
      for (int b = 0; b < value_size; ++b) {
        EIGEN_GPU_RUNTIME_CHECK(gpuMemset2DAsync(buffer + b, value_size, value_bytes[b], 1, count, stream_->stream()));
      }
    }
#else
    EIGEN_UNUSED_VARIABLE(begin);
    EIGEN_UNUSED_VARIABLE(end);
    EIGEN_UNUSED_VARIABLE(value);
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  // The warp (wavefront) width, which is what "a thread" means to the block-size heuristics that ask.
  EIGEN_STRONG_INLINE size_t numThreads() const { return static_cast<size_t>(warpSize()); }

  EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    // FIXME: Return a more accurate cache size.
    return 48 * 1024;
  }

  EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on hip/cuda devices.
    return firstLevelCacheSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
#ifndef EIGEN_GPU_COMPILE_PHASE
    EIGEN_GPU_RUNTIME_CHECK(gpuStreamSynchronize(stream_->stream()));
#else
    gpu_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE int getNumGpuMultiProcessors() const { return stream_->deviceProperties().multiProcessorCount; }
  EIGEN_STRONG_INLINE int maxGpuThreadsPerBlock() const { return stream_->deviceProperties().maxThreadsPerBlock; }
  EIGEN_STRONG_INLINE int maxGpuThreadsPerMultiProcessor() const {
    return stream_->deviceProperties().maxThreadsPerMultiProcessor;
  }
  EIGEN_STRONG_INLINE int sharedMemPerBlock() const {
    return static_cast<int>(stream_->deviceProperties().sharedMemPerBlock);
  }
  EIGEN_STRONG_INLINE int majorDeviceVersion() const { return stream_->deviceProperties().major; }
  EIGEN_STRONG_INLINE int minorDeviceVersion() const { return stream_->deviceProperties().minor; }

  // Read from the StreamInterface's attributes rather than from its properties: gpuDeviceProp_t does not carry
  // the opt-in shared-memory limit or memory-pool support portably. Like the properties above, they describe the
  // device the stream and its allocations belong to, which need not be the device the calling thread is bound to.
  EIGEN_STRONG_INLINE int warpSize() const { return stream_->deviceAttributes().warpSize; }
  // The shared memory a kernel may request with gpuFuncSetAttribute, which exceeds sharedMemPerBlock on every
  // architecture since Volta.
  EIGEN_STRONG_INLINE int sharedMemPerBlockOptin() const { return stream_->deviceAttributes().sharedMemPerBlockOptin; }
  // Whether gpuMallocAsync and its pool are available on this device.
  EIGEN_STRONG_INLINE bool memoryPoolsSupported() const {
    return stream_->deviceAttributes().memoryPoolsSupported != 0;
  }

  EIGEN_DEPRECATED EIGEN_STRONG_INLINE int maxBlocks() const { return max_blocks_; }

  // This function checks if the GPU runtime recorded an error for the
  // underlying stream device.
  inline bool ok() const {
#ifdef EIGEN_GPUCC
    gpuError_t error = gpuStreamQuery(stream_->stream());
    return (error == gpuSuccess) || (error == gpuErrorNotReady);
#else
    return false;
#endif
  }

 private:
  const StreamInterface* stream_;
  int max_blocks_;
};

// Launches `kernel` on the device's stream through internal::gpu_launch (GpuRuntime.h), which reports a failed
// launch through EIGEN_GPU_RUNTIME_CHECK. A single statement, so it composes with an unbraced `if`.
#define LAUNCH_GPU_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)                               \
  do {                                                                                                       \
    ::Eigen::internal::gpu_launch((kernel), dim3(gridsize), dim3(blocksize), (sharedmem), (device).stream(), \
                                  __VA_ARGS__);                                                              \
  } while (0)

}  // end namespace Eigen

// undefine all the gpu* macros we defined at the beginning of the file
#include "../../../../Eigen/src/Core/util/GpuHipCudaUndefines.inc"

#endif  // EIGEN_TENSOR_TENSOR_DEVICE_GPU_H
