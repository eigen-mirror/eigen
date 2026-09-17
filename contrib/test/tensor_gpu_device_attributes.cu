// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_USE_GPU
#define EIGEN_NO_DEBUG

#if defined(EIGEN_USE_HIP)
#include <hip/hip_runtime.h>
using TestGpuError = hipError_t;
using TestGpuAttribute = hipDeviceAttribute_t;
#else
#include <cuda_runtime.h>
using TestGpuError = cudaError_t;
using TestGpuAttribute = cudaDeviceAttr;
#endif

static int driver_version = 11010;
static int pool_support = 1;
static int pool_queries = 0;
static int runtime_errors = 0;
static TestGpuError injected_error = static_cast<TestGpuError>(0);
static TestGpuError last_error = static_cast<TestGpuError>(0);

static TestGpuError test_device_get_attribute(int* value, TestGpuAttribute attribute, int device);
static TestGpuError test_get_device_count(int* count) {
  *count = 2;
  return static_cast<TestGpuError>(0);
}
static TestGpuError test_get_last_error() {
  const TestGpuError result = last_error;
  last_error = static_cast<TestGpuError>(0);
  return result;
}
#if defined(EIGEN_USE_HIP)
#define hipDeviceGetAttribute test_device_get_attribute
#define hipGetDeviceCount test_get_device_count
#define hipGetLastError test_get_last_error
#else
static cudaError_t test_driver_get_version(int* version) {
  *version = driver_version;
  return cudaSuccess;
}
#define cudaDeviceGetAttribute test_device_get_attribute
#define cudaGetDeviceCount test_get_device_count
#define cudaGetLastError test_get_last_error
#define cudaDriverGetVersion test_driver_get_version
#endif

static void record_runtime_error(TestGpuError error) {
  if (static_cast<int>(error) != 0) ++runtime_errors;
}
#define EIGEN_GPU_RUNTIME_CHECK(expr) record_runtime_error(expr)

#include "main.h"
#include <contrib/Eigen/Tensor>

// Inject old-runtime responses before the public umbrella is parsed, without requiring old drivers or hardware.
static TestGpuError test_device_get_attribute(int* value, TestGpuAttribute attribute, int device) {
  *value = -1;
  if (injected_error != gpuSuccess) return injected_error;
  if (attribute == gpuDevAttrMaxSharedMemoryPerBlockOptin) {
#if defined(EIGEN_USE_HIP)
    if (device == 0) return last_error = hipErrorInvalidValue;
#endif
    *value = 98304;
  } else if (attribute == gpuDevAttrMaxSharedMemoryPerBlock) {
    *value = 49152;
  } else if (attribute == gpuDevAttrMemoryPoolsSupported) {
    ++pool_queries;
#if !defined(EIGEN_USE_HIP)
    if (driver_version < 11020) return last_error = cudaErrorInvalidValue;
#endif
    *value = pool_support;
  } else if (attribute == gpuDevAttrWarpSize) {
    *value = 32 * (device + 1);
  } else {
    *value = 1;
  }
  return gpuSuccess;
}

static void test_attribute_cache() {
  // numThreads() initializes every cached field, even on runtimes missing an optional attribute.
  for (int device = 0; device < 2; ++device) {
    Eigen::GpuStreamDevice stream(device);
    Eigen::GpuDevice gpu_device(&stream);
    VERIFY_IS_EQUAL(gpu_device.numThreads(), static_cast<size_t>(32 * (device + 1)));
    VERIFY_IS_EQUAL(runtime_errors, 0);
    VERIFY_IS_EQUAL(test_get_last_error(), gpuSuccess);
#if defined(EIGEN_USE_HIP)
    VERIFY_IS_EQUAL(gpu_device.sharedMemPerBlockOptin(), device == 0 ? 49152 : 98304);
    VERIFY(gpu_device.memoryPoolsSupported());
#else
    VERIFY_IS_EQUAL(gpu_device.sharedMemPerBlockOptin(), 98304);
    VERIFY(!gpu_device.memoryPoolsSupported());
    VERIFY_IS_EQUAL(pool_queries, 0);
#endif
  }
}

static void test_attribute_queries() {
  driver_version = 11020;
  for (int supported = 0; supported <= 1; ++supported) {
    pool_support = supported;
    const int queries_before = pool_queries;
    VERIFY_IS_EQUAL(Eigen::GetGpuDeviceAttribute(gpuDevAttrMemoryPoolsSupported, 0), supported);
    VERIFY_IS_EQUAL(pool_queries, queries_before + 1);
  }
  VERIFY_IS_EQUAL(runtime_errors, 0);

#if defined(EIGEN_USE_HIP)
  injected_error = hipErrorInvalidValue;
#else
  injected_error = cudaErrorInvalidValue;
#endif
  // An invalid-value error on a required attribute or a supported pool query is still a runtime failure.
  Eigen::GetGpuDeviceAttribute(gpuDevAttrWarpSize, 0);
  VERIFY_IS_EQUAL(runtime_errors, 1);
  Eigen::GetGpuDeviceAttribute(gpuDevAttrMemoryPoolsSupported, 0);
  VERIFY_IS_EQUAL(runtime_errors, 2);
#if defined(EIGEN_USE_HIP)
  injected_error = hipErrorInvalidDevice;
#else
  injected_error = cudaErrorInvalidDevice;
#endif
  Eigen::GetGpuDeviceAttribute(gpuDevAttrMaxSharedMemoryPerBlockOptin, 0);
  VERIFY_IS_EQUAL(runtime_errors, 3);
  injected_error = gpuSuccess;
  runtime_errors = 0;
}

EIGEN_DECLARE_TEST(tensor_gpu_device_attributes) {
  CALL_SUBTEST(test_attribute_cache());
  CALL_SUBTEST(test_attribute_queries());
}
