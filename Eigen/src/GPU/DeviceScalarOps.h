// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Device-resident scalar and element-wise operations via NPP signals.
// Header-only — no custom CUDA kernels needed. Uses nppsDiv, nppsMul,
// nppsMulC from the NPP library (CUDA::npps, part of the CUDA toolkit).

#ifndef EIGEN_GPU_DEVICE_SCALAR_OPS_H
#define EIGEN_GPU_DEVICE_SCALAR_OPS_H

#include <cuda_runtime.h>
#include <npps_arithmetic_and_logical_operations.h>

namespace Eigen {
namespace internal {

// ---- NppStreamContext helper ------------------------------------------------

inline NppStreamContext make_npp_stream_ctx(cudaStream_t stream) {
  // Cache device attributes (constant for process lifetime) in a thread-local.
  // Only the stream and its flags vary per call.
  struct CachedDeviceInfo {
    bool initialized = false;
    int device_id = 0;
    int cc_major = 0;
    int cc_minor = 0;
    int mp_count = 0;
    int max_threads_per_mp = 0;
    int max_threads_per_block = 0;
    int shared_mem_per_block = 0;

    void init() {
      if (initialized) return;
      cudaGetDevice(&device_id);
      cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id);
      cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device_id);
      cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, device_id);
      cudaDeviceGetAttribute(&max_threads_per_mp, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
      cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id);
      cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
      initialized = true;
    }
  };
  thread_local CachedDeviceInfo cached;
  cached.init();

  NppStreamContext ctx = {};
  ctx.hStream = stream;
  ctx.nCudaDeviceId = cached.device_id;
  ctx.nCudaDevAttrComputeCapabilityMajor = cached.cc_major;
  ctx.nCudaDevAttrComputeCapabilityMinor = cached.cc_minor;
  ctx.nMultiProcessorCount = cached.mp_count;
  ctx.nMaxThreadsPerMultiProcessor = cached.max_threads_per_mp;
  ctx.nMaxThreadsPerBlock = cached.max_threads_per_block;
  ctx.nSharedMemPerBlock = cached.shared_mem_per_block;
  cudaStreamGetFlags(stream, &ctx.nStreamFlags);
  return ctx;
}

// ---- Scalar division: c = a / b (device-resident, async) --------------------

inline void device_scalar_div(const float* a, const float* b, float* c, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsDiv_32f_Ctx(b, a, c, 1, npp_ctx);  // NPP: pDst[i] = pSrc2[i] / pSrc1[i]
}

inline void device_scalar_div(const double* a, const double* b, double* c, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsDiv_64f_Ctx(b, a, c, 1, npp_ctx);  // NPP: pDst[i] = pSrc2[i] / pSrc1[i]
}

// ---- Scalar negation: c = -a (device-resident, async) -----------------------

inline void device_scalar_neg(const float* a, float* c, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsMulC_32f_Ctx(a, -1.0f, c, 1, npp_ctx);
}

inline void device_scalar_neg(const double* a, double* c, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsMulC_64f_Ctx(a, -1.0, c, 1, npp_ctx);
}

// ---- Element-wise vector multiply: c[i] = a[i] * b[i] ----------------------

inline void device_cwiseProduct(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsMul_32f_Ctx(a, b, c, static_cast<size_t>(n), npp_ctx);
}

inline void device_cwiseProduct(const double* a, const double* b, double* c, int n, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsMul_64f_Ctx(a, b, c, static_cast<size_t>(n), npp_ctx);
}

// ---- Element-wise vector division: c[i] = a[i] / b[i] ----------------------

inline void device_cwiseQuotient(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsDiv_32f_Ctx(b, a, c, static_cast<size_t>(n), npp_ctx);  // NPP: dst = src2 / src1
}

inline void device_cwiseQuotient(const double* a, const double* b, double* c, int n, cudaStream_t stream) {
  NppStreamContext npp_ctx = make_npp_stream_ctx(stream);
  nppsDiv_64f_Ctx(b, a, c, static_cast<size_t>(n), npp_ctx);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_SCALAR_OPS_H
