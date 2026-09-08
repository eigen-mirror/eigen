// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// What a temporary costs. GpuDevice::allocate is gpuMalloc, which synchronizes the device, against
// cudaMallocAsync, which is ordered on a stream and recycles through a pool. These are host-side API costs with no
// device work to time, so they are measured as wall time per allocate-and-free pair rather than with events.

#include "gpu_bench_common.h"

namespace {

const std::vector<int64_t> kSizes = {1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 28};

void BM_DeviceAllocateFree(benchmark::State& state) {
  const int64_t bytes = state.range(0);
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  // One untimed pair, so the driver's first-touch cost stays out of the loop.
  void* warm = device.allocate(bytes);
  device.deallocate(warm);
  for (auto _ : state) {
    void* p = device.allocate(bytes);
    benchmark::DoNotOptimize(p);
    device.deallocate(p);
  }
  state.SetBytesProcessed(state.iterations() * bytes);
}

void BM_MallocAsyncFreeAsync(benchmark::State& state) {
  int device = 0, pools_supported = 0;
  EIGEN_GPU_RUNTIME_CHECK(cudaGetDevice(&device));
  EIGEN_GPU_RUNTIME_CHECK(cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, device));
  if (!pools_supported) {
    state.SkipWithError("Device does not support stream-ordered allocation");
    return;
  }
  const int64_t bytes = state.range(0);
  cudaStream_t stream = nullptr;
  EIGEN_GPU_RUNTIME_CHECK(cudaStreamCreate(&stream));
  {
    void* warm = nullptr;
    EIGEN_GPU_RUNTIME_CHECK(cudaMallocAsync(&warm, bytes, stream));
    EIGEN_GPU_RUNTIME_CHECK(cudaFreeAsync(warm, stream));
    EIGEN_GPU_RUNTIME_CHECK(cudaStreamSynchronize(stream));
  }
  for (auto _ : state) {
    void* p = nullptr;
    EIGEN_GPU_RUNTIME_CHECK(cudaMallocAsync(&p, bytes, stream));
    benchmark::DoNotOptimize(p);
    EIGEN_GPU_RUNTIME_CHECK(cudaFreeAsync(p, stream));
  }
  EIGEN_GPU_RUNTIME_CHECK(cudaStreamSynchronize(stream));
  state.SetBytesProcessed(state.iterations() * bytes);
  EIGEN_GPU_RUNTIME_CHECK(cudaStreamDestroy(stream));
}

// The 1 KiB scratch buffer the reductions take: allocated on first touch and kept for the life of the stream
// device, so this measures what every later call costs.
void BM_Scratchpad(benchmark::State& state) {
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  benchmark::DoNotOptimize(device.scratchpad());
  for (auto _ : state) {
    benchmark::DoNotOptimize(device.scratchpad());
  }
}

}  // namespace

BENCHMARK(BM_DeviceAllocateFree)->ArgsProduct({kSizes})->UseRealTime();
BENCHMARK(BM_MallocAsyncFreeAsync)->ArgsProduct({kSizes})->UseRealTime();
BENCHMARK(BM_Scratchpad)->UseRealTime();

EIGEN_GPU_BENCHMARK_MAIN()
