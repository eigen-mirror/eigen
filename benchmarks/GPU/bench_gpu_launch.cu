// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Launch overhead: the cost of evaluating a tiny expression through the Tensor executor (one element and one
// packet-sized block), against a raw <<<>>> launch of an empty kernel, the same launch through LAUNCH_GPU_KERNEL
// (which adds a cudaGetLastError) and cudaLaunchKernel. 1000 launches per iteration; the device time per launch is
// the reported time, the host time per enqueue is the host_us_per_launch counter.

#include "gpu_bench_common.h"

namespace {

using eigen_bench::DeviceBuffer;

const int kLaunchesPerIteration = 1000;
const int kIterations = 20;
const std::vector<int64_t> kSizes = {1, 1024};

template <int MapOptions>
void BM_ExecutorAssign(benchmark::State& state) {
  const int64_t n = state.range(0);
  DeviceBuffer<float> a(n), out(n);
  a.fillRandom();
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, MapOptions> ta(a.data(), n), tout(out.data(), n);
  const auto enqueue = [&] { tout.device(device) = ta; };
  eigen_bench::warmUp(device.stream(), enqueue);
  eigen_bench::requireClose(out.toHost(), a.toHost(), 0.0, "assign");
  eigen_bench::timeLaunches(state, device.stream(), enqueue, kLaunchesPerIteration);
}

__global__ void noop_kernel(int) {}

void BM_RawLaunch(benchmark::State& state) {
  cudaStream_t stream = nullptr;
  const auto enqueue = [&] { noop_kernel<<<1, 32, 0, stream>>>(0); };
  eigen_bench::warmUp(stream, enqueue);
  eigen_bench::timeLaunches(state, stream, enqueue, kLaunchesPerIteration);
}

void BM_CheckedLaunch(benchmark::State& state) {
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  const auto enqueue = [&] { LAUNCH_GPU_KERNEL(noop_kernel, 1, 32, 0, device, 0); };
  eigen_bench::warmUp(device.stream(), enqueue);
  eigen_bench::timeLaunches(state, device.stream(), enqueue, kLaunchesPerIteration);
}

void BM_CudaLaunchKernel(benchmark::State& state) {
  cudaStream_t stream = nullptr;
  int arg = 0;
  void* args[] = {&arg};
  const auto enqueue = [&] {
    EIGEN_GPU_RUNTIME_CHECK(
        cudaLaunchKernel(reinterpret_cast<const void*>(&noop_kernel), dim3(1), dim3(32), args, 0, stream));
  };
  eigen_bench::warmUp(stream, enqueue);
  eigen_bench::timeLaunches(state, stream, enqueue, kLaunchesPerIteration);
}

}  // namespace

EIGEN_GPU_BENCHMARK_SIZES(BM_ExecutorAssign, Eigen::Aligned);
EIGEN_GPU_BENCHMARK_SIZES(BM_ExecutorAssign, Eigen::Unaligned);
EIGEN_GPU_BENCHMARK(BM_RawLaunch);
EIGEN_GPU_BENCHMARK(BM_CheckedLaunch);
EIGEN_GPU_BENCHMARK(BM_CudaLaunchKernel);

EIGEN_GPU_BENCHMARK_MAIN()
