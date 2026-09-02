// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Elementwise expressions through the Tensor executor: out = a + b, out = a * b + c and out = exp(a) over 2^16 to
// 2^26 elements, for float and half on aligned maps (the packet path) and unaligned ones (the scalar path), and for
// bfloat16, which has no device packet and takes the scalar path whatever the map. Baselines: a hand-written
// grid-stride float4 kernel and cudaMemcpyAsync device to device, which bounds what a memory-bound kernel can reach.
// Bytes per launch count every operand read once and the result written once.

#include "gpu_bench_common.h"

namespace {

using eigen_bench::DeviceBuffer;

const int kLaunchesPerIteration = 4;
const int kIterations = 10;
const std::vector<int64_t> kSizes = {1 << 16, 1 << 20, 1 << 24, 1 << 26};
const double kExactTolerance = 0.0;
// Result checks in units of the type's epsilon: the device may fuse a * b + c (one rounding fewer than the host's
// two), and its exp is within a few ulps of the host's before the rounding to a narrow type.
template <typename T>
double epsilons(int count) {
  return count * static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <typename T, int MapOptions>
using Map1D = Eigen::TensorMap<Eigen::Tensor<T, 1>, MapOptions>;

// One elementwise benchmark: NumInputs random device buffers, `expression(device, out, inputs)` enqueued on the
// executor, `reference(host_inputs, i)` checked once outside the timed loop, then the timing.
template <typename T, int MapOptions, int NumInputs, typename Expression, typename Reference>
void bench_elementwise(benchmark::State& state, Expression expression, Reference reference, double tolerance,
                       const char* name) {
  const int64_t n = state.range(0);
  std::vector<DeviceBuffer<T>> inputs;
  std::vector<Map1D<T, MapOptions>> maps;
  inputs.reserve(NumInputs);
  for (int k = 0; k < NumInputs; ++k) {
    inputs.emplace_back(n);
    inputs.back().fillRandom();
    maps.push_back(Map1D<T, MapOptions>(inputs.back().data(), n));
  }
  DeviceBuffer<T> out(n);
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  Map1D<T, MapOptions> tout(out.data(), n);
  const auto enqueue = [&] { expression(device, tout, maps); };
  eigen_bench::warmUp(device.stream(), enqueue);
  {
    std::vector<std::vector<T>> host;
    for (const auto& input : inputs) host.push_back(input.toHost());
    std::vector<T> expected(n);
    for (int64_t i = 0; i < n; ++i) expected[i] = reference(host, i);
    eigen_bench::requireClose(out.toHost(), expected, tolerance, name);
  }
  eigen_bench::timeLaunches(state, device.stream(), enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond((NumInputs + 1.0) * n * sizeof(T));
}

template <typename T, int MapOptions>
void BM_Add(benchmark::State& state) {
  using Map = Map1D<T, MapOptions>;
  bench_elementwise<T, MapOptions, 2>(
      state,
      [](const Eigen::GpuDevice& device, Map& out, const std::vector<Map>& in) { out.device(device) = in[0] + in[1]; },
      [](const std::vector<std::vector<T>>& x, int64_t i) { return static_cast<T>(x[0][i] + x[1][i]); },
      kExactTolerance, "add");
}

template <typename T, int MapOptions>
void BM_MulAdd(benchmark::State& state) {
  using Map = Map1D<T, MapOptions>;
  bench_elementwise<T, MapOptions, 3>(
      state,
      [](const Eigen::GpuDevice& device, Map& out, const std::vector<Map>& in) {
        out.device(device) = in[0] * in[1] + in[2];
      },
      [](const std::vector<std::vector<T>>& x, int64_t i) { return static_cast<T>(x[0][i] * x[1][i] + x[2][i]); },
      epsilons<T>(4), "mul_add");
}

template <typename T, int MapOptions>
void BM_Exp(benchmark::State& state) {
  using Map = Map1D<T, MapOptions>;
  bench_elementwise<T, MapOptions, 1>(
      state,
      [](const Eigen::GpuDevice& device, Map& out, const std::vector<Map>& in) { out.device(device) = in[0].exp(); },
      [](const std::vector<std::vector<T>>& x, int64_t i) {
        return static_cast<T>(std::exp(static_cast<float>(x[0][i])));
      },
      epsilons<T>(8), "exp");
}

// What a kernel written by hand for this one expression reaches: 16-byte loads and stores, a grid sized to keep
// every multiprocessor busy, no evaluator.
__global__ void add_float4_kernel(const float4* a, const float4* b, float4* out, int n4) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    const float4 x = a[i];
    const float4 y = b[i];
    out[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

void BM_AddHandWritten(benchmark::State& state) {
  const int64_t n = state.range(0);
  DeviceBuffer<float> a(n), b(n), out(n);
  a.fillRandom();
  b.fillRandom();
  int device_id = 0;
  EIGEN_GPU_RUNTIME_CHECK(cudaGetDevice(&device_id));
  int multiprocessors = 0;
  EIGEN_GPU_RUNTIME_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device_id));
  const int n4 = static_cast<int>(n / 4);
  const int block = 256;
  const int grid = std::min((n4 + block - 1) / block, multiprocessors * 8);
  cudaStream_t stream = nullptr;
  const auto enqueue = [&] {
    add_float4_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float4*>(a.data()),
                                                  reinterpret_cast<const float4*>(b.data()),
                                                  reinterpret_cast<float4*>(out.data()), n4);
    EIGEN_GPU_RUNTIME_CHECK(cudaGetLastError());
  };
  eigen_bench::warmUp(stream, enqueue);
  {
    const std::vector<float> ha = a.toHost(), hb = b.toHost();
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) expected[i] = ha[i] + hb[i];
    eigen_bench::requireClose(out.toHost(), expected, kExactTolerance, "add_hand_written");
  }
  eigen_bench::timeLaunches(state, stream, enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond(3.0 * n * sizeof(float));
}

// The copy roofline: one read and one write per element, no arithmetic.
void BM_D2DMemcpy(benchmark::State& state) {
  const int64_t n = state.range(0);
  DeviceBuffer<float> a(n), out(n);
  a.fillRandom();
  cudaStream_t stream = nullptr;
  const auto enqueue = [&] {
    EIGEN_GPU_RUNTIME_CHECK(cudaMemcpyAsync(out.data(), a.data(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  };
  eigen_bench::warmUp(stream, enqueue);
  eigen_bench::timeLaunches(state, stream, enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond(2.0 * n * sizeof(float));
}

}  // namespace

// bfloat16 has no device packet, so it is registered once, as the scalar path it is.
// bfloat16 has no device packet, so it is registered once, as the scalar path it is.
#define EIGEN_GPU_ELEMENTWISE_BENCHMARKS(NAME)                    \
  EIGEN_GPU_BENCHMARK_SIZES(NAME, float, Eigen::Aligned);         \
  EIGEN_GPU_BENCHMARK_SIZES(NAME, float, Eigen::Unaligned);       \
  EIGEN_GPU_BENCHMARK_SIZES(NAME, Eigen::half, Eigen::Aligned);   \
  EIGEN_GPU_BENCHMARK_SIZES(NAME, Eigen::half, Eigen::Unaligned); \
  EIGEN_GPU_BENCHMARK_SIZES(NAME, Eigen::bfloat16, Eigen::Unaligned);

EIGEN_GPU_ELEMENTWISE_BENCHMARKS(BM_Add)
EIGEN_GPU_ELEMENTWISE_BENCHMARKS(BM_MulAdd)
EIGEN_GPU_ELEMENTWISE_BENCHMARKS(BM_Exp)
EIGEN_GPU_BENCHMARK(BM_AddHandWritten)->ArgsProduct({kSizes});
EIGEN_GPU_BENCHMARK(BM_D2DMemcpy)->ArgsProduct({kSizes});

EIGEN_GPU_BENCHMARK_MAIN()
