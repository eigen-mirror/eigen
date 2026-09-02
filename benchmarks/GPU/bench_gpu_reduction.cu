// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Tensor reductions on GpuDevice against CUB, which is the reference implementation of these kernels: full sum and
// max over 2^20 to 2^26 elements, and the two partial reductions of a matrix, over the inner dimension (unit
// stride, the coalesced direction) and the outer one. Bytes per launch count the input read once.

#include "gpu_bench_common.h"

#include <cub/cub.cuh>

namespace {

using eigen_bench::DeviceBuffer;

const int kLaunchesPerIteration = 4;
const int kIterations = 10;
const std::vector<int64_t> kFullSizes = {1 << 20, 1 << 24, 1 << 26};
// The three matrix shapes the partial reductions consume are registered as (rows, columns) pairs rather than as a
// product of two lists, whose cross product would ask for tensors of up to 2^40 elements: the square case, a wide
// one whose reduced extent is short, and a tall one whose reduced extent is long.

template <typename T>
using Vec = Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned>;
template <typename T>
using Mat = Eigen::TensorMap<Eigen::Tensor<T, 2>, Eigen::Aligned>;
template <typename T>
using Scalar0 = Eigen::TensorMap<Eigen::Tensor<T, 0>, Eigen::Aligned>;

// Compare against a compensated Neumaier sum. For these random inputs, use an error scale of
// eps * log2(n) * ||x||_2 for trees and eps * sqrt(n) * ||x||_2 for sequential accumulation, with a safety factor.
// These are statistical tolerances for this benchmark's inputs, not worst-case bounds for arbitrary data.
enum Accumulation { kTree, kInOrder };

struct SumReference {
  double value;
  double tolerance;
};

template <typename T, typename Element>
SumReference referenceSum(Accumulation accumulation, int64_t count, Element element) {
  double sum = 0.0;
  double compensation = 0.0;
  double sum_squares = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    const double x = static_cast<double>(element(i));
    const double total = sum + x;
    compensation += (std::fabs(sum) >= std::fabs(x)) ? (sum - total) + x : (x - total) + sum;
    sum = total;
    sum_squares += x * x;
  }
  const double depth = std::log2(static_cast<double>(count)) + 1.0;
  const double growth = (accumulation == kTree) ? depth : std::fmax(depth, std::sqrt(static_cast<double>(count)));
  const double epsilon = static_cast<double>(std::numeric_limits<T>::epsilon());
  return {sum + compensation, 16.0 * growth * epsilon * std::sqrt(sum_squares)};
}

void requireSum(double got, const SumReference& reference, const char* what, int64_t index) {
  if (!(std::fabs(got - reference.value) <= reference.tolerance)) {
    std::fprintf(stderr, "%s: output %lld is %g, expected %g (tolerance %g)\n", what, static_cast<long long>(index),
                 got, reference.value, reference.tolerance);
    std::abort();
  }
}

template <typename T>
void checkSum(const DeviceBuffer<T>& out, const std::vector<T>& input, int64_t count, const char* what) {
  const SumReference reference = referenceSum<T>(kTree, count, [&](int64_t i) { return input[i]; });
  requireSum(static_cast<double>(out.toHost()[0]), reference, what, 0);
}

// One reference per output: dimension 0 sums each column of the column-major matrix, dimension 1 each row. A
// benchmark that never reads its output cannot tell a fast kernel from a wrong one, so both partial reductions
// and the CUB baseline are checked once before the timed loop.
template <typename T, int Dim>
void checkPartialSums(const DeviceBuffer<T>& out, const std::vector<T>& input, int64_t rows, int64_t cols,
                      const char* what) {
  const std::vector<T> got = out.toHost();
  const int64_t out_size = (Dim == 0) ? cols : rows;
  const int64_t reduced = (Dim == 0) ? rows : cols;
  for (int64_t o = 0; o < out_size; ++o) {
    const SumReference reference = referenceSum<T>(
        kInOrder, reduced, [&](int64_t i) { return (Dim == 0) ? input[i + o * rows] : input[o + i * rows]; });
    requireSum(static_cast<double>(got[o]), reference, what, o);
  }
}

template <typename T>
void BM_FullReduceSum(benchmark::State& state) {
  const int64_t n = state.range(0);
  DeviceBuffer<T> in(n), out(1);
  in.fillRandom();
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  Vec<T> tin(in.data(), n);
  Scalar0<T> tout(out.data());
  const auto enqueue = [&] { tout.device(device) = tin.sum(); };
  eigen_bench::warmUp(device.stream(), enqueue);
  checkSum(out, in.toHost(), n, "full_sum");
  eigen_bench::timeLaunches(state, device.stream(), enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond(static_cast<double>(n) * sizeof(T));
}

template <typename T>
void BM_FullReduceMax(benchmark::State& state) {
  const int64_t n = state.range(0);
  DeviceBuffer<T> in(n), out(1);
  in.fillRandom();
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  Vec<T> tin(in.data(), n);
  Scalar0<T> tout(out.data());
  const auto enqueue = [&] { tout.device(device) = tin.maximum(); };
  eigen_bench::warmUp(device.stream(), enqueue);
  {
    const std::vector<T> host = in.toHost();
    const T expected = *std::max_element(host.begin(), host.end());
    eigen_bench::requireClose(out.toHost(), std::vector<T>{expected}, 0.0, "full_max");
  }
  eigen_bench::timeLaunches(state, device.stream(), enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond(static_cast<double>(n) * sizeof(T));
}

// Dimension 0 of a column-major tensor is unit stride within a column; dimension 1 strides by the number of rows,
// which is the case the double specialization used to fall back to the generic evaluator for.
template <typename T, int Dim>
void bench_partial_reduce(benchmark::State& state) {
  const int64_t rows = state.range(0), cols = state.range(1);
  const int64_t out_size = (Dim == 0) ? cols : rows;
  DeviceBuffer<T> in(rows * cols), out(out_size);
  in.fillRandom();
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  Mat<T> tin(in.data(), rows, cols);
  Vec<T> tout(out.data(), out_size);
  const Eigen::array<Eigen::Index, 1> dims{Dim};
  const auto enqueue = [&] { tout.device(device) = tin.sum(dims); };
  eigen_bench::warmUp(device.stream(), enqueue);
  checkPartialSums<T, Dim>(out, in.toHost(), rows, cols, Dim == 0 ? "reduce_inner" : "reduce_outer");
  eigen_bench::timeLaunches(state, device.stream(), enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond(static_cast<double>(rows * cols) * sizeof(T));
}

template <typename T>
void BM_ReduceInner(benchmark::State& state) {
  bench_partial_reduce<T, 0>(state);
}

template <typename T>
void BM_ReduceOuter(benchmark::State& state) {
  bench_partial_reduce<T, 1>(state);
}

// CUB's own kernels, the baseline these are measured against.
template <typename T>
void BM_CubFullReduceSum(benchmark::State& state) {
  const int64_t n = state.range(0);
  DeviceBuffer<T> in(n), out(1);
  in.fillRandom();
  cudaStream_t stream = nullptr;
  size_t temp_bytes = 0;
  EIGEN_GPU_RUNTIME_CHECK(
      cub::DeviceReduce::Sum(nullptr, temp_bytes, in.data(), out.data(), static_cast<int>(n), stream));
  DeviceBuffer<char> temp(temp_bytes);
  const auto enqueue = [&] {
    size_t bytes = temp_bytes;
    EIGEN_GPU_RUNTIME_CHECK(
        cub::DeviceReduce::Sum(temp.data(), bytes, in.data(), out.data(), static_cast<int>(n), stream));
  };
  eigen_bench::warmUp(stream, enqueue);
  checkSum(out, in.toHost(), n, "cub_full_sum");
  eigen_bench::timeLaunches(state, stream, enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond(static_cast<double>(n) * sizeof(T));
}

// One segment per column, which is what an inner reduction of a column-major matrix computes.
template <typename T>
void BM_CubSegmentedReduceSum(benchmark::State& state) {
  const int64_t rows = state.range(0), cols = state.range(1);
  DeviceBuffer<T> in(rows * cols), out(cols);
  in.fillRandom();
  std::vector<int> offsets(cols + 1);
  for (int64_t c = 0; c <= cols; ++c) offsets[c] = static_cast<int>(c * rows);
  DeviceBuffer<int> d_offsets(offsets.size());
  EIGEN_GPU_RUNTIME_CHECK(
      cudaMemcpy(d_offsets.data(), offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  cudaStream_t stream = nullptr;
  size_t temp_bytes = 0;
  EIGEN_GPU_RUNTIME_CHECK(cub::DeviceSegmentedReduce::Sum(nullptr, temp_bytes, in.data(), out.data(),
                                                          static_cast<int>(cols), d_offsets.data(),
                                                          d_offsets.data() + 1, stream));
  DeviceBuffer<char> temp(temp_bytes);
  const auto enqueue = [&] {
    size_t bytes = temp_bytes;
    EIGEN_GPU_RUNTIME_CHECK(cub::DeviceSegmentedReduce::Sum(temp.data(), bytes, in.data(), out.data(),
                                                            static_cast<int>(cols), d_offsets.data(),
                                                            d_offsets.data() + 1, stream));
  };
  eigen_bench::warmUp(stream, enqueue);
  checkPartialSums<T, 0>(out, in.toHost(), rows, cols, "cub_segmented_sum");
  eigen_bench::timeLaunches(state, stream, enqueue, kLaunchesPerIteration);
  state.counters["bytes_per_second"] = eigen_bench::bytesPerSecond(static_cast<double>(rows * cols) * sizeof(T));
}

}  // namespace

#define EIGEN_GPU_FULL_REDUCTION_BENCHMARKS(NAME)                                                       \
  BENCHMARK_TEMPLATE(NAME, float)->ArgsProduct({kFullSizes})->UseManualTime()->Iterations(kIterations); \
  BENCHMARK_TEMPLATE(NAME, double)->ArgsProduct({kFullSizes})->UseManualTime()->Iterations(kIterations);

// Explicit shapes rather than a product: the product of the two ranges asks for tensors of up to 2^40 elements.
// Args is (rows, columns), so for the outer reduction it reads as (outputs, reduced extent) and the last three
// shapes straddle the band OuterReducer::run admits doubles to OuterReductionKernel in: a reduced extent below
// its floor, a shape inside the band, and an output count above its ceiling on a 36-multiprocessor device. For
// the inner reduction and the CUB baseline the two extents swap roles.
#define EIGEN_GPU_PARTIAL_REDUCTION_SHAPES(NAME, TYPE) \
  BENCHMARK_TEMPLATE(NAME, TYPE)                       \
      ->Args({1 << 10, 1 << 10})                       \
      ->Args({1 << 4, 1 << 20})                        \
      ->Args({1 << 20, 1 << 4})                        \
      ->Args({1 << 10, 33})                            \
      ->Args({1 << 9, 1 << 7})                         \
      ->Args({1 << 13, 1 << 9})                        \
      ->UseManualTime()                                \
      ->Iterations(kIterations);

#define EIGEN_GPU_PARTIAL_REDUCTION_BENCHMARKS(NAME) \
  EIGEN_GPU_PARTIAL_REDUCTION_SHAPES(NAME, float)    \
  EIGEN_GPU_PARTIAL_REDUCTION_SHAPES(NAME, double)

EIGEN_GPU_FULL_REDUCTION_BENCHMARKS(BM_FullReduceSum)
EIGEN_GPU_FULL_REDUCTION_BENCHMARKS(BM_FullReduceMax)
EIGEN_GPU_FULL_REDUCTION_BENCHMARKS(BM_CubFullReduceSum)
EIGEN_GPU_PARTIAL_REDUCTION_BENCHMARKS(BM_ReduceInner)
EIGEN_GPU_PARTIAL_REDUCTION_BENCHMARKS(BM_ReduceOuter)
EIGEN_GPU_PARTIAL_REDUCTION_BENCHMARKS(BM_CubSegmentedReduceSum)

EIGEN_GPU_BENCHMARK_MAIN()
