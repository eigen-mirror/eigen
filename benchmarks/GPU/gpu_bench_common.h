// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BENCHMARKS_GPU_BENCH_COMMON_H
#define EIGEN_BENCHMARKS_GPU_BENCH_COMMON_H

// Shared by the benchmarks under benchmarks/GPU: device buffers,
// event-based timing of a batch of launches, result checking outside the timed region, and a main() that records
// the device in the benchmark context. EIGEN_USE_GPU is defined by the CMake target.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <contrib/Eigen/Tensor>

// Every runtime call goes through EIGEN_GPU_RUNTIME_CHECK (Eigen/src/Core/util/GpuRuntime.h, in scope through the
// Tensor header): under the NDEBUG these targets define it reports `file:line: call: name: description` and aborts.

namespace eigen_bench {

// A device allocation of `size` elements, optionally filled from the host with values in [-1, 1].
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) : size_(size) {
    EIGEN_GPU_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)));
  }
  ~DeviceBuffer() { EIGEN_GPU_RUNTIME_CHECK(cudaFree(data_)); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : size_(other.size_), data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;  // cudaFree(nullptr) succeeds, so the moved-from destructor is fine
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }

  // Values in [-1, 1] from Eigen's own generator, which is deterministic across runs.
  void fillRandom() {
    const Eigen::Array<T, Eigen::Dynamic, 1> host = Eigen::Array<T, Eigen::Dynamic, 1>::Random(size_);
    EIGEN_GPU_RUNTIME_CHECK(cudaMemcpy(data_, host.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  std::vector<T> toHost() const {
    std::vector<T> host(size_);
    EIGEN_GPU_RUNTIME_CHECK(cudaMemcpy(host.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
  }

 private:
  size_t size_;
  T* data_ = nullptr;
};

// Elapsed device time between two events recorded on one stream.
class GpuEventTimer {
 public:
  GpuEventTimer() {
    EIGEN_GPU_RUNTIME_CHECK(cudaEventCreate(&start_));
    EIGEN_GPU_RUNTIME_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuEventTimer() {
    EIGEN_GPU_RUNTIME_CHECK(cudaEventDestroy(start_));
    EIGEN_GPU_RUNTIME_CHECK(cudaEventDestroy(stop_));
  }
  void begin(cudaStream_t stream) { EIGEN_GPU_RUNTIME_CHECK(cudaEventRecord(start_, stream)); }
  // Records the end, waits for it and returns the elapsed seconds.
  double end(cudaStream_t stream) {
    EIGEN_GPU_RUNTIME_CHECK(cudaEventRecord(stop_, stream));
    EIGEN_GPU_RUNTIME_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    EIGEN_GPU_RUNTIME_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms * 1e-3;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

// Times `enqueue()` on the device: each benchmark iteration records events around `launches_per_iteration` calls
// and reports the device time per call through SetIterationTime, so the registration needs UseManualTime(). Event
// time measures the GPU-side duration including the gaps between launches, not the cost of the API calls; the
// latter is reported separately as the host time per enqueue, measured before the synchronization. Because the
// reported time is per launch, a registration must fix its iteration count (Iterations(n)): left to the
// framework's minimum-time rule, microsecond launches would be iterated hundreds of thousands of times.
template <typename Enqueue>
void timeLaunches(benchmark::State& state, cudaStream_t stream, Enqueue enqueue, int launches_per_iteration) {
  GpuEventTimer timer;
  double host_seconds = 0.0;
  int64_t iterations = 0;
  for (auto _ : state) {
    timer.begin(stream);
    const auto host_begin = std::chrono::steady_clock::now();
    for (int k = 0; k < launches_per_iteration; ++k) enqueue();
    host_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - host_begin).count();
    state.SetIterationTime(timer.end(stream) / launches_per_iteration);
    ++iterations;
  }
  state.counters["host_us_per_launch"] = host_seconds * 1e6 / (iterations * launches_per_iteration);
}

// One untimed launch, waited for: module load and any lazy setup stay out of the timed loop, and the result is
// what a correctness check reads. GpuDevice::synchronize() is this same wait on the device's own stream, so a
// benchmark holding a GpuDevice passes device.stream().
template <typename Enqueue>
void warmUp(cudaStream_t stream, Enqueue enqueue) {
  enqueue();
  EIGEN_GPU_RUNTIME_CHECK(cudaStreamSynchronize(stream));
}

// Bytes moved per launch, as a rate against the manual (device) time.
inline benchmark::Counter bytesPerSecond(double bytes_per_launch) {
  return benchmark::Counter(bytes_per_launch, benchmark::Counter::kIsIterationInvariantRate,
                            benchmark::Counter::kIs1024);
}

// Aborts when two results disagree beyond `tolerance` (relative to the larger magnitude, absolute below 1): a
// faster wrong kernel is not a result. Called once per configuration, outside the timed loop.
template <typename T>
void requireClose(const std::vector<T>& result, const std::vector<T>& reference, double tolerance, const char* what) {
  if (result.size() != reference.size()) {
    std::fprintf(stderr, "%s: size mismatch %zu vs %zu\n", what, result.size(), reference.size());
    std::abort();
  }
  for (size_t i = 0; i < result.size(); ++i) {
    const double a = static_cast<double>(result[i]);
    const double b = static_cast<double>(reference[i]);
    const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
    if (!std::isfinite(a) || !std::isfinite(b) || !(std::fabs(a - b) <= tolerance * scale)) {
      std::fprintf(stderr, "%s: element %zu is %g, expected %g\n", what, i, a, b);
      std::abort();
    }
  }
}

// The device a run reports, as the "gpu" context of the JSON output and a line on stdout, since a number without
// the device it was measured on is not a result.
inline void describeDevice() {
  int device = 0;
  EIGEN_GPU_RUNTIME_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  EIGEN_GPU_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device));
  int driver = 0, runtime = 0;
  EIGEN_GPU_RUNTIME_CHECK(cudaDriverGetVersion(&driver));
  EIGEN_GPU_RUNTIME_CHECK(cudaRuntimeGetVersion(&runtime));
  const std::string capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  benchmark::AddCustomContext("gpu", prop.name);
  benchmark::AddCustomContext("gpu_compute_capability", capability);
  benchmark::AddCustomContext("gpu_multiprocessors", std::to_string(prop.multiProcessorCount));
  benchmark::AddCustomContext("cuda_driver_version", std::to_string(driver));
  benchmark::AddCustomContext("cuda_runtime_version", std::to_string(runtime));
  std::printf("GPU: %s, compute capability %s, %d multiprocessors, driver %d, runtime %d\n", prop.name,
              capability.c_str(), prop.multiProcessorCount, driver, runtime);
}

inline int benchmarkMain(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  describeDevice();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

}  // namespace eigen_bench

// Every GPU benchmark reports device time per launch, so each registration needs manual time and a fixed
// iteration count (see timeLaunches). Both forms take the kIterations, and the sizes form the kSizes, that the
// benchmark file defines.
#define EIGEN_GPU_BENCHMARK(...) BENCHMARK(__VA_ARGS__)->UseManualTime()->Iterations(kIterations)
#define EIGEN_GPU_BENCHMARK_SIZES(...) \
  BENCHMARK_TEMPLATE(__VA_ARGS__)->ArgsProduct({kSizes})->UseManualTime()->Iterations(kIterations)

#define EIGEN_GPU_BENCHMARK_MAIN() \
  int main(int argc, char** argv) { return eigen_bench::benchmarkMain(argc, argv); }

#endif  // EIGEN_BENCHMARKS_GPU_BENCH_COMMON_H
