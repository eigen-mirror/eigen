// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Tensor contraction on GpuDevice against cuBLAS, for square matrices from 256 to 2048 and a skinny shape. Eigen's
// contraction is a hand-written tile kernel from 2014 with no tensor cores, so the ratio here is the gap a library
// dispatch would close; reporting FLOP/s for both makes it a one-line read.

#include "gpu_bench_common.h"

#include <cublas_v2.h>

namespace {

using eigen_bench::DeviceBuffer;

const int kLaunchesPerIteration = 2;
const int kIterations = 10;

#define EIGEN_BENCH_CUBLAS_CHECK(expr)                                                 \
  do {                                                                                 \
    const cublasStatus_t eigen_bench_status = (expr);                                  \
    if (eigen_bench_status != CUBLAS_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "%s:%d: %s: cuBLAS status %d\n", __FILE__, __LINE__, #expr, \
                   static_cast<int>(eigen_bench_status));                              \
      std::abort();                                                                    \
    }                                                                                  \
  } while (0)

template <typename T>
using Mat = Eigen::TensorMap<Eigen::Tensor<T, 2>, Eigen::Aligned>;

template <typename T>
cublasStatus_t gemm(cublasHandle_t handle, int m, int n, int k, const T* a, const T* b, T* c);
template <>
cublasStatus_t gemm<float>(cublasHandle_t handle, int m, int n, int k, const float* a, const float* b, float* c) {
  const float alpha = 1.0f, beta = 0.0f;
  return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, m, b, k, &beta, c, m);
}
template <>
cublasStatus_t gemm<double>(cublasHandle_t handle, int m, int n, int k, const double* a, const double* b, double* c) {
  const double alpha = 1.0, beta = 0.0;
  return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, m, b, k, &beta, c, m);
}

// C = A * B, column major in both implementations. Validate every timed shape, including filtered runs.
template <typename T, bool UseCublas>
void runContraction(benchmark::State& state) {
  const int64_t m = state.range(0), k = state.range(1), n = state.range(2);
  DeviceBuffer<T> a(m * k), b(k * n), eigen_c(m * n), cublas_c(m * n);
  a.fillRandom();
  b.fillRandom();
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  Mat<T> ta(a.data(), m, k), tb(b.data(), k, n), tc(eigen_c.data(), m, n);
  const Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> dims{Eigen::IndexPair<Eigen::Index>(1, 0)};
  const auto eigen_enqueue = [&] { tc.device(device) = ta.contract(tb, dims); };
  cublasHandle_t handle = nullptr;
  EIGEN_BENCH_CUBLAS_CHECK(cublasCreate(&handle));
  EIGEN_BENCH_CUBLAS_CHECK(cublasSetStream(handle, device.stream()));
  const auto cublas_enqueue = [&] {
    EIGEN_BENCH_CUBLAS_CHECK(gemm<T>(handle, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), a.data(),
                                     b.data(), cublas_c.data()));
  };
  eigen_enqueue();
  cublas_enqueue();
  device.synchronize();
  // The implementations accumulate in different orders; allow error proportional to the contracted extent.
  eigen_bench::requireClose(eigen_c.toHost(), cublas_c.toHost(),
                            k * static_cast<double>(std::numeric_limits<T>::epsilon()), "contraction");
  const auto enqueue = [&] {
    if (UseCublas) {
      cublas_enqueue();
    } else {
      eigen_enqueue();
    }
  };
  eigen_bench::warmUp(device.stream(), enqueue);
  eigen_bench::timeLaunches(state, device.stream(), enqueue, kLaunchesPerIteration);
  state.counters["FLOP/s"] = benchmark::Counter(2.0 * m * n * k, benchmark::Counter::kIsIterationInvariantRate);
  EIGEN_BENCH_CUBLAS_CHECK(cublasDestroy(handle));
}

template <typename T>
void BM_Contraction(benchmark::State& state) {
  runContraction<T, false>(state);
}

template <typename T>
void BM_CublasGemm(benchmark::State& state) {
  runContraction<T, true>(state);
}

}  // namespace

#define EIGEN_GPU_CONTRACTION_SHAPES(NAME, TYPE) \
  BENCHMARK_TEMPLATE(NAME, TYPE)                 \
      ->Args({256, 256, 256})                    \
      ->Args({512, 512, 512})                    \
      ->Args({1024, 1024, 1024})                 \
      ->Args({2048, 2048, 2048})                 \
      ->Args({4096, 64, 4096})                   \
      ->UseManualTime()                          \
      ->Iterations(kIterations);

EIGEN_GPU_CONTRACTION_SHAPES(BM_Contraction, float)
EIGEN_GPU_CONTRACTION_SHAPES(BM_Contraction, double)
EIGEN_GPU_CONTRACTION_SHAPES(BM_CublasGemm, float)
EIGEN_GPU_CONTRACTION_SHAPES(BM_CublasGemm, double)
EIGEN_GPU_BENCHMARK_MAIN()
