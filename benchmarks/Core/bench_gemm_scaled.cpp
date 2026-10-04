// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

using Scalar = SCALAR;
using Mat = Matrix<Scalar, Dynamic, Dynamic>;

template <typename T>
struct ScaledGemmAlpha {
  static T value() { return T(1.375); }
};
template <typename T>
struct ScaledGemmAlpha<std::complex<T>> {
  static std::complex<T> value() { return std::complex<T>(T(1.375), T(-0.625)); }
};

template <typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm_scaled(const A& a, const B& b, C& c, Scalar alpha) {
  c.noalias() += alpha * (a * b);
}

static void BM_EigenGemmScaled(benchmark::State& state) {
  const Index m = state.range(0), n = state.range(1), k = state.range(2);
  const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
  const Scalar alpha = ScaledGemmAlpha<Scalar>::value();
  Mat c = Mat::Ones(m, n);
  // Check a sampled column independently without timing the reference product.
  const Matrix<Scalar, Dynamic, 1> expected = Matrix<Scalar, Dynamic, 1>::Ones(m) + alpha * a.lazyProduct(b.col(0));
  gemm_scaled(a, b, c, alpha);
  if (!c.col(0).isApprox(expected)) {
    state.SkipWithError("scaled GEMM validation failed");
    return;
  }
  for (auto _ : state) {
    c.setZero();
    gemm_scaled(a, b, c, alpha);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  constexpr double flops = NumTraits<Scalar>::IsComplex ? 8.0 : 2.0;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops * m * n * k, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

template <typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm_subtract(const A& a, const B& b, C& c) {
  c.noalias() -= a * b;
}

// C -= A*B: alpha = -1, the real-alpha store path for complex scalars.
static void BM_EigenGemmSubtract(benchmark::State& state) {
  const Index m = state.range(0), n = state.range(1), k = state.range(2);
  const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
  Mat c = Mat::Ones(m, n);
  const Matrix<Scalar, Dynamic, 1> expected = Matrix<Scalar, Dynamic, 1>::Ones(m) - a.lazyProduct(b.col(0));
  gemm_subtract(a, b, c);
  if (!c.col(0).isApprox(expected)) {
    state.SkipWithError("GEMM subtraction validation failed");
    return;
  }
  for (auto _ : state) {
    c.setZero();
    gemm_subtract(a, b, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  constexpr double flops = NumTraits<Scalar>::IsComplex ? 8.0 : 2.0;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops * m * n * k, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// clang-format off
#define EIGEN_GEMM_SCALED_SHAPES \
    ->Args({8, 8, 8})->Args({16, 16, 16})->Args({32, 32, 32})->Args({64, 64, 64})->Args({128, 128, 128}) \
    ->ArgsProduct({{8, 24, 40, 264, 520, 16, 32, 256, 1024}, {256, 1024}, {256, 1024}}) \
    ->Args({4096, 96, 96})->Args({4096, 128, 128})->Args({4096, 144, 144}) \
    ->Args({4096, 160, 160})->Args({4096, 176, 176})->Args({8192, 128, 128})
BENCHMARK(BM_EigenGemmScaled) EIGEN_GEMM_SCALED_SHAPES;
BENCHMARK(BM_EigenGemmSubtract) EIGEN_GEMM_SCALED_SHAPES;
// clang-format on
