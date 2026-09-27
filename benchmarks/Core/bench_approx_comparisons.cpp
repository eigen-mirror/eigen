// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar, int Size, int Operation>
static void BM_ApproxComparison(benchmark::State& state) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Size, 1>;
  const Index size = Index(state.range(0));
  const Real scales[] = {Real(1), NumTraits<Real>::highest() / Real(32),
                         (numext::numeric_limits<Real>::min)() * Real(32)};
  const Real scale = scales[state.range(1)];
  const Real precision = Real(0.125);
  Vector x = Vector::Constant(size, Scalar(scale));
  Vector y = x * (Real(1) + precision / Real(2));
  if (Operation != 0) x *= precision / Real(4);
  const Real reference = Real(4) * scale;
  const auto compare = [&]() {
    if (Operation == 0) return x.isApprox(y, precision);
    if (Operation == 1) return x.isMuchSmallerThan(y, precision);
    return x.isMuchSmallerThan(reference, precision);
  };
  // The scalar reference has room for at most 256 coefficients at this scale.
  const bool expected = Operation != 2 || size <= 256;
  if (compare() != expected) {
    state.SkipWithError("Incorrect approximate comparison");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(compare());
  }
  state.SetItemsProcessed(state.iterations() * size);
}

#define APPROX_COMPARISON_BENCHMARKS(Scalar, Operation)                                         \
  BENCHMARK_TEMPLATE(BM_ApproxComparison, Scalar, 4, Operation)->ArgsProduct({{4}, {0, 1, 2}}); \
  BENCHMARK_TEMPLATE(BM_ApproxComparison, Scalar, Dynamic, Operation)->ArgsProduct({{16, 256, 4096}, {0, 1, 2}})

APPROX_COMPARISON_BENCHMARKS(float, 0);
APPROX_COMPARISON_BENCHMARKS(double, 0);
APPROX_COMPARISON_BENCHMARKS(std::complex<double>, 0);
APPROX_COMPARISON_BENCHMARKS(double, 1);
APPROX_COMPARISON_BENCHMARKS(double, 2);
APPROX_COMPARISON_BENCHMARKS(half, 0);
APPROX_COMPARISON_BENCHMARKS(bfloat16, 0);

#undef APPROX_COMPARISON_BENCHMARKS
