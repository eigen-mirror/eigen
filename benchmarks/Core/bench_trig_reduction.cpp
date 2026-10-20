// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <cmath>
#include <random>

using namespace Eigen;

// Domains exercise the small reduction, double-word reduction, and libm fallback.
template <int Function>
static void BM_TrigReduction(benchmark::State& state) {
  const Index size = state.range(0);
  const int domain = static_cast<int>(state.range(1));
  const double lower[] = {-10.0, 15.0, 1e4, 1e15};
  const double upper[] = {10.0, 1e4, 1e13, 2e15};
  // MPFR 4.2.1 at 256 bits, rounded to binary64; checked outside the timed loop.
  const double reference[][4] = {
      {0.3, 0.29552020666133955, 0.95533648912560598, 0.30933624960962325},
      {1000.0, 0.82687954053200252, 0.56237907629070294, 1.4703241557027185},
      {1e12, -0.61123870237688949, 0.79144630185289022, -0.77230596813187613},
      {1e15, 0.85827279317023586, -0.51319373778697031, -1.672414782127583},
  };
  std::mt19937_64 random(0x3389);
  ArrayXd x(size), y(size);
  for (Index i = 0; i < size; ++i) {
    const double fraction = double(random() >> 11) / 9007199254740992.0;
    x(i) = lower[domain] + (upper[domain] - lower[domain]) * fraction;
    if (i % 2) x(i) = -x(i);
  }
  x(0) = reference[domain][0];
  auto evaluate = [&]() {
    EIGEN_IF_CONSTEXPR (Function == 0) y = x.sin();
    EIGEN_IF_CONSTEXPR (Function == 1) y = x.cos();
    EIGEN_IF_CONSTEXPR (Function == 2) y = x.tan();
  };
  evaluate();
  const double expected = reference[domain][Function + 1];
  const double bound = 4 * NumTraits<double>::epsilon() * (std::max)(1.0, std::abs(expected));
  if (!(std::abs(y(0) - expected) <= bound)) {
    state.SkipWithError("trig result disagrees with MPFR reference");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(x.data());
    evaluate();
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK_TEMPLATE(BM_TrigReduction, 0)
    ->Name("SinDoubleReduction")
    ->ArgsProduct({{256, 4096, 65536}, {0, 1, 2, 3}})
    ->ArgNames({"size", "domain"});
BENCHMARK_TEMPLATE(BM_TrigReduction, 1)
    ->Name("CosDoubleReduction")
    ->ArgsProduct({{256, 4096, 65536}, {0, 1, 2, 3}})
    ->ArgNames({"size", "domain"});
BENCHMARK_TEMPLATE(BM_TrigReduction, 2)
    ->Name("TanDoubleReduction")
    ->ArgsProduct({{256, 4096, 65536}, {0, 1, 2, 3}})
    ->ArgNames({"size", "domain"});
