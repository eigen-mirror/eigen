// Benchmarks for vectorized casts between integer and floating-point arrays.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>

using namespace Eigen;

// 64-bit integers of every magnitude; about one in six is at least 2^53, where double cannot hold it exactly.
template <typename Src>
static void BM_CastToFloat(benchmark::State& state) {
  const Index size = state.range(0);
  Array<Src, Dynamic, 1> input(size);
  std::mt19937_64 rng(42);
  for (Index i = 0; i < size; ++i) {
    const uint64_t bits = rng() >> (rng() % 64);
    input(i) = static_cast<Src>(std::is_signed<Src>::value && (rng() & 1) ? uint64_t(0) - (bits >> 1) : bits);
  }
  ArrayXf output = input.template cast<float>();
  for (Index i = 0; i < size; ++i) {
    // Within an ulp: a conversion that rounds twice through double is off by at most one ulp.
    const float expected = static_cast<float>(static_cast<double>(input(i)));
    if (!(std::abs(output(i) - expected) <= std::abs(expected) * NumTraits<float>::epsilon())) {
      state.SkipWithError("cast failed reference validation");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    output = input.template cast<float>();
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK_TEMPLATE(BM_CastToFloat, int64_t)->Arg(64)->Arg(4096)->Arg(262144);
BENCHMARK_TEMPLATE(BM_CastToFloat, uint64_t)->Arg(64)->Arg(4096)->Arg(262144);
