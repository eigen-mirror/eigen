// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

namespace Eigen {
namespace {

template <typename Scalar>
void BM_Count(benchmark::State& state) {
  const Index size = state.range(0);
  Array<Scalar, Dynamic, 1> input(size);
  Index expected = 0;
  for (Index i = 0; i < size; ++i) {
    input[i] = Scalar(i % 3 != 0);
    expected += i % 3 != 0;
  }
  if (input.count() != expected) {
    state.SkipWithError("count does not match the scalar reference");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(input.count());
  }
  state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK_TEMPLATE(BM_Count, half)->Arg(16)->Arg(256)->Arg(4099);
BENCHMARK_TEMPLATE(BM_Count, bfloat16)->Arg(16)->Arg(256)->Arg(4099);

}  // namespace
}  // namespace Eigen
