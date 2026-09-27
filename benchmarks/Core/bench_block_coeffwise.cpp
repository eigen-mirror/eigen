// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

template <typename Scalar, bool UseBlock>
static void BM_BlockCoefficientProduct(benchmark::State& state) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  const Eigen::Index rows = state.range(0), cols = 20;
  const Mat big = Mat::Random(5 * rows, cols);
  const Mat packed = big.topRows(rows), rhs = Mat::Random(rows, cols);
  Mat out(rows, cols);
  auto evaluate = [&] {
    if (UseBlock)
      out = big.block(0, 0, rows, cols).array() * rhs.array();
    else
      out = packed.array() * rhs.array();
  };
  evaluate();
  const Mat expected = packed.cwiseProduct(rhs);
  if (out != expected) {
    state.SkipWithError("incorrect coefficient product");
    return;
  }
  for (auto _ : state) {
    evaluate();
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * rows * cols);
}

#define BLOCK_COEFFICIENT_SIZES ->Arg(96)->Arg(100)->Arg(101)->Arg(104)->Arg(128)
BENCHMARK_TEMPLATE(BM_BlockCoefficientProduct, float, false) BLOCK_COEFFICIENT_SIZES;
BENCHMARK_TEMPLATE(BM_BlockCoefficientProduct, float, true) BLOCK_COEFFICIENT_SIZES;
BENCHMARK_TEMPLATE(BM_BlockCoefficientProduct, double, false) BLOCK_COEFFICIENT_SIZES;
BENCHMARK_TEMPLATE(BM_BlockCoefficientProduct, double, true) BLOCK_COEFFICIENT_SIZES;
#undef BLOCK_COEFFICIENT_SIZES
