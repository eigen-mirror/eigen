// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <algorithm>
#include <numeric>

template <int Stride, bool Reverse>
static void BM_StridedIterator(benchmark::State& state) {
  const Eigen::Index size = state.range(0);
  const Eigen::Index stride = state.range(1);
  // Padding also makes the pre-fix iterator's end pointer valid for the comparison build.
  Eigen::VectorXd storage(size * Eigen::numext::abs(stride) + 1);
  Eigen::Map<Eigen::VectorXd, Eigen::Unaligned, Eigen::InnerStride<Stride>> values(
      storage.data() + (stride < 0 ? storage.size() - 1 : 0), size, Eigen::InnerStride<Stride>(stride));
  for (Eigen::Index i = 0; i < size; ++i) values[i] = double(i + 1);
  const double expected = double(size) * double(size + 1) / 2;
  if (std::accumulate(values.begin(), values.end(), 0.0) != expected) {
    state.SkipWithError("Incorrect iterator sum");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    EIGEN_IF_CONSTEXPR (Reverse) {
      std::reverse(values.begin(), values.end());
      benchmark::DoNotOptimize(values.data());
    } else {
      double sum = std::accumulate(values.begin(), values.end(), 0.0);
      benchmark::DoNotOptimize(sum);
    }
  }
  const bool reversed = Reverse && state.iterations() % 2 != 0;
  for (Eigen::Index i = 0; i < size; ++i) {
    if (values[i] != double(reversed ? size - i : i + 1)) {
      state.SkipWithError("Incorrect iterator traversal");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK_TEMPLATE(BM_StridedIterator, 1, false)->ArgsProduct({{16, 256, 4096}, {1}});
BENCHMARK_TEMPLATE(BM_StridedIterator, 2, false)->ArgsProduct({{16, 256, 4096}, {2}});
BENCHMARK_TEMPLATE(BM_StridedIterator, Eigen::Dynamic, false)->ArgsProduct({{16, 256, 4096}, {-2, 1, 2}});
BENCHMARK_TEMPLATE(BM_StridedIterator, 1, true)->ArgsProduct({{16, 256, 4096}, {1}});
BENCHMARK_TEMPLATE(BM_StridedIterator, 2, true)->ArgsProduct({{16, 256, 4096}, {2}});
BENCHMARK_TEMPLATE(BM_StridedIterator, Eigen::Dynamic, true)->ArgsProduct({{16, 256, 4096}, {-2, 1, 2}});
