// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <random>
namespace {
template <typename StorageIndex, int Size = Eigen::Dynamic>
void BM_PermutationInverseProduct(benchmark::State& state) {
  using Permutation = Eigen::PermutationMatrix<Size, Size, StorageIndex>;
  const Eigen::Index size = Size == Eigen::Dynamic ? state.range(0) : Size;
  Permutation lhs(size), rhs(size);
  lhs.setIdentity();
  rhs.setIdentity();
  std::mt19937 random(42);
  std::shuffle(lhs.indices().data(), lhs.indices().data() + size, random);
  std::shuffle(rhs.indices().data(), rhs.indices().data() + size, random);
  const Permutation reference = lhs * rhs.inverse();
  for (Eigen::Index i = 0; i < size; ++i) {
    if (reference.indices()[rhs.indices()[i]] != lhs.indices()[i]) {
      state.SkipWithError("Incorrect inverse permutation product");
      return;
    }
  }
  for (auto _ : state) {
    Permutation result = lhs * rhs.inverse();
    benchmark::DoNotOptimize(result.indices().data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK_TEMPLATE(BM_PermutationInverseProduct, int, 4);
BENCHMARK_TEMPLATE(BM_PermutationInverseProduct, int, 16);
BENCHMARK_TEMPLATE(BM_PermutationInverseProduct, int)->Arg(8)->Arg(32)->Arg(256)->Arg(4096)->Arg(65536)->Arg(1048576);
BENCHMARK_TEMPLATE(BM_PermutationInverseProduct, std::int64_t, 4);
BENCHMARK_TEMPLATE(BM_PermutationInverseProduct, std::int64_t, 16);
BENCHMARK_TEMPLATE(BM_PermutationInverseProduct, std::int64_t)
    ->Arg(8)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576);
}  // namespace
