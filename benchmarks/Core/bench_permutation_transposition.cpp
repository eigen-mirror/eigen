// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <utility>
namespace {
template <typename StorageIndex, int Size = Eigen::Dynamic, bool SameIndex = false>
void BM_PermutationLeftTransposition(benchmark::State& state) {
  using Permutation = Eigen::PermutationMatrix<Size, Size, StorageIndex>;
  const Eigen::Index size = Size == Eigen::Dynamic ? state.range(0) : Size;
  Permutation permutation(size);
  permutation.setIdentity();
  std::mt19937 random(42);
  std::shuffle(permutation.indices().data(), permutation.indices().data() + size, random);
  std::array<std::pair<Eigen::Index, Eigen::Index>, 1024> pairs;
  for (auto& pair : pairs) {
    pair.first = random() % size;
    pair.second = SameIndex ? pair.first : (pair.first + 1 + random() % (size - 1)) % size;
  }
  Permutation expected = permutation;
  for (std::size_t k = 0; k < 16; ++k) {
    const auto& pair = pairs[k];
    permutation.applyTranspositionOnTheLeft(pair.first, pair.second);
    for (Eigen::Index i = 0; i < size; ++i) {
      auto& index = expected.indices()[i];
      if (index == pair.first)
        index = StorageIndex(pair.second);
      else if (index == pair.second)
        index = StorageIndex(pair.first);
    }
  }
  if (permutation.indices() != expected.indices()) {
    state.SkipWithError("Incorrect left transposition");
    return;
  }
  std::size_t k = 0;
  for (auto _ : state) {
    const auto& pair = pairs[k++ % pairs.size()];
    permutation.applyTranspositionOnTheLeft(pair.first, pair.second);
    benchmark::DoNotOptimize(permutation.indices().data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK_TEMPLATE(BM_PermutationLeftTransposition, int, 4);
BENCHMARK_TEMPLATE(BM_PermutationLeftTransposition, int, 8);
BENCHMARK_TEMPLATE(BM_PermutationLeftTransposition, int, 16);
BENCHMARK_TEMPLATE(BM_PermutationLeftTransposition, int)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576);
BENCHMARK_TEMPLATE(BM_PermutationLeftTransposition, std::int64_t)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576);
BENCHMARK_TEMPLATE(BM_PermutationLeftTransposition, int, Eigen::Dynamic, true)->Arg(8)->Arg(4096);
}  // namespace
