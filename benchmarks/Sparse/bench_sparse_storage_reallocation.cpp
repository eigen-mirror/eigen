// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/SparseCore>

static void BM_SparseStorageReallocation(benchmark::State& state) {
  const Eigen::Index size = state.range(0);
  Eigen::SparseVector<double> vector(size);
  vector.reserve(size);
  for (Eigen::Index i = 0; i < size; ++i) vector.insertBack(i) = double(i + 1);
  for (auto _ : state) {
    vector.reserve(size);
    vector.data().squeeze();
    benchmark::DoNotOptimize(vector.valuePtr());
    benchmark::DoNotOptimize(vector.innerIndexPtr());
    benchmark::ClobberMemory();
  }
  if (vector.nonZeros() != size || vector.coeff(size - 1) != double(size))
    state.SkipWithError("reallocation changed the sparse coefficients");
}

BENCHMARK(BM_SparseStorageReallocation)->Arg(16)->Arg(256)->Arg(4096)->Arg(65536);
