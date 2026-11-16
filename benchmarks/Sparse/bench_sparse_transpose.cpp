// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Sparse>
#include <random>
#include <set>
#include <vector>

using namespace Eigen;

using Scalar = double;
using SpMat = SparseMatrix<Scalar>;
using Triplets = std::vector<Triplet<Scalar, SpMat::StorageIndex>>;

static void fillMatrix(float density, int rows, int cols, SpMat& dst) {
  Triplets t;
  t.reserve(static_cast<std::size_t>(double(rows) * cols * density));
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (internal::random<float>(0, 1) < density) t.emplace_back(i, j, internal::random<Scalar>());
    }
  }
  dst.resize(rows, cols);
  dst.setFromTriplets(t.begin(), t.end());
}

static void BM_SparseTranspose(benchmark::State& state) {
  int n = state.range(0);
  float density = state.range(1) / 10000.0f;
  SpMat sm(n, n), result(n, n);
  fillMatrix(density, n, n, sm);
  for (auto _ : state) {
    result = sm.transpose();
    benchmark::DoNotOptimize(result.valuePtr());
  }
  state.counters["nnz"] = sm.nonZeros();
  state.counters["density%"] = density * 100;
}

// Args: {size, density*10000}: 1%, 0.5%, 0.1%, 0.04%
BENCHMARK(BM_SparseTranspose)->ArgsProduct({{1000, 10000}, {100, 50, 10, 4}});

static SpMat fromTriplets(Index rows, Index cols, const Triplets& t) {
  SpMat m(rows, cols);
  m.setFromTriplets(t.begin(), t.end());
  return m;
}

// 5-point Laplacian on an n x n grid.
static SpMat laplacian2d(int n) {
  Triplets t;
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x) {
      int i = y * n + x;
      t.emplace_back(i, i, 4);
      if (x > 0) t.emplace_back(i, i - 1, -1);
      if (x + 1 < n) t.emplace_back(i, i + 1, -1);
      if (y > 0) t.emplace_back(i, i - n, -1);
      if (y + 1 < n) t.emplace_back(i, i + n, -1);
    }
  return fromTriplets(n * n, n * n, t);
}

// perCol entries per column in uniformly random rows.
static SpMat randomColumns(int rows, int cols, int perCol, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<int> row(0, rows - 1);
  Triplets t;
  for (int j = 0; j < cols; ++j)
    for (int q = 0; q < perCol; ++q) t.emplace_back(row(g), j, Scalar(q + 1));
  return fromTriplets(rows, cols, t);
}

// A storage-order change of matrices whose destination vectors are written close together (banded) or anywhere
// (random rows, and a stencil times a random prolongator, the result an AMG setup converts).
static const SpMat& orderChangeCase(int c) {
  static const SpMat cases[] = {laplacian2d(500), randomColumns(200000, 200000, 8, 1),
                                SpMat(laplacian2d(500) * randomColumns(250000, 62500, 4, 2))};
  return cases[c];
}

static void BM_SparseStorageOrderChange(benchmark::State& state) {
  const SpMat& m = orderChangeCase(int(state.range(0)));
  SparseMatrix<Scalar, RowMajor> result;
  for (auto _ : state) {
    result = m;
    benchmark::DoNotOptimize(result.valuePtr());
  }
  if (result.nonZeros() != m.nonZeros() || !result.isApprox(m)) state.SkipWithError("wrong result");
  static const char* names[] = {"lap2d500", "random200k_8", "lap2d500_x_randP"};
  state.SetLabel(names[state.range(0)]);
  state.counters["nnz"] = double(m.nonZeros());
  state.SetItemsProcessed(state.iterations() * m.nonZeros());
}

BENCHMARK(BM_SparseStorageOrderChange)->DenseRange(0, 2)->Unit(benchmark::kMillisecond);
