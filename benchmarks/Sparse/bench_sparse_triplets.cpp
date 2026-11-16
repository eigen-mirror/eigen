// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// setFromTriplets: a counting sort of the triplets into the transposed storage order, duplicate collapsing, then a
// storage-order change.

#include <benchmark/benchmark.h>
#include <Eigen/Sparse>
#include <algorithm>
#include <random>
#include <vector>

using namespace Eigen;

using Scalar = double;
using SpMat = SparseMatrix<Scalar>;
using Triplets = std::vector<Triplet<Scalar, SpMat::StorageIndex>>;

struct TripletCase {
  Index rows, cols;
  Triplets triplets;
};

// 5-point Laplacian on an n x n grid, every entry repeated `copies` times.
static TripletCase laplacian2d(int n, int copies) {
  TripletCase c{n * n, n * n, {}};
  for (int k = 0; k < copies; ++k)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        int i = y * n + x;
        c.triplets.emplace_back(i, i, 4);
        if (x > 0) c.triplets.emplace_back(i, i - 1, -1);
        if (x + 1 < n) c.triplets.emplace_back(i, i + 1, -1);
        if (y > 0) c.triplets.emplace_back(i, i - n, -1);
        if (y + 1 < n) c.triplets.emplace_back(i, i + n, -1);
      }
  return c;
}

// count triplets at uniformly random positions.
static TripletCase randomPositions(int rows, int cols, int count, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<int> row(0, rows - 1), col(0, cols - 1);
  TripletCase c{rows, cols, {}};
  for (int k = 0; k < count; ++k) c.triplets.emplace_back(row(g), col(g), Scalar(k % 7 + 1));
  return c;
}

// Banded, scattered, hypersparse (most vectors empty), and a banded assembly in which every entry occurs four times.
static const TripletCase& tripletCase(int c) {
  static const TripletCase cases[] = {laplacian2d(500, 1), randomPositions(200000, 200000, 1600000, 1),
                                      randomPositions(1000000, 1000000, 200000, 2), laplacian2d(500, 4)};
  return cases[c];
}

static void BM_SetFromTriplets(benchmark::State& state) {
  const TripletCase& c = tripletCase(int(state.range(0)));
  SpMat m(c.rows, c.cols);
  for (auto _ : state) {
    m.setFromTriplets(c.triplets.begin(), c.triplets.end());
    benchmark::DoNotOptimize(m.valuePtr());
  }
  // Reference through the sorted-triplet path, which shares no code with the unsorted one.
  Triplets sorted = c.triplets;
  std::sort(sorted.begin(), sorted.end(), [](const Triplets::value_type& a, const Triplets::value_type& b) {
    return a.col() != b.col() ? a.col() < b.col() : a.row() < b.row();
  });
  SpMat reference(c.rows, c.cols);
  reference.setFromSortedTriplets(sorted.begin(), sorted.end());
  if (m.nonZeros() != reference.nonZeros() || !m.isApprox(reference) || !m.isCompressed())
    state.SkipWithError("wrong result");
  static const char* names[] = {"lap2d500", "random200k_1.6M", "hyper1M_200k", "lap2d500_dup4"};
  state.SetLabel(names[state.range(0)]);
  state.counters["triplets"] = double(c.triplets.size());
  state.counters["nnz"] = double(m.nonZeros());
  state.SetItemsProcessed(state.iterations() * Index(c.triplets.size()));
}

BENCHMARK(BM_SetFromTriplets)->DenseRange(0, 3)->Unit(benchmark::kMillisecond);
