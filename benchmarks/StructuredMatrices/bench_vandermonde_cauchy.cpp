// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Vandermonde and Cauchy operators with n x n nodes: products Y = A X with r
// right-hand sides (also for a tall m x 32 Cauchy), the GKO factorization
// CauchyLU and the closed-form determinants.

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

using Vec = VectorXd;
using Mat = MatrixXd;

namespace {

// Interlaced nodes x_i = i + 1/2, y_j = j: a well-conditioned Cauchy matrix
// (separated node sets are numerically singular well before n = 256).
Cauchy<double> interlacedCauchy(Index n) {
  return Cauchy<double>(Vec(Vec::LinSpaced(n, 0.5, double(n) - 0.5)), Vec(Vec::LinSpaced(n, 0.0, double(n - 1))));
}

template <typename Op>
void runProduct(benchmark::State& state, const Op& A) {
  const Mat X = Mat::Random(A.cols(), state.range(1));
  Mat Y(A.rows(), X.cols());
  for (auto _ : state) {
    Y.noalias() = A * X;
    benchmark::DoNotOptimize(Y.data());
    benchmark::ClobberMemory();
  }
}

void BM_VandermondeProduct(benchmark::State& state) {
  runProduct(state, Vandermonde<double>(Vec(0.9 * Vec::Random(state.range(0)))));
}

void BM_CauchyProduct(benchmark::State& state) { runProduct(state, interlacedCauchy(state.range(0))); }

// A tall m x 32 Cauchy matrix (m = range(0)), where the r destination columns
// outgrow the cache.
void BM_CauchyProductTall(benchmark::State& state) {
  const Index m = state.range(0), n = 32;
  runProduct(state,
             Cauchy<double>(Vec(Vec::LinSpaced(m, 0.5, double(m) - 0.5)), Vec(Vec::LinSpaced(n, 0.0, double(n - 1)))));
}

void BM_CauchyLU(benchmark::State& state) {
  const Cauchy<double> C = interlacedCauchy(state.range(0));
  for (auto _ : state) {
    CauchyLU<double> lu(C);
    benchmark::DoNotOptimize(&lu);
  }
}

template <typename Op>
void runDeterminant(benchmark::State& state, const Op& A) {
  for (auto _ : state) benchmark::DoNotOptimize(A.determinant());
}

void BM_VandermondeDeterminant(benchmark::State& state) {
  runDeterminant(state, Vandermonde<double>(Vec(Vec::Random(state.range(0)))));
}

void BM_CauchyDeterminant(benchmark::State& state) { runDeterminant(state, interlacedCauchy(state.range(0))); }

void Products(benchmark::Benchmark* b) {
  for (int n : {64, 256, 1024})
    for (int r : {1, 16}) b->Args({n, r});
}

}  // namespace

BENCHMARK(BM_VandermondeProduct)->Apply(Products);
BENCHMARK(BM_CauchyProduct)->Apply(Products);
BENCHMARK(BM_CauchyProductTall)->ArgsProduct({{1 << 16, 1 << 20}, {4, 16}});
BENCHMARK(BM_CauchyLU)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_VandermondeDeterminant)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_CauchyDeterminant)->Arg(64)->Arg(256)->Arg(1024);
