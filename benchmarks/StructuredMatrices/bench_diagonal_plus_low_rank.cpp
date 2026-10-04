// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// DiagonalPlusLowRank D + U V^H of size n and correction rank k: the Woodbury
// solve with one right-hand side, the inverse and the determinant.

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

using Vec = VectorXd;
using Mat = MatrixXd;

namespace {

DiagonalPlusLowRank<double> makeOperator(Index n, Index k) {
  return DiagonalPlusLowRank<double>(Vec(Vec::Random(n).cwiseAbs().array() + 1.0), Mat(0.1 * Mat::Random(n, k)),
                                     Mat(0.1 * Mat::Random(n, k)));
}

void BM_Solve(benchmark::State& state) {
  const DiagonalPlusLowRank<double> A = makeOperator(state.range(0), state.range(1));
  const Vec b = Vec::Random(A.rows());
  Vec x(A.rows());
  for (auto _ : state) {
    x = A.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
}

void BM_Inverse(benchmark::State& state) {
  const DiagonalPlusLowRank<double> A = makeOperator(state.range(0), state.range(1));
  for (auto _ : state) {
    DiagonalPlusLowRank<double> Ainv = A.inverse();
    benchmark::DoNotOptimize(Ainv.factorU().data());
  }
}

void BM_Determinant(benchmark::State& state) {
  const DiagonalPlusLowRank<double> A = makeOperator(state.range(0), state.range(1));
  for (auto _ : state) benchmark::DoNotOptimize(A.determinant());
}

void Shapes(benchmark::Benchmark* b) {
  for (int n : {1000, 100000})
    for (int k : {4, 16}) b->Args({n, k});
}

}  // namespace

BENCHMARK(BM_Solve)->Apply(Shapes);
BENCHMARK(BM_Inverse)->Apply(Shapes);
BENCHMARK(BM_Determinant)->Apply(Shapes);
