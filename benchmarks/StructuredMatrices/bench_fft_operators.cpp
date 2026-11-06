// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// FFT-based structured operators: products y = A x and solves x = A^+ b with
// r right-hand sides. Circulant and Toeplitz take n, Hankel an n x n operator,
// Bccb an n x n grid of n x n blocks (N = n^2). Sizes 97 and 101 are prime,
// exercising the padded product embedding.

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

using Vec = VectorXd;
using Mat = MatrixXd;

namespace {

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

template <typename Op>
void runSolve(benchmark::State& state, const Op& A) {
  const Mat B = Mat::Random(A.rows(), state.range(1));
  Mat X(A.cols(), B.cols());
  for (auto _ : state) {
    X = A.solve(B);
    benchmark::DoNotOptimize(X.data());
    benchmark::ClobberMemory();
  }
}

void BM_CirculantProduct(benchmark::State& state) {
  runProduct(state, Circulant<double>(Vec(Vec::Random(state.range(0)))));
}

void BM_ToeplitzProduct(benchmark::State& state) {
  const Index n = state.range(0);
  runProduct(state, Toeplitz<double>(Vec(Vec::Random(n)), Vec(Vec::Random(n))));
}

void BM_HankelProduct(benchmark::State& state) {
  const Index n = state.range(0);
  runProduct(state, Hankel<double>(Vec(Vec::Random(n)), Vec(Vec::Random(n))));
}

void BM_BccbProduct(benchmark::State& state) {
  const Index n = state.range(0);
  runProduct(state, Bccb<double>(Mat(Mat::Random(n, n))));
}

void BM_CirculantSolve(benchmark::State& state) {
  const Index n = state.range(0);
  Vec c = Vec::Random(n);
  c[0] += double(2 * n);
  runSolve(state, Circulant<double>(c));
}

void BM_BccbSolve(benchmark::State& state) {
  const Index n = state.range(0);
  Mat G = Mat::Random(n, n);
  G(0, 0) += double(2 * n * n);
  runSolve(state, Bccb<double>(G));
}

}  // namespace

BENCHMARK(BM_CirculantProduct)->ArgsProduct({{64, 97, 256, 1024, 4096}, {1, 16}});
BENCHMARK(BM_ToeplitzProduct)->ArgsProduct({{64, 97, 256, 1024, 4096}, {1, 16}});
BENCHMARK(BM_HankelProduct)->ArgsProduct({{64, 97, 256, 1024, 4096}, {1, 16}});
BENCHMARK(BM_CirculantSolve)->ArgsProduct({{64, 97, 256, 1024, 4096}, {1, 16}});
BENCHMARK(BM_BccbProduct)->ArgsProduct({{16, 64, 101}, {1, 16}});
BENCHMARK(BM_BccbSolve)->ArgsProduct({{16, 64, 101}, {1, 16}});
