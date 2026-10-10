// Benchmarks for KroneckerSparseView, the lazy SparseMatrixBase expression of
// a Kronecker product or sum: assembling the implicit Euler matrix
// I + tau (Dy (+) Dx) of an n x n grid in one sparse expression through the
// view, against materializing Dy (+) Dx first, and the sparse-dense product
// through the view against the materialized matrix and the operator itself.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Sparse>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

using Vec = VectorXd;
using SpMat = SparseMatrix<double>;

static SpMat tridiagonal(Index n) {
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 3));
  for (Index j = 0; j < n; ++j) {
    if (j > 0) A.insert(j - 1, j) = -1.0;
    A.insert(j, j) = 2.0;
    if (j + 1 < n) A.insert(j + 1, j) = -1.0;
  }
  A.makeCompressed();
  return A;
}

static SpMat identity(Index n) {
  SpMat I(n, n);
  I.setIdentity();
  return I;
}

static void BM_KroneckerSumAssembleView(benchmark::State& state) {
  const Index n = state.range(0);
  auto L = makeKroneckerSum(tridiagonal(n), tridiagonal(n));
  const SpMat Id = identity(n * n);
  SpMat M;
  for (auto _ : state) {
    M = Id + 0.25 * L.sparseView();
    benchmark::DoNotOptimize(M.valuePtr());
  }
}
BENCHMARK(BM_KroneckerSumAssembleView)->Arg(64)->Arg(256)->Arg(1024);

static void BM_KroneckerSumAssembleMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  auto L = makeKroneckerSum(tridiagonal(n), tridiagonal(n));
  const SpMat Id = identity(n * n);
  SpMat Lm, M;
  for (auto _ : state) {
    Lm = L;
    M = Id + 0.25 * Lm;
    benchmark::DoNotOptimize(M.valuePtr());
  }
}
BENCHMARK(BM_KroneckerSumAssembleMaterialized)->Arg(64)->Arg(256)->Arg(1024);

static void BM_KroneckerSumProductView(benchmark::State& state) {
  const Index n = state.range(0);
  auto L = makeKroneckerSum(tridiagonal(n), tridiagonal(n));
  const auto V = L.sparseView();
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = V * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerSumProductView)->Arg(64)->Arg(256)->Arg(1024);

static void BM_KroneckerSumProductMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat Lm;
  Lm = makeKroneckerSum(tridiagonal(n), tridiagonal(n));
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = Lm * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerSumProductMaterialized)->Arg(64)->Arg(256)->Arg(1024);

static void BM_KroneckerSumProductOperator(benchmark::State& state) {
  const Index n = state.range(0);
  auto L = makeKroneckerSum(tridiagonal(n), tridiagonal(n));
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = L * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerSumProductOperator)->Arg(64)->Arg(256)->Arg(1024);
