// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <benchmark/benchmark.h>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

template <typename Scalar>
void BM_KroneckerLeastSquaresMultiRhs(benchmark::State& state) {
  using Real = typename NumTraits<Scalar>::Real;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = state.range(0), nrhs = state.range(1);
  Mat a = Mat::Random(n + 1, n), b = Mat::Random(n, n);
  a.topRows(n).diagonal().array() += Real(2 * n);
  b.diagonal().array() += Real(2 * n);
  const auto op = makeKroneckerOperator(a, b);
  const Mat expected = Mat::Random(n * n, nrhs);
  const Mat rhs = op * expected;
  Mat x = op.leastSquaresSolve(rhs);
  const Real error = (x - expected).norm() / expected.norm();
  if (!(error <= Real(100 * n) * NumTraits<Real>::epsilon())) {
    state.SkipWithError("incorrect least-squares solution");
    return;
  }
  state.counters["relative_error"] = double(error);
  for (auto _ : state) {
    x = op.leastSquaresSolve(rhs);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK_TEMPLATE(BM_KroneckerLeastSquaresMultiRhs, double)->ArgsProduct({{8, 16, 32}, {1, 8, 64}});
BENCHMARK_TEMPLATE(BM_KroneckerLeastSquaresMultiRhs, std::complex<double>)->ArgsProduct({{8, 16}, {1, 8, 64}});

template <typename Scalar>
void BM_KroneckerRank(benchmark::State& state) {
  using Real = typename NumTraits<Scalar>::Real;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = state.range(0);
  Mat a = Mat::Random(n, n), b = Mat::Random(n, n);
  a.diagonal().array() += Real(2 * n);
  b.diagonal().array() += Real(2 * n);
  const auto op = makeKroneckerOperator(a, b);
  if (op.rank() != n * n) {
    state.SkipWithError("incorrect rank");
    return;
  }
  for (auto _ : state) benchmark::DoNotOptimize(op.rank());
}
BENCHMARK_TEMPLATE(BM_KroneckerRank, double)->Arg(8)->Arg(16)->Arg(32);
BENCHMARK_TEMPLATE(BM_KroneckerRank, std::complex<double>)->Arg(8)->Arg(16);
