// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <cstdlib>

template <int Order, bool SelfAdjoint, bool Left>
static void BM_StructuredPermutationAccumulation(benchmark::State& state) {
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Order>;
  using View = std::conditional_t<SelfAdjoint, Eigen::SelfAdjointView<const Matrix, Eigen::Lower>,
                                  Eigen::TriangularView<const Matrix, Eigen::UnitLower>>;
  const Eigen::Index n = state.range(0);
  std::srand(1);
  const Matrix source = Matrix::Random(n, n);
  const View view(source);
  Eigen::PermutationMatrix<Eigen::Dynamic> p(n);
  for (Eigen::Index i = 0; i < n; ++i) p.indices()(i) = (i + 1) % n;
  Matrix expected(n, n), dst = Matrix::Zero(n, n);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < n; ++j) {
      const Eigen::Index column = Left ? j : p.indices()(j);
      expected(Left ? p.indices()(i) : i, j) = SelfAdjoint
                                                   ? (i >= column ? source(i, column) : source(column, i))
                                                   : (i == column ? 1.0 : (i > column ? source(i, column) : 0.0));
    }
  EIGEN_IF_CONSTEXPR (Left)
    dst.noalias() += p * view;
  else
    dst.noalias() += view * p;
  if (dst != expected) {
    state.SkipWithError("Structured permutation differs from scalar reference");
    return;
  }
  dst.setZero();
  for (auto _ : state) {
    EIGEN_IF_CONSTEXPR (Left)
      dst.noalias() += p * view;
    else
      dst.noalias() += view * p;
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
    EIGEN_IF_CONSTEXPR (Left)
      dst.noalias() -= p * view;
    else
      dst.noalias() -= view * p;
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
}

#define EIGEN_BENCH_STRUCTURED_PERMUTATION(Order, SelfAdjoint, Left) \
  BENCHMARK(BM_StructuredPermutationAccumulation<Order, SelfAdjoint, Left>)->Arg(7)->Arg(64)->Arg(512)

EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::ColMajor, false, false);
EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::ColMajor, false, true);
EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::ColMajor, true, false);
EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::ColMajor, true, true);
EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::RowMajor, false, false);
EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::RowMajor, false, true);
EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::RowMajor, true, false);
EIGEN_BENCH_STRUCTURED_PERMUTATION(Eigen::RowMajor, true, true);

#undef EIGEN_BENCH_STRUCTURED_PERMUTATION
