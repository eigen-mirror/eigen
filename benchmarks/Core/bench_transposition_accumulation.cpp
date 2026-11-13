// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <random>
#include <cstdlib>

template <int Order, int MaxSize, bool Left, bool Inverse>
static void BM_TranspositionAccumulation(benchmark::State& state) {
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Order>;
  const Eigen::Index n = state.range(0);
  Eigen::Transpositions<Eigen::Dynamic, MaxSize> transpositions(n);
  std::mt19937 generator(12345);
  std::uniform_int_distribution<int> index(0, static_cast<int>(n - 1));
  for (Eigen::Index i = 0; i < n; ++i) transpositions.coeffRef(i) = index(generator);
  std::srand(1);
  const Matrix source = Matrix::Random(n, n);
  Matrix dst = Matrix::Zero(n, n), reference = source;
  for (Eigen::Index step = 0; step < n; ++step) {
    const Eigen::Index k = Inverse ? n - 1 - step : step, p = transpositions.coeff(k);
    for (Eigen::Index j = 0; j < n; ++j) {
      if (Left)
        std::swap(reference(k, j), reference(p, j));
      else
        std::swap(reference(j, k), reference(j, p));
    }
  }
  EIGEN_IF_CONSTEXPR (Left) {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() += transpositions.inverse() * source;
    else
      dst.noalias() += transpositions * source;
  } else {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() += source * transpositions.inverse();
    else
      dst.noalias() += source * transpositions;
  }
  if (!(dst.array() == reference.array()).all()) {
    state.SkipWithError("Transposition accumulation differs from scalar reference");
    return;
  }
  EIGEN_IF_CONSTEXPR (Left) {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() -= transpositions.inverse() * source;
    else
      dst.noalias() -= transpositions * source;
  } else {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() -= source * transpositions.inverse();
    else
      dst.noalias() -= source * transpositions;
  }
  if (!dst.isZero(0)) {
    state.SkipWithError("Transposition subtraction differs from scalar reference");
    return;
  }
  dst.setZero();
  for (auto _ : state) {
    EIGEN_IF_CONSTEXPR (Left) {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() += transpositions.inverse() * source;
      else
        dst.noalias() += transpositions * source;
    } else {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() += source * transpositions.inverse();
      else
        dst.noalias() += source * transpositions;
    }
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
    EIGEN_IF_CONSTEXPR (Left) {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() -= transpositions.inverse() * source;
      else
        dst.noalias() -= transpositions * source;
    } else {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() -= source * transpositions.inverse();
      else
        dst.noalias() -= source * transpositions;
    }
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
}

#define EIGEN_BENCH_TRANSPOSITION(Order, MaxSize, Left, Inverse) \
  BENCHMARK(BM_TranspositionAccumulation<Order, MaxSize, Left, Inverse>)->ArgsProduct({{7, 64, 512}})

EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, Eigen::Dynamic, false, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, Eigen::Dynamic, false, true);
EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, Eigen::Dynamic, true, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, Eigen::Dynamic, true, true);
EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, 512, false, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, 512, false, true);
EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, 512, true, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::ColMajor, 512, true, true);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, Eigen::Dynamic, false, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, Eigen::Dynamic, false, true);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, Eigen::Dynamic, true, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, Eigen::Dynamic, true, true);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, 512, false, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, 512, false, true);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, 512, true, false);
EIGEN_BENCH_TRANSPOSITION(Eigen::RowMajor, 512, true, true);

#undef EIGEN_BENCH_TRANSPOSITION
