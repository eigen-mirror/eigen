// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstdlib>
#include <random>

template <int Order, bool Left, bool Inverse, int Size = Eigen::Dynamic>
static void BM_PermutationAccumulation(benchmark::State& state) {
  using Matrix = Eigen::Matrix<double, Size, Size, Order>;
  const Eigen::Index n = state.range(0);
  Eigen::PermutationMatrix<Eigen::Dynamic> permutation(n);
  for (Eigen::Index i = 0; i < n; ++i) permutation.indices()(i) = (i + 1) % n;
  if (state.range(1)) {
    std::mt19937 generator(12345);
    std::shuffle(permutation.indices().data(), permutation.indices().data() + n, generator);
  }
  std::srand(1);
  const Matrix source = Matrix::Random(n, n);
  Matrix dst = Matrix::Zero(n, n), reference(n, n);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < n; ++j) {
      const Eigen::Index p = permutation.indices()(i);
      if (Left) {
        reference(Inverse ? i : p, j) = source(Inverse ? p : i, j);
      } else {
        reference(j, Inverse ? p : i) = source(j, Inverse ? i : p);
      }
    }
  EIGEN_IF_CONSTEXPR (Left) {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() += permutation.inverse() * source;
    else
      dst.noalias() += permutation * source;
  } else {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() += source * permutation.inverse();
    else
      dst.noalias() += source * permutation;
  }
  if (!(dst.array() == reference.array()).all()) {
    state.SkipWithError("Permutation accumulation differs from scalar reference");
    return;
  }
  EIGEN_IF_CONSTEXPR (Left) {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() -= permutation.inverse() * source;
    else
      dst.noalias() -= permutation * source;
  } else {
    EIGEN_IF_CONSTEXPR (Inverse)
      dst.noalias() -= source * permutation.inverse();
    else
      dst.noalias() -= source * permutation;
  }
  if (!dst.isZero(0)) {
    state.SkipWithError("Permutation subtraction differs from scalar reference");
    return;
  }
  dst.setZero();
  for (auto _ : state) {
    EIGEN_IF_CONSTEXPR (Left) {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() += permutation.inverse() * source;
      else
        dst.noalias() += permutation * source;
    } else {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() += source * permutation.inverse();
      else
        dst.noalias() += source * permutation;
    }
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
    EIGEN_IF_CONSTEXPR (Left) {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() -= permutation.inverse() * source;
      else
        dst.noalias() -= permutation * source;
    } else {
      EIGEN_IF_CONSTEXPR (Inverse)
        dst.noalias() -= source * permutation.inverse();
      else
        dst.noalias() -= source * permutation;
    }
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
}

#define EIGEN_BENCH_PERMUTATION(Order, Left, Inverse) \
  BENCHMARK(BM_PermutationAccumulation<Order, Left, Inverse>)->ArgsProduct({{7, 64, 512}, {0, 1}})

EIGEN_BENCH_PERMUTATION(Eigen::ColMajor, false, false);
EIGEN_BENCH_PERMUTATION(Eigen::ColMajor, false, true);
EIGEN_BENCH_PERMUTATION(Eigen::ColMajor, true, false);
EIGEN_BENCH_PERMUTATION(Eigen::ColMajor, true, true);
EIGEN_BENCH_PERMUTATION(Eigen::RowMajor, false, false);
EIGEN_BENCH_PERMUTATION(Eigen::RowMajor, false, true);
EIGEN_BENCH_PERMUTATION(Eigen::RowMajor, true, false);
EIGEN_BENCH_PERMUTATION(Eigen::RowMajor, true, true);

#undef EIGEN_BENCH_PERMUTATION

#define EIGEN_BENCH_FIXED_PERMUTATION(Order, Left, Inverse)                                   \
  BENCHMARK(BM_PermutationAccumulation<Order, Left, Inverse, 1>)->ArgsProduct({{1}, {0, 1}}); \
  BENCHMARK(BM_PermutationAccumulation<Order, Left, Inverse, 2>)->ArgsProduct({{2}, {0, 1}}); \
  BENCHMARK(BM_PermutationAccumulation<Order, Left, Inverse, 3>)->ArgsProduct({{3}, {0, 1}}); \
  BENCHMARK(BM_PermutationAccumulation<Order, Left, Inverse, 4>)->ArgsProduct({{4}, {0, 1}}); \
  BENCHMARK(BM_PermutationAccumulation<Order, Left, Inverse, 7>)->ArgsProduct({{7}, {0, 1}})

EIGEN_BENCH_FIXED_PERMUTATION(Eigen::ColMajor, false, false);
EIGEN_BENCH_FIXED_PERMUTATION(Eigen::ColMajor, false, true);
EIGEN_BENCH_FIXED_PERMUTATION(Eigen::ColMajor, true, false);
EIGEN_BENCH_FIXED_PERMUTATION(Eigen::ColMajor, true, true);
EIGEN_BENCH_FIXED_PERMUTATION(Eigen::RowMajor, false, false);
EIGEN_BENCH_FIXED_PERMUTATION(Eigen::RowMajor, false, true);
EIGEN_BENCH_FIXED_PERMUTATION(Eigen::RowMajor, true, false);
EIGEN_BENCH_FIXED_PERMUTATION(Eigen::RowMajor, true, true);

#undef EIGEN_BENCH_FIXED_PERMUTATION
