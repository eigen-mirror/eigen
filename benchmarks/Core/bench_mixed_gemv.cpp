// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

// Kind: complex x real, real x complex, real x real, complex x complex.
template <typename Real, int Order, int Kind>
static void BM_MixedGemv(benchmark::State& state) {
  using Complex = std::complex<Real>;
  using LhsScalar = std::conditional_t<Kind == 1 || Kind == 2, Real, Complex>;
  using RhsScalar = std::conditional_t<Kind == 0 || Kind == 2, Real, Complex>;
  using Scalar = std::conditional_t<Kind == 2, Real, Complex>;
  using Matrix = Eigen::Matrix<LhsScalar, Eigen::Dynamic, Eigen::Dynamic, Order>;
  using Rhs = Eigen::Matrix<RhsScalar, Eigen::Dynamic, 1>;
  using Result = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const Matrix lhs = Matrix::Random(state.range(0), state.range(1));
  const Rhs rhs = Rhs::Random(state.range(1));
  Result result(lhs.rows());
  result.noalias() = lhs * rhs;
  for (Eigen::Index i = 0; i < lhs.rows(); ++i) {
    std::complex<long double> reference(0);
    long double magnitude = 0;
    for (Eigen::Index j = 0; j < lhs.cols(); ++j) {
      const auto term = std::complex<long double>(lhs(i, j)) * std::complex<long double>(rhs(j));
      reference += term;
      magnitude += std::abs(term);
    }
    const long double bound = 16 * (lhs.cols() + 1) * Eigen::NumTraits<Real>::epsilon() * (1 + magnitude);
    if (!(std::abs(std::complex<long double>(result(i)) - reference) <= bound)) {
      state.SkipWithError("incorrect mixed GEMV");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(lhs.data());
    benchmark::DoNotOptimize(rhs.data());
    benchmark::ClobberMemory();
    result.noalias() = lhs * rhs;
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

#define EIGEN_BENCH_MIXED_GEMV(Real, Order, Kind) \
  BENCHMARK_TEMPLATE(BM_MixedGemv, Real, Order, Kind)->Args({8, 8})->Args({31, 17})->Args({128, 128})->Args({513, 257})

EIGEN_BENCH_MIXED_GEMV(float, Eigen::ColMajor, 0);
EIGEN_BENCH_MIXED_GEMV(float, Eigen::ColMajor, 1);
EIGEN_BENCH_MIXED_GEMV(float, Eigen::RowMajor, 0);
EIGEN_BENCH_MIXED_GEMV(float, Eigen::RowMajor, 1);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::ColMajor, 0);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::ColMajor, 1);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::RowMajor, 0);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::RowMajor, 1);
EIGEN_BENCH_MIXED_GEMV(float, Eigen::ColMajor, 2);
EIGEN_BENCH_MIXED_GEMV(float, Eigen::ColMajor, 3);
EIGEN_BENCH_MIXED_GEMV(float, Eigen::RowMajor, 2);
EIGEN_BENCH_MIXED_GEMV(float, Eigen::RowMajor, 3);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::ColMajor, 2);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::ColMajor, 3);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::RowMajor, 2);
EIGEN_BENCH_MIXED_GEMV(double, Eigen::RowMajor, 3);

#undef EIGEN_BENCH_MIXED_GEMV
