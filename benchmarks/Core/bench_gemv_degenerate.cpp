// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

template <typename Scalar, int StorageOrder = Eigen::ColMajor>
static void BM_GemvTranspose(benchmark::State& state) {
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const Eigen::Index rows = state.range(0), cols = state.range(1);
  const Matrix a = Matrix::Random(rows, cols);
  const Vector x = Vector::Random(rows);
  Vector y = Vector::Zero(cols);
  const Scalar alpha(0.75);
  y.noalias() += alpha * a.transpose() * x;
  using Wide = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex, std::complex<long double>, long double>;
  using Real = typename Eigen::NumTraits<Scalar>::Real;
  for (Eigen::Index j = 0; j < cols; ++j) {
    Wide expected(0);
    long double magnitude = 0;
    for (Eigen::Index i = 0; i < rows; ++i) {
      const Wide term = Wide(alpha) * Wide(a(i, j)) * Wide(x[i]);
      expected += term;
      magnitude += Eigen::numext::abs(term);
    }
    const long double bound = 8 * static_cast<long double>(rows + 1) *
                              static_cast<long double>(Eigen::NumTraits<Real>::epsilon()) * magnitude;
    if (!(Eigen::numext::abs(Wide(y[j]) - expected) <= bound)) {
      state.SkipWithError("GEMV differs from scalar reference");
      return;
    }
  }
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(x.data());
  for (auto _ : state) {
    y.noalias() += alpha * a.transpose() * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_GemvTranspose, float)->ArgsProduct({{1, 2, 4}, {16, 256, 1024}});
BENCHMARK_TEMPLATE(BM_GemvTranspose, double)->ArgsProduct({{1, 2, 4}, {16, 256, 1024}});
BENCHMARK_TEMPLATE(BM_GemvTranspose, std::complex<float>)->ArgsProduct({{1, 2, 4}, {16, 256, 1024}});
BENCHMARK_TEMPLATE(BM_GemvTranspose, float, Eigen::RowMajor)->ArgsProduct({{1, 2, 4}, {16, 256, 1024}});
BENCHMARK_TEMPLATE(BM_GemvTranspose, double, Eigen::RowMajor)->ArgsProduct({{1, 2, 4}, {16, 256, 1024}});
BENCHMARK_TEMPLATE(BM_GemvTranspose, std::complex<float>, Eigen::RowMajor)->ArgsProduct({{1, 2, 4}, {16, 256, 1024}});

template <typename Scalar>
static void BM_GemvStrided(benchmark::State& state) {
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const Eigen::Index rows = state.range(0), cols = state.range(1);
  const Matrix a = Matrix::Random(rows, cols);
  const Vector x = Vector::Random(cols);
  Vector storage = Vector::Zero(2 * rows);
  Eigen::Map<Vector, Eigen::Unaligned, Eigen::InnerStride<2>> y(storage.data(), rows);
  const Scalar alpha(0.75);
  const Vector expected = alpha * a * x;
  y.noalias() += alpha * a * x;
  if (!y.isApprox(expected)) {
    state.SkipWithError("Strided GEMV differs from contiguous result");
    return;
  }
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(x.data());
  for (auto _ : state) {
    y.noalias() += alpha * a * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_GemvStrided, float)->ArgsProduct({{16, 256, 1024}, {1, 2, 4}});
BENCHMARK_TEMPLATE(BM_GemvStrided, double)->ArgsProduct({{16, 256, 1024}, {1, 2, 4}});
BENCHMARK_TEMPLATE(BM_GemvStrided, std::complex<float>)->ArgsProduct({{16, 256, 1024}, {1, 2, 4}});
