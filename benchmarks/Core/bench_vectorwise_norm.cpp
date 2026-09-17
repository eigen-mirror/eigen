// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar, int Order>
static void BM_ColumnNorms(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Real = typename NumTraits<Scalar>::Real;
  const Mat matrix = Mat::Random(state.range(0), state.range(1));
  Matrix<Real, 1, Dynamic> norms = matrix.colwise().norm();
  for (Index j = 0; j < matrix.cols(); ++j) {
    long double reference = 0;
    for (Index i = 0; i < matrix.rows(); ++i) {
      const long double real = static_cast<long double>(numext::real(matrix(i, j)));
      const long double imag = static_cast<long double>(numext::imag(matrix(i, j)));
      reference += real * real + imag * imag;
    }
    const long double expected = numext::sqrt(reference);
    const long double bound = (4 * matrix.rows() + 4) * static_cast<long double>(NumTraits<Real>::epsilon()) * expected;
    if (!(numext::abs(static_cast<long double>(norms(j)) - expected) <= bound)) {
      state.SkipWithError("column norm disagrees with the reference");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(matrix.data());
    norms = matrix.colwise().norm();
    benchmark::DoNotOptimize(norms.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * matrix.size() * sizeof(Scalar));
}

#define NORM_CASES(Scalar, Order)                   \
  BENCHMARK_TEMPLATE(BM_ColumnNorms, Scalar, Order) \
      ->Args({4, 4})                                \
      ->Args({17, 19})                              \
      ->Args({128, 128})                            \
      ->Args({512, 512})                            \
      ->Args({1024, 32})

NORM_CASES(float, ColMajor);
NORM_CASES(float, RowMajor);
NORM_CASES(double, ColMajor);
NORM_CASES(double, RowMajor);
NORM_CASES(std::complex<float>, ColMajor);
NORM_CASES(std::complex<float>, RowMajor);
NORM_CASES(std::complex<double>, ColMajor);
NORM_CASES(std::complex<double>, RowMajor);

#undef NORM_CASES
