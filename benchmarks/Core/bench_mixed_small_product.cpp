// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

template <typename Real, int Size, bool ComplexLeft, bool ExpandReal = false>
static void BM_MixedSmallProduct(benchmark::State& state) {
  using Complex = std::complex<Real>;
  using MatrixScalar = std::conditional_t<ExpandReal, Real, Complex>;
  using FactorScalar = std::conditional_t<ExpandReal, Complex, Real>;
  using Lhs = std::conditional_t<ComplexLeft, Eigen::Matrix<MatrixScalar, Size, Size, Eigen::ColMajor>,
                                 Eigen::Matrix<FactorScalar, 1, Size>>;
  using Rhs = std::conditional_t<ComplexLeft, Eigen::Matrix<FactorScalar, Size, 1>,
                                 Eigen::Matrix<MatrixScalar, Size, Size, Eigen::RowMajor>>;
  using Dst = Eigen::Matrix<Complex, ComplexLeft ? Size : 1, ComplexLeft ? 1 : Size>;
  const Lhs lhs = Lhs::Random();
  const Rhs rhs = Rhs::Random();
  Dst result;
  result.noalias() = lhs * rhs;
  for (Eigen::Index j = 0; j < result.cols(); ++j) {
    for (Eigen::Index i = 0; i < result.rows(); ++i) {
      std::complex<long double> reference(0);
      for (Eigen::Index k = 0; k < Size; ++k) {
        reference += std::complex<long double>(lhs(i, k)) * std::complex<long double>(rhs(k, j));
      }
      const long double bound = 64 * Size * Eigen::NumTraits<Real>::epsilon();
      if (!(std::abs(std::complex<long double>(result(i, j)) - reference) <= bound)) {
        state.SkipWithError("incorrect mixed product");
        return;
      }
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

BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 2, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 3, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 4, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 8, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 2, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 3, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 4, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 8, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 2, false);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 3, false);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 4, false);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 8, false);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 2, false);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 3, false);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 4, false);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 8, false);

BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 2, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 3, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 4, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 8, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 2, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 3, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 4, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 8, true, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 2, false, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 3, false, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 4, false, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, float, 8, false, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 2, false, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 3, false, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 4, false, true);
BENCHMARK_TEMPLATE(BM_MixedSmallProduct, double, 8, false, true);
