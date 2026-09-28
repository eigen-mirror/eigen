// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <int Operation, typename Mat, typename Vec>
static void scaled_product(Mat& dst, const Mat& lhs, const Mat& rhs, const Vec& diagonal) {
  using Scalar = typename Mat::Scalar;
  if (Operation == 0) dst.noalias() = Scalar(2) * (lhs * rhs);
  if (Operation == 1) dst = Scalar(2) * (lhs * rhs) + Mat::Zero(dst.rows(), dst.cols());
  if (Operation == 2) dst.noalias() = Scalar(2) * (lhs.template triangularView<Lower>() * rhs);
  if (Operation == 3) dst.noalias() = Scalar(2) * (diagonal.asDiagonal() * rhs);
}

template <typename Scalar, int Order, int Operation>
static void BM_ScaledProduct(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Real = typename NumTraits<Scalar>::Real;
  const Index n = state.range(0);
  const Mat lhs = Mat::Random(n, n), rhs = Mat::Random(n, n);
  const Vec diagonal = Vec::Random(n);
  Mat dst(n, n);
  scaled_product<Operation>(dst, lhs, rhs, diagonal);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) {
      Scalar expected = Scalar(0);
      Real magnitude = Real(0);
      if (Operation == 3) {
        expected = diagonal(i) * rhs(i, j);
        magnitude = numext::abs(diagonal(i)) * numext::abs(rhs(i, j));
      } else {
        for (Index k = 0; k < n; ++k) {
          if (Operation == 2 && k > i) continue;
          expected += lhs(i, k) * rhs(k, j);
          magnitude += numext::abs(lhs(i, k)) * numext::abs(rhs(k, j));
        }
      }
      expected *= Scalar(2);
      const Real bound = Real(8 * n + 4) * NumTraits<Real>::epsilon() * (Real(1) + Real(2) * magnitude);
      if (!(numext::abs(dst(i, j) - expected) <= bound)) {
        state.SkipWithError("Scaled product disagrees with the scalar reference");
        return;
      }
    }
  }
  for (auto _ : state) {
    scaled_product<Operation>(dst, lhs, rhs, diagonal);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
}

#define SCALED_PRODUCT_SIZES ->Arg(3)->Arg(16)->Arg(17)->Arg(32)->Arg(65)->Arg(128)
#define SCALED_PRODUCT_CASES(Scalar, Order)                                    \
  BENCHMARK_TEMPLATE(BM_ScaledProduct, Scalar, Order, 0) SCALED_PRODUCT_SIZES; \
  BENCHMARK_TEMPLATE(BM_ScaledProduct, Scalar, Order, 1) SCALED_PRODUCT_SIZES; \
  BENCHMARK_TEMPLATE(BM_ScaledProduct, Scalar, Order, 2) SCALED_PRODUCT_SIZES; \
  BENCHMARK_TEMPLATE(BM_ScaledProduct, Scalar, Order, 3) SCALED_PRODUCT_SIZES;

SCALED_PRODUCT_CASES(double, ColMajor)
SCALED_PRODUCT_CASES(double, RowMajor)
SCALED_PRODUCT_CASES(std::complex<double>, ColMajor)
SCALED_PRODUCT_CASES(std::complex<double>, RowMajor)

#undef SCALED_PRODUCT_CASES
#undef SCALED_PRODUCT_SIZES
