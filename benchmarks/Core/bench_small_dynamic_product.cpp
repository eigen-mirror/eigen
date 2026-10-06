// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Products of tiny dynamic-size matrices. Square operands with n <= 6 take the coefficient-based
// path below EIGEN_GEMM_TO_COEFFBASED_THRESHOLD; n = 7 and 8 take GEMM.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <complex>

using namespace Eigen;

template <bool Accumulate, typename Mat>
EIGEN_DONT_INLINE void product(Mat& dst, const Mat& lhs, const Mat& rhs) {
  if (Accumulate)
    dst.noalias() += lhs * rhs;
  else
    dst.noalias() = lhs * rhs;
}

template <typename Scalar, bool Accumulate>
static void BM_SmallDynamicProduct(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Real = typename NumTraits<Scalar>::Real;
  using RealMat = Matrix<Real, Dynamic, Dynamic>;
  const Index n = state.range(0);
  const Mat lhs = Mat::Random(n, n), rhs = Mat::Random(n, n);
  Mat dst = Mat::Random(n, n);

  Mat expected = Mat::Zero(n, n);
  RealMat bound = lhs.cwiseAbs() * rhs.cwiseAbs();
  if (Accumulate) {
    expected = dst;
    bound += dst.cwiseAbs();
  }
  for (Index j = 0; j < n; ++j)
    for (Index k = 0; k < n; ++k)
      for (Index i = 0; i < n; ++i) expected(i, j) += lhs(i, k) * rhs(k, j);
  product<Accumulate>(dst, lhs, rhs);
  const Real tolerance = Real(4 * (n + 1)) * NumTraits<Real>::epsilon();
  if (!((dst - expected).cwiseAbs().array() <= tolerance * bound.array()).all()) {
    state.SkipWithError("product disagrees with the scalar reference");
    return;
  }

  for (auto _ : state) {
    product<Accumulate>(dst, lhs, rhs);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * n * n * n);
}

BENCHMARK_TEMPLATE(BM_SmallDynamicProduct, float, false)->DenseRange(1, 8);
BENCHMARK_TEMPLATE(BM_SmallDynamicProduct, float, true)->DenseRange(1, 8);
BENCHMARK_TEMPLATE(BM_SmallDynamicProduct, double, false)->DenseRange(1, 8);
BENCHMARK_TEMPLATE(BM_SmallDynamicProduct, double, true)->DenseRange(1, 8);
BENCHMARK_TEMPLATE(BM_SmallDynamicProduct, std::complex<double>, false)->DenseRange(1, 8);
BENCHMARK_TEMPLATE(BM_SmallDynamicProduct, std::complex<double>, true)->DenseRange(1, 8);
