// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/QR>

using namespace Eigen;

template <typename Scalar, int Order>
static void BM_ColPivNorm(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Real = typename NumTraits<Scalar>::Real;
  const Mat matrix = Mat::Random(state.range(0), state.range(1));
  ColPivHouseholderQR<Mat> qr(matrix);
  const Mat reconstructed =
      qr.householderQ() * Mat(qr.matrixQR().template triangularView<Upper>()) * qr.colsPermutation().inverse();
  const Real bound = Real(16 * matrix.rows()) * NumTraits<Real>::epsilon() * matrix.norm();
  if (!((reconstructed - matrix).norm() <= bound)) {
    state.SkipWithError("QR reconstruction residual exceeds the bound");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(matrix.data());
    qr.compute(matrix);
    benchmark::DoNotOptimize(qr.matrixQR().data());
    benchmark::ClobberMemory();
  }
}

#define QR_NORM_CASES(Scalar, Order)               \
  BENCHMARK_TEMPLATE(BM_ColPivNorm, Scalar, Order) \
      ->Args({4, 4})                               \
      ->Args({17, 19})                             \
      ->Args({128, 128})                           \
      ->Args({512, 512})                           \
      ->Args({1024, 32})

QR_NORM_CASES(float, ColMajor);
QR_NORM_CASES(float, RowMajor);
QR_NORM_CASES(double, ColMajor);
QR_NORM_CASES(double, RowMajor);
QR_NORM_CASES(std::complex<float>, ColMajor);
QR_NORM_CASES(std::complex<float>, RowMajor);
QR_NORM_CASES(std::complex<double>, ColMajor);
QR_NORM_CASES(std::complex<double>, RowMajor);

#undef QR_NORM_CASES
