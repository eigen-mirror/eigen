// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Dense>

using namespace Eigen;

// Benchmark JacobiSVD and BDCSVD for various scalar types, matrix shapes,
// and computation options.

// ---------- helpers ----------

template <typename Scalar>
using Mat = Matrix<Scalar, Dynamic, Dynamic>;

template <typename SVD>
EIGEN_DONT_INLINE void do_compute(SVD& svd, const typename SVD::MatrixType& A) {
  svd.compute(A);
}

// ---------- JacobiSVD ----------

template <typename Scalar, int Options>
static void BM_JacobiSVD(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  Mat<Scalar> A = Mat<Scalar>::Random(rows, cols);
  JacobiSVD<Mat<Scalar>, Options> svd(rows, cols);
  for (auto _ : state) {
    do_compute(svd, A);
    benchmark::DoNotOptimize(svd.singularValues().data());
  }
  state.SetItemsProcessed(state.iterations());
}

// Square A = Q1 diag(sigma) Q2^T, sigma_i = kappa^(-i/(n-1)): the graded spectrum on which
// PreconditionSquareMatrix pays off. Args: n, log10(kappa).
template <int Options>
static void BM_JacobiSVD_Graded(benchmark::State& state) {
  const Index n = state.range(0);
  const double log10Kappa = double(state.range(1));
  VectorXd sigma(n);
  for (Index i = 0; i < n; ++i) sigma(i) = std::pow(10.0, -log10Kappa * double(i) / double(n > 1 ? n - 1 : 1));
  const MatrixXd q1 = HouseholderQR<MatrixXd>(MatrixXd::Random(n, n)).householderQ();
  const MatrixXd q2 = HouseholderQR<MatrixXd>(MatrixXd::Random(n, n)).householderQ();
  const MatrixXd A = q1 * sigma.asDiagonal() * q2.transpose();
  JacobiSVD<MatrixXd, Options> svd(n, n);
  for (auto _ : state) {
    do_compute(svd, A);
    benchmark::DoNotOptimize(svd.singularValues().data());
  }
  state.SetItemsProcessed(state.iterations());
}

// ---------- BDCSVD ----------

template <typename Scalar, int Options>
static void BM_BDCSVD(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  Mat<Scalar> A = Mat<Scalar>::Random(rows, cols);
  BDCSVD<Mat<Scalar>, Options> svd(rows, cols);
  for (auto _ : state) {
    do_compute(svd, A);
    benchmark::DoNotOptimize(svd.singularValues().data());
  }
  state.SetItemsProcessed(state.iterations());
}

// ---------- Size configurations ----------

// ---------- Register benchmarks ----------

// clang-format off
// JacobiSVD sizes: square + tall-skinny (expensive for large n).
#define JACOBI_SIZES \
    ->Args({4, 4})->Args({8, 8})->Args({16, 16})->Args({32, 32})->Args({64, 64}) \
    ->Args({128, 128})->Args({256, 256})->Args({512, 512}) \
    ->Args({100, 4})->Args({1000, 4})->Args({1000, 10})

// BDCSVD sizes: square + tall-skinny (triggers R-bidiagonalization when aspect ratio >= 4).
#define BDC_SIZES \
    ->Args({4, 4})->Args({8, 8})->Args({16, 16})->Args({32, 32})->Args({64, 64}) \
    ->Args({128, 128})->Args({256, 256})->Args({512, 512})->Args({1024, 1024}) \
    ->Args({100, 4})->Args({1000, 4})->Args({1000, 10})->Args({1000, 100}) \
    ->Args({10000, 10})->Args({10000, 100})

// Complex JacobiSVD above the shapes bench_jacobisvd_rotations covers.
#define JACOBI_COMPLEX_SIZES ->Args({128, 128})->Args({256, 256})->Args({512, 512})

// JacobiSVD — float
BENCHMARK(BM_JacobiSVD<float, ComputeThinU | ComputeThinV>) JACOBI_SIZES ->Name("JacobiSVD_float_ThinUV");
BENCHMARK(BM_JacobiSVD<float, 0>) JACOBI_SIZES ->Name("JacobiSVD_float_ValuesOnly");

// JacobiSVD — double
BENCHMARK(BM_JacobiSVD<double, ComputeThinU | ComputeThinV>) JACOBI_SIZES ->Name("JacobiSVD_double_ThinUV");
BENCHMARK(BM_JacobiSVD<double, 0>) JACOBI_SIZES ->Name("JacobiSVD_double_ValuesOnly");

// JacobiSVD — complex
BENCHMARK(BM_JacobiSVD<std::complex<float>, ComputeThinU | ComputeThinV>) JACOBI_COMPLEX_SIZES ->Name("JacobiSVD_cfloat_ThinUV");
BENCHMARK(BM_JacobiSVD<std::complex<float>, 0>) JACOBI_COMPLEX_SIZES ->Name("JacobiSVD_cfloat_ValuesOnly");
BENCHMARK(BM_JacobiSVD<std::complex<double>, ComputeThinU | ComputeThinV>) JACOBI_COMPLEX_SIZES ->Name("JacobiSVD_cdouble_ThinUV");
BENCHMARK(BM_JacobiSVD<std::complex<double>, 0>) JACOBI_COMPLEX_SIZES ->Name("JacobiSVD_cdouble_ValuesOnly");

// BDCSVD — float
BENCHMARK(BM_BDCSVD<float, ComputeThinU | ComputeThinV>) BDC_SIZES ->Name("BDCSVD_float_ThinUV");
BENCHMARK(BM_BDCSVD<float, 0>) BDC_SIZES ->Name("BDCSVD_float_ValuesOnly");

// BDCSVD — double
BENCHMARK(BM_BDCSVD<double, ComputeThinU | ComputeThinV>) BDC_SIZES ->Name("BDCSVD_double_ThinUV");
BENCHMARK(BM_BDCSVD<double, 0>) BDC_SIZES ->Name("BDCSVD_double_ValuesOnly");

#undef JACOBI_SIZES
#undef BDC_SIZES
#undef JACOBI_COMPLEX_SIZES
// clang-format on

// JacobiSVD — QR preconditioner comparison (double, 64x64, ThinUV)
BENCHMARK(BM_JacobiSVD<double, ComputeThinU | ComputeThinV | ColPivHouseholderQRPreconditioner>)
    ->Args({64, 64})
    ->Args({1000, 10})
    ->Name("JacobiSVD_double_ColPivQR");
BENCHMARK(BM_JacobiSVD<double, ComputeThinU | ComputeThinV | HouseholderQRPreconditioner>)
    ->Args({64, 64})
    ->Args({1000, 10})
    ->Name("JacobiSVD_double_HouseholderQR");
BENCHMARK(BM_JacobiSVD<double, ComputeFullU | ComputeFullV | FullPivHouseholderQRPreconditioner>)
    ->Args({64, 64})
    ->Name("JacobiSVD_double_FullPivQR");

// JacobiSVD — PreconditionSquareMatrix against the default on graded square inputs (double)
BENCHMARK(BM_JacobiSVD_Graded<0>)
    ->ArgsProduct({{4, 8, 16, 24, 32}, {2, 6, 12}})
    ->Name("JacobiSVD_double_Graded_ValuesOnly");
BENCHMARK(BM_JacobiSVD_Graded<PreconditionSquareMatrix>)
    ->ArgsProduct({{4, 8, 16, 24, 32}, {2, 6, 12}})
    ->Name("JacobiSVD_double_Graded_PrecondSquare_ValuesOnly");
BENCHMARK(BM_JacobiSVD_Graded<ComputeThinU | ComputeThinV>)
    ->ArgsProduct({{4, 8, 16, 24, 32}, {2, 6, 12}})
    ->Name("JacobiSVD_double_Graded_ThinUV");
BENCHMARK(BM_JacobiSVD_Graded<PreconditionSquareMatrix | ComputeThinU | ComputeThinV>)
    ->ArgsProduct({{4, 8, 16, 24, 32}, {2, 6, 12}})
    ->Name("JacobiSVD_double_Graded_PrecondSquare_ThinUV");
