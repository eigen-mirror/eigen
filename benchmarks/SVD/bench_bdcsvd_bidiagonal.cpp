// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/SVD>

template <typename Scalar, int Options>
static void BM_BDCSVDBidiagonal(benchmark::State& state) {
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const Eigen::Index size = state.range(0);
  const int pattern = static_cast<int>(state.range(1));
  Vector diagonal(size), superdiagonal(size - 1);
  for (Eigen::Index i = 0; i < size; ++i) diagonal(i) = Scalar(1) + Scalar((i * 17) % 31) / Scalar(32);
  for (Eigen::Index i = 0; i + 1 < size; ++i) {
    superdiagonal(i) = Scalar(0.25) + Scalar((i * 13) % 23) / Scalar(32);
    if (pattern == 1) superdiagonal(i) *= Eigen::NumTraits<Scalar>::epsilon();
    if (pattern == 2 && i % 7 == 3) superdiagonal(i) = Scalar(0);
  }

  Matrix input = Matrix::Zero(size, size);
  input.diagonal() = diagonal;
  input.template diagonal<1>() = superdiagonal;
  Eigen::BDCSVD<Matrix, Eigen::ComputeThinU | Eigen::ComputeThinV> reference(diagonal, superdiagonal);
  const Scalar tolerance = Scalar(32) * Scalar(size) * Eigen::NumTraits<Scalar>::epsilon();
  const Matrix identity = Matrix::Identity(size, size);
  if (reference.info() != Eigen::Success ||
      !((reference.matrixU() * reference.singularValues().asDiagonal() * reference.matrixV().transpose() - input)
            .norm() <= tolerance * input.norm()) ||
      !((reference.matrixU().transpose() * reference.matrixU() - identity).norm() <= tolerance) ||
      !((reference.matrixV().transpose() * reference.matrixV() - identity).norm() <= tolerance)) {
    state.SkipWithError("BDCSVD failed reconstruction or orthogonality check");
    return;
  }

  Eigen::BDCSVD<Matrix, Options> svd(size, size);
  svd.compute(diagonal, superdiagonal);
  if (svd.info() != Eigen::Success ||
      !((svd.singularValues() - reference.singularValues()).norm() <= tolerance * input.norm())) {
    state.SkipWithError("BDCSVD failed singular-value check");
    return;
  }
  for (auto _ : state) {
    svd.compute(diagonal, superdiagonal);
    benchmark::DoNotOptimize(svd.singularValues().data());
    if (Options & Eigen::ComputeThinU) benchmark::DoNotOptimize(svd.matrixU().data());
    if (Options & Eigen::ComputeThinV) benchmark::DoNotOptimize(svd.matrixV().data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}

// Patterns: coupled, near deflation, and split into small blocks.
#define BDC_BIDIAGONAL_CASES \
  ->ArgsProduct({{15, 16, 17, 31, 32, 33, 64, 65, 128, 129, 256, 512, 1024}, {0, 1, 2}})->ArgNames({"size", "pattern"})

BENCHMARK_TEMPLATE(BM_BDCSVDBidiagonal, float, 0) BDC_BIDIAGONAL_CASES;
BENCHMARK_TEMPLATE(BM_BDCSVDBidiagonal, double, 0) BDC_BIDIAGONAL_CASES;
BENCHMARK_TEMPLATE(BM_BDCSVDBidiagonal, float, Eigen::ComputeThinU | Eigen::ComputeThinV) BDC_BIDIAGONAL_CASES;
BENCHMARK_TEMPLATE(BM_BDCSVDBidiagonal, double, Eigen::ComputeThinU | Eigen::ComputeThinV) BDC_BIDIAGONAL_CASES;

#undef BDC_BIDIAGONAL_CASES
