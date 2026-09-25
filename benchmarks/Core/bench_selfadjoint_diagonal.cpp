// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <int Operation, typename Mat, typename Vec>
static void selfadjoint_diagonal(Mat& dst, const Mat& matrix, const Vec& diagonal) {
  if (Operation == 0) dst.noalias() = matrix.template selfadjointView<Lower>() * diagonal.asDiagonal();
  if (Operation == 1)
    dst = (matrix.template selfadjointView<Lower>() * diagonal.asDiagonal()) + Mat::Zero(dst.rows(), dst.cols());
  if (Operation == 2)
    dst = ((typename Mat::Scalar(2) * matrix.template selfadjointView<Lower>()) * diagonal.asDiagonal()) +
          Mat::Zero(dst.rows(), dst.cols());
}

template <typename Scalar, int Order, int Operation>
static void BM_SelfAdjointDiagonal(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Real = typename NumTraits<Scalar>::Real;
  const Index n = state.range(0);
  Mat matrix = Mat::Random(n, n), dst(n, n);
  const Vec diagonal = Vec::Random(n);
  matrix.diagonal() = matrix.diagonal().real().template cast<Scalar>();
  selfadjoint_diagonal<Operation>(dst, matrix, diagonal);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) {
      Scalar value = (i >= j ? matrix(i, j) : numext::conj(matrix(j, i))) * diagonal(j);
      if (Operation == 2) value *= Scalar(2);
      const Real bound = Real(16) * NumTraits<Real>::epsilon() * (Real(1) + numext::abs(value));
      if (!(numext::abs(dst(i, j) - value) <= bound)) {
        state.SkipWithError("Selfadjoint/diagonal product disagrees with the scalar reference");
        return;
      }
    }
  }
  for (auto _ : state) {
    selfadjoint_diagonal<Operation>(dst, matrix, diagonal);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * n * n);
}

#define SELFADJOINT_DIAGONAL_SIZES ->Arg(3)->Arg(16)->Arg(33)->Arg(65)->Arg(128)->Arg(512)
#define SELFADJOINT_DIAGONAL_OPERATIONS(SCALAR, ORDER)                                     \
  BENCHMARK_TEMPLATE(BM_SelfAdjointDiagonal, SCALAR, ORDER, 0) SELFADJOINT_DIAGONAL_SIZES; \
  BENCHMARK_TEMPLATE(BM_SelfAdjointDiagonal, SCALAR, ORDER, 1) SELFADJOINT_DIAGONAL_SIZES; \
  BENCHMARK_TEMPLATE(BM_SelfAdjointDiagonal, SCALAR, ORDER, 2) SELFADJOINT_DIAGONAL_SIZES

SELFADJOINT_DIAGONAL_OPERATIONS(double, ColMajor);
SELFADJOINT_DIAGONAL_OPERATIONS(double, RowMajor);
SELFADJOINT_DIAGONAL_OPERATIONS(std::complex<double>, ColMajor);
SELFADJOINT_DIAGONAL_OPERATIONS(std::complex<double>, RowMajor);

#undef SELFADJOINT_DIAGONAL_OPERATIONS
#undef SELFADJOINT_DIAGONAL_SIZES
