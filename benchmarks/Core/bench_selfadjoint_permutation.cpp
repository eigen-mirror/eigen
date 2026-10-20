// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstdlib>
#include <random>

namespace {
using namespace Eigen;

template <bool InPlace, unsigned int UpLo, typename MatrixType, typename PermutationType>
void permute(MatrixType& matrix, const PermutationType& permutation, bool inverse) {
  EIGEN_IF_CONSTEXPR (InPlace) {
    if (inverse)
      matrix.template selfadjointView<UpLo>().applyPermutation(permutation.inverse());
    else
      matrix.template selfadjointView<UpLo>().applyPermutation(permutation);
  } else {
    if (inverse)
      matrix.template selfadjointView<UpLo>() =
          (permutation.inverse() * matrix.template selfadjointView<UpLo>() * permutation).eval();
    else
      matrix.template selfadjointView<UpLo>() =
          (permutation * matrix.template selfadjointView<UpLo>() * permutation.transpose()).eval();
  }
}

template <typename Scalar, int Order, unsigned int UpLo, bool InPlace>
void BM_SelfAdjointPermutation(benchmark::State& state) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Index size = state.range(0);
  std::srand(42);
  MatrixType matrix = MatrixType::Random(size, size);
  matrix.diagonal() = matrix.diagonal().real().template cast<Scalar>();
  PermutationMatrix<Dynamic> permutation(size);
  permutation.setIdentity();
  std::mt19937 random(42);
  if (state.range(1) == 0) {
    std::shuffle(permutation.indices().data(), permutation.indices().data() + size, random);
  } else {
    permutation.indices().reverseInPlace();
  }
  const MatrixType full = matrix.template selfadjointView<UpLo>();
  const bool inverse = state.range(2) != 0;
  MatrixType reference(size, size);
  if (inverse)
    reference = permutation.inverse() * full * permutation;
  else
    reference = permutation * full * permutation.transpose();
  MatrixType expected = matrix;
  expected.template triangularView<UpLo>() = reference;
  permute<InPlace, UpLo>(matrix, permutation, inverse);
  if (matrix != expected) {
    state.SkipWithError("Incorrect self-adjoint permutation or modified opposite triangle");
    return;
  }
  for (auto _ : state) {
    permute<InPlace, UpLo>(matrix, permutation, inverse);
    benchmark::DoNotOptimize(matrix.data());
    benchmark::ClobberMemory();
  }
}

// Permutation kind: 0 = shuffled (seed 42), 1 = reversal. Last argument selects the inverse.
#define EIGEN_BENCH_SELFADJOINT_PERMUTATION(Scalar, Order, UpLo)            \
  BENCHMARK_TEMPLATE(BM_SelfAdjointPermutation, Scalar, Order, UpLo, false) \
      ->ArgsProduct({{4, 16, 64, 256, 1024}, {0, 1}, {0, 1}});              \
  BENCHMARK_TEMPLATE(BM_SelfAdjointPermutation, Scalar, Order, UpLo, true)  \
      ->ArgsProduct({{4, 16, 64, 256, 1024}, {0, 1}, {0, 1}})

EIGEN_BENCH_SELFADJOINT_PERMUTATION(float, ColMajor, Lower);
EIGEN_BENCH_SELFADJOINT_PERMUTATION(float, RowMajor, Upper);
EIGEN_BENCH_SELFADJOINT_PERMUTATION(double, ColMajor, Lower);
EIGEN_BENCH_SELFADJOINT_PERMUTATION(double, RowMajor, Upper);
EIGEN_BENCH_SELFADJOINT_PERMUTATION(std::complex<float>, ColMajor, Upper);
EIGEN_BENCH_SELFADJOINT_PERMUTATION(std::complex<float>, RowMajor, Lower);
EIGEN_BENCH_SELFADJOINT_PERMUTATION(std::complex<double>, ColMajor, Upper);
EIGEN_BENCH_SELFADJOINT_PERMUTATION(std::complex<double>, RowMajor, Lower);

#undef EIGEN_BENCH_SELFADJOINT_PERMUTATION
}  // namespace
