// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <cstdlib>
#include <type_traits>

namespace {
using namespace Eigen;

template <typename Scalar, int Order, int UpLo, bool Indefinite>
void BM_CholeskyReconstruction(benchmark::State& state) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using RealScalar = typename MatrixType::RealScalar;
  using Factorization =
      typename std::conditional<Indefinite, BunchKaufman<MatrixType, UpLo>, LDLT<MatrixType, UpLo>>::type;
  const Index size = state.range(0);
  std::srand(3395);
  const MatrixType random = MatrixType::Random(size, size);
  MatrixType input(size, size);
  if (Indefinite) {
    input = random + random.adjoint();
  } else {
    input.noalias() = random * random.adjoint();
    input.diagonal().array() += Scalar(1);
  }
  const Factorization factorization(input);
  const MatrixType reconstructed = factorization.reconstructedMatrix();
  const RealScalar tolerance = RealScalar(64) * RealScalar(size) * NumTraits<RealScalar>::epsilon() * input.norm();
  if (factorization.info() != Success || !reconstructed.allFinite() || !((reconstructed - input).norm() <= tolerance)) {
    state.SkipWithError("Incorrect Cholesky reconstruction");
    return;
  }
  for (auto _ : state) {
    const MatrixType result = factorization.reconstructedMatrix();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

// Indefinite=false uses LDLT; true uses BunchKaufman. Factorization is outside timing.
#define EIGEN_BENCH_RECONSTRUCTION(Scalar, Order, UpLo)                     \
  BENCHMARK_TEMPLATE(BM_CholeskyReconstruction, Scalar, Order, UpLo, false) \
      ->ArgsProduct({{4, 16, 64, 128, 256, 1024}});                         \
  BENCHMARK_TEMPLATE(BM_CholeskyReconstruction, Scalar, Order, UpLo, true)->ArgsProduct({{4, 16, 64, 128, 256, 1024}})

EIGEN_BENCH_RECONSTRUCTION(double, ColMajor, Lower);
EIGEN_BENCH_RECONSTRUCTION(double, RowMajor, Upper);
EIGEN_BENCH_RECONSTRUCTION(std::complex<double>, ColMajor, Upper);
EIGEN_BENCH_RECONSTRUCTION(std::complex<double>, RowMajor, Lower);

#undef EIGEN_BENCH_RECONSTRUCTION
}  // namespace
