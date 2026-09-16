// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <cstdlib>

using namespace Eigen;

template <typename Scalar, int StorageOrder, unsigned int UpLo>
static void BM_SelfAdjointL1Norm(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const Index n = state.range(0);
  std::srand(1);
  Mat matrix = Mat::Random(n, n);
  matrix.diagonal() = matrix.diagonal().real().template cast<Scalar>();
  const Mat full = matrix.template selfadjointView<UpLo>();
  const RealScalar reference = full.cwiseAbs().colwise().sum().maxCoeff();
  const RealScalar actual = matrix.template selfadjointView<UpLo>().l1Norm();
  const RealScalar bound = RealScalar(8 * n) * NumTraits<RealScalar>::epsilon() * reference;
  if (!(numext::abs(actual - reference) <= bound)) {
    state.SkipWithError("Self-adjoint norm differs from the full-matrix norm");
    return;
  }
  for (auto _ : state) {
    RealScalar norm = matrix.template selfadjointView<UpLo>().l1Norm();
    benchmark::DoNotOptimize(norm);
    benchmark::ClobberMemory();
  }
}

#define EIGEN_BENCH_SELFADJOINT_NORM(Scalar, StorageOrder, UpLo) \
  BENCHMARK_TEMPLATE(BM_SelfAdjointL1Norm, Scalar, StorageOrder, UpLo)->Arg(4)->Arg(16)->Arg(17)->Arg(128)->Arg(512)

EIGEN_BENCH_SELFADJOINT_NORM(float, ColMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(float, ColMajor, Upper);
EIGEN_BENCH_SELFADJOINT_NORM(float, RowMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(float, RowMajor, Upper);
EIGEN_BENCH_SELFADJOINT_NORM(double, ColMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(double, ColMajor, Upper);
EIGEN_BENCH_SELFADJOINT_NORM(double, RowMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(double, RowMajor, Upper);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<float>, ColMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<float>, ColMajor, Upper);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<float>, RowMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<float>, RowMajor, Upper);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<double>, ColMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<double>, ColMajor, Upper);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<double>, RowMajor, Lower);
EIGEN_BENCH_SELFADJOINT_NORM(std::complex<double>, RowMajor, Upper);

#undef EIGEN_BENCH_SELFADJOINT_NORM
