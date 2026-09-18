// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>
#include <cstdlib>

using namespace Eigen;

template <typename MatrixType, template <typename> class QZ>
static void BM_QZ(benchmark::State& state) {
  using RealScalar = typename MatrixType::RealScalar;
  const Index n = state.range(0);
  const bool computeQZ = state.range(1) != 0;
  std::srand(3163);
  const MatrixType A = MatrixType::Random(n, n), B = MatrixType::Random(n, n);
  QZ<MatrixType> qz(n);
  qz.compute(A, B);
  if (qz.info() != Success) {
    state.SkipWithError("QZ did not converge");
    return;
  }
  // Allow for rounding error accumulated over the QZ sweeps.
  const RealScalar tolerance = RealScalar(128 * n) * NumTraits<RealScalar>::epsilon();
  if (!((qz.matrixQ() * qz.matrixS() * qz.matrixZ() - A).norm() <= tolerance * A.norm()) ||
      !((qz.matrixQ() * qz.matrixT() * qz.matrixZ() - B).norm() <= tolerance * B.norm()) ||
      !((qz.matrixQ() * qz.matrixQ().adjoint() - MatrixType::Identity(n, n)).norm() <= tolerance) ||
      !((qz.matrixZ() * qz.matrixZ().adjoint() - MatrixType::Identity(n, n)).norm() <= tolerance)) {
    state.SkipWithError("QZ reconstruction or orthogonality check failed");
    return;
  }
  for (auto _ : state) {
    qz.compute(A, B, computeQZ);
    benchmark::DoNotOptimize(qz.matrixS().data());
    benchmark::DoNotOptimize(qz.matrixT().data());
    benchmark::ClobberMemory();
  }
}

using RowMatrixXd = Matrix<double, Dynamic, Dynamic, RowMajor>;
using RowMatrixXcd = Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>;

BENCHMARK_TEMPLATE(BM_QZ, MatrixXd, RealQZ)->ArgsProduct({{16, 24, 48, 64, 128, 256}, {0, 1}});
BENCHMARK_TEMPLATE(BM_QZ, RowMatrixXd, RealQZ)->ArgsProduct({{16, 24, 48, 64, 128, 256}, {0, 1}});
BENCHMARK_TEMPLATE(BM_QZ, MatrixXcd, ComplexQZ)->ArgsProduct({{16, 24, 48, 64, 128, 256}, {0, 1}});
BENCHMARK_TEMPLATE(BM_QZ, RowMatrixXcd, ComplexQZ)->ArgsProduct({{16, 24, 48, 64, 128, 256}, {0, 1}});
