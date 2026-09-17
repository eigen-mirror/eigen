// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>
#include <cstdlib>

using namespace Eigen;

template <typename Scalar, int StorageOrder>
static void BM_RealSchurFromHessenberg(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  const Index n = state.range(0);
  const bool triangular = state.range(1) != 0;
  std::srand(1);
  Mat h = Mat::Random(n, n);
  h.template triangularView<StrictlyLower>().setZero();
  if (!triangular) h.template diagonal<-1>().setRandom();
  const Mat identity = Mat::Identity(n, n);
  RealSchur<Mat> solver(n);
  solver.computeFromHessenberg(h, identity, true);
  if (solver.info() != Success) {
    state.SkipWithError("RealSchur did not converge");
    return;
  }
  const Scalar bound = Scalar(64 * n) * NumTraits<Scalar>::epsilon();
  if (!((solver.matrixU() * solver.matrixT() * solver.matrixU().transpose() - h).norm() <= bound * h.norm()) ||
      !((solver.matrixU().transpose() * solver.matrixU() - identity).norm() <= bound)) {
    state.SkipWithError("Schur reconstruction or orthogonality failed");
    return;
  }
  const Mat referenceT = solver.matrixT();
  solver.computeFromHessenberg(h, identity, false);
  if (solver.info() != Success || !((solver.matrixT() - referenceT).norm() <= bound * h.norm())) {
    state.SkipWithError("RealSchur values-only result differs");
    return;
  }
  // An already triangular input isolates the O(n^2) copy and norm traversal; a Hessenberg input also exercises QR.
  for (auto _ : state) {
    solver.computeFromHessenberg(h, identity, false);
    benchmark::DoNotOptimize(solver.matrixT().data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_RealSchurFromHessenberg, float, ColMajor)->ArgsProduct({{32, 128, 511, 512}, {0, 1}});
BENCHMARK_TEMPLATE(BM_RealSchurFromHessenberg, float, RowMajor)->ArgsProduct({{32, 128, 511, 512}, {0, 1}});
BENCHMARK_TEMPLATE(BM_RealSchurFromHessenberg, double, ColMajor)->ArgsProduct({{32, 128, 511, 512}, {0, 1}});
BENCHMARK_TEMPLATE(BM_RealSchurFromHessenberg, double, RowMajor)->ArgsProduct({{32, 128, 511, 512}, {0, 1}});
