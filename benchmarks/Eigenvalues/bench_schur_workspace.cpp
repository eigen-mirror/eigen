// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>
#include <cstdlib>

#ifndef SCALAR
#define SCALAR double
#endif

// ExtraStride = -1 uses owning storage; 0 and 1 bind a Ref with leading dimension n and n+1.
template <int ExtraStride>
static void BM_RealSchur(benchmark::State& state) {
  using Scalar = SCALAR;
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using SolverMatrix = std::conditional_t<(ExtraStride < 0), Mat, Eigen::Ref<Mat>>;
  const Eigen::Index n = state.range(0);
  const bool vectors = state.range(1) != 0;
  const bool fromHessenberg = state.range(2) == 1;
  std::srand(5489);
  Mat matrix = Mat::Random(n, n);
  if (fromHessenberg) {
    Eigen::HessenbergDecomposition<Mat> hess(matrix);
    matrix = hess.matrixH();
  } else if (state.range(2) == 2) {
    matrix.template triangularView<Eigen::StrictlyLower>().setZero();
  }
  const Mat identity = Mat::Identity(n, n);
  Mat storage(n + (ExtraStride > 0 ? ExtraStride : 0), n);
  auto workspace = storage.topRows(n);
  workspace = identity;
  Eigen::RealSchur<SolverMatrix> solver(workspace);
  const auto compute = [&](bool computeU) {
    if (fromHessenberg)
      solver.computeFromHessenberg(matrix, identity, computeU);
    else
      solver.compute(matrix, computeU);
  };
  compute(true);
  const Scalar bound = Scalar(64 * n) * Eigen::NumTraits<Scalar>::epsilon();
  if (solver.info() != Eigen::Success ||
      !((matrix - solver.matrixU() * solver.matrixT() * solver.matrixU().transpose()).norm() <=
        bound * matrix.norm()) ||
      !((solver.matrixU().transpose() * solver.matrixU() - identity).norm() <= bound)) {
    state.SkipWithError("RealSchur failed reconstruction or orthogonality check");
    return;
  }
  const Mat referenceT = solver.matrixT();
  compute(vectors);
  if (solver.info() != Eigen::Success || !((solver.matrixT() - referenceT).norm() <= bound * matrix.norm())) {
    state.SkipWithError("RealSchur failed values-only check");
    return;
  }
  for (auto _ : state) {
    compute(vectors);
    benchmark::DoNotOptimize(solver.matrixT().data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_RealSchur, -1)->ArgsProduct({{128, 129, 256, 257, 512, 1024}, {0, 1}, {0, 1, 2}})->UseRealTime();
BENCHMARK_TEMPLATE(BM_RealSchur, 0)->ArgsProduct({{128, 129, 256, 257, 512, 1024}, {0, 1}, {0, 1, 2}})->UseRealTime();
BENCHMARK_TEMPLATE(BM_RealSchur, 1)->ArgsProduct({{128, 129, 256, 257, 512, 1024}, {0, 1}, {0, 1, 2}})->UseRealTime();
