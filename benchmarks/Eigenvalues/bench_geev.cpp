// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>
#include <cstdlib>

#ifndef SCALAR
#define SCALAR double
#endif

static void BM_GEEV(benchmark::State& state) {
  using Scalar = SCALAR;
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  const Eigen::Index n = state.range(0);
  const bool vectors = state.range(1) != 0;
  std::srand(5489);
  const Mat matrix = Mat::Random(n, n);
  Eigen::EigenSolver<Mat> solver(matrix);
  const Scalar bound = Scalar(64 * n) * Eigen::NumTraits<Scalar>::epsilon();
  if (solver.info() != Eigen::Success ||
      !((matrix * solver.pseudoEigenvectors() - solver.pseudoEigenvectors() * solver.pseudoEigenvalueMatrix()).norm() <=
        bound * matrix.norm() * solver.pseudoEigenvectors().norm())) {
    state.SkipWithError("GEEV failed eigenpair residual check");
    return;
  }
  if (!vectors) {
    const auto reference = solver.eigenvalues().eval();
    solver.compute(matrix, false);
    if (solver.info() != Eigen::Success || !((solver.eigenvalues() - reference).norm() <= bound * matrix.norm())) {
      state.SkipWithError("GEEV failed eigenvalue check");
      return;
    }
  }
  for (auto _ : state) {
    solver.compute(matrix, vectors);
    benchmark::DoNotOptimize(solver.eigenvalues().data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_GEEV)->ArgsProduct({{32, 128, 256, 257, 500, 512, 768, 1000, 1024}, {0, 1}})->UseRealTime();
