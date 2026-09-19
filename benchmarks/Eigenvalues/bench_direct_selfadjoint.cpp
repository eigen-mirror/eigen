// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>
#include <array>
#include <limits>
#include <random>

#ifndef SCALAR
#define SCALAR float
#endif
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 2
#endif

static void BM_DirectSelfAdjoint(benchmark::State& state) {
  using Scalar = SCALAR;
  using MatrixType = Eigen::Matrix<Scalar, MATRIX_SIZE, MATRIX_SIZE>;
  using WideMatrix = Eigen::Matrix<long double, MATRIX_SIZE, MATRIX_SIZE>;
  using WideVector = Eigen::Matrix<long double, MATRIX_SIZE, 1>;
  const int options = static_cast<int>(state.range(0));
  Scalar scale = Scalar(1);
  switch (state.range(1)) {
    case 1:
      scale = Eigen::numext::ldexp(Scalar(1), -20);
      break;
    case 2:
      scale = (std::numeric_limits<Scalar>::min)();
      break;
    case 3:
      scale = Eigen::NumTraits<Scalar>::highest() / Scalar(2);
      break;
    case 4:
      scale = Scalar(0.001);
      break;
    case 5:
      scale = Scalar(0.1);
      break;
    case 6:
      scale = Scalar(8);
      break;
    case 7:
      scale = Scalar(1000);
      break;
    case 8:
      scale = Scalar(1000000);
      break;
  }
  std::mt19937 generator(42);
  std::uniform_real_distribution<double> random(-0.125, 0.125);
  std::array<MatrixType, 32> inputs;
  Eigen::SelfAdjointEigenSolver<MatrixType> solver;
  const long double tolerance = 128 * MATRIX_SIZE * static_cast<long double>(Eigen::NumTraits<Scalar>::epsilon());
  const long double quantum =
      static_cast<long double>(std::numeric_limits<Scalar>::denorm_min()) / static_cast<long double>(scale);
  for (auto& input : inputs) {
    for (Eigen::Index col = 0; col < MATRIX_SIZE; ++col) {
      for (Eigen::Index row = col; row < MATRIX_SIZE; ++row) {
        input(row, col) = input(col, row) = Scalar(random(generator));
      }
      input(col, col) += Scalar(col) - Scalar(MATRIX_SIZE - 1) / Scalar(2);
    }
    input *= scale;
    solver.computeDirect(input, options);
    const WideMatrix normalized = input.template cast<long double>() / static_cast<long double>(scale);
    Eigen::SelfAdjointEigenSolver<WideMatrix> reference(normalized, Eigen::EigenvaluesOnly);
    const WideVector values = solver.eigenvalues().template cast<long double>() / static_cast<long double>(scale);
    if (solver.info() != Eigen::Success || !values.allFinite() ||
        !((values - reference.eigenvalues()).norm() <= tolerance * normalized.norm() + MATRIX_SIZE * quantum)) {
      state.SkipWithError("Eigenvalue validation failed");
      return;
    }
    if (options == Eigen::ComputeEigenvectors) {
      const WideMatrix vectors = solver.eigenvectors().template cast<long double>();
      if (!((normalized * vectors - vectors * values.asDiagonal()).norm() <=
            tolerance * normalized.norm() + MATRIX_SIZE * quantum) ||
          !((vectors.transpose() * vectors - WideMatrix::Identity()).norm() <= tolerance)) {
        state.SkipWithError("Eigenvector validation failed");
        return;
      }
    }
  }
  unsigned int index = 0;
  for (auto _ : state) {
    const MatrixType& input = inputs[index++ & 31];
    benchmark::DoNotOptimize(input.data());
    solver.computeDirect(input, options);
    benchmark::DoNotOptimize(solver.eigenvalues().data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_DirectSelfAdjoint)
    ->ArgsProduct({{Eigen::EigenvaluesOnly, Eigen::ComputeEigenvectors}, {0, 1, 2, 3, 4, 5, 6, 7, 8}})
    ->ArgNames({"options", "scale_case"});
