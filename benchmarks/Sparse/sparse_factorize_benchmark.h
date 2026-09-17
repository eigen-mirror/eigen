// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_FACTORIZE_BENCHMARK_H
#define EIGEN_SPARSE_FACTORIZE_BENCHMARK_H

#include <benchmark/benchmark.h>
#include <Eigen/SparseCore>
#include <contrib/Eigen/SparseExtra>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

template <typename Solver>
void sparseFactorizeBenchmark(benchmark::State& state) {
  using Scalar = typename Solver::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using SparseMatrix = typename Solver::MatrixType;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const Eigen::Index n = state.range(0), bandwidth = state.range(1);
  std::srand(1);
  std::vector<Eigen::Triplet<Scalar>> entries;
  for (Eigen::Index j = 0; j < n; ++j) {
    entries.emplace_back(j, j, Scalar(2 * bandwidth + 1));
    for (Eigen::Index i = j + 1; i < (std::min)(n, j + bandwidth + 1); ++i) {
      const Scalar value = Eigen::internal::random<Scalar>();
      entries.emplace_back(i, j, value);
      entries.emplace_back(j, i, Eigen::numext::conj(value));
    }
  }
  SparseMatrix matrix(n, n);
  matrix.setFromTriplets(entries.begin(), entries.end());
  Solver solver;
  solver.compute(matrix);
  if (solver.info() != Eigen::Success) {
    state.SkipWithError("factorization failed");
    return;
  }
  const Vector rhs = Vector::Ones(n);
  const Vector solution = solver.solve(rhs);
  // Allow O(n * epsilon) roundoff from factorization and the triangular solves.
  const RealScalar bound = RealScalar(32 * n) * Eigen::NumTraits<RealScalar>::epsilon();
  if (solver.info() != Eigen::Success || !((matrix * solution - rhs).norm() <= bound * rhs.norm())) {
    state.SkipWithError("solve residual exceeds bound");
    return;
  }
  for (auto _ : state) {
    solver.factorize(matrix);
    benchmark::DoNotOptimize(solver.info());
    benchmark::ClobberMemory();
  }
}

template <typename Solver, bool RequireSquare = true>
void sparseFactorizeFileBenchmark(benchmark::State& state) {
  using Scalar = typename Solver::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using SparseMatrix = typename Solver::MatrixType;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const char* filename = std::getenv("EIGEN_BENCHMARK_MATRIX");
  if (!filename) {
    state.SkipWithError("Set EIGEN_BENCHMARK_MATRIX to a MatrixMarket file");
    return;
  }
  std::ifstream input(filename);
  std::string banner, object, format, field, symmetry;
  if (!(input >> banner >> object >> format >> field >> symmetry) || banner != "%%MatrixMarket" || object != "matrix" ||
      format != "coordinate" || field != "real" || (symmetry != "general" && symmetry != "symmetric")) {
    state.SkipWithError("Expected a real coordinate MatrixMarket matrix");
    return;
  }
  SparseMatrix matrix;
  if (!Eigen::loadMarket(matrix, filename) || matrix.rows() == 0 || matrix.cols() == 0 ||
      ((RequireSquare || symmetry == "symmetric") && matrix.rows() != matrix.cols())) {
    state.SkipWithError("Matrix loading or shape check failed");
    return;
  }
  // loadMarket reads the stored triangle without expanding MatrixMarket symmetry.
  if (symmetry == "symmetric") {
    for (Eigen::Index j = 0; j < matrix.outerSize(); ++j) {
      for (typename SparseMatrix::InnerIterator it(matrix, j); it; ++it) {
        if (it.row() < it.col()) {
          state.SkipWithError("Expected lower-triangular storage for a symmetric MatrixMarket matrix");
          return;
        }
      }
    }
    SparseMatrix full = matrix.template selfadjointView<Eigen::Lower>();
    matrix.swap(full);
  }
  matrix.makeCompressed();
  Solver solver;
  solver.compute(matrix);
  Vector reference(matrix.cols());
  for (Eigen::Index i = 0; i < reference.size(); ++i) reference[i] = Scalar((17 * i) % 101 - 50) / Scalar(50);
  const Vector rhs = matrix * reference;
  const RealScalar matrixNorm = matrix.norm(), rhsNorm = rhs.norm();
  // Normwise backward error: ||Ax-b|| / (||A|| ||x|| + ||b||) <= O(n * epsilon).
  const RealScalar bound =
      RealScalar(64 * (std::max)(matrix.rows(), matrix.cols())) * Eigen::NumTraits<RealScalar>::epsilon();
  const auto validate = [&]() {
    if (solver.info() != Eigen::Success) return false;
    const Vector solution = solver.solve(rhs);
    const RealScalar residual = (matrix * solution - rhs).norm();
    const RealScalar error = residual / (matrixNorm * solution.norm() + rhsNorm);
    state.counters["backward_error"] = error;
    state.counters["relative_residual"] = residual / rhsNorm;
    state.counters["relative_solution_error"] = (solution - reference).norm() / reference.norm();
    state.counters["solution_norm"] = solution.norm();
    return solver.info() == Eigen::Success && error <= bound;
  };
  state.SetLabel(filename);
  state.counters["rows"] = double(matrix.rows());
  state.counters["cols"] = double(matrix.cols());
  state.counters["nnz"] = double(matrix.nonZeros());
  if (!validate()) {
    state.SkipWithError("Initial factorization or solve validation failed");
    return;
  }
  for (auto _ : state) {
    solver.factorize(matrix);
    benchmark::DoNotOptimize(solver.info());
    benchmark::ClobberMemory();
  }
  if (!validate()) state.SkipWithError("Repeated factorization or solve validation failed");
}

#endif
