// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>
#include <random>

template <typename Scalar>
static void BM_TridiagonalScaling(benchmark::State& state) {
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  const Eigen::Index n = state.range(0);
  const int exponent = static_cast<int>(state.range(1));
  const bool vectors = state.range(2) != 0;
  std::mt19937 random(42);
  Vector diag(n), sub(n - 1);
  for (Eigen::Index i = 0; i < n; ++i) diag[i] = Scalar(int(random() % 2001) - 1000) / Scalar(1000);
  for (Eigen::Index i = 0; i < n - 1; ++i) sub[i] = Scalar(int(random() % 2001) - 1000) / Scalar(1000);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> reference;
  reference.computeFromTridiagonal(diag.template cast<double>(), sub.template cast<double>(), Eigen::EigenvaluesOnly);
  if (reference.info() != Eigen::Success) {
    state.SkipWithError("Reference failed");
    return;
  }
  const Vector expected = reference.eigenvalues().template cast<Scalar>();
  const Scalar scale = Eigen::numext::ldexp(Scalar(1), exponent);
  const Vector d = diag * scale, e = sub * scale, values = expected * scale;
  Eigen::TridiagonalEigenSolver<Scalar> solver(n);
  const auto compute = [&]() {
    if (vectors)
      solver.computeEigenvectors(d, e, values);
    else
      solver.computeEigenvalues(d, e);
  };
  compute();
  if (solver.info() != Eigen::Success) {
    state.SkipWithError("Solver failed");
    return;
  }
  const Scalar tolerance = Scalar(64 * n) * Eigen::NumTraits<Scalar>::epsilon();
  if (vectors) {
    Matrix matrix = Matrix::Zero(n, n);
    matrix.diagonal() = diag;
    matrix.diagonal(1) = sub;
    matrix.diagonal(-1) = sub;
    const Matrix& v = solver.eigenvectors();
    if (!v.allFinite() || !((matrix * v - v * expected.asDiagonal()).norm() <= tolerance * matrix.norm()) ||
        !((v.transpose() * v - Matrix::Identity(n, n)).norm() <= tolerance)) {
      state.SkipWithError("Invalid eigenvectors");
      return;
    }
  } else {
    const Vector normalized = solver.eigenvalues() / scale;
    if (!normalized.allFinite() || !((normalized - expected).norm() <= tolerance * expected.norm())) {
      state.SkipWithError("Invalid eigenvalues");
      return;
    }
  }
  for (auto _ : state) {
    compute();
    benchmark::DoNotOptimize(solver);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_TridiagonalScaling, float)->ArgsProduct({{3, 32, 128}, {-60, 0, 60}, {0, 1}});
BENCHMARK_TEMPLATE(BM_TridiagonalScaling, double)->ArgsProduct({{3, 32, 128}, {-60, 0, 60}, {0, 1}});
