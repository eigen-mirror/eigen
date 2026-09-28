// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Benchmarks the eigenvector stage of the spectral-bisection path: inverse iteration on a real
// symmetric tridiagonal matrix via TridiagonalEigenSolver::computeEigenvectors(diag, subdiag, evals).
// The eigenvalues are precomputed (bisection) outside the timed region, so only the inverse-iteration
// work (LU factor + back-solves + intra-cluster reorthogonalization) is measured.
//
//   - BM_invit_rand_*    : random tridiagonal, well-separated spectrum (factor/solve bound);
//   - BM_invit_cluster_* : glued-Wilkinson blocks, tight clusters (reorthogonalization bound).
// Suffix _f / _d selects float / double.
//
// A second group times the full eigendecomposition (eigenvalues AND eigenvectors) of a random
// tridiagonal, comparing against the implicit-QR algorithm:
//   - BM_full_qr_*         : SelfAdjointEigenSolver::computeFromTridiagonal(ComputeEigenvectors);
//   - BM_full_bisect_*     : TridiagonalEigenSolver::compute(), all eigenpairs;
//   - BM_full_bisect_sub_* : TridiagonalEigenSolver::compute() of only the 10% smallest eigenpairs
//                            (a query QR cannot answer without the whole decomposition).
//
// When built with OpenMP, bisection and inverse iteration parallelize across Eigen::nbThreads()
// threads (the QR path is serial). To measure thread scaling, run the binary once per thread count
// via the environment (one process per data point; never sweep with setNbThreads() in-process):
//     for t in 1 2 4 8; do OMP_NUM_THREADS=$t ./bench_tridiagonal_inverse_iteration \
//         --benchmark_filter='BM_full_bisect_d/' --benchmark_repetitions=5; done

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>

using namespace Eigen;

namespace {

enum Kind { kRandom, kClustered };

template <typename Scalar>
void make_matrix(Kind kind, Index n, Matrix<Scalar, Dynamic, 1>& diag, Matrix<Scalar, Dynamic, 1>& sub) {
  diag.resize(n);
  sub.resize(n > 1 ? n - 1 : 0);
  if (kind == kRandom) {
    diag.setRandom();
    sub.setRandom();
  } else {
    // Glued Wilkinson W+_{blk}: blocks laid end to end with a small inter-block "glue" off-diagonal.
    // The large eigenvalues of each block nearly coincide across copies -> clusters at working
    // precision, the stress case for intra-cluster reorthogonalization.
    const Index blk = 15;
    const Index m = (blk - 1) / 2;
    for (Index i = 0; i < n; ++i) diag(i) = Scalar(numext::abs(m - (i % blk)));
    for (Index i = 0; i < n - 1; ++i) sub(i) = ((i + 1) % blk == 0) ? Scalar(1e-3) : Scalar(1);
  }
}

template <typename Scalar>
void run(benchmark::State& state, Kind kind) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> diag, sub;
  make_matrix<Scalar>(kind, n, diag, sub);

  // Precompute the eigenvalues once (not timed).
  TridiagonalEigenSolver<Scalar> evsolver(n);
  evsolver.computeEigenvalues(diag, sub);
  const Matrix<Scalar, Dynamic, 1> evals = evsolver.eigenvalues();

  TridiagonalEigenSolver<Scalar> solver(n);
  for (auto _ : state) {
    solver.computeEigenvectors(diag, sub, evals);
    benchmark::DoNotOptimize(solver.eigenvectors().data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * n);
}

void BM_invit_rand_f(benchmark::State& s) { run<float>(s, kRandom); }
void BM_invit_rand_d(benchmark::State& s) { run<double>(s, kRandom); }
void BM_invit_cluster_f(benchmark::State& s) { run<float>(s, kClustered); }
void BM_invit_cluster_d(benchmark::State& s) { run<double>(s, kClustered); }

void BM_invit_zero_block(benchmark::State& state) {
  const Index n = state.range(0);
  const double scale = numext::ldexp(1.0, int(state.range(1)));
  VectorXd diag = VectorXd::Constant(n + 1, 2.0 * scale);
  VectorXd sub = VectorXd::Constant(n, scale), evals(n + 1);
  diag(0) = sub(0) = evals(0) = 0.0;
  const double pi = std::acos(-1.0);
  for (Index i = 1; i <= n; ++i) evals(i) = (2.0 - 2.0 * std::cos(pi * double(i) / double(n + 1))) * scale;

  TridiagonalEigenSolver<double> solver;
  solver.computeEigenvectors(diag, sub, evals);
  const MatrixXd& vectors = solver.eigenvectors();
  MatrixXd residual = (diag / scale).asDiagonal() * vectors - vectors * (evals / scale).asDiagonal();
  residual.topRows(n) += (sub / scale).asDiagonal() * vectors.bottomRows(n);
  residual.bottomRows(n) += (sub / scale).asDiagonal() * vectors.topRows(n);
  const double tolerance = 128.0 * double(n + 1) * NumTraits<double>::epsilon();
  if (solver.info() != Success || !vectors.allFinite() || !(residual.norm() <= tolerance) ||
      !((vectors.transpose() * vectors - MatrixXd::Identity(n + 1, n + 1)).norm() <= tolerance)) {
    state.SkipWithError("Invalid zero-block eigenvectors");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(diag.data());
    benchmark::DoNotOptimize(sub.data());
    benchmark::DoNotOptimize(evals.data());
    benchmark::ClobberMemory();
    solver.computeEigenvectors(diag, sub, evals);
    benchmark::DoNotOptimize(solver.eigenvectors().data());
  }
  state.SetItemsProcessed(state.iterations() * (n + 1));
}

void BM_invit_supplied_small_block(benchmark::State& state) {
  const Index n = state.range(0);
  VectorXd diag = VectorXd::Constant(n + 1, 2.0);
  VectorXd sub = VectorXd::Ones(n), evals(n);
  diag(0) = numext::ldexp(1.0, 60);
  sub(0) = 0;
  const double pi = std::acos(-1.0);
  for (Index i = 0; i < n; ++i) evals(i) = 4 * numext::abs2(std::sin(pi * double(i + 1) / double(2 * (n + 1))));
  TridiagonalEigenSolver<double> solver;
  solver.computeEigenvectors(diag, sub, evals);
  const MatrixXd& vectors = solver.eigenvectors();
  MatrixXd residual = diag.asDiagonal() * vectors - vectors * evals.asDiagonal();
  residual.topRows(n) += sub.asDiagonal() * vectors.bottomRows(n);
  residual.bottomRows(n) += sub.asDiagonal() * vectors.topRows(n);
  const double tolerance = 128.0 * double(n) * NumTraits<double>::epsilon();
  if (solver.info() != Success || !vectors.allFinite() || !(residual.norm() <= tolerance) ||
      !((vectors.transpose() * vectors - MatrixXd::Identity(n, n)).norm() <= tolerance)) {
    state.SkipWithError("Invalid supplied small-block eigenvectors");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(diag.data());
    benchmark::DoNotOptimize(sub.data());
    benchmark::DoNotOptimize(evals.data());
    benchmark::ClobberMemory();
    solver.computeEigenvectors(diag, sub, evals);
    benchmark::DoNotOptimize(solver.eigenvectors().data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

enum FullMode { kFullQr, kFullBisect, kFullBisectSubset };

// Full eigendecomposition (eigenvalues + eigenvectors) of a random symmetric tridiagonal.
template <typename Scalar>
void run_full(benchmark::State& state, FullMode mode) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> diag, sub;
  make_matrix<Scalar>(kRandom, n, diag, sub);

  if (mode == kFullQr) {
    SelfAdjointEigenSolver<Matrix<Scalar, Dynamic, Dynamic>> solver(n);
    for (auto _ : state) {
      solver.computeFromTridiagonal(diag, sub, ComputeEigenvectors);
      benchmark::DoNotOptimize(solver.eigenvectors().data());
      benchmark::ClobberMemory();
    }
  } else {
    TridiagonalEigenSolver<Scalar> solver(n);
    const EigenvalueRange range =
        mode == kFullBisectSubset ? EigenvalueRange::indices(0, n / 10) : EigenvalueRange::all();
    for (auto _ : state) {
      solver.compute(diag, sub, ComputeEigenvectors, range);
      benchmark::DoNotOptimize(solver.eigenvectors().data());
      benchmark::ClobberMemory();
    }
  }
  state.SetItemsProcessed(state.iterations() * n);
}

void BM_full_qr_f(benchmark::State& s) { run_full<float>(s, kFullQr); }
void BM_full_qr_d(benchmark::State& s) { run_full<double>(s, kFullQr); }
void BM_full_bisect_f(benchmark::State& s) { run_full<float>(s, kFullBisect); }
void BM_full_bisect_d(benchmark::State& s) { run_full<double>(s, kFullBisect); }
void BM_full_bisect_sub_f(benchmark::State& s) { run_full<float>(s, kFullBisectSubset); }
void BM_full_bisect_sub_d(benchmark::State& s) { run_full<double>(s, kFullBisectSubset); }

#define EIGEN_BENCH_SIZES ArgsProduct({{16, 32, 64, 128, 256, 512, 1024, 2048}})
// QR with eigenvectors is O(n^3) (rotation accumulation), so cap the full-solve sizes lower than
// the eigenvalues-only benchmark's.
#define EIGEN_BENCH_FULL_SIZES ArgsProduct({{256, 512, 1024, 2048, 4096}})

BENCHMARK(BM_invit_rand_f)->EIGEN_BENCH_SIZES->UseRealTime();
BENCHMARK(BM_invit_rand_d)->EIGEN_BENCH_SIZES->UseRealTime();
BENCHMARK(BM_invit_cluster_f)->EIGEN_BENCH_SIZES->UseRealTime();
BENCHMARK(BM_invit_cluster_d)->EIGEN_BENCH_SIZES->UseRealTime();
BENCHMARK(BM_invit_zero_block)->ArgsProduct({{128, 512}, {0, -60}})->UseRealTime();
BENCHMARK(BM_invit_supplied_small_block)->Arg(512)->Arg(1024)->UseRealTime();
BENCHMARK(BM_full_qr_f)->EIGEN_BENCH_FULL_SIZES->UseRealTime();
BENCHMARK(BM_full_qr_d)->EIGEN_BENCH_FULL_SIZES->UseRealTime();
BENCHMARK(BM_full_bisect_f)->EIGEN_BENCH_FULL_SIZES->UseRealTime();
BENCHMARK(BM_full_bisect_d)->EIGEN_BENCH_FULL_SIZES->UseRealTime();
BENCHMARK(BM_full_bisect_sub_f)->EIGEN_BENCH_FULL_SIZES->UseRealTime();
BENCHMARK(BM_full_bisect_sub_d)->EIGEN_BENCH_FULL_SIZES->UseRealTime();

}  // namespace
