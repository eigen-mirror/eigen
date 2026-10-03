// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// KroneckerOperator products and solves, (A (x) B) vec(X) = vec(B X A^T), with
// n x n factors and r right-hand sides. The product GFLOPS count
// 2 r (work(B) n1 + m2 work(A)) flops, work(F) = stored entries of F. Each
// benchmark first checks its result outside the timed loop.

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

using Mat = MatrixXd;
using SpMat = SparseMatrix<double>;
using Diag = DiagonalMatrix<double, Dynamic>;

namespace {

// The 1-D Laplacian tridiag(-1, 2, -1): its Kronecker sums are the 2-D
// finite-difference operators.
SpMat laplacian(Index n) {
  SpMat L(n, n);
  L.reserve(VectorXi::Constant(n, 3));
  for (Index j = 0; j < n; ++j) {
    if (j > 0) L.insert(j - 1, j) = -1.0;
    L.insert(j, j) = 2.0;
    if (j + 1 < n) L.insert(j + 1, j) = -1.0;
  }
  L.makeCompressed();
  return L;
}

Mat dominant(Index n) { return Mat::Random(n, n) + double(2 * n) * Mat::Identity(n, n); }

void setFlops(benchmark::State& state, double flopsPerIteration) {
  state.counters["GFLOPS"] =
      benchmark::Counter(1e-9 * flopsPerIteration, benchmark::Counter::kIsIterationInvariantRate);
}

// The batched result must match the operator applied to one column at a time.
template <typename Lhs, typename Rhs>
bool productMatchesColumns(const KroneckerOperator<Lhs, Rhs>& K, const Mat& X, const Mat& Y) {
  Mat expected(K.rows(), X.cols());
  for (Index k = 0; k < X.cols(); ++k) expected.col(k) = K * X.col(k);
  return (Y - expected).norm() <= 1e-10 * expected.norm();
}

// K x = b up to the conditioning of the (diagonally dominant) factors.
template <typename Lhs, typename Rhs>
bool solveHasSmallResidual(const KroneckerOperator<Lhs, Rhs>& K, const Mat& x, const Mat& b) {
  const Mat residual = K * x - b;
  return residual.norm() <= 1e-8 * b.norm();
}

template <typename Lhs, typename Rhs>
void runProduct(benchmark::State& state, const KroneckerOperator<Lhs, Rhs>& K, double workA, double workB) {
  const Index r = state.range(1);
  const Mat X = Mat::Random(K.cols(), r);
  Mat Y = K * X;
  if (!productMatchesColumns(K, X, Y)) {
    state.SkipWithError("batched product differs from the column-by-column product");
    return;
  }
  for (auto _ : state) {
    Y.noalias() = K * X;
    benchmark::DoNotOptimize(Y.data());
    benchmark::ClobberMemory();
  }
  const double n1 = double(K.lhs().cols()), m2 = double(K.rhs().rows());
  setFlops(state, 2.0 * double(r) * (workB * n1 + m2 * workA));
}

void BM_ProductDense(benchmark::State& state) {
  const Index n = state.range(0);
  const KroneckerOperator<Mat, Mat> K(Mat::Random(n, n), Mat::Random(n, n));
  runProduct(state, K, double(n * n), double(n * n));
}

void BM_ProductSparse(benchmark::State& state) {
  const Index n = state.range(0);
  const SpMat L = laplacian(n);
  const KroneckerOperator<SpMat, SpMat> K(L, L);
  runProduct(state, K, double(L.nonZeros()), double(L.nonZeros()));
}

// I_n (x) B, the block-diagonal operator of n independent n x n systems.
void BM_ProductIdentityDense(benchmark::State& state) {
  const Index n = state.range(0);
  const KroneckerOperator<Diag, Mat> K(Diag(VectorXd::Ones(n)), Mat::Random(n, n));
  runProduct(state, K, double(n), double(n * n));
}

// L (x) I_n and I_n (x) L: a sparse factor against a diagonal one.
void BM_ProductSparseIdentity(benchmark::State& state) {
  const SpMat L = laplacian(state.range(0));
  const KroneckerOperator<SpMat, Diag> K(L, Diag(VectorXd::Ones(L.rows())));
  runProduct(state, K, double(L.nonZeros()), double(L.rows()));
}

void BM_ProductIdentitySparse(benchmark::State& state) {
  const SpMat L = laplacian(state.range(0));
  const KroneckerOperator<Diag, SpMat> K(Diag(VectorXd::Ones(L.rows())), L);
  runProduct(state, K, double(L.rows()), double(L.nonZeros()));
}

template <typename Lhs, typename Rhs>
void runSolve(benchmark::State& state, const KroneckerOperator<Lhs, Rhs>& K) {
  const Mat b = Mat::Random(K.rows(), state.range(1));
  Mat x = K.solve(b);
  if (!solveHasSmallResidual(K, x, b)) {
    state.SkipWithError("solve residual too large");
    return;
  }
  for (auto _ : state) {
    x = K.solve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
}

void BM_SolveDense(benchmark::State& state) {
  const Index n = state.range(0);
  runSolve(state, KroneckerOperator<Mat, Mat>(dominant(n), dominant(n)));
}

void BM_SolveSparse(benchmark::State& state) {
  const SpMat L = laplacian(state.range(0));
  runSolve(state, KroneckerOperator<SpMat, SpMat>(L, L));
}

void BM_LeastSquaresDense(benchmark::State& state) {
  const Index n = state.range(0);
  const KroneckerOperator<Mat, Mat> K(dominant(n), dominant(n));  // full rank: least squares is the solve
  const Mat b = Mat::Random(K.rows(), state.range(1));
  Mat x = K.leastSquaresSolve(b);
  if (!solveHasSmallResidual(K, x, b)) {
    state.SkipWithError("least-squares residual too large");
    return;
  }
  for (auto _ : state) {
    x = K.leastSquaresSolve(b);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
}

}  // namespace

BENCHMARK(BM_ProductDense)->ArgsProduct({{8, 32, 128}, {1, 32}});
BENCHMARK(BM_ProductSparse)->ArgsProduct({{64, 512}, {1, 32}});
BENCHMARK(BM_ProductIdentityDense)->ArgsProduct({{8, 32, 128}, {1, 32}});
BENCHMARK(BM_ProductSparseIdentity)->ArgsProduct({{64, 512}, {1, 32}});
BENCHMARK(BM_ProductIdentitySparse)->ArgsProduct({{64, 512}, {1, 32}});
BENCHMARK(BM_SolveDense)->ArgsProduct({{8, 32, 128}, {1, 32}});
BENCHMARK(BM_SolveSparse)->ArgsProduct({{64, 512}, {1, 32}});
BENCHMARK(BM_LeastSquaresDense)->ArgsProduct({{32}, {1, 32}});
