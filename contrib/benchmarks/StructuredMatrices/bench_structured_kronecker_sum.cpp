// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Benchmarks for the implicit Kronecker sum A (+) B = A (x) I + I (x) B and its
// BartelsStewart solver on finite-difference operators: the 2-D Laplacian
// Dy (+) Dx and the 3-D Laplacian Dz (+) Dy (+) Dx of tridiagonal 1-D factors,
// and a nonsymmetric 2-D convection-diffusion operator, against the
// materialized sparse matrix and SparseLU. The products cost
// O(n1 nnz(B) + n2 nnz(A)) either way; the direct solve costs one O(n^3)
// decomposition per factor and O(N sum_k n_k) per right-hand side, against the
// fill-in of a sparse LU of the N x N matrix.

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

using Vec = VectorXd;
using SpMat = SparseMatrix<double>;

// tridiag(-1 - c, 2, -1 + c): the negated second difference plus c times a
// centered first difference, SPD for c = 0.
static SpMat tridiagonal(Index n, double c = 0.0) {
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 3));
  for (Index j = 0; j < n; ++j) {
    if (j > 0) A.insert(j - 1, j) = -1.0 + c;
    A.insert(j, j) = 2.0;
    if (j + 1 < n) A.insert(j + 1, j) = -1.0 - c;
  }
  A.makeCompressed();
  return A;
}

// --- y = (Dy (+) Dx) x on an n x n grid ---
static void BM_KroneckerSumProduct2D(benchmark::State& state) {
  const Index n = state.range(0);
  auto L = makeKroneckerSum(tridiagonal(n), tridiagonal(n));
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = L * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerSumProduct2D)->Arg(64)->Arg(256)->Arg(1024);

static void BM_KroneckerSumProduct2DMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat L;
  L = makeKroneckerSum(tridiagonal(n), tridiagonal(n));
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = L * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerSumProduct2DMaterialized)->Arg(64)->Arg(256)->Arg(1024);

// --- y = ((Dz (+) Dy) (+) Dx) x on an n^3 grid: a left-nested sum, applied
// from the right inside the outer sum ---
static void BM_KroneckerSumProduct3DLeftNested(benchmark::State& state) {
  const Index n = state.range(0);
  const KroneckerSum<KroneckerSum<SpMat, SpMat>, SpMat> L(makeKroneckerSum(tridiagonal(n), tridiagonal(n)),
                                                          tridiagonal(n));
  Vec x = Vec::Random(n * n * n), y(n * n * n);
  for (auto _ : state) {
    y.noalias() = L * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerSumProduct3DLeftNested)->Arg(32)->Arg(64)->Arg(128);

// --- Implicit Euler step (I + tau D) u = b, D = -L the SPD negated Laplacian,
// decompositions set up once. Symmetric factors: the fast diagonalization path.
static auto heatStep2D(Index n, double tau) {
  SpMat I(n, n);
  I.setIdentity();
  const SpMat D = tridiagonal(n);
  return makeKroneckerSum(SpMat(I + tau * D), SpMat(tau * D));
}

static void BM_KroneckerSumSolveHeat2D(benchmark::State& state) {
  const Index n = state.range(0);
  auto M = heatStep2D(n, 0.25);
  BartelsStewart<decltype(M)> solver(M);
  Vec b = Vec::Random(n * n), u(n * n);
  for (auto _ : state) {
    u = solver.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_KroneckerSumSolveHeat2D)->Arg(64)->Arg(128)->Arg(256);

static void BM_KroneckerSumSolveHeat2DSparseLU(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat M;
  M = heatStep2D(n, 0.25);
  SparseLU<SpMat> lu(M);
  Vec b = Vec::Random(n * n), u(n * n);
  for (auto _ : state) {
    u = lu.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_KroneckerSumSolveHeat2DSparseLU)->Arg(64)->Arg(128)->Arg(256);

// The one-time setup: two n x n eigendecompositions against the sparse LU.
static void BM_KroneckerSumSetupHeat2D(benchmark::State& state) {
  const Index n = state.range(0);
  auto M = heatStep2D(n, 0.25);
  for (auto _ : state) {
    BartelsStewart<decltype(M)> solver(M);
    benchmark::DoNotOptimize(&solver);
  }
}
BENCHMARK(BM_KroneckerSumSetupHeat2D)->Arg(64)->Arg(128)->Arg(256);

static void BM_KroneckerSumSetupHeat2DSparseLU(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat M;
  M = heatStep2D(n, 0.25);
  for (auto _ : state) {
    SparseLU<SpMat> lu(M);
    benchmark::DoNotOptimize(&lu);
  }
}
BENCHMARK(BM_KroneckerSumSetupHeat2DSparseLU)->Arg(64)->Arg(128)->Arg(256);

// --- 3-D: (I + tau (Dz (+) Dy (+) Dx)) u = b on an n^3 grid ---
static auto heatStep3D(Index n, double tau) {
  SpMat I(n, n);
  I.setIdentity();
  const SpMat D = tridiagonal(n);
  return makeKroneckerSum(SpMat(I + tau * D), SpMat(tau * D), SpMat(tau * D));
}

static void BM_KroneckerSumSolveHeat3D(benchmark::State& state) {
  const Index n = state.range(0);
  auto M = heatStep3D(n, 0.25);
  BartelsStewart<decltype(M)> solver(M);
  Vec b = Vec::Random(n * n * n), u(n * n * n);
  for (auto _ : state) {
    u = solver.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_KroneckerSumSolveHeat3D)->Arg(16)->Arg(32)->Arg(48);

static void BM_KroneckerSumSolveHeat3DSparseLU(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat M;
  M = heatStep3D(n, 0.25);
  SparseLU<SpMat> lu(M);
  Vec b = Vec::Random(n * n * n), u(n * n * n);
  for (auto _ : state) {
    u = lu.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_KroneckerSumSolveHeat3DSparseLU)->Arg(16)->Arg(32)->Arg(48);

// --- Nonsymmetric factors: the complex Schur path ---
static void BM_KroneckerSumSolveConvection2D(benchmark::State& state) {
  const Index n = state.range(0);
  auto M = makeKroneckerSum(tridiagonal(n, 0.5), tridiagonal(n, 0.3));
  BartelsStewart<decltype(M)> solver(M);
  Vec b = Vec::Random(n * n), u(n * n);
  for (auto _ : state) {
    u = solver.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_KroneckerSumSolveConvection2D)->Arg(64)->Arg(128)->Arg(256);

// The transposed system on the same decompositions: a forward substitution on
// the transposed Schur forms.
static void BM_KroneckerSumSolveConvection2DTransposed(benchmark::State& state) {
  const Index n = state.range(0);
  auto M = makeKroneckerSum(tridiagonal(n, 0.5), tridiagonal(n, 0.3));
  BartelsStewart<decltype(M)> solver(M);
  Vec b = Vec::Random(n * n), u(n * n);
  for (auto _ : state) {
    u = solver.transpose().solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_KroneckerSumSolveConvection2DTransposed)->Arg(64)->Arg(128)->Arg(256);

static void BM_KroneckerSumSolveConvection2DSparseLU(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat M;
  M = makeKroneckerSum(tridiagonal(n, 0.5), tridiagonal(n, 0.3));
  SparseLU<SpMat> lu(M);
  Vec b = Vec::Random(n * n), u(n * n);
  for (auto _ : state) {
    u = lu.solve(b);
    benchmark::DoNotOptimize(u.data());
  }
}
BENCHMARK(BM_KroneckerSumSolveConvection2DSparseLU)->Arg(64)->Arg(128)->Arg(256);

// --- Sparse assembly: M = Dz (+) Dy (+) Dx and M = I (x) (Dy (+) Dx) ---
// A Kronecker-sum factor is materialized once, not once per entry of the
// factor it meets.
static void BM_KroneckerSumAssemble3D(benchmark::State& state) {
  const Index n = state.range(0);
  const SpMat D = tridiagonal(n);
  const auto L = makeKroneckerSum(D, D, D);
  SpMat M;
  for (auto _ : state) {
    M = L;
    benchmark::DoNotOptimize(M.valuePtr());
  }
}
BENCHMARK(BM_KroneckerSumAssemble3D)->Arg(32)->Arg(64)->Arg(96);

static void BM_KroneckerSumAssembleIdentityKron(benchmark::State& state) {
  const Index n = state.range(0);
  const SpMat D = tridiagonal(n);
  const auto K = makeKroneckerOperator(MatrixXd::Identity(n, n), makeKroneckerSum(D, D));
  SpMat M;
  for (auto _ : state) {
    M = K;
    benchmark::DoNotOptimize(M.valuePtr());
  }
}
BENCHMARK(BM_KroneckerSumAssembleIdentityKron)->Arg(32)->Arg(64)->Arg(96);

// --- Spectrum of I_2 (x) (A (+) B) with nonsymmetric tridiagonal n x n A, B ---
// One n x n eigenvalue solve per factor of the sum, against the dense
// eigenvalue solve of the materialized n^2 x n^2 sum.
static void BM_KroneckerSumFactorEigenvalues(benchmark::State& state) {
  const Index n = state.range(0);
  const auto K = makeKroneckerOperator(MatrixXd::Identity(2, 2),
                                       makeKroneckerSum(MatrixXd(tridiagonal(n, 0.5)), MatrixXd(tridiagonal(n, 0.3))));
  for (auto _ : state) {
    VectorXcd lambda = K.eigenvalues();
    benchmark::DoNotOptimize(lambda.data());
  }
}
BENCHMARK(BM_KroneckerSumFactorEigenvalues)->Arg(16)->Arg(24);

static void BM_KroneckerSumFactorEigenvaluesMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  const MatrixXcd S = MatrixXd(makeKroneckerSum(tridiagonal(n, 0.5), tridiagonal(n, 0.3))).cast<std::complex<double>>();
  for (auto _ : state) {
    ComplexEigenSolver<MatrixXcd> es(S, /*computeEigenvectors=*/false);
    benchmark::DoNotOptimize(es.eigenvalues().data());
  }
}
BENCHMARK(BM_KroneckerSumFactorEigenvaluesMaterialized)->Arg(16)->Arg(24);
