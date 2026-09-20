// Benchmarks for the implicit Kronecker operator: applying and solving with
// A (x) B through the vec identity (A (x) B) vec(X) = vec(B X A^T) [Van Loan
// 2000] against materializing the dense Kronecker product. With n x n factors
// the implicit product costs O(n^3) and touches O(n^2) memory, while the dense
// operator costs O(n^4) to apply (plus O(n^4) to form and store); the direct
// solve factors two n x n matrices instead of one n^2 x n^2 matrix.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> Vec;
typedef Matrix<double, Dynamic, Dynamic> Mat;
typedef DiagonalMatrix<double, Dynamic> Diag;

// --- Matrix-vector product y = (A (x) B) * x, n x n factors ---
static void BM_KroneckerProductImplicit(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), B = Mat::Random(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductImplicit)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

static void BM_KroneckerProductDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), B = Mat::Random(n, n);
  Mat dense = KroneckerOperator<Mat, Mat>(A, B);  // materialized once, outside the loop
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = dense * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductDense)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

// Forming the dense Kronecker product: the up-front cost (and O(n^4) storage)
// the implicit operator avoids entirely.
static void BM_KroneckerMaterializeDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), B = Mat::Random(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat dense(n * n, n * n);
  for (auto _ : state) {
    dense = K;
    benchmark::DoNotOptimize(dense.data());
  }
}
BENCHMARK(BM_KroneckerMaterializeDense)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

// --- Multiple right-hand sides ---
// Both the product and the direct solve walk the right-hand side column by
// column through the vec identity, so these cover the path where the per-column
// vec-trick workspaces are reused instead of reallocated.
static void BM_KroneckerProductImplicitMultiRhs(benchmark::State& state) {
  const Index n = state.range(0), nrhs = state.range(1);
  Mat A = Mat::Random(n, n), B = Mat::Random(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat X = Mat::Random(n * n, nrhs), Y(n * n, nrhs);
  for (auto _ : state) {
    Y.noalias() = K * X;
    benchmark::DoNotOptimize(Y.data());
  }
}
BENCHMARK(BM_KroneckerProductImplicitMultiRhs)->ArgsProduct({{8, 16, 32}, {8, 64}});

static void BM_KroneckerSolveImplicitMultiRhs(benchmark::State& state) {
  const Index n = state.range(0), nrhs = state.range(1);
  Mat A = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  Mat B = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Mat rhs = Mat::Random(n * n, nrhs), X(n * n, nrhs);
  for (auto _ : state) {
    X = K.solve(rhs);
    benchmark::DoNotOptimize(X.data());
  }
}
BENCHMARK(BM_KroneckerSolveImplicitMultiRhs)->ArgsProduct({{8, 16, 32}, {8, 64}});

// --- Identity and diagonal factors ---
// The I (x) A and A (x) I operators of finite-difference discretizations. With
// the identity stored as a dense factor the vec trick still pays a full GEMM
// for the identity side; stored as a DiagonalMatrix (unit diagonal) that side
// degenerates to a diagonal scaling, leaving one GEMM of the other factor.
static void BM_KroneckerProductIdentityLeftDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), Id = Mat::Identity(n, n);
  KroneckerOperator<Mat, Mat> K(Id, A);  // I_n (x) A, identity as a dense factor
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityLeftDense)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

static void BM_KroneckerProductIdentityLeftDiag(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n);
  KroneckerOperator<Diag, Mat> K(Vec::Ones(n).asDiagonal(), A);  // I_n (x) A, diagonal identity
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityLeftDiag)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

static void BM_KroneckerProductIdentityRightDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n), Id = Mat::Identity(n, n);
  KroneckerOperator<Mat, Mat> K(A, Id);  // A (x) I_n, identity as a dense factor
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityRightDense)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

static void BM_KroneckerProductIdentityRightDiag(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n);
  KroneckerOperator<Mat, Diag> K(A, Vec::Ones(n).asDiagonal());  // A (x) I_n, diagonal identity
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityRightDiag)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

// Solving (D (x) B) x = b: a densely stored diagonal factor costs a full LU
// per solve call; the DiagonalMatrix factor is normalized once and divided
// entrywise, leaving the single LU of the dense factor.
static void BM_KroneckerSolveDiagFactorDense(benchmark::State& state) {
  const Index n = state.range(0);
  Vec d = Vec::Random(n) + Vec::Constant(n, 2.0);
  Mat Dd = d.asDiagonal();
  Mat B = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  KroneckerOperator<Mat, Mat> K(Dd, B);
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = K.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveDiagFactorDense)->Arg(8)->Arg(16)->Arg(32);

static void BM_KroneckerSolveDiagFactorDiag(benchmark::State& state) {
  const Index n = state.range(0);
  Vec d = Vec::Random(n) + Vec::Constant(n, 2.0);
  Mat B = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  KroneckerOperator<Diag, Mat> K(d.asDiagonal(), B);
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = K.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveDiagFactorDiag)->Arg(8)->Arg(16)->Arg(32);

// --- Direct solve (A (x) B) x = b, n x n invertible factors ---
// The dense product is materialized once outside the timed loop. Each iteration
// then factorizes two n x n matrices for the implicit operator, versus one
// n^2 x n^2 matrix for the dense baseline.
static void BM_KroneckerSolveImplicit(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  Mat B = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  KroneckerOperator<Mat, Mat> K(A, B);
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = K.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveImplicit)->Arg(8)->Arg(16)->Arg(32);

static void BM_KroneckerSolveDense(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  Mat B = Mat::Random(n, n) + 2.0 * double(n) * Mat::Identity(n, n);
  Mat dense = KroneckerOperator<Mat, Mat>(A, B);  // materialized once, outside the loop
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = dense.partialPivLu().solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveDense)->Arg(8)->Arg(16)->Arg(32);

// ---- Sparse factors -----------------------------------------------------------
// I_n (x) A and A (x) I_n for a tridiagonal sparse A of order n: the
// sparse-factor operator against the materialized sparse Kronecker product, for
// the matrix-vector product and the one-shot direct solve (SparseLU per call).

using SpMat = SparseMatrix<double>;

static SpMat tridiagonal(Index n) {
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 3));
  for (Index j = 0; j < n; ++j) {
    if (j > 0) A.insert(j - 1, j) = -1.0;
    A.insert(j, j) = 2.0;
    if (j + 1 < n) A.insert(j + 1, j) = -1.0;
  }
  A.makeCompressed();
  return A;
}

static void BM_KroneckerProductIdentityLeftSparse(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  auto K = makeKroneckerOperator(Vec::Ones(n).asDiagonal(), A);
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityLeftSparse)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_KroneckerProductIdentityLeftSparseMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  SpMat K;
  K = makeKroneckerOperator(Vec::Ones(n).asDiagonal(), A);  // materialized once, outside the loop
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityLeftSparseMaterialized)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_KroneckerProductIdentityRightSparse(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  auto K = makeKroneckerOperator(A, Vec::Ones(n).asDiagonal());
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityRightSparse)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_KroneckerProductIdentityRightSparseMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  SpMat K;
  K = makeKroneckerOperator(A, Vec::Ones(n).asDiagonal());
  Vec x = Vec::Random(n * n), y(n * n);
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductIdentityRightSparseMaterialized)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_KroneckerMaterializeSparse(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  auto K = makeKroneckerOperator(Vec::Ones(n).asDiagonal(), A);
  SpMat M;
  for (auto _ : state) {
    M = K;
    benchmark::DoNotOptimize(M.valuePtr());
  }
}
BENCHMARK(BM_KroneckerMaterializeSparse)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_KroneckerSolveIdentityLeftSparse(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  auto K = makeKroneckerOperator(Vec::Ones(n).asDiagonal(), A);
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = K.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveIdentityLeftSparse)->Arg(64)->Arg(128)->Arg(256);

static void BM_KroneckerSolveIdentityLeftSparseMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  SpMat K;
  K = makeKroneckerOperator(Vec::Ones(n).asDiagonal(), A);
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    SparseLU<SpMat> lu(K);
    x = lu.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveIdentityLeftSparseMaterialized)->Arg(64)->Arg(128)->Arg(256);

static void BM_KroneckerSolveIdentityRightSparse(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  auto K = makeKroneckerOperator(A, Vec::Ones(n).asDiagonal());
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    x = K.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveIdentityRightSparse)->Arg(64)->Arg(128)->Arg(256);

static void BM_KroneckerSolveIdentityRightSparseMaterialized(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A = tridiagonal(n);
  SpMat K;
  K = makeKroneckerOperator(A, Vec::Ones(n).asDiagonal());
  Vec b = Vec::Random(n * n), x(n * n);
  for (auto _ : state) {
    SparseLU<SpMat> lu(K);
    x = lu.solve(b);
    benchmark::DoNotOptimize(x.data());
  }
}
BENCHMARK(BM_KroneckerSolveIdentityRightSparseMaterialized)->Arg(64)->Arg(128)->Arg(256);

static void BM_KroneckerProductNonFiniteSparse(benchmark::State& state) {
  const Index n = state.range(0);
  SpMat A(n, n);
  for (Index i = 1; i < n; ++i) A.insert(i, i) = 2.0;
  Mat B = Mat::Identity(2, 2);
  B(0, 0) = NumTraits<double>::infinity();
  auto K = makeKroneckerOperator(B, A);
  Vec x = Vec::Ones(2 * n), y = K * x;
  Vec expected = Vec::Constant(2 * n, 2.0);
  expected.head(n).setConstant(NumTraits<double>::infinity());
  expected[0] = expected[n] = 0.0;  // The empty sparse row annihilates Inf.
  if (!(y.array() == expected.array()).all()) {
    state.SkipWithError("non-finite product did not preserve structural zeros");
    return;
  }
  for (auto _ : state) {
    y.noalias() = K * x;
    benchmark::DoNotOptimize(y.data());
  }
}
BENCHMARK(BM_KroneckerProductNonFiniteSparse)->Arg(64)->Arg(128)->Arg(256)->Arg(512);
