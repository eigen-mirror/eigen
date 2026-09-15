// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
//
// Times the reductions, eigensolvers and Bunch-Kaufman factorization three ways: reusing a solver through
// compute() (steady state), constructing a plain solver from the matrix (allocation, copy and computation),
// and constructing the Ref<> instantiation within the matrix's own memory (computation only, the input is
// destroyed and refreshed outside the timed region). One family per binary, selected by the BENCH_* definition;
// BENCH_PLAIN_ONLY skips the Ref<> variants, to build the same file against a tree without them.
#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

using namespace Eigen;

template <typename MatrixType>
static MatrixType generalMatrix(Index n) {
  return MatrixType::Random(n, n);
}

template <typename MatrixType>
static MatrixType selfadjointMatrix(Index n) {
  MatrixType a = MatrixType::Random(n, n);
  return a + a.adjoint();
}

template <typename MatrixType>
static MatrixType positiveMatrix(Index n) {
  using RealScalar = typename MatrixType::RealScalar;
  MatrixType a = MatrixType::Random(n, n);
  return a * a.adjoint() + RealScalar(n) * MatrixType::Identity(n, n);
}

// An adapter names the solver, generates its input(s) and reads one scalar back from the result.

struct HessenbergAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = HessenbergDecomposition<M>;
  static MatrixType input(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return s.packedMatrix().coeff(0, 0);
  }
};

struct TridiagonalizationAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = Tridiagonalization<M>;
  static MatrixType input(Index n) { return selfadjointMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return s.packedMatrix().coeff(0, 0);
  }
};

struct RealSchurAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = RealSchur<M>;
  static MatrixType input(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return s.matrixT().coeff(0, 0);
  }
};

struct ComplexSchurAdapter {
  using MatrixType = MatrixXcd;
  template <typename M>
  using Solver = ComplexSchur<M>;
  static MatrixType input(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return numext::real(s.matrixT().coeff(0, 0));
  }
};

struct SelfAdjointEigenSolverAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = SelfAdjointEigenSolver<M>;
  static MatrixType input(Index n) { return selfadjointMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return s.eigenvalues().coeff(0);
  }
};

struct EigenSolverAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = EigenSolver<M>;
  static MatrixType input(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return numext::real(s.eigenvalues().coeff(0));
  }
};

struct ComplexEigenSolverAdapter {
  using MatrixType = MatrixXcd;
  template <typename M>
  using Solver = ComplexEigenSolver<M>;
  static MatrixType input(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return numext::real(s.eigenvalues().coeff(0));
  }
};

struct BunchKaufmanAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = BunchKaufman<M>;
  static MatrixType input(Index n) { return selfadjointMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return s.matrixLDLT().coeff(0, 0);
  }
};

struct RealQZAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = RealQZ<M>;
  static MatrixType inputA(Index n) { return generalMatrix<MatrixType>(n); }
  static MatrixType inputB(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return s.matrixS().coeff(0, 0);
  }
};

struct ComplexQZAdapter {
  using MatrixType = MatrixXcd;
  template <typename M>
  using Solver = ComplexQZ<M>;
  static MatrixType inputA(Index n) { return generalMatrix<MatrixType>(n); }
  static MatrixType inputB(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return numext::real(s.matrixS().coeff(0, 0));
  }
};

struct GeneralizedEigenSolverAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = GeneralizedEigenSolver<M>;
  static MatrixType inputA(Index n) { return generalMatrix<MatrixType>(n); }
  static MatrixType inputB(Index n) { return generalMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return numext::real(s.alphas().coeff(0));
  }
};

struct GeneralizedSelfAdjointEigenSolverAdapter {
  using MatrixType = MatrixXd;
  template <typename M>
  using Solver = GeneralizedSelfAdjointEigenSolver<M>;
  static MatrixType inputA(Index n) { return selfadjointMatrix<MatrixType>(n); }
  static MatrixType inputB(Index n) { return positiveMatrix<MatrixType>(n); }
  template <typename S>
  static double probe(const S& s) {
    return s.eigenvalues().coeff(0);
  }
};

// Single-matrix classes.

template <typename Adapter>
static void BM_compute(benchmark::State& state) {
  using MatrixType = typename Adapter::MatrixType;
  const Index n = state.range(0);
  const MatrixType A = Adapter::input(n);
  typename Adapter::template Solver<MatrixType> solver(A);
  double acc = 0;
  for (auto _ : state) {
    solver.compute(A);
    acc += Adapter::probe(solver);
    benchmark::DoNotOptimize(acc);
  }
}

template <typename Adapter>
static void BM_construct(benchmark::State& state) {
  using MatrixType = typename Adapter::MatrixType;
  const Index n = state.range(0);
  const MatrixType A = Adapter::input(n);
  double acc = 0;
  for (auto _ : state) {
    typename Adapter::template Solver<MatrixType> solver(A);
    acc += Adapter::probe(solver);
    benchmark::DoNotOptimize(acc);
  }
}

template <typename Adapter>
static void BM_inplace(benchmark::State& state) {
  using MatrixType = typename Adapter::MatrixType;
  const Index n = state.range(0);
  const MatrixType A = Adapter::input(n);
  MatrixType W = A;
  double acc = 0;
  for (auto _ : state) {
    state.PauseTiming();
    W = A;
    state.ResumeTiming();
    typename Adapter::template Solver<Ref<MatrixType> > solver(W);
    acc += Adapter::probe(solver);
    benchmark::DoNotOptimize(acc);
  }
}

// Matrix-pencil classes.

template <typename Adapter>
static void BM2_compute(benchmark::State& state) {
  using MatrixType = typename Adapter::MatrixType;
  const Index n = state.range(0);
  const MatrixType A = Adapter::inputA(n), B = Adapter::inputB(n);
  typename Adapter::template Solver<MatrixType> solver(A, B);
  double acc = 0;
  for (auto _ : state) {
    solver.compute(A, B);
    acc += Adapter::probe(solver);
    benchmark::DoNotOptimize(acc);
  }
}

template <typename Adapter>
static void BM2_construct(benchmark::State& state) {
  using MatrixType = typename Adapter::MatrixType;
  const Index n = state.range(0);
  const MatrixType A = Adapter::inputA(n), B = Adapter::inputB(n);
  double acc = 0;
  for (auto _ : state) {
    typename Adapter::template Solver<MatrixType> solver(A, B);
    acc += Adapter::probe(solver);
    benchmark::DoNotOptimize(acc);
  }
}

template <typename Adapter>
static void BM2_inplace(benchmark::State& state) {
  using MatrixType = typename Adapter::MatrixType;
  const Index n = state.range(0);
  const MatrixType A = Adapter::inputA(n), B = Adapter::inputB(n);
  MatrixType WA = A, WB = B;
  double acc = 0;
  for (auto _ : state) {
    state.PauseTiming();
    WA = A;
    WB = B;
    state.ResumeTiming();
    typename Adapter::template Solver<Ref<MatrixType> > solver(WA, WB);
    acc += Adapter::probe(solver);
    benchmark::DoNotOptimize(acc);
  }
}

#define EIGEN_BENCH_SIZES \
  { 32, 128, 512 }

#ifdef BENCH_PLAIN_ONLY
#define EIGEN_BENCH_INPLACE(Bench, Adapter)
#else
#define EIGEN_BENCH_INPLACE(Bench, Adapter) \
  BENCHMARK_TEMPLATE(Bench##_inplace, Adapter)->ArgsProduct({EIGEN_BENCH_SIZES});
#endif

#define EIGEN_BENCH_CLASS(Bench, Adapter)                                           \
  BENCHMARK_TEMPLATE(Bench##_compute, Adapter)->ArgsProduct({EIGEN_BENCH_SIZES});   \
  BENCHMARK_TEMPLATE(Bench##_construct, Adapter)->ArgsProduct({EIGEN_BENCH_SIZES}); \
  EIGEN_BENCH_INPLACE(Bench, Adapter)

#if defined(BENCH_REDUCTIONS)
EIGEN_BENCH_CLASS(BM, HessenbergAdapter)
EIGEN_BENCH_CLASS(BM, TridiagonalizationAdapter)
#elif defined(BENCH_SCHUR)
EIGEN_BENCH_CLASS(BM, RealSchurAdapter)
EIGEN_BENCH_CLASS(BM, ComplexSchurAdapter)
#elif defined(BENCH_SELFADJOINT)
EIGEN_BENCH_CLASS(BM, SelfAdjointEigenSolverAdapter)
EIGEN_BENCH_CLASS(BM2, GeneralizedSelfAdjointEigenSolverAdapter)
#elif defined(BENCH_EIGENSOLVER)
EIGEN_BENCH_CLASS(BM, EigenSolverAdapter)
EIGEN_BENCH_CLASS(BM, ComplexEigenSolverAdapter)
#elif defined(BENCH_QZ)
// A QZ iteration costs several times a Schur iteration of the same size; n = 256 keeps a run within minutes.
#undef EIGEN_BENCH_SIZES
#define EIGEN_BENCH_SIZES \
  { 32, 128, 256 }
EIGEN_BENCH_CLASS(BM2, RealQZAdapter)
EIGEN_BENCH_CLASS(BM2, ComplexQZAdapter)
EIGEN_BENCH_CLASS(BM2, GeneralizedEigenSolverAdapter)
#elif defined(BENCH_BUNCHKAUFMAN)
EIGEN_BENCH_CLASS(BM, BunchKaufmanAdapter)
#else
#error "Define one of BENCH_REDUCTIONS, BENCH_SCHUR, BENCH_SELFADJOINT, BENCH_EIGENSOLVER, BENCH_QZ, BENCH_BUNCHKAUFMAN"
#endif

BENCHMARK_MAIN();
