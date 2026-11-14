// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include "sparse_factorize_benchmark.h"

template <typename Scalar>
using Natural = Eigen::MultifrontalLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower, Eigen::NaturalOrdering<int>>;
template <typename Scalar>
using SimplicialNatural = Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower, Eigen::NaturalOrdering<int>>;

// Positive definite band matrices, where SimplicialLDLT is the baseline.
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, Natural<float>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, Natural<double>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, Natural<std::complex<double>>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, SimplicialNatural<double>)->ArgsProduct({{128, 512}, {3, 17, 63}});

// ||P A P^T - L D L^*||_F <= 64 n epsilon (||A||_F + || |L| |D| |L^*| ||_F), the normwise form of the backward error
// bound of block LDL^T (N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM, 2002,
// Theorem 11.3).
template <typename MatrixType, int UpLo, typename Ordering>
bool validFactors(const Eigen::MultifrontalLDLT<MatrixType, UpLo, Ordering>& solver, const MatrixType& matrix) {
  using Factor = typename Eigen::MultifrontalLDLT<MatrixType, UpLo, Ordering>::FactorType;
  using Scalar = typename Factor::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  const Eigen::Index size = matrix.rows();

  std::vector<Eigen::Triplet<Scalar, typename Factor::StorageIndex>> entries;
  for (Eigen::Index k = 0; k < size; ++k) {
    entries.emplace_back(k, k, Scalar(solver.vectorD()[k]));
    if (solver.subDiagonal()[k] != Scalar(0)) {
      entries.emplace_back(k + 1, k, solver.subDiagonal()[k]);
      entries.emplace_back(k, k + 1, Eigen::numext::conj(solver.subDiagonal()[k]));
    }
  }
  Factor D(size, size);
  D.setFromTriplets(entries.begin(), entries.end());

  const Factor L = solver.matrixL(), U = solver.matrixU();
  const Factor LD = L * D, absLD = L.cwiseAbs() * D.cwiseAbs();
  const Factor error = Factor(solver.permutationP() * matrix * solver.permutationPinv()) - LD * U;
  const Factor scale = absLD * U.cwiseAbs();
  const RealScalar bound = RealScalar(64 * size) * Eigen::NumTraits<RealScalar>::epsilon();
  return error.norm() <= bound * (matrix.norm() + scale.norm());
}

// The baselines' factors are validated by their own benchmarks.
template <typename Solver>
bool validFactors(const Solver&, const typename Solver::MatrixType&) {
  return true;
}

// Validates the last factorization's factors.
template <typename Solver>
void factorizeLoop(benchmark::State& state, const typename Solver::MatrixType& matrix) {
  Solver solver;
  solver.analyzePattern(matrix);
  solver.factorize(matrix);
  if (solver.info() != Eigen::Success) {
    state.SkipWithError("factorization failed");
    return;
  }
  for (auto _ : state) {
    solver.factorize(matrix);
    benchmark::DoNotOptimize(solver.info());
    benchmark::ClobberMemory();
  }
  if (solver.info() != Eigen::Success) {
    state.SkipWithError("repeated factorization failed");
  } else if (!validFactors(solver, matrix)) {
    state.SkipWithError("factors exceed the backward error bound");
  }
}

// Times solve() with state.range(1) right-hand sides.
template <typename Solver>
void solveLoop(benchmark::State& state, const typename Solver::MatrixType& matrix) {
  using Scalar = typename Solver::Scalar;
  using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Solver solver(matrix);
  if (solver.info() != Eigen::Success) {
    state.SkipWithError("factorization failed");
    return;
  }
  const Dense rhs = Dense::Ones(matrix.rows(), state.range(1));
  Dense solution(matrix.rows(), state.range(1));
  for (auto _ : state) {
    solution = solver.solve(rhs);
    benchmark::DoNotOptimize(solution.data());
    benchmark::ClobberMemory();
  }
  // ||A X - B||_F <= 64 n epsilon (||A||_F ||X||_F + ||B||_F), a normwise backward error bound (J. L. Rigal and
  // J. Gaches, "On the compatibility of a given solution with the data of a linear system", J. ACM 14(3), 1967). The
  // bound scales with ||X||, which grows like N^3 on the KKT chain.
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  const RealScalar bound = RealScalar(64 * matrix.rows()) * Eigen::NumTraits<RealScalar>::epsilon();
  if (!((matrix * solution - rhs).norm() <= bound * (matrix.norm() * solution.norm() + rhs.norm()))) {
    state.SkipWithError("solve backward error exceeds bound");
  }
}

// analyzePattern() alone. The analysis depends only on the pattern, so there's nothing to validate here.
template <typename Solver>
void analyzeLoop(benchmark::State& state, const typename Solver::MatrixType& matrix) {
  for (auto _ : state) {
    Solver solver;
    solver.analyzePattern(matrix);
    benchmark::DoNotOptimize(solver);
  }
}

// The KKT matrix [delta I, A^T; A, 0] of a chain of N states, where A has a row for each of the N - 1 constraints
// x_k - x_{k+1} = 0. Its fronts have a few columns each, so it times the small-front paths. delta = 1 takes 1x1 pivots
// for the states; delta = 1e-6 fails their 1x1 tests, so it times 2x2 pivots and delayed columns.
template <typename Solver>
typename Solver::MatrixType kktChainMatrix(Eigen::Index numStates, const typename Solver::Scalar& delta) {
  using Scalar = typename Solver::Scalar;
  using SparseMatrix = typename Solver::MatrixType;
  const Eigen::Index size = 2 * numStates - 1;
  std::vector<Eigen::Triplet<Scalar>> entries;
  for (Eigen::Index i = 0; i < numStates; ++i) entries.emplace_back(i, i, delta);
  for (Eigen::Index r = 0; r + 1 < numStates; ++r) {
    const Eigen::Index row = numStates + r;
    entries.emplace_back(row, r, Scalar(1));
    entries.emplace_back(r, row, Scalar(1));
    entries.emplace_back(row, r + 1, Scalar(-1));
    entries.emplace_back(r + 1, row, Scalar(-1));
  }
  SparseMatrix matrix(size, size);
  matrix.setFromTriplets(entries.begin(), entries.end());
  matrix.makeCompressed();
  return matrix;
}

template <typename Solver>
void kktChainFactorize(benchmark::State& state) {
  using Scalar = typename Solver::Scalar;
  factorizeLoop<Solver>(state, kktChainMatrix<Solver>(state.range(0), state.range(1) == 0 ? Scalar(1) : Scalar(1e-6)));
}

template <typename Solver>
void kktChainSolve(benchmark::State& state) {
  solveLoop<Solver>(state, kktChainMatrix<Solver>(state.range(0), typename Solver::Scalar(1)));
}

template <typename Solver>
void kktChainAnalyze(benchmark::State& state) {
  analyzeLoop<Solver>(state, kktChainMatrix<Solver>(state.range(0), typename Solver::Scalar(1)));
}

// A shifted Laplacian on an N^3 grid, which is indefinite. Its root fronts have O(N^2) columns, so it times the level-3
// updates.
template <typename Solver>
typename Solver::MatrixType shiftedGridMatrix(Eigen::Index n) {
  using Scalar = typename Solver::Scalar;
  using SparseMatrix = typename Solver::MatrixType;
  const auto id = [n](Eigen::Index i, Eigen::Index j, Eigen::Index k) { return (i * n + j) * n + k; };
  std::vector<Eigen::Triplet<Scalar>> entries;
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      for (Eigen::Index k = 0; k < n; ++k) {
        const Eigen::Index node = id(i, j, k);
        entries.emplace_back(node, node, Scalar(4.5));
        const auto couple = [&](Eigen::Index other) {
          entries.emplace_back(other, node, Scalar(-1));
          entries.emplace_back(node, other, Scalar(-1));
        };
        if (i + 1 < n) couple(id(i + 1, j, k));
        if (j + 1 < n) couple(id(i, j + 1, k));
        if (k + 1 < n) couple(id(i, j, k + 1));
      }
    }
  }
  SparseMatrix matrix(n * n * n, n * n * n);
  matrix.setFromTriplets(entries.begin(), entries.end());
  matrix.makeCompressed();
  return matrix;
}

template <typename Solver>
void shiftedGridFactorize(benchmark::State& state) {
  factorizeLoop<Solver>(state, shiftedGridMatrix<Solver>(state.range(0)));
}

template <typename Solver>
void shiftedGridAnalyze(benchmark::State& state) {
  analyzeLoop<Solver>(state, shiftedGridMatrix<Solver>(state.range(0)));
}

template <typename Solver>
void shiftedGridSolve(benchmark::State& state) {
  solveLoop<Solver>(state, shiftedGridMatrix<Solver>(state.range(0)));
}

// The dense KKT matrix [-delta I, A^T; A, h I] in a single front, with h / |A| above 1 / threshold. Constraint columns
// fail both pivot tests until most of the variables are eliminated, while each variable passes its 1x1 test. Ordering
// the constraints first (second argument 1) puts the columns that keep failing ahead of the pivots, and ordering them
// last is the same factorization without those tests. The zero entries are stored so that all columns share one front.
template <typename Solver>
void stiffKktFactorize(benchmark::State& state) {
  using Scalar = typename Solver::Scalar;
  using SparseMatrix = typename Solver::MatrixType;
  const Eigen::Index numConstraints = state.range(0);
  const Eigen::Index numVariables = state.range(0);
  const bool constraintsFirst = state.range(1) != 0;
  const Eigen::Index firstConstraint = constraintsFirst ? 0 : numVariables;
  const Eigen::Index firstVariable = constraintsFirst ? numConstraints : 0;
  std::srand(1);
  std::vector<Eigen::Triplet<Scalar>> entries;
  const auto symmetric = [&](Eigen::Index row, Eigen::Index col, const Scalar& value) {
    entries.emplace_back(row, col, value);
    entries.emplace_back(col, row, value);
  };
  for (Eigen::Index i = 0; i < numConstraints; ++i) {
    entries.emplace_back(firstConstraint + i, firstConstraint + i, Scalar(-1e-10));
    for (Eigen::Index j = 0; j < i; ++j) symmetric(firstConstraint + i, firstConstraint + j, Scalar(0));
    for (Eigen::Index j = 0; j < numVariables; ++j) {
      symmetric(firstConstraint + i, firstVariable + j, Scalar(1) + Eigen::internal::random<Scalar>() / Scalar(2));
    }
  }
  for (Eigen::Index i = 0; i < numVariables; ++i) {
    entries.emplace_back(firstVariable + i, firstVariable + i, Scalar(1e6));
    for (Eigen::Index j = 0; j < i; ++j) symmetric(firstVariable + i, firstVariable + j, Scalar(0));
  }
  const Eigen::Index size = numConstraints + numVariables;
  SparseMatrix matrix(size, size);
  matrix.setFromTriplets(entries.begin(), entries.end());
  matrix.makeCompressed();
  factorizeLoop<Solver>(state, matrix);
}

// The KKT matrix of an N^3 grid with Hessian Laplacian + h I and a difference constraint at every other node. With h =
// 1e6 (second argument 1), most constraint columns are delayed, so fronts hold many columns that fail repeatedly. h = 8
// has fewer of them.
template <typename Solver>
void stiffGridKktFactorize(benchmark::State& state) {
  using Scalar = typename Solver::Scalar;
  using SparseMatrix = typename Solver::MatrixType;
  const Eigen::Index n = state.range(0);
  const Scalar h = state.range(1) == 0 ? Scalar(8) : Scalar(1e6);
  const auto id = [n](Eigen::Index i, Eigen::Index j, Eigen::Index k) { return (i * n + j) * n + k; };
  std::vector<Eigen::Triplet<Scalar>> entries;
  const auto couple = [&](Eigen::Index a, Eigen::Index b, const Scalar& value) {
    entries.emplace_back(a, b, value);
    entries.emplace_back(b, a, value);
  };
  Eigen::Index size = n * n * n;
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      for (Eigen::Index k = 0; k < n; ++k) {
        const Eigen::Index node = id(i, j, k);
        entries.emplace_back(node, node, h);
        if (i + 1 < n) couple(node, id(i + 1, j, k), Scalar(-1));
        if (j + 1 < n) couple(node, id(i, j + 1, k), Scalar(-1));
        if (k + 1 < n) couple(node, id(i, j, k + 1), Scalar(-1));
        if ((i + j + k) % 2 == 0) {
          const Eigen::Index axis = (i + 2 * j + 3 * k) % 3;
          const Eigen::Index ni = i + (axis == 0), nj = j + (axis == 1), nk = k + (axis == 2);
          if (ni < n && nj < n && nk < n) {
            entries.emplace_back(size, size, Scalar(-1e-10));
            couple(size, node, Scalar(1));
            couple(size, id(ni, nj, nk), Scalar(-1));
            ++size;
          }
        }
      }
    }
  }
  SparseMatrix matrix(size, size);
  matrix.setFromTriplets(entries.begin(), entries.end());
  matrix.makeCompressed();
  factorizeLoop<Solver>(state, matrix);
}

// An unpivoted LDLT can't factor these, so the baseline is SparseLU.
using MultifrontalAmd = Eigen::MultifrontalLDLT<Eigen::SparseMatrix<double>>;
using SparseLUAmd = Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>>;
BENCHMARK_TEMPLATE(kktChainFactorize, MultifrontalAmd)->ArgsProduct({{100, 1000, 10000}, {0, 1}});
BENCHMARK_TEMPLATE(kktChainFactorize, SparseLUAmd)->ArgsProduct({{100, 1000, 10000}, {0, 1}});
BENCHMARK_TEMPLATE(shiftedGridFactorize, MultifrontalAmd)->Arg(10)->Arg(20)->Arg(30)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(shiftedGridFactorize, SparseLUAmd)->Arg(10)->Arg(20)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(kktChainSolve, MultifrontalAmd)->ArgsProduct({{1000, 100000}, {1, 16}});
BENCHMARK_TEMPLATE(shiftedGridSolve, MultifrontalAmd)->ArgsProduct({{20, 30}, {1, 16}});
BENCHMARK_TEMPLATE(stiffKktFactorize, Natural<double>)
    ->ArgsProduct({{100, 500, 1000}, {0, 1}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(stiffGridKktFactorize, MultifrontalAmd)
    ->ArgsProduct({{12, 20}, {0, 1}})
    ->Unit(benchmark::kMillisecond);

// The symbolic analysis, with SimplicialLDLT's as the baseline. Natural ordering leaves out the time in AMD.
using SimplicialAmd = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
BENCHMARK_TEMPLATE(kktChainAnalyze, MultifrontalAmd)->Arg(10000)->Arg(100000)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(kktChainAnalyze, SimplicialAmd)->Arg(10000)->Arg(100000)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(shiftedGridAnalyze, MultifrontalAmd)->Arg(20)->Arg(40)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(shiftedGridAnalyze, SimplicialAmd)->Arg(20)->Arg(40)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(shiftedGridAnalyze, Natural<double>)->Arg(20)->Arg(40)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(shiftedGridAnalyze, SimplicialNatural<double>)->Arg(20)->Arg(40)->Unit(benchmark::kMillisecond);
