// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 Kolja Brix <brix@igpm.rwth-aaachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T>
void test_idrs_T() {
  IDRS<SparseMatrix<T>, DiagonalPreconditioner<T> > idrs_colmajor_diag;
  IDRS<SparseMatrix<T>, IncompleteLUT<T> > idrs_colmajor_ilut;

  CALL_SUBTEST(check_sparse_square_solving(idrs_colmajor_diag));
  CALL_SUBTEST(check_sparse_square_solving(idrs_colmajor_ilut));

  CALL_SUBTEST(check_sparse_zero_tolerance(idrs_colmajor_diag));
  CALL_SUBTEST(check_sparse_zero_tolerance(idrs_colmajor_ilut));

  CALL_SUBTEST(check_sparse_default_tolerance(idrs_colmajor_diag));
}

// Residual smoothing used to scale its line search by |t| instead of |t|^2 and diverged.
template <typename T>
void test_idrs_smoothing() {
  using RealScalar = typename NumTraits<T>::Real;
  const Index n = 100;
  SparseMatrix<T> A(n, n);
  for (Index i = 0; i < n; ++i) {
    A.insert(i, i) = T(4);
    if (i > 0) A.insert(i, i - 1) = T(-1);
  }
  Matrix<T, Dynamic, 1> b = Matrix<T, Dynamic, 1>::Random(n);
  IDRS<SparseMatrix<T> > solver(A);
  solver.setSmoothing(true);
  solver.setTolerance(RealScalar(1e-10));
  Matrix<T, Dynamic, 1> x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(A * x, b);
}

// In exact arithmetic IDR(s) reaches the solution of an n x n system within n / s cycles. The bi-orthogonalization used
// to skip the latest direction, which lost that property: these systems took 7 or more cycles, and for s = n often did
// not converge.
template <typename T>
void test_idrs_finite_termination() {
  using RealScalar = typename NumTraits<T>::Real;
  using MatrixType = Matrix<T, Dynamic, Dynamic>;
  using VectorType = Matrix<T, Dynamic, 1>;
  const Index n = 8;
  // Singular values 1, ..., n and random singular vectors, so no early convergence hides a lost termination property.
  const MatrixType q1 = HouseholderQR<MatrixType>(MatrixType::Random(n, n)).householderQ();
  const MatrixType q2 = HouseholderQR<MatrixType>(MatrixType::Random(n, n)).householderQ();
  const MatrixType A =
      q1 * Matrix<RealScalar, Dynamic, 1>::LinSpaced(n, RealScalar(1), RealScalar(n)).asDiagonal() * q2.adjoint();
  const VectorType b = VectorType::Random(n);
  const RealScalar tol = RealScalar(1e4) * NumTraits<RealScalar>::epsilon();
  for (Index s : {Index(4), n}) {
    IDRS<MatrixType, IdentityPreconditioner> solver(A);
    solver.setS(s);
    solver.setTolerance(tol);
    solver.setResidualUpdate(true);
    const VectorType x = solver.solve(b);
    VERIFY_IS_EQUAL(solver.info(), Success);
    // Rounding, and a restart when b - A x does not confirm convergence, cost an occasional cycle: over 10^6 seeds no
    // solve took more than n / s + 2 of them.
    VERIFY_LE(solver.iterations(), n / s + 2);
    // Success is confirmed with b - A x itself; the factor 2 covers rounding in recomputing it here.
    VERIFY_LE((b - A * x).norm(), RealScalar(2) * tol * b.norm());
  }
}

// The recurrences for r and x drift apart when A U(:,k) = G(:,k) loses accuracy, which happens on ill-conditioned
// systems when the bi-orthogonalization cancels heavily. IDRS used to report Success from the recurrence residual
// alone, with b - A x up to 10^7 times the tolerance. For cond(A) = 10^6, s = 2 and a tolerance of 10^6 eps, a quarter
// to a third of the solves did.
template <typename T>
void test_idrs_true_residual() {
  using RealScalar = typename NumTraits<T>::Real;
  using MatrixType = Matrix<T, Dynamic, Dynamic>;
  using VectorType = Matrix<T, Dynamic, 1>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;
  const Index n = 8;
  // Singular values 10^0, ..., 10^-6, logarithmically spaced.
  const RealVector sigma =
      (RealVector::LinSpaced(n, RealScalar(0), RealScalar(-6)).array() * numext::log(RealScalar(10))).exp();
  // At this tolerance, about eps cond(A), a solve can legitimately stop with NoConvergence.
  const RealScalar tol = RealScalar(1e6) * NumTraits<RealScalar>::epsilon();
  for (int trial = 0; trial < 10; ++trial) {
    const MatrixType q1 = HouseholderQR<MatrixType>(MatrixType::Random(n, n)).householderQ();
    const MatrixType q2 = HouseholderQR<MatrixType>(MatrixType::Random(n, n)).householderQ();
    const MatrixType A = q1 * sigma.asDiagonal() * q2.adjoint();
    const VectorType b = VectorType::Random(n);
    for (Index s : {Index(2), Index(4), n}) {
      IDRS<MatrixType, IdentityPreconditioner> solver(A);
      solver.setS(s);
      solver.setTolerance(tol);
      const VectorType x = solver.solve(b);
      if (solver.info() == Success) {
        VERIFY_LE((b - A * x).norm(), RealScalar(2) * tol * b.norm());
      } else {
        VERIFY_IS_EQUAL(solver.info(), NoConvergence);
      }
    }
  }
}

template <typename T>
void test_idrs_default_tolerance() {
  using RealScalar = typename NumTraits<T>::Real;
  using MatrixType = Matrix<T, Dynamic, Dynamic>;
  using VectorType = Matrix<T, Dynamic, 1>;
  const Index n = 16;
  const MatrixType A = MatrixType::Identity(n, n) + RealScalar(0.1) * MatrixType::Random(n, n);
  IDRS<MatrixType, IdentityPreconditioner> solver(A);
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar confirmable = RealScalar(8) * numext::sqrt(RealScalar(n)) * eps;
  VERIFY_IS_EQUAL(solver.tolerance(), eps);

  const VectorType b = VectorType::Random(n);
  VectorType x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_LE((b - A * x).norm(), RealScalar(2) * confirmable * b.norm());

  // A negative tolerance selects the default, see MINRES::tolerance().
  solver.setTolerance(RealScalar(-1));
  VERIFY_IS_EQUAL(solver.tolerance(), eps);

  solver.setTolerance(RealScalar(1e3) * eps);
  VERIFY_IS_EQUAL(solver.tolerance(), RealScalar(1e3) * eps);

  // A residual that is not finite ends the solve at once, rather than restarting until maxIterations().
  VectorType bnan = b;
  bnan(0) = NumTraits<RealScalar>::quiet_NaN();
  x = solver.solve(bnan);
  VERIFY(solver.info() != Success);
  VERIFY_IS_EQUAL(solver.iterations(), 0);
}

// See check_sparse_small_rhs: the solvers returned NoConvergence or an inaccurate x for small |b|.
template <typename = void>
void test_idrs_small_rhs() {
  IDRS<SparseMatrix<double>, DiagonalPreconditioner<double>> solver_double;
  IDRS<SparseMatrix<float>, DiagonalPreconditioner<float>> solver_float;
  check_sparse_small_rhs(solver_double);
  check_sparse_small_rhs(solver_float);
  check_sparse_tiny_rhs(solver_float);
}

// The default tolerance is eps, so that a well-conditioned system is solved to the accuracy of the arithmetic: for
// tridiag(-1, 4, -1) and x = 1 in float, a default of 8 sqrt(n) eps stopped at a forward error of 3e-5 for n = 10000,
// and eps gives 3e-8.
template <typename = void>
void test_idrs_default_accuracy() {
  using SpMat = SparseMatrix<float>;
  const float eps = NumTraits<float>::epsilon();
  for (Index n : {40, 10000}) {
    std::vector<Triplet<float>> triplets;
    for (Index i = 0; i < n; ++i) {
      triplets.emplace_back(i, i, 4.0f);
      if (i + 1 < n) {
        triplets.emplace_back(i, i + 1, -1.0f);
        triplets.emplace_back(i + 1, i, -1.0f);
      }
    }
    SpMat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const VectorXf ones = VectorXf::Ones(n);
    const VectorXf b = A * ones;
    IDRS<SpMat, DiagonalPreconditioner<float>> solver(A);
    VERIFY_IS_EQUAL(solver.tolerance(), eps);
    const VectorXf x = solver.solve(b);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_LE((x - ones).norm() / ones.norm(), 10 * eps);
  }
}

// On D T D for T = tridiag(-1, 4, -1) and D spread over 10^4, without a preconditioner, the recurrence residual of IDRS
// drifts from b - A x, and a solve that ended at maxIterations() reported the recurrence residual: error() was up to
// 10^4 times too small (about one solve in seven had it off by more than a factor of 2). error() is now the computed
// |b - A x| / |b| at every exit but a breakdown.
template <typename = void>
void test_idrs_final_residual() {
  using SpMat = SparseMatrix<double>;
  const Index n = 120;
  VectorXd d(n);
  for (Index i = 0; i < n; ++i) d(i) = std::pow(1e4, double((i * 7) % 9) / 8.0 - 0.5);
  std::vector<Triplet<double>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, d(i) * d(i) * 4);
    if (i + 1 < n) {
      triplets.emplace_back(i, i + 1, -d(i) * d(i + 1));
      triplets.emplace_back(i + 1, i, -d(i) * d(i + 1));
    }
  }
  SpMat A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  for (int trial = 0; trial < 20; ++trial) {
    const VectorXd b = VectorXd::Random(n);
    for (Index maxit : {60, 80, 100}) {
      IDRS<SpMat, IdentityPreconditioner> solver(A);
      solver.setTolerance(1e-14);
      solver.setMaxIterations(maxit);
      const VectorXd x = solver.solve(b);
      if (solver.info() == NumericalIssue) continue;  // a breakdown leaves error() at the tolerance
      const double res = (b - A * x).norm() / b.norm();
      VERIFY_LE(numext::abs(solver.error() - res), res / 2);
    }
  }
}

// The 5-point Laplacian on an m x m grid.
template <typename SpMat>
SpMat idrs_poisson2d(Index m) {
  std::vector<Triplet<double>> triplets;
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < m; ++j) {
      const Index k = i * m + j;
      triplets.emplace_back(k, k, 4.0);
      if (j + 1 < m) triplets.emplace_back(k, k + 1, -1.0);
      if (j > 0) triplets.emplace_back(k, k - 1, -1.0);
      if (i + 1 < m) triplets.emplace_back(k, k + m, -1.0);
      if (i > 0) triplets.emplace_back(k, k - m, -1.0);
    }
  }
  SpMat A(m * m, m * m);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

// A tolerance of eps is below the rounding error of b - A x on this matrix. IDRS used to restart after every failed
// confirmation until maxIterations() = 2 n (1800 iterations for m = 30, against about 25 that suffice). It now stops
// when a failed confirmation comes 5 iterations after |b - A x| last halved, and error() is the computed
// |b - A x| / |b| whichever way it ends. A tolerance set by the caller is not relaxed: Success exactly when it is met.
template <typename = void>
void test_idrs_stagnation() {
  using SpMat = SparseMatrix<double>;
  const double eps = NumTraits<double>::epsilon();
  for (Index m : {30, 40}) {
    const SpMat A = idrs_poisson2d<SpMat>(m);
    const Index n = A.rows();
    const VectorXd b = VectorXd::Random(n);
    IDRS<SpMat, DiagonalPreconditioner<double>> solver(A);
    solver.setTolerance(eps);
    const VectorXd x = solver.solve(b);
    VERIFY(x.allFinite());
    VERIFY_LE(solver.iterations(), 5 * m);
    const double res = (b - A * x).norm() / b.norm();
    VERIFY_LE(res, double(n) * eps);
    VERIFY_LE(numext::abs(solver.error() - res), res / 2 + eps);
    VERIFY_IS_EQUAL(solver.info(), solver.error() <= eps ? Success : NoConvergence);
  }

  // The default tolerance is eps too. The residual stagnates above it, and IDRS returns Success there as it is at most
  // 8 sqrt(n) eps. That error() exceeds eps, which proves that the relaxed branch ran, holds for almost every solve but
  // not for all (a solve confirms eps in about one of a few thousand for m = 40, and in 3% for m = 25), so it is
  // required of at least one of four.
  const Index m = 40;
  const SpMat A = idrs_poisson2d<SpMat>(m);
  const Index n = A.rows();
  const double confirmable = 8 * numext::sqrt(double(n)) * eps;
  bool relaxed = false;
  for (int trial = 0; trial < 4; ++trial) {
    const VectorXd b = VectorXd::Random(n);
    IDRS<SpMat, DiagonalPreconditioner<double>> solver(A);
    const VectorXd x = solver.solve(b);
    VERIFY_IS_EQUAL(solver.tolerance(), eps);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_LE(solver.error(), confirmable);
    VERIFY_LE(solver.iterations(), 5 * m);
    VERIFY_LE((b - A * x).norm(), 2 * confirmable * b.norm());
    relaxed |= solver.error() > eps;
  }
  VERIFY(relaxed);
}

EIGEN_DECLARE_TEST(idrs) {
  CALL_SUBTEST_1(test_idrs_T<double>());
  CALL_SUBTEST_2(test_idrs_T<std::complex<double> >());
  CALL_SUBTEST_1(test_idrs_smoothing<double>());
  CALL_SUBTEST_2(test_idrs_smoothing<std::complex<double> >());
  CALL_SUBTEST_1(test_idrs_finite_termination<double>());
  CALL_SUBTEST_2(test_idrs_finite_termination<std::complex<double> >());
  CALL_SUBTEST_1(test_idrs_true_residual<double>());
  CALL_SUBTEST_2(test_idrs_true_residual<std::complex<double> >());
  CALL_SUBTEST_1(test_idrs_default_tolerance<double>());
  CALL_SUBTEST_2(test_idrs_default_tolerance<std::complex<double> >());
  CALL_SUBTEST_1(test_idrs_small_rhs<>());
  CALL_SUBTEST_1(test_idrs_stagnation<>());
  CALL_SUBTEST_1(test_idrs_final_residual<>());
  CALL_SUBTEST_1(test_idrs_default_accuracy<>());
  CALL_SUBTEST_1((check_matrix_free_accessors<IDRS<MatrixFreeOperator, IdentityPreconditioner>>()));
}
