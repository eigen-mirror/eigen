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
void test_gmres_T() {
  GMRES<SparseMatrix<T>, DiagonalPreconditioner<T> > gmres_colmajor_diag;
  GMRES<SparseMatrix<T>, IdentityPreconditioner> gmres_colmajor_I;
  GMRES<SparseMatrix<T>, IncompleteLUT<T> > gmres_colmajor_ilut;
  // GMRES<SparseMatrix<T>, SSORPreconditioner<T> >     gmres_colmajor_ssor;

  CALL_SUBTEST(check_sparse_square_solving(gmres_colmajor_diag));
  //   CALL_SUBTEST( check_sparse_square_solving(gmres_colmajor_I)     );
  CALL_SUBTEST(check_sparse_square_solving(gmres_colmajor_ilut));
  // CALL_SUBTEST( check_sparse_square_solving(gmres_colmajor_ssor)     );

  CALL_SUBTEST(check_sparse_zero_tolerance(gmres_colmajor_diag));
  CALL_SUBTEST(check_sparse_zero_tolerance(gmres_colmajor_ilut));

  CALL_SUBTEST(check_sparse_default_tolerance(gmres_colmajor_diag));
}

template <typename = void>
void test_gmres_solve_with_guess_tolerance() {
  const SparseMatrix<double>::StorageIndex size = 20;
  SparseMatrix<double> matrix(size, size);
  std::vector<Triplet<double> > triplets;
  for (SparseMatrix<double>::StorageIndex i = 0; i < size; ++i) {
    triplets.emplace_back(i, i, 2.0 + double(i) / double(size));
    if (i > 0) triplets.emplace_back(i, i - 1, -0.25);
    if (i + 1 < size) triplets.emplace_back(i, i + 1, 0.125);
  }
  matrix.setFromTriplets(triplets.begin(), triplets.end());

  VectorXd rhs = VectorXd::Ones(size);
  VectorXd guess(size);
  for (Index i = 0; i < size; ++i) {
    guess[i] = ((i % 2) ? 1.0 : -1.0) * 1e8;
  }

  GMRES<SparseMatrix<double>, DiagonalPreconditioner<double> > gmres;
  gmres.compute(matrix);
  gmres.setTolerance(1e-6);
  gmres.setMaxIterations(size);
  gmres.set_restart(size);

  VectorXd x = gmres.solveWithGuess(rhs, guess);
  DiagonalPreconditioner<double> preconditioner(matrix);
  const VectorXd residual = matrix * x - rhs;
  const double relativeResidual = preconditioner.solve(residual).norm() / preconditioner.solve(rhs).norm();
  VERIFY(gmres.info() == Success);
  VERIFY(gmres.iterations() > 1);
  VERIFY(relativeResidual < gmres.tolerance());
  VERIFY(gmres.error() < gmres.tolerance());
}

template <typename = void>
void test_gmres_large_restart() {
  const SparseMatrix<double>::StorageIndex size = 4;
  SparseMatrix<double> matrix(size, size);
  std::vector<Triplet<double> > triplets;
  for (SparseMatrix<double>::StorageIndex i = 0; i < size; ++i) {
    triplets.emplace_back(i, i, 2.0 + double(i));
  }
  matrix.setFromTriplets(triplets.begin(), triplets.end());

  const VectorXd rhs = VectorXd::LinSpaced(size, 1.0, 4.0);
  GMRES<SparseMatrix<double>, IdentityPreconditioner> gmres;
  gmres.compute(matrix);
  gmres.setTolerance(1e-12);
  gmres.setMaxIterations(size);
  gmres.set_restart(1000000);

  VectorXd x = gmres.solve(rhs);
  VERIFY(gmres.info() == Success);
  VERIFY_IS_APPROX(matrix * x, rhs);
}

// A zero right-hand side is solved by x = 0 without iterating; iterations() used to report the cap.
void test_zero_rhs() {
  SparseMatrix<double> A(4, 4);
  for (Index i = 0; i < 4; ++i) A.insert(i, i) = 2.0;
  GMRES<SparseMatrix<double> > solver(A);
  VectorXd x = solver.solve(VectorXd::Zero(4));
  VERIFY(x.isZero());
  VERIFY_IS_EQUAL(solver.iterations(), Index(0));
  VERIFY_IS_EQUAL(solver.error(), 0.0);
}

// The iteration limit is honored, including 0: GMRES used to test iters == maxIters only after incrementing, so 0 never
// matched. It ran to convergence, and with an unreachable tolerance restarted forever.
template <typename = void>
void test_gmres_max_iterations() {
  const Index n = 60;
  SparseMatrix<double> A(n, n);
  for (Index i = 0; i < n; ++i) {
    A.insert(i, i) = 4.0 + double(i % 7);
    if (i > 0) A.insert(i, i - 1) = -0.5;
    if (i + 1 < n) A.insert(i, i + 1) = 1.0;
  }
  const VectorXd b = VectorXd::LinSpaced(n, 1.0, 2.0);
  const VectorXd guess = VectorXd::Constant(n, 0.25);
  for (double tol : {NumTraits<double>::epsilon(), 0.0}) {
    for (Index maxit : {0, 1, 2, 5}) {
      GMRES<SparseMatrix<double>> solver(A);
      solver.set_restart(10);
      solver.setTolerance(tol);
      solver.setMaxIterations(maxit);
      VectorXd x = solver.solve(b);
      VERIFY(x.allFinite());
      VERIFY_LE(solver.iterations(), maxit);
      VERIFY_IS_EQUAL(solver.info(), NoConvergence);
      if (maxit == 0) {
        VERIFY(x.isZero());
        x = solver.solveWithGuess(b, guess);
        VERIFY_IS_EQUAL(solver.iterations(), Index(0));
        VERIFY_IS_EQUAL(x, guess);
      }
    }
  }
}

// See check_sparse_small_rhs: the solvers returned NoConvergence or an inaccurate x for small |b|.
template <typename = void>
void test_gmres_small_rhs() {
  GMRES<SparseMatrix<double>, DiagonalPreconditioner<double>> solver_double;
  GMRES<SparseMatrix<float>, DiagonalPreconditioner<float>> solver_float;
  check_sparse_small_rhs(solver_double);
  check_sparse_small_rhs(solver_float);
}

EIGEN_DECLARE_TEST(gmres) {
  CALL_SUBTEST_1(test_gmres_T<double>());
  CALL_SUBTEST_2(test_gmres_T<std::complex<double> >());
  CALL_SUBTEST_3(test_gmres_solve_with_guess_tolerance<>());
  CALL_SUBTEST_4(test_gmres_large_restart<>());
  CALL_SUBTEST_4(test_zero_rhs());
  CALL_SUBTEST_4(test_gmres_max_iterations<>());
  CALL_SUBTEST_4(test_gmres_small_rhs<>());
  CALL_SUBTEST_4((check_matrix_free_accessors<GMRES<MatrixFreeOperator, IdentityPreconditioner>>()));
}
