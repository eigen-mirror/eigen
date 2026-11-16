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
void test_bicgstabl_T() {
  BiCGSTABL<SparseMatrix<T>, DiagonalPreconditioner<T> > bicgstabl_colmajor_diag;
  BiCGSTABL<SparseMatrix<T>, IncompleteLUT<T> > bicgstabl_colmajor_ilut;

  // This does not change the tolerance of the test, only the tolerance of the solver.
  bicgstabl_colmajor_diag.setTolerance(NumTraits<T>::epsilon() * 20);
  bicgstabl_colmajor_ilut.setTolerance(NumTraits<T>::epsilon() * 20);

  CALL_SUBTEST(check_sparse_square_solving(bicgstabl_colmajor_diag));
  CALL_SUBTEST(check_sparse_square_solving(bicgstabl_colmajor_ilut));

  // check_sparse_zero_tolerance is not run: BiCGSTABL returns the iterate x_min with the smallest recurrence residual
  // without comparing true residuals, so for tolerance 0 its b - A x can exceed 100 n eps |b|.

  CALL_SUBTEST(check_sparse_default_tolerance(bicgstabl_colmajor_diag));
}

template <typename = void>
void test_bicgstabl_solve_with_guess_restart() {
  // Regression for a solveWithGuess case where the recursive residual reached
  // the tolerance before the true residual did.
  srand(555248885);
  test_bicgstabl_T<double>();
}

// A zero right-hand side is solved by x = 0 without iterating; iterations() used to report the cap.
template <typename = void>
void test_zero_rhs() {
  SparseMatrix<double> A(4, 4);
  for (Index i = 0; i < 4; ++i) A.insert(i, i) = 2.0;
  BiCGSTABL<SparseMatrix<double> > solver(A);
  VectorXd x = solver.solve(VectorXd::Zero(4));
  VERIFY(x.isZero());
  VERIFY_IS_EQUAL(solver.iterations(), Index(0));
  VERIFY_IS_EQUAL(solver.error(), 0.0);
}

// See check_sparse_small_rhs: the solvers returned NoConvergence or an inaccurate x for small |b|.
template <typename = void>
void test_bicgstabl_small_rhs() {
  BiCGSTABL<SparseMatrix<double>, DiagonalPreconditioner<double>> solver_double;
  BiCGSTABL<SparseMatrix<float>, DiagonalPreconditioner<float>> solver_float;
  check_sparse_small_rhs(solver_double);
  check_sparse_small_rhs(solver_float);
}

EIGEN_DECLARE_TEST(bicgstabl) {
  CALL_SUBTEST_1(test_bicgstabl_T<double>());
  CALL_SUBTEST_2(test_bicgstabl_T<std::complex<double> >());
  CALL_SUBTEST_3(test_bicgstabl_solve_with_guess_restart<>());
  CALL_SUBTEST_3(test_zero_rhs<>());
  CALL_SUBTEST_3(test_bicgstabl_small_rhs<>());
  CALL_SUBTEST_3((check_matrix_free_accessors<BiCGSTABL<MatrixFreeOperator, IdentityPreconditioner>>()));
}
