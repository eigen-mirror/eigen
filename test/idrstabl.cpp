// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Uninitialized workspace entries read by the solver show up as NaN in the solution.
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T>
void test_idrstabl_T() {
  IDRSTABL<SparseMatrix<T>, DiagonalPreconditioner<T> > idrstabl_colmajor_diag;
  IDRSTABL<SparseMatrix<T>, IncompleteLUT<T> > idrstabl_colmajor_ilut;

  idrstabl_colmajor_diag.setTolerance(NumTraits<T>::epsilon() * 4);
  idrstabl_colmajor_ilut.setTolerance(NumTraits<T>::epsilon() * 4);

  CALL_SUBTEST(check_sparse_square_solving(idrstabl_colmajor_diag));
  CALL_SUBTEST(check_sparse_square_solving(idrstabl_colmajor_ilut));
}

// Neither a zero right-hand side nor the direct solve for n <= S iterates; iterations() used to report the cap.
void test_zero_rhs() {
  SparseMatrix<double> A(4, 4);
  for (Index i = 0; i < 4; ++i) A.insert(i, i) = 2.0;
  IDRSTABL<SparseMatrix<double> > solver(A);
  VectorXd x = solver.solve(VectorXd::Zero(4));
  VERIFY(x.isZero());
  VERIFY_IS_EQUAL(solver.iterations(), Index(0));
  VERIFY_IS_EQUAL(solver.error(), 0.0);
  // n <= S takes the direct dense solve, which performs no iteration either.
  x = solver.solve(VectorXd::Ones(4));
  VERIFY_IS_APPROX(A * x, VectorXd::Ones(4));
  VERIFY_IS_EQUAL(solver.iterations(), Index(0));
}

// An exact zero residual with zero tolerance breaks down normalization in the first IDR step.
void test_normalization_breakdown() {
  const Index n = 10;
  SparseMatrix<double> A(n, n);
  for (Index i = 0; i < n; ++i) A.insert(i, i) = 1.0;
  const VectorXd b = VectorXd::Unit(n, 0);
  for (Index L = 1; L <= 3; ++L) {
    IDRSTABL<SparseMatrix<double> > solver(A);
    solver.setL(L);
    solver.setS(2);
    solver.setTolerance(0.0);
    solver.setMaxIterations(2);
    VectorXd x = solver.solve(b);
    VERIFY(x.allFinite());
    VERIFY_IS_EQUAL(x, b);
  }
}

EIGEN_DECLARE_TEST(idrstabl) {
  CALL_SUBTEST_1((test_idrstabl_T<double>()));
  CALL_SUBTEST_2((test_idrstabl_T<std::complex<double> >()));
  CALL_SUBTEST_1(test_zero_rhs());
  CALL_SUBTEST_1(test_normalization_breakdown());
}
