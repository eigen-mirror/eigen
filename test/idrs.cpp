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
  VERIFY_IS_EQUAL(solver.tolerance(), RealScalar(8) * numext::sqrt(RealScalar(n)) * eps);

  const VectorType b = VectorType::Random(n);
  VectorType x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_LE((b - A * x).norm(), RealScalar(2) * solver.tolerance() * b.norm());

  solver.setTolerance(RealScalar(1e3) * eps);
  VERIFY_IS_EQUAL(solver.tolerance(), RealScalar(1e3) * eps);

  // A residual that is not finite ends the solve at once, rather than restarting until maxIterations().
  VectorType bnan = b;
  bnan(0) = NumTraits<RealScalar>::quiet_NaN();
  x = solver.solve(bnan);
  VERIFY(solver.info() != Success);
  VERIFY_IS_EQUAL(solver.iterations(), 0);
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
}
