// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"
#include <Eigen/PaStiXSupport>
#include <contrib/Eigen/SparseExtra>

template <typename Solver>
void test_pastix_reuse() {
  using Scalar = typename Solver::Scalar;
  using Real = typename NumTraits<Scalar>::Real;
  using Sparse = typename Solver::MatrixType;
  using Dense = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  Sparse matrix(5, 5);
  const Scalar offDiagonal = Scalar(-1) + internal::random<Scalar>() * Scalar(0.25);
  for (Index i = 0; i < matrix.rows(); ++i) {
    matrix.insert(i, i) = Scalar(4);
    if (i > 0) matrix.insert(i, i - 1) = offDiagonal;
    if (i + 1 < matrix.rows()) matrix.insert(i, i + 1) = numext::conj(offDiagonal);
  }
  Dense expected(5, 2);
  for (Index i = 0; i < expected.rows(); ++i) {
    expected(i, 0) = Scalar(i + 1);
    expected(i, 1) = Scalar(2 - i);
  }
  Solver solver;
  solver.iparm(IPARM_THREAD_NBR) = 1;
  solver.dparm(DPARM_EPSILON_REFINEMENT) = NumTraits<Real>::epsilon();
  solver.analyzePattern(matrix);
  VERIFY_IS_EQUAL(solver.info(), Success);
  for (int iteration = 0; iteration < 2; ++iteration) {
    for (Index i = 0; i < matrix.rows(); ++i) matrix.coeffRef(i, i) += Scalar(1);
    solver.factorize(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    const Dense rhs = matrix * expected;
    const Dense result = solver.solve(rhs);
    VERIFY_IS_EQUAL(solver.info(), Success);
    const Real errorBound = Real(128) * NumTraits<Real>::epsilon() * expected.norm();
    VERIFY((result - expected).norm() <= errorBound);
  }
  solver.compute(matrix);
  VERIFY_IS_EQUAL(solver.info(), Success);
  const Dense rhs = matrix * expected;
  const Dense result = solver.solve(rhs);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(result, expected);

  Sparse diagonal(2, 2);
  diagonal.insert(0, 0) = Scalar(2);
  diagonal.insert(1, 1) = Scalar(3);
  solver.compute(diagonal);
  VERIFY_IS_EQUAL(solver.info(), Success);
  const Matrix<Scalar, 2, 1> diagonalRhs(Scalar(2), Scalar(3));
  const Matrix<Scalar, 2, 1> diagonalResult = solver.solve(diagonalRhs);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(diagonalResult, (Matrix<Scalar, 2, 1>::Ones()));

  Sparse empty(0, 0);
  solver.compute(empty);
  VERIFY_IS_EQUAL(solver.info(), Success);
  const Dense emptyRhs(0, 2);
  const Dense emptyResult = solver.solve(emptyRhs);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(emptyResult.rows(), 0);
  VERIFY_IS_EQUAL(emptyResult.cols(), 2);
  solver.analyzePattern(empty);
  solver.factorize(empty);
  VERIFY_IS_EQUAL(solver.info(), Success);
  solver.compute(matrix);
  const Dense noRhs(matrix.rows(), 0);
  const Dense noResult = solver.solve(noRhs);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(noResult.cols(), 0);
}

template <typename Scalar>
void test_pastix_special_cases() {
  using Sparse = SparseMatrix<Scalar, RowMajor, long long>;
  test_pastix_reuse<PastixLU<Sparse>>();
  test_pastix_reuse<PastixLLT<Sparse, Lower>>();
  test_pastix_reuse<PastixLDLT<Sparse, Upper>>();
  for (int emptyRow = 0; emptyRow < 2; ++emptyRow) {
    Sparse singular(2, 2);
    singular.insert(0, 0) = Scalar(1);
    if (!emptyRow) {
      singular.insert(0, 1) = Scalar(1);
      singular.insert(1, 0) = Scalar(1);
      singular.insert(1, 1) = Scalar(1);
    }
    PastixLU<Sparse> solver;
    solver.iparm(IPARM_THREAD_NBR) = 1;
    // A fixed tiny perturbation makes these inconsistent systems overflow during solve.
    solver.dparm(DPARM_EPSILON_MAGN_CTRL) = -(std::numeric_limits<double>::min)();
    solver.compute(singular);
    if (solver.info() == Success) {
      const Matrix<Scalar, 2, 1> rhs(Scalar(1), Scalar(8));
      const Matrix<Scalar, 2, 1> result = solver.solve(rhs);
      EIGEN_UNUSED_VARIABLE(result);
      VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    } else {
      VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    }
  }
}

template <typename T>
void test_pastix_T() {
  PastixLLT<SparseMatrix<T, ColMajor>, Eigen::Lower> pastix_llt_lower;
  PastixLDLT<SparseMatrix<T, ColMajor>, Eigen::Lower> pastix_ldlt_lower;
  PastixLLT<SparseMatrix<T, ColMajor>, Eigen::Upper> pastix_llt_upper;
  PastixLDLT<SparseMatrix<T, ColMajor>, Eigen::Upper> pastix_ldlt_upper;
  PastixLU<SparseMatrix<T, ColMajor> > pastix_lu;

  pastix_llt_lower.iparm(IPARM_THREAD_NBR) = 1;
  pastix_ldlt_lower.iparm(IPARM_THREAD_NBR) = 1;
  pastix_llt_upper.iparm(IPARM_THREAD_NBR) = 1;
  pastix_ldlt_upper.iparm(IPARM_THREAD_NBR) = 1;
  pastix_lu.iparm(IPARM_THREAD_NBR) = 1;

  check_sparse_spd_solving(pastix_llt_lower);
  check_sparse_spd_solving(pastix_ldlt_lower);
  check_sparse_spd_solving(pastix_llt_upper);
  check_sparse_spd_solving(pastix_ldlt_upper);
  check_sparse_square_solving(pastix_lu);

  // Some compilation check:
  pastix_llt_lower.iparm();
  pastix_llt_lower.dparm();
  pastix_ldlt_lower.iparm();
  pastix_ldlt_lower.dparm();
  pastix_lu.iparm();
  pastix_lu.dparm();
}

EIGEN_DECLARE_TEST(pastix_support) {
  CALL_SUBTEST_1(test_pastix_T<float>());
  CALL_SUBTEST_2(test_pastix_T<double>());
  CALL_SUBTEST_3((test_pastix_T<std::complex<float>>()));
  CALL_SUBTEST_4(test_pastix_T<std::complex<double>>());
  CALL_SUBTEST_1(test_pastix_special_cases<float>());
  CALL_SUBTEST_2(test_pastix_special_cases<double>());
  CALL_SUBTEST_3(test_pastix_special_cases<std::complex<float>>());
  CALL_SUBTEST_4(test_pastix_special_cases<std::complex<double>>());
}
