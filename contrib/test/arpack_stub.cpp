// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Tests the state handling of ArpackGeneralizedSelfAdjointEigenSolver without the ARPACK library: the two ARPACK
// entry points the double-precision solver calls are replaced by stubs that return a prescribed status.

#include "sparse.h"

#include <Eigen/ArpackSupport>

namespace {
int g_saupd_info = 0;  // status returned by the stub dsaupd_
}

extern "C" {
void dsaupd_(int* ido, char*, int*, char*, int*, double*, double*, int*, double*, int*, int* iparam, int*, double*,
             double*, int*, int* info) {
  *ido = 99;
  *info = g_saupd_info;
  iparam[2] = 7;
  iparam[4] = 3;
}

void dseupd_(int*, char*, int*, double* d, double* z, int* ldz, double*, char*, int*, char*, int* nev, double*, double*,
             int*, double*, int*, int*, int*, double*, double*, int*, int* ierr) {
  for (int i = 0; i < *nev; ++i) {
    d[i] = i + 1;
    for (int j = 0; j < *ldz; ++j) z[i * (*ldz) + j] = (i == j) ? 1 : 0;
  }
  *ierr = 0;
}
}

using ArpackSolver = ArpackGeneralizedSelfAdjointEigenSolver<SparseMatrix<double> >;

static SparseMatrix<double> diagonal_matrix(Index n, double sign) {
  SparseMatrix<double> A(n, n);
  for (Index i = 0; i < n; ++i) A.insert(i, i) = sign * double(i + 1);
  return A;
}

void test_arpack_info_after_failure() {
  const Index n = 30, nev = 3;
  const SparseMatrix<double> A = diagonal_matrix(n, 1.0);

  // info() reports every failure code of compute(), also when assertions are enabled.
  ArpackSolver solver;
  VERIFY(solver.info() != Success);
  const struct {
    int arpackInfo;
    ComputationInfo expected;
  } failures[] = {{1, NoConvergence}, {3, NumericalIssue}, {-1, InvalidInput}};
  for (const auto& failure : failures) {
    g_saupd_info = failure.arpackInfo;
    solver.compute(A, nev, "LM");
    VERIFY_IS_EQUAL(solver.info(), failure.expected);
    VERIFY_RAISES_ASSERT(solver.eigenvalues());
    VERIFY_RAISES_ASSERT(solver.eigenvectors());

    ArpackSolver constructed(A, nev, "LM");
    VERIFY_IS_EQUAL(constructed.info(), failure.expected);
  }

  // Failure of the factorization used by the shift-and-invert mode.
  g_saupd_info = 0;
  solver.compute(diagonal_matrix(n, -1.0), nev, "SM");
  VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
  VERIFY_RAISES_ASSERT(solver.eigenvalues());
}

void test_arpack_state_across_computations() {
  const Index n = 30, nev = 3;
  const SparseMatrix<double> A = diagonal_matrix(n, 1.0);

  ArpackSolver solver;
  g_saupd_info = 0;
  solver.compute(A, nev, "LM", ComputeEigenvectors);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(solver.eigenvalues().size(), nev);
  VERIFY_IS_EQUAL(solver.eigenvectors().cols(), nev);
  VERIFY_IS_EQUAL(solver.getNbrConvergedEigenValues(), size_t(3));
  VERIFY_IS_EQUAL(solver.getNbrIterations(), size_t(7));

  // A later failure invalidates the previous result.
  g_saupd_info = 1;
  solver.compute(A, nev, "LM", ComputeEigenvectors);
  VERIFY_IS_EQUAL(solver.info(), NoConvergence);
  VERIFY_RAISES_ASSERT(solver.eigenvalues());
  VERIFY_IS_EQUAL(solver.getNbrConvergedEigenValues(), size_t(0));
  VERIFY_IS_EQUAL(solver.getNbrIterations(), size_t(0));

  // So does a failed factorization, which returns before ARPACK runs.
  g_saupd_info = 0;
  solver.compute(A, nev, "LM", ComputeEigenvectors);
  VERIFY_IS_EQUAL(solver.getNbrConvergedEigenValues(), size_t(3));
  solver.compute(diagonal_matrix(n, -1.0), nev, "SM");
  VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
  VERIFY_IS_EQUAL(solver.getNbrConvergedEigenValues(), size_t(0));
  VERIFY_IS_EQUAL(solver.getNbrIterations(), size_t(0));

  // So does a later computation without eigenvectors.
  g_saupd_info = 0;
  solver.compute(A, nev, "LM", ComputeEigenvectors);
  solver.compute(A, nev, "LM", EigenvaluesOnly);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(solver.eigenvalues().size(), nev);
  VERIFY_RAISES_ASSERT(solver.eigenvectors());
}

EIGEN_DECLARE_TEST(arpack_stub) {
  CALL_SUBTEST_1(test_arpack_info_after_failure());
  CALL_SUBTEST_1(test_arpack_state_across_computations());
}
