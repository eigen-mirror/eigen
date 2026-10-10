// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"

#include <Eigen/SuperLUSupport>

// The last pivot of this rank-2 matrix is exactly zero; determinant() used to skip it and return 1.
void test_superlu_singular_determinant() {
  SparseMatrix<double> A(3, 3);
  A.insert(0, 0) = 1;
  A.insert(0, 1) = 2;
  A.insert(1, 1) = 1;
  A.insert(1, 2) = 1;
  A.insert(2, 0) = 1;
  A.insert(2, 1) = 3;
  A.insert(2, 2) = 1;
  A.makeCompressed();
  SuperLU<SparseMatrix<double> > lu(A);
  VERIFY_IS_EQUAL(lu.determinant(), 0.0);
}

EIGEN_DECLARE_TEST(superlu_support) {
  SuperLU<SparseMatrix<double> > superlu_double_colmajor;
  SuperLU<SparseMatrix<std::complex<double> > > superlu_cplxdouble_colmajor;
  CALL_SUBTEST_1(check_sparse_square_solving(superlu_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_solving(superlu_cplxdouble_colmajor));
  CALL_SUBTEST_1(check_sparse_square_determinant(superlu_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_determinant(superlu_cplxdouble_colmajor));
  CALL_SUBTEST_1(test_superlu_singular_determinant());
}
