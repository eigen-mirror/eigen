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

#include <Eigen/UmfPackSupport>

template <typename T1, typename T2>
void test_umfpack_support_T() {
  UmfPackLU<SparseMatrix<T1, ColMajor, T2> > umfpack_colmajor;
  UmfPackLU<SparseMatrix<T1, RowMajor, T2> > umfpack_rowmajor;

  check_sparse_square_solving(umfpack_colmajor);
  check_sparse_square_solving(umfpack_rowmajor);

  check_sparse_square_determinant(umfpack_colmajor);
  check_sparse_square_determinant(umfpack_rowmajor);
}

// UMFPACK computes (L * U)(i, j) = s(P(i)) * A(P(i), Q(j)), with L unit lower triangular and U upper triangular.
template <typename Scalar, typename StorageIndex>
void check_umfpack_factors(int scaling) {
  using MatrixType = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = internal::random<Index>(5, 30);
  MatrixType A(n, n);
  DenseMatrix refA = DenseMatrix::Zero(n, n);
  initSparse<Scalar>(0.2, refA, A, ForceNonZeroDiag);
  A.makeCompressed();

  UmfPackLU<MatrixType> lu;
  lu.umfpackControl()[UMFPACK_SCALE] = scaling;
  lu.compute(A);
  VERIFY_IS_EQUAL(lu.info(), Success);

  const DenseMatrix L = lu.matrixL();
  const DenseMatrix U = lu.matrixU();
  VERIFY_IS_EQUAL(L.rows(), n);
  VERIFY_IS_EQUAL(L.cols(), n);
  VERIFY_IS_EQUAL(U.rows(), n);
  VERIFY_IS_EQUAL(U.cols(), n);
  VERIFY(L.isApprox(DenseMatrix(L.template triangularView<UnitLower>())));
  VERIFY(L.diagonal().isApprox(Matrix<Scalar, Dynamic, 1>::Ones(n)));
  VERIFY(U.isApprox(DenseMatrix(U.template triangularView<Upper>())));

  const auto& P = lu.permutationP();
  const auto& Q = lu.permutationQ();
  VERIFY_IS_EQUAL(P.size(), n);
  VERIFY_IS_EQUAL(Q.size(), n);
  Matrix<bool, Dynamic, 1> seenRow = Matrix<bool, Dynamic, 1>::Constant(n, false);
  Matrix<bool, Dynamic, 1> seenCol = Matrix<bool, Dynamic, 1>::Constant(n, false);
  for (Index k = 0; k < n; ++k) {
    VERIFY(P(k) >= 0 && P(k) < n && Q(k) >= 0 && Q(k) < n);
    seenRow(P(k)) = true;
    seenCol(Q(k)) = true;
  }
  VERIFY(seenRow.all() && seenCol.all());

  const auto& s = lu.rowScaling();
  VERIFY_IS_EQUAL(s.size(), n);
  if (scaling == UMFPACK_SCALE_NONE) VERIFY(s.isOnes());

  DenseMatrix scaledPermutedA(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) scaledPermutedA(i, j) = s(P(i)) * refA(P(i), Q(j));
  VERIFY_IS_APPROX(L * U, scaledPermutedA);
}

template <typename Scalar, typename StorageIndex>
void check_umfpack_factors() {
  check_umfpack_factors<Scalar, StorageIndex>(UMFPACK_SCALE_NONE);
  check_umfpack_factors<Scalar, StorageIndex>(UMFPACK_SCALE_SUM);
  check_umfpack_factors<Scalar, StorageIndex>(UMFPACK_SCALE_MAX);
}

EIGEN_DECLARE_TEST(umfpack_support) {
  CALL_SUBTEST_1((test_umfpack_support_T<double, int>()));
  CALL_SUBTEST_2((test_umfpack_support_T<std::complex<double>, int>()));
  CALL_SUBTEST_3((test_umfpack_support_T<double, long>()));
  CALL_SUBTEST_4((test_umfpack_support_T<std::complex<double>, long>()));
  CALL_SUBTEST_5((check_umfpack_factors<double, int>()));
  CALL_SUBTEST_5((check_umfpack_factors<std::complex<double>, int>()));
  CALL_SUBTEST_6((check_umfpack_factors<double, long>()));
  CALL_SUBTEST_6((check_umfpack_factors<std::complex<double>, long>()));
}
