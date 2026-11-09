// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
#include "common.h"

namespace {

// The uplo triangle of the n-by-n C += alpha op(A) op(B), with op(A) n-by-k and opa, opb in {NOTR, TR, ADJ}.
void triangular_product(int uplo, int opa, int opb, Eigen::Index n, Eigen::Index k, const Scalar &alpha,
                        const Scalar *a, Eigen::Index lda, const Scalar *b, Eigen::Index ldb, Scalar *c,
                        Eigen::Index ldc) {
  using Eigen::ColMajor;
  using Eigen::DenseIndex;
  using Eigen::Dynamic;
  using Eigen::Lower;
  using Eigen::RowMajor;
  using Eigen::Upper;
  using functype = void (*)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *,
                            DenseIndex, DenseIndex, const Scalar &, Eigen::internal::level3_blocking<Scalar, Scalar> &);
  static const functype func[24] = {
      // Upper-triangular result.
      // array index: NOTR  | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, ColMajor, false, Scalar, ColMajor,
                                                                 false, ColMajor, 1, Upper>::run),
      // array index: TR    | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, false, Scalar, ColMajor,
                                                                 false, ColMajor, 1, Upper>::run),
      // array index: ADJ   | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, ColMajor,
                                                                 false, ColMajor, 1, Upper>::run),
      nullptr,
      // array index: NOTR  | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, ColMajor, false, Scalar, RowMajor,
                                                                 false, ColMajor, 1, Upper>::run),
      // array index: TR    | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, false, Scalar, RowMajor,
                                                                 false, ColMajor, 1, Upper>::run),
      // array index: ADJ   | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, RowMajor,
                                                                 false, ColMajor, 1, Upper>::run),
      nullptr,
      // array index: NOTR  | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, ColMajor, false, Scalar, RowMajor,
                                                                 Conj, ColMajor, 1, Upper>::run),
      // array index: TR    | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, false, Scalar, RowMajor,
                                                                 Conj, ColMajor, 1, Upper>::run),
      // array index: ADJ   | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, RowMajor,
                                                                 Conj, ColMajor, 1, Upper>::run),
      nullptr,

      // Lower-triangular result.
      // array index: NOTR  | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, ColMajor, false, Scalar, ColMajor,
                                                                 false, ColMajor, 1, Lower>::run),
      // array index: TR    | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, false, Scalar, ColMajor,
                                                                 false, ColMajor, 1, Lower>::run),
      // array index: ADJ   | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, ColMajor,
                                                                 false, ColMajor, 1, Lower>::run),
      nullptr,
      // array index: NOTR  | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, ColMajor, false, Scalar, RowMajor,
                                                                 false, ColMajor, 1, Lower>::run),
      // array index: TR    | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, false, Scalar, RowMajor,
                                                                 false, ColMajor, 1, Lower>::run),
      // array index: ADJ   | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, RowMajor,
                                                                 false, ColMajor, 1, Lower>::run),
      nullptr,
      // array index: NOTR  | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, ColMajor, false, Scalar, RowMajor,
                                                                 Conj, ColMajor, 1, Lower>::run),
      // array index: TR    | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, false, Scalar, RowMajor,
                                                                 Conj, ColMajor, 1, Lower>::run),
      // array index: ADJ   | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_triangular_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, RowMajor,
                                                                 Conj, ColMajor, 1, Lower>::run),
      nullptr,
  };

  Eigen::internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic> blocking(n, n, k, 1, false);
  func[opa + 4 * opb + 12 * uplo](n, k, a, lda, b, ldb, c, 1, ldc, alpha, blocking);
}

// The uplo triangle of the n-by-n C := beta C, column by column; beta == 0 overwrites. A real beta (herk, her2k)
// scales the real and imaginary parts separately, so an infinite part does not make the other one NaN.
template <typename Beta>
void scale_triangle(int uplo, Eigen::Index n, Scalar *c, Eigen::Index ldc, const Beta &beta) {
  if (beta == Beta(1)) return;
  for (Eigen::Index j = 0; j < n; ++j) {
    auto col = uplo == UP ? make_vector(c + j * ldc, j + 1) : make_vector(c + j * ldc + j, n - j);
    if (beta == Beta(0))
      col.setZero();
    else
      col *= beta;
  }
}

// The m-by-n C := beta C; beta == 0 overwrites.
void scale_matrix(Eigen::Index m, Eigen::Index n, Scalar *c, Eigen::Index ldc, const Scalar &beta) {
  if (beta == Scalar(1)) return;
  if (beta == Scalar(0))
    matrix(c, m, n, ldc).setZero();
  else
    matrix(c, m, n, ldc) *= beta;
}

// C += alpha A B (side == LEFT) or alpha B A (side == RIGHT) for the m-by-n B and C and the selfadjoint A stored in
// its uplo triangle.
void selfadjoint_product(int side, int uplo, Eigen::Index m, Eigen::Index n, const Scalar &alpha, const Scalar *a,
                         Eigen::Index lda, const Scalar *b, Eigen::Index ldb, Scalar *c, Eigen::Index ldc) {
  using Eigen::ColMajor;
  using Eigen::DenseIndex;
  using Eigen::Dynamic;
  using Eigen::RowMajor;
  using Eigen::internal::product_selfadjoint_matrix;
  Eigen::internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic> blocking(
      m, n, side == LEFT ? m : n, 1, false);

  if (side == LEFT) {
    if (uplo == UP)
      product_selfadjoint_matrix<Scalar, DenseIndex, RowMajor, true, Conj, ColMajor, false, false, ColMajor, 1>::run(
          m, n, a, lda, b, ldb, c, 1, ldc, alpha, blocking);
    else
      product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor, true, false, ColMajor, false, false, ColMajor, 1>::run(
          m, n, a, lda, b, ldb, c, 1, ldc, alpha, blocking);
  } else {
    if (uplo == UP)
      product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor, false, false, RowMajor, true, Conj, ColMajor, 1>::run(
          m, n, b, ldb, a, lda, c, 1, ldc, alpha, blocking);
    else
      product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor, false, false, ColMajor, true, false, ColMajor, 1>::run(
          m, n, b, ldb, a, lda, c, 1, ldc, alpha, blocking);
  }
}

}  // namespace

EIGEN_BLAS_FUNC(gemm)
(const char *opa, const char *opb, const EIGEN_BLAS_INT *m, const EIGEN_BLAS_INT *n, const EIGEN_BLAS_INT *k,
 const RealScalar *palpha, const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pb,
 const EIGEN_BLAS_INT *ldb, const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  using Eigen::ColMajor;
  using Eigen::DenseIndex;
  using Eigen::Dynamic;
  using Eigen::RowMajor;
  using functype =
      void (*)(DenseIndex, DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *,
               DenseIndex, DenseIndex, Scalar, Eigen::internal::level3_blocking<Scalar, Scalar> &,
               Eigen::internal::GemmParallelInfo<DenseIndex> *);
  static const functype func[12] = {
      // array index: NOTR  | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, ColMajor, false, Scalar, ColMajor, false,
                                                      ColMajor, 1>::run),
      // array index: TR    | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, RowMajor, false, Scalar, ColMajor, false,
                                                      ColMajor, 1>::run),
      // array index: ADJ   | (NOTR << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, ColMajor, false,
                                                      ColMajor, 1>::run),
      0,
      // array index: NOTR  | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, ColMajor, false, Scalar, RowMajor, false,
                                                      ColMajor, 1>::run),
      // array index: TR    | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, RowMajor, false, Scalar, RowMajor, false,
                                                      ColMajor, 1>::run),
      // array index: ADJ   | (TR   << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, RowMajor, false,
                                                      ColMajor, 1>::run),
      0,
      // array index: NOTR  | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, ColMajor, false, Scalar, RowMajor, Conj,
                                                      ColMajor, 1>::run),
      // array index: TR    | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, RowMajor, false, Scalar, RowMajor, Conj,
                                                      ColMajor, 1>::run),
      // array index: ADJ   | (ADJ  << 2)
      (Eigen::internal::general_matrix_matrix_product<DenseIndex, Scalar, RowMajor, Conj, Scalar, RowMajor, Conj,
                                                      ColMajor, 1>::run),
      0};

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *b = reinterpret_cast<const Scalar *>(pb);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  EIGEN_BLAS_INT info = 0;
  if (OP(*opa) == INVALID)
    info = 1;
  else if (OP(*opb) == INVALID)
    info = 2;
  else if (*m < 0)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*k < 0)
    info = 5;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (OP(*opa) == NOTR) ? *m : *k))
    info = 8;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, (OP(*opb) == NOTR) ? *k : *n))
    info = 10;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 13;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GEMM ", &info, kBlasNameLength);

  if (*m == 0 || *n == 0) return;

  scale_matrix(*m, *n, c, *ldc, beta);
  if (*k == 0) return;

  Eigen::internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic> blocking(*m, *n, *k, 1,
                                                                                                     true);

  int code = OP(*opa) | (OP(*opb) << 2);
  func[code](*m, *n, *k, a, *lda, b, *ldb, c, 1, *ldc, alpha, blocking, 0);
}

EIGEN_BLAS_FUNC(gemmtr)
(const char *uplo, const char *opa, const char *opb, const EIGEN_BLAS_INT *n, const EIGEN_BLAS_INT *k,
 const RealScalar *palpha, const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pb,
 const EIGEN_BLAS_INT *ldb, const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *b = reinterpret_cast<const Scalar *>(pb);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*opa) == INVALID)
    info = 2;
  else if (OP(*opb) == INVALID)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*k < 0)
    info = 5;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (OP(*opa) == NOTR) ? *n : *k))
    info = 8;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, (OP(*opb) == NOTR) ? *k : *n))
    info = 10;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 13;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "GEMMTR", &info, kBlasNameLength + 1);

  if (*n == 0) return;

  scale_triangle(UPLO(*uplo), *n, c, *ldc, beta);
  if (*k == 0 || alpha == Scalar(0)) return;

  triangular_product(UPLO(*uplo), OP(*opa), OP(*opb), *n, *k, alpha, a, *lda, b, *ldb, c, *ldc);
}

EIGEN_BLAS_FUNC(trsm)
(const char *side, const char *uplo, const char *opa, const char *diag, const EIGEN_BLAS_INT *m,
 const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *pa, const EIGEN_BLAS_INT *lda, RealScalar *pb,
 const EIGEN_BLAS_INT *ldb) {
  using Eigen::ColMajor;
  using Eigen::DenseIndex;
  using Eigen::Dynamic;
  using Eigen::Lower;
  using Eigen::OnTheLeft;
  using Eigen::OnTheRight;
  using Eigen::RowMajor;
  using Eigen::UnitDiag;
  using Eigen::Upper;
  using functype = void (*)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, DenseIndex,
                            Eigen::internal::level3_blocking<Scalar, Scalar> &);
  static const functype func[32] = {
      // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Upper | 0, false, ColMajor, ColMajor,
                                                1>::run),
      // array index: TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Lower | 0, false, RowMajor, ColMajor,
                                                1>::run),
      // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Lower | 0, Conj, RowMajor, ColMajor,
                                                1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Upper | 0, false, ColMajor, ColMajor,
                                                1>::run),
      // array index: TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Lower | 0, false, RowMajor, ColMajor,
                                                1>::run),
      // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Lower | 0, Conj, RowMajor, ColMajor,
                                                1>::run),
      0,
      // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Lower | 0, false, ColMajor, ColMajor,
                                                1>::run),
      // array index: TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Upper | 0, false, RowMajor, ColMajor,
                                                1>::run),
      // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Upper | 0, Conj, RowMajor, ColMajor,
                                                1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Lower | 0, false, ColMajor, ColMajor,
                                                1>::run),
      // array index: TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Upper | 0, false, RowMajor, ColMajor,
                                                1>::run),
      // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Upper | 0, Conj, RowMajor, ColMajor,
                                                1>::run),
      0,
      // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Upper | UnitDiag, false, ColMajor,
                                                ColMajor, 1>::run),
      // array index: TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Lower | UnitDiag, false, RowMajor,
                                                ColMajor, 1>::run),
      // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Lower | UnitDiag, Conj, RowMajor,
                                                ColMajor, 1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Upper | UnitDiag, false, ColMajor,
                                                ColMajor, 1>::run),
      // array index: TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Lower | UnitDiag, false, RowMajor,
                                                ColMajor, 1>::run),
      // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Lower | UnitDiag, Conj, RowMajor,
                                                ColMajor, 1>::run),
      0,
      // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Lower | UnitDiag, false, ColMajor,
                                                ColMajor, 1>::run),
      // array index: TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Upper | UnitDiag, false, RowMajor,
                                                ColMajor, 1>::run),
      // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheLeft, Upper | UnitDiag, Conj, RowMajor,
                                                ColMajor, 1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Lower | UnitDiag, false, ColMajor,
                                                ColMajor, 1>::run),
      // array index: TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Upper | UnitDiag, false, RowMajor,
                                                ColMajor, 1>::run),
      // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::triangular_solve_matrix<Scalar, DenseIndex, OnTheRight, Upper | UnitDiag, Conj, RowMajor,
                                                ColMajor, 1>::run),
      0};

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  Scalar *b = reinterpret_cast<Scalar *>(pb);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (SIDE(*side) == INVALID)
    info = 1;
  else if (UPLO(*uplo) == INVALID)
    info = 2;
  else if (OP(*opa) == INVALID)
    info = 3;
  else if (DIAG(*diag) == INVALID)
    info = 4;
  else if (*m < 0)
    info = 5;
  else if (*n < 0)
    info = 6;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (SIDE(*side) == LEFT) ? *m : *n))
    info = 9;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 11;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TRSM ", &info, kBlasNameLength);

  if (*m == 0 || *n == 0) return;
  // When alpha is zero, A is not referenced and B need not be set on entry.
  if (alpha == Scalar(0)) {
    matrix(b, *m, *n, *ldb).setZero();
    return;
  }

  // B := alpha B before the solve, as the reference does: solving first can overflow where the result does not.
  if (alpha != Scalar(1)) matrix(b, *m, *n, *ldb) *= alpha;

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);

  if (SIDE(*side) == LEFT) {
    Eigen::internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic, 4> blocking(*m, *n, *m, 1,
                                                                                                          false);
    func[code](*m, *n, a, *lda, b, 1, *ldb, blocking);
  } else {
    Eigen::internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic, 4> blocking(*m, *n, *n, 1,
                                                                                                          false);
    func[code](*n, *m, a, *lda, b, 1, *ldb, blocking);
  }
}

// b = alpha*op(a)*b  for side = 'L'or'l'
// b = alpha*b*op(a)  for side = 'R'or'r'
EIGEN_BLAS_FUNC(trmm)
(const char *side, const char *uplo, const char *opa, const char *diag, const EIGEN_BLAS_INT *m,
 const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *pa, const EIGEN_BLAS_INT *lda, RealScalar *pb,
 const EIGEN_BLAS_INT *ldb) {
  using Eigen::ColMajor;
  using Eigen::DenseIndex;
  using Eigen::Dynamic;
  using Eigen::Lower;
  using Eigen::RowMajor;
  using Eigen::UnitDiag;
  using Eigen::Upper;
  using functype =
      void (*)(DenseIndex, DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *,
               DenseIndex, DenseIndex, const Scalar &, Eigen::internal::level3_blocking<Scalar, Scalar> &);
  static const functype func[32] = {
      // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | 0, true, ColMajor, false, ColMajor,
                                                         false, ColMajor, 1>::run),
      // array index: TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | 0, true, RowMajor, false, ColMajor,
                                                         false, ColMajor, 1>::run),
      // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | 0, true, RowMajor, Conj, ColMajor,
                                                         false, ColMajor, 1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | 0, false, ColMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | 0, false, ColMajor, false,
                                                         RowMajor, false, ColMajor, 1>::run),
      // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | 0, false, ColMajor, false,
                                                         RowMajor, Conj, ColMajor, 1>::run),
      0,
      // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | 0, true, ColMajor, false, ColMajor,
                                                         false, ColMajor, 1>::run),
      // array index: TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | 0, true, RowMajor, false, ColMajor,
                                                         false, ColMajor, 1>::run),
      // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | 0, true, RowMajor, Conj, ColMajor,
                                                         false, ColMajor, 1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | 0, false, ColMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | 0, false, ColMajor, false,
                                                         RowMajor, false, ColMajor, 1>::run),
      // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | 0, false, ColMajor, false,
                                                         RowMajor, Conj, ColMajor, 1>::run),
      0,
      // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | UnitDiag, true, ColMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | UnitDiag, true, RowMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | UnitDiag, true, RowMajor, Conj,
                                                         ColMajor, false, ColMajor, 1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | UnitDiag, false, ColMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | UnitDiag, false, ColMajor, false,
                                                         RowMajor, false, ColMajor, 1>::run),
      // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | UnitDiag, false, ColMajor, false,
                                                         RowMajor, Conj, ColMajor, 1>::run),
      0,
      // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | UnitDiag, true, ColMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | UnitDiag, true, RowMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | UnitDiag, true, RowMajor, Conj,
                                                         ColMajor, false, ColMajor, 1>::run),
      0,
      // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Lower | UnitDiag, false, ColMajor, false,
                                                         ColMajor, false, ColMajor, 1>::run),
      // array index: TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | UnitDiag, false, ColMajor, false,
                                                         RowMajor, false, ColMajor, 1>::run),
      // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
      (Eigen::internal::product_triangular_matrix_matrix<Scalar, DenseIndex, Upper | UnitDiag, false, ColMajor, false,
                                                         RowMajor, Conj, ColMajor, 1>::run),
      0};

  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  Scalar *b = reinterpret_cast<Scalar *>(pb);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);

  EIGEN_BLAS_INT info = 0;
  if (SIDE(*side) == INVALID)
    info = 1;
  else if (UPLO(*uplo) == INVALID)
    info = 2;
  else if (OP(*opa) == INVALID)
    info = 3;
  else if (DIAG(*diag) == INVALID)
    info = 4;
  else if (*m < 0)
    info = 5;
  else if (*n < 0)
    info = 6;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (SIDE(*side) == LEFT) ? *m : *n))
    info = 9;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 11;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "TRMM ", &info, kBlasNameLength);

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);

  if (*m == 0 || *n == 0) return;
  // When alpha is zero, A is not referenced and B need not be set on entry.
  if (alpha == Scalar(0)) {
    matrix(b, *m, *n, *ldb).setZero();
    return;
  }

  // FIXME: find a way to avoid this copy
  Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor> tmp = matrix(b, *m, *n, *ldb);
  matrix(b, *m, *n, *ldb).setZero();

  if (SIDE(*side) == LEFT) {
    Eigen::internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic, 4> blocking(*m, *n, *m, 1,
                                                                                                          false);
    func[code](*m, *n, *m, a, *lda, tmp.data(), tmp.outerStride(), b, 1, *ldb, alpha, blocking);
  } else {
    Eigen::internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic, 4> blocking(*m, *n, *n, 1,
                                                                                                          false);
    func[code](*m, *n, *n, tmp.data(), tmp.outerStride(), a, *lda, b, 1, *ldb, alpha, blocking);
  }
}

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
EIGEN_BLAS_FUNC(symm)
(const char *side, const char *uplo, const EIGEN_BLAS_INT *m, const EIGEN_BLAS_INT *n, const RealScalar *palpha,
 const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pb, const EIGEN_BLAS_INT *ldb,
 const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *b = reinterpret_cast<const Scalar *>(pb);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  EIGEN_BLAS_INT info = 0;
  if (SIDE(*side) == INVALID)
    info = 1;
  else if (UPLO(*uplo) == INVALID)
    info = 2;
  else if (*m < 0)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (SIDE(*side) == LEFT) ? *m : *n))
    info = 7;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 9;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 12;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYMM ", &info, kBlasNameLength);

  scale_matrix(*m, *n, c, *ldc, beta);
  if (*m == 0 || *n == 0 || alpha == Scalar(0)) return;

#if ISCOMPLEX
  // Eigen's selfadjoint kernels assume a Hermitian A, so a complex symmetric one is expanded first.
  const Eigen::Index size = (SIDE(*side) == LEFT) ? (*m) : (*n);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matA(size, size);
  if (UPLO(*uplo) == UP) {
    matA.triangularView<Eigen::Upper>() = matrix(a, size, size, *lda);
    matA.triangularView<Eigen::StrictlyLower>() = matrix(a, size, size, *lda).transpose();
  } else {
    matA.triangularView<Eigen::Lower>() = matrix(a, size, size, *lda);
    matA.triangularView<Eigen::StrictlyUpper>() = matrix(a, size, size, *lda).transpose();
  }
  if (SIDE(*side) == LEFT)
    matrix(c, *m, *n, *ldc).noalias() += alpha * matA * matrix(b, *m, *n, *ldb);
  else
    matrix(c, *m, *n, *ldc).noalias() += alpha * matrix(b, *m, *n, *ldb) * matA;
#else
  selfadjoint_product(SIDE(*side), UPLO(*uplo), *m, *n, alpha, a, *lda, b, *ldb, c, *ldc);
#endif
}

// c = alpha*a*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*a + beta*c  for op = 'T'or't','C'or'c'
EIGEN_BLAS_FUNC(syrk)
(const char *uplo, const char *op, const EIGEN_BLAS_INT *n, const EIGEN_BLAS_INT *k, const RealScalar *palpha,
 const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*op) == INVALID || (ISCOMPLEX && OP(*op) == ADJ))
    info = 2;
  else if (*n < 0)
    info = 3;
  else if (*k < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (OP(*op) == NOTR) ? *n : *k))
    info = 7;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 10;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYRK ", &info, kBlasNameLength);

  scale_triangle(UPLO(*uplo), *n, c, *ldc, beta);
  if (*n == 0 || *k == 0 || alpha == Scalar(0)) return;

  // C += alpha A A^T or alpha A^T A, without conjugation also for complex A (TR and ADJ coincide for real A).
  const int opa = OP(*op) == NOTR ? NOTR : TR;
  triangular_product(UPLO(*uplo), opa, opa == NOTR ? TR : NOTR, *n, *k, alpha, a, *lda, a, *lda, c, *ldc);
}

// c = alpha*a*b' + alpha*b*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*b + alpha*b'*a + beta*c  for op = 'T'or't'
EIGEN_BLAS_FUNC(syr2k)
(const char *uplo, const char *op, const EIGEN_BLAS_INT *n, const EIGEN_BLAS_INT *k, const RealScalar *palpha,
 const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pb, const EIGEN_BLAS_INT *ldb,
 const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *b = reinterpret_cast<const Scalar *>(pb);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if (OP(*op) == INVALID || (ISCOMPLEX && OP(*op) == ADJ))
    info = 2;
  else if (*n < 0)
    info = 3;
  else if (*k < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (OP(*op) == NOTR) ? *n : *k))
    info = 7;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, (OP(*op) == NOTR) ? *n : *k))
    info = 9;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 12;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "SYR2K", &info, kBlasNameLength);

  scale_triangle(UPLO(*uplo), *n, c, *ldc, beta);
  if (*n == 0 || *k == 0 || alpha == Scalar(0)) return;

  // C += alpha A B^T + alpha B A^T, or alpha A^T B + alpha B^T A for TR and ADJ, as two triangular products.
  const int opa = OP(*op) == NOTR ? NOTR : TR, opb = opa == NOTR ? TR : NOTR;
  triangular_product(UPLO(*uplo), opa, opb, *n, *k, alpha, a, *lda, b, *ldb, c, *ldc);
  triangular_product(UPLO(*uplo), opa, opb, *n, *k, alpha, b, *ldb, a, *lda, c, *ldc);
}

#if ISCOMPLEX

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
EIGEN_BLAS_FUNC(hemm)
(const char *side, const char *uplo, const EIGEN_BLAS_INT *m, const EIGEN_BLAS_INT *n, const RealScalar *palpha,
 const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pb, const EIGEN_BLAS_INT *ldb,
 const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *b = reinterpret_cast<const Scalar *>(pb);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);

  EIGEN_BLAS_INT info = 0;
  if (SIDE(*side) == INVALID)
    info = 1;
  else if (UPLO(*uplo) == INVALID)
    info = 2;
  else if (*m < 0)
    info = 3;
  else if (*n < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (SIDE(*side) == LEFT) ? *m : *n))
    info = 7;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 9;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *m))
    info = 12;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HEMM ", &info, kBlasNameLength);

  scale_matrix(*m, *n, c, *ldc, beta);
  if (*m == 0 || *n == 0 || alpha == Scalar(0)) return;

  selfadjoint_product(SIDE(*side), UPLO(*uplo), *m, *n, alpha, a, *lda, b, *ldb, c, *ldc);
}

// c = alpha*a*conj(a') + beta*c  for op = 'N'or'n'
// c = alpha*conj(a')*a + beta*c  for op  = 'C'or'c'
EIGEN_BLAS_FUNC(herk)
(const char *uplo, const char *op, const EIGEN_BLAS_INT *n, const EIGEN_BLAS_INT *k, const RealScalar *palpha,
 const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  RealScalar alpha = *palpha;
  RealScalar beta = *pbeta;

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if ((OP(*op) == INVALID) || (OP(*op) == TR))
    info = 2;
  else if (*n < 0)
    info = 3;
  else if (*k < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (OP(*op) == NOTR) ? *n : *k))
    info = 7;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 10;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HERK ", &info, kBlasNameLength);

  const int uplo_code = UPLO(*uplo);
  // The diagonal of a Hermitian C is real on exit whenever C is referenced.
  scale_triangle(uplo_code, *n, c, *ldc, beta);
  if (beta != RealScalar(1)) matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
  if (*n == 0 || *k == 0 || alpha == RealScalar(0)) return;

  // C += alpha A A^H, or alpha A^H A.
  const int opa = OP(*op), opb = opa == NOTR ? ADJ : NOTR;
  triangular_product(uplo_code, opa, opb, *n, *k, Scalar(alpha), a, *lda, a, *lda, c, *ldc);
  matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
}

// c = alpha*a*conj(b') + conj(alpha)*b*conj(a') + beta*c,  for op = 'N'or'n'
// c = alpha*conj(a')*b + conj(alpha)*conj(b')*a + beta*c,  for op = 'C'or'c'
EIGEN_BLAS_FUNC(her2k)
(const char *uplo, const char *op, const EIGEN_BLAS_INT *n, const EIGEN_BLAS_INT *k, const RealScalar *palpha,
 const RealScalar *pa, const EIGEN_BLAS_INT *lda, const RealScalar *pb, const EIGEN_BLAS_INT *ldb,
 const RealScalar *pbeta, RealScalar *pc, const EIGEN_BLAS_INT *ldc) {
  const Scalar *a = reinterpret_cast<const Scalar *>(pa);
  const Scalar *b = reinterpret_cast<const Scalar *>(pb);
  Scalar *c = reinterpret_cast<Scalar *>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  RealScalar beta = *pbeta;

  EIGEN_BLAS_INT info = 0;
  if (UPLO(*uplo) == INVALID)
    info = 1;
  else if ((OP(*op) == INVALID) || (OP(*op) == TR))
    info = 2;
  else if (*n < 0)
    info = 3;
  else if (*k < 0)
    info = 4;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, (OP(*op) == NOTR) ? *n : *k))
    info = 7;
  else if (*ldb < std::max<EIGEN_BLAS_INT>(1, (OP(*op) == NOTR) ? *n : *k))
    info = 9;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *n))
    info = 12;
  if (info) return xerbla_(SCALAR_SUFFIX_UP "HER2K", &info, kBlasNameLength);

  const int uplo_code = UPLO(*uplo);
  scale_triangle(uplo_code, *n, c, *ldc, beta);
  if (beta != RealScalar(1)) matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
  if (*n == 0 || *k == 0 || alpha == Scalar(0)) return;

  // C += alpha A B^H + conj(alpha) B A^H, or alpha A^H B + conj(alpha) B^H A. The two products round differently,
  // so the diagonal is made real again afterwards.
  const int opa = OP(*op), opb = opa == NOTR ? ADJ : NOTR;
  triangular_product(uplo_code, opa, opb, *n, *k, alpha, a, *lda, b, *ldb, c, *ldc);
  triangular_product(uplo_code, opa, opb, *n, *k, Eigen::numext::conj(alpha), b, *ldb, a, *lda, c, *ldc);
  matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
}

#endif
