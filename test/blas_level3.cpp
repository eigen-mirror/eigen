// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Runs the level-3 BLAS adapters (?gemm, ?symm and ?hemm, ?trmm) against the BLAS library the tests link: Eigen's own
// shared library unless EIGEN_TEST_EXTERNAL_BLAS. Parts 1 and 2 use column-major default storage, parts 3 and 4
// row-major default storage, under which the adapters' temporaries must still be column-major.
#if (defined(EIGEN_TEST_PART_3) || defined(EIGEN_TEST_PART_4)) && !defined(EIGEN_DEFAULT_TO_ROW_MAJOR)
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#endif
#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif

#include "main.h"

// Integer entries in [-3, 3] keep every sum exact in any order, so each BLAS product must equal the
// coefficient-based lazyProduct exactly.

template <typename Scalar, int LhsOrder, int RhsOrder, int DstOrder>
void blas_level3_gemm(Index rows, Index depth, Index cols) {
  using Lhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<Scalar, Dynamic, Dynamic, RhsOrder>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, DstOrder>;
  const Lhs a = RandomIntegerMatrix<Lhs>(rows, depth, -3, 3);
  const Lhs a_t = RandomIntegerMatrix<Lhs>(depth, rows, -3, 3);
  const Rhs b = RandomIntegerMatrix<Rhs>(depth, cols, -3, 3);
  const Rhs b_t = RandomIntegerMatrix<Rhs>(cols, depth, -3, 3);
  const Dst c = RandomIntegerMatrix<Dst>(rows, cols, -3, 3);
  Dst dst(rows, cols);

  dst.noalias() = a * b;
  VERIFY_IS_CWISE_EQUAL(dst, Dst(a.lazyProduct(b)));
  dst.noalias() = a.conjugate() * b;
  VERIFY_IS_CWISE_EQUAL(dst, Dst(a.conjugate().lazyProduct(b)));
  dst.noalias() = a * b.conjugate();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(a.lazyProduct(b.conjugate())));
  dst.noalias() = a_t.adjoint() * b_t.adjoint();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(a_t.adjoint().lazyProduct(b_t.adjoint())));
  dst = c;
  dst.noalias() += Scalar(2) * a.conjugate() * b.conjugate();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(c + Scalar(2) * a.conjugate().lazyProduct(b.conjugate())));
}

template <typename Scalar, int SelfadjointOrder, int OtherOrder, int DstOrder>
void blas_level3_symm(Index rows, Index cols) {
  using Selfadjoint = Matrix<Scalar, Dynamic, Dynamic, SelfadjointOrder>;
  using Other = Matrix<Scalar, Dynamic, Dynamic, OtherOrder>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, DstOrder>;
  using Real = typename NumTraits<Scalar>::Real;
  Selfadjoint s_left = RandomIntegerMatrix<Selfadjoint>(rows, rows, -3, 3);
  Selfadjoint s_right = RandomIntegerMatrix<Selfadjoint>(cols, cols, -3, 3);
  // ?hemm reads only the real part of the diagonal.
  s_left.diagonal() = s_left.diagonal().real().template cast<Scalar>();
  s_right.diagonal() = s_right.diagonal().real().template cast<Scalar>();
  const Selfadjoint lower_left = s_left.template selfadjointView<Lower>();
  const Selfadjoint upper_left = s_left.template selfadjointView<Upper>();
  const Selfadjoint lower_right = s_right.template selfadjointView<Lower>();
  const Selfadjoint upper_right = s_right.template selfadjointView<Upper>();
  const Other b = RandomIntegerMatrix<Other>(rows, cols, -3, 3);
  const Dst c = RandomIntegerMatrix<Dst>(rows, cols, -3, 3);
  Dst dst(rows, cols);

  dst.noalias() = s_left.template selfadjointView<Lower>() * b;
  VERIFY_IS_CWISE_EQUAL(dst, Dst(lower_left.lazyProduct(b)));
  dst.noalias() = s_left.template selfadjointView<Upper>() * b.conjugate();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(upper_left.lazyProduct(b.conjugate())));
  dst.noalias() = b * s_right.template selfadjointView<Upper>();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(b.lazyProduct(upper_right)));
  dst = c;
  dst.noalias() += Scalar(Real(-2)) * b.conjugate() * s_right.template selfadjointView<Lower>();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(c + Scalar(Real(-2)) * b.conjugate().lazyProduct(lower_right)));
}

template <int Mode, typename Triangular, typename Other, typename Dst>
void blas_level3_trmm_mode(const Triangular& t_left, const Triangular& t_right, const Other& b, const Dst& c) {
  using Scalar = typename Dst::Scalar;
  const Triangular left = t_left.template triangularView<Mode>();
  const Triangular right = t_right.template triangularView<Mode>();
  Dst dst(c.rows(), c.cols());

  dst.noalias() = t_left.template triangularView<Mode>() * b;
  VERIFY_IS_CWISE_EQUAL(dst, Dst(left.lazyProduct(b)));
  dst.noalias() = t_left.template triangularView<Mode>().conjugate() * b.conjugate();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(left.conjugate().lazyProduct(b.conjugate())));
  dst.noalias() = b * t_right.template triangularView<Mode>();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(b.lazyProduct(right)));
  dst = c;
  dst.noalias() += Scalar(2) * b.conjugate() * t_right.template triangularView<Mode>().conjugate();
  VERIFY_IS_CWISE_EQUAL(dst, Dst(c + Scalar(2) * b.conjugate().lazyProduct(right.conjugate())));
}

template <typename Scalar, int TriangularOrder, int OtherOrder, int DstOrder>
void blas_level3_trmm(Index rows, Index cols) {
  using Triangular = Matrix<Scalar, Dynamic, Dynamic, TriangularOrder>;
  using Other = Matrix<Scalar, Dynamic, Dynamic, OtherOrder>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, DstOrder>;
  const Triangular t_left = RandomIntegerMatrix<Triangular>(rows, rows, -3, 3);
  const Triangular t_right = RandomIntegerMatrix<Triangular>(cols, cols, -3, 3);
  const Other b = RandomIntegerMatrix<Other>(rows, cols, -3, 3);
  const Dst c = RandomIntegerMatrix<Dst>(rows, cols, -3, 3);
  blas_level3_trmm_mode<Lower>(t_left, t_right, b, c);
  blas_level3_trmm_mode<Upper>(t_left, t_right, b, c);
  blas_level3_trmm_mode<UnitLower>(t_left, t_right, b, c);
  blas_level3_trmm_mode<StrictlyUpper>(t_left, t_right, b, c);
}

template <typename Scalar, int LhsOrder, int RhsOrder, int DstOrder>
void blas_level3_case() {
  // Distinct sizes, so that a transposed or misstrided operand cannot pass.
  blas_level3_gemm<Scalar, LhsOrder, RhsOrder, DstOrder>(23, 17, 11);
  blas_level3_gemm<Scalar, LhsOrder, RhsOrder, DstOrder>(11, 17, 23);
  blas_level3_symm<Scalar, LhsOrder, RhsOrder, DstOrder>(23, 11);
  blas_level3_symm<Scalar, LhsOrder, RhsOrder, DstOrder>(11, 23);
  blas_level3_trmm<Scalar, LhsOrder, RhsOrder, DstOrder>(23, 11);
  blas_level3_trmm<Scalar, LhsOrder, RhsOrder, DstOrder>(11, 23);
}

template <typename Scalar>
void blas_level3() {
  blas_level3_case<Scalar, ColMajor, ColMajor, ColMajor>();
  blas_level3_case<Scalar, ColMajor, RowMajor, ColMajor>();
  blas_level3_case<Scalar, RowMajor, ColMajor, ColMajor>();
  blas_level3_case<Scalar, RowMajor, RowMajor, ColMajor>();
  blas_level3_case<Scalar, ColMajor, ColMajor, RowMajor>();
  blas_level3_case<Scalar, ColMajor, RowMajor, RowMajor>();
  blas_level3_case<Scalar, RowMajor, ColMajor, RowMajor>();
  blas_level3_case<Scalar, RowMajor, RowMajor, RowMajor>();
}

EIGEN_DECLARE_TEST(blas_level3) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(blas_level3<float>());
    CALL_SUBTEST_1(blas_level3<double>());
    CALL_SUBTEST_2(blas_level3<std::complex<float>>());
    CALL_SUBTEST_2(blas_level3<std::complex<double>>());
    CALL_SUBTEST_3(blas_level3<float>());
    CALL_SUBTEST_3(blas_level3<double>());
    CALL_SUBTEST_4(blas_level3<std::complex<float>>());
    CALL_SUBTEST_4(blas_level3<std::complex<double>>());
  }
}
