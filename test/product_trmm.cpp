// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "product_test_helpers.h"

template <typename T>
int get_random_size() {
  const int factor = NumTraits<T>::ReadCost;
  constexpr int max_test_size = EIGEN_TEST_MAX_SIZE > 2 * factor ? EIGEN_TEST_MAX_SIZE / factor : EIGEN_TEST_MAX_SIZE;
  return internal::random<int>(1, max_test_size);
}

template <typename Scalar, int Mode, int TriOrder, int OtherOrder, int ResOrder, int OtherCols>
void trmm(int rows = get_random_size<Scalar>(), int cols = get_random_size<Scalar>(),
          int otherCols = OtherCols == Dynamic ? get_random_size<Scalar>() : OtherCols) {
  using TriMatrix = Matrix<Scalar, Dynamic, Dynamic, TriOrder>;
  using OnTheRight = Matrix<Scalar, Dynamic, OtherCols, OtherCols == 1 ? ColMajor : OtherOrder>;
  using OnTheLeft = Matrix<Scalar, OtherCols, Dynamic, OtherCols == 1 ? RowMajor : OtherOrder>;

  using ResXS = Matrix<Scalar, Dynamic, OtherCols, OtherCols == 1 ? ColMajor : ResOrder>;
  using ResSX = Matrix<Scalar, OtherCols, Dynamic, OtherCols == 1 ? RowMajor : ResOrder>;

  TriMatrix mat(rows, cols), tri(rows, cols), triTr(cols, rows), s1tri(rows, cols), s1triTr(cols, rows);

  OnTheRight ge_right(cols, otherCols);
  OnTheLeft ge_left(otherCols, rows);
  ResSX ge_sx, ge_sx_save;
  ResXS ge_xs, ge_xs_save;

  Scalar s1 = internal::random<Scalar>(), s2 = internal::random<Scalar>();

  mat.setRandom();
  tri = mat.template triangularView<Mode>();
  triTr = mat.transpose().template triangularView<Mode>();
  s1tri = (s1 * mat).template triangularView<Mode>();
  s1triTr = (s1 * mat).transpose().template triangularView<Mode>();
  ge_right.setRandom();
  ge_left.setRandom();

  VERIFY_IS_APPROX(ge_xs = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
  VERIFY_IS_APPROX(ge_sx = ge_left * mat.template triangularView<Mode>(), ge_left * tri);

  VERIFY_IS_APPROX(ge_xs.noalias() = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
  VERIFY_IS_APPROX(ge_sx.noalias() = ge_left * mat.template triangularView<Mode>(), ge_left * tri);

  if ((Mode & UnitDiag) == 0)
    VERIFY_IS_APPROX(
        ge_xs.noalias() = (s1 * mat.adjoint()).template triangularView<Mode>() * (s2 * ge_left.transpose()),
        s1 * triTr.conjugate() * (s2 * ge_left.transpose()));

  VERIFY_IS_APPROX(
      ge_xs.noalias() = (s1 * mat.transpose()).template triangularView<Mode>() * (s2 * ge_left.transpose()),
      s1triTr * (s2 * ge_left.transpose()));
  VERIFY_IS_APPROX(ge_sx.noalias() = (s2 * ge_left) * (s1 * mat).template triangularView<Mode>(),
                   (s2 * ge_left) * s1tri);

  VERIFY_IS_APPROX(ge_sx.noalias() = ge_right.transpose() * mat.adjoint().template triangularView<Mode>(),
                   ge_right.transpose() * triTr.conjugate());
  VERIFY_IS_APPROX(ge_sx.noalias() = ge_right.adjoint() * mat.adjoint().template triangularView<Mode>(),
                   ge_right.adjoint() * triTr.conjugate());

  ge_xs_save = ge_xs;
  if ((Mode & UnitDiag) == 0)
    VERIFY_IS_APPROX(
        (ge_xs_save + s1 * triTr.conjugate() * (s2 * ge_left.adjoint())).eval(),
        ge_xs.noalias() += (s1 * mat.adjoint()).template triangularView<Mode>() * (s2 * ge_left.adjoint()));
  ge_xs_save = ge_xs;
  VERIFY_IS_APPROX(
      (ge_xs_save + s1triTr * (s2 * ge_left.adjoint())).eval(),
      ge_xs.noalias() += (s1 * mat.transpose()).template triangularView<Mode>() * (s2 * ge_left.adjoint()));
  ge_sx.setRandom();
  ge_sx_save = ge_sx;
  if ((Mode & UnitDiag) == 0)
    VERIFY_IS_APPROX(
        ge_sx_save - (ge_right.adjoint() * (-s1 * triTr).conjugate()).eval(),
        ge_sx.noalias() -= (ge_right.adjoint() * (-s1 * mat).adjoint().template triangularView<Mode>()).eval());

  if ((Mode & UnitDiag) == 0)
    VERIFY_IS_APPROX(ge_xs = (s1 * mat).adjoint().template triangularView<Mode>() * ge_left.adjoint(),
                     numext::conj(s1) * triTr.conjugate() * ge_left.adjoint());
  VERIFY_IS_APPROX(ge_xs = (s1 * mat).transpose().template triangularView<Mode>() * ge_left.adjoint(),
                   s1triTr * ge_left.adjoint());

  // Subtracting a product whose triangle has a folded scalar factor, which a unit diagonal does not share.
  ge_xs.setRandom();
  ge_xs_save = ge_xs;
  VERIFY_IS_APPROX(
      (ge_xs_save - s1triTr * (s2 * ge_left.adjoint())).eval(),
      ge_xs.noalias() -= (s1 * mat.transpose()).template triangularView<Mode>() * (s2 * ge_left.adjoint()));
  ge_sx.setRandom(ge_left.rows(), mat.cols());
  ge_sx_save = ge_sx;
  VERIFY_IS_APPROX((ge_sx_save - (s2 * ge_left) * s1tri).eval(),
                   ge_sx.noalias() -= (s2 * ge_left) * (s1 * mat).template triangularView<Mode>());

  // TODO check with sub-matrix expressions ?

  // Triangular times triangular or self-adjoint view: the right factor is densified and the product
  // runs the TRMM kernel of the left one.
  {
    using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;
    using ResXX = Matrix<Scalar, Dynamic, Dynamic, ResOrder>;
    // Hermitian operands, so that the self-adjoint views below have the real diagonal the SYMM kernel assumes.
    TriMatrix mat2 = TriMatrix::Random(cols, cols), mat3 = TriMatrix::Random(rows, rows);
    mat2 = (mat2 + mat2.adjoint()).eval();
    mat3 = (mat3 + mat3.adjoint()).eval();
    const MatrixX tri2 = mat2.template triangularView<UnitUpper>();
    const MatrixX tri3 = mat3.template triangularView<StrictlyLower>();
    const MatrixX sa2 = mat2.template selfadjointView<Lower>();
    const MatrixX sa3 = mat3.template selfadjointView<Upper>();
    ResXX res(rows, cols), res_save(rows, cols);

    VERIFY_IS_APPROX(res = mat.template triangularView<Mode>() * mat2.template triangularView<UnitUpper>(), tri * tri2);
    VERIFY_IS_APPROX(
        res.noalias() = mat3.template triangularView<StrictlyLower>() * mat.template triangularView<Mode>(),
        tri3 * tri);
    VERIFY_IS_APPROX(res = mat.template triangularView<Mode>() * mat2.template selfadjointView<Lower>(), tri * sa2);
    VERIFY_IS_APPROX(res.noalias() = mat3.template selfadjointView<Upper>() * mat.template triangularView<Mode>(),
                     sa3 * tri);
    VERIFY_IS_APPROX(res = (s1 * mat.transpose()).template triangularView<Mode>() *
                           mat3.transpose().template triangularView<StrictlyUpper>(),
                     s1triTr * tri3.transpose());

    res.setRandom(rows, cols);
    res_save = res;
    VERIFY_IS_APPROX(res.noalias() += mat.template triangularView<Mode>() * mat2.template triangularView<UnitUpper>(),
                     res_save + tri * tri2);
    res_save = res;
    VERIFY_IS_APPROX(res.noalias() -= mat3.template selfadjointView<Upper>() * mat.template triangularView<Mode>(),
                     res_save - sa3 * tri);

    // s1 cannot fold into a unit diagonal, so s1 * (UnitTri * V) evaluates the product first and scales it afterwards.
    const MatrixX unit3 = mat3.template triangularView<UnitLower>();
    VERIFY_IS_APPROX(res = s1 * (mat3.template triangularView<UnitLower>() * mat.template triangularView<Mode>()),
                     s1 * (unit3 * tri));
    ResXX res3(rows, rows);
    VERIFY_IS_APPROX(
        res3.noalias() = s1 * (mat3.template triangularView<UnitLower>() * mat3.template selfadjointView<Upper>()),
        s1 * (unit3 * sa3));
    res.setRandom(rows, cols);
    res_save = res;
    VERIFY_IS_APPROX(
        res.noalias() -= s1 * (mat3.template triangularView<UnitLower>() * mat.template triangularView<Mode>()),
        res_save - s1 * (unit3 * tri));
  }

  // destination with a non-default inner-stride
  // see bug 1741
  {
    VERIFY_IS_APPROX(ge_xs.noalias() = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
    using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;
    MatrixX buffer(2 * ge_xs.rows(), 2 * ge_xs.cols());
    Map<ResXS, 0, Stride<Dynamic, 2> > map1(buffer.data(), ge_xs.rows(), ge_xs.cols(),
                                            Stride<Dynamic, 2>(2 * ge_xs.outerStride(), 2));
    buffer.setZero();
    VERIFY_IS_APPROX(map1.noalias() = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
  }
}

template <typename Scalar, int Mode, int TriOrder>
void trmv(int rows = get_random_size<Scalar>(), int cols = get_random_size<Scalar>()) {
  trmm<Scalar, Mode, TriOrder, ColMajor, ColMajor, 1>(rows, cols, 1);
}

template <typename Scalar, int Mode, int TriOrder, int OtherOrder, int ResOrder>
void trmm(int rows = get_random_size<Scalar>(), int cols = get_random_size<Scalar>(),
          int otherCols = get_random_size<Scalar>()) {
  trmm<Scalar, Mode, TriOrder, OtherOrder, ResOrder, Dynamic>(rows, cols, otherCols);
}

// Maps whose types fix or bound some dimensions. When every maximum size is finite, TRMM packs into inline
// buffers sized by the operands, so only the partially fixed maps are large.
template <int Mode, typename LhsType, typename RhsType, typename DstType>
void trmm_maps(Index rows, Index depth, Index cols) {
  using Scalar = typename DstType::Scalar;
  using LhsX = Matrix<Scalar, Dynamic, Dynamic, LhsType::IsRowMajor ? RowMajor : ColMajor>;
  using RhsX = Matrix<Scalar, Dynamic, Dynamic, RhsType::IsRowMajor ? RowMajor : ColMajor>;
  using DstX = Matrix<Scalar, Dynamic, Dynamic, DstType::IsRowMajor ? RowMajor : ColMajor>;
  const LhsX lhs = LhsX::Random(rows, depth);
  const RhsX rhs = RhsX::Random(depth, cols);
  DstX dst(rows, cols);
  const Map<const LhsType> lhs_map(lhs.data(), rows, depth);
  const Map<const RhsType> rhs_map(rhs.data(), depth, cols);
  Map<DstType> dst_map(dst.data(), rows, cols);
  const LhsX tri_lhs = lhs.template triangularView<Mode>();
  const RhsX tri_rhs = rhs.template triangularView<Mode>();

  VERIFY_IS_APPROX(dst_map.noalias() = lhs_map.template triangularView<Mode>() * rhs_map, tri_lhs * rhs);
  VERIFY_IS_APPROX(dst_map.noalias() = lhs_map * rhs_map.template triangularView<Mode>(), lhs * tri_rhs);
}

template <typename Scalar, int Mode, int Order>
void trmm_panel_boundaries() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Real = typename NumTraits<Scalar>::Real;
  {
    // The coefficient-based path must evaluate nested products before accessing their coefficients.
    Mat a = Mat::Random(3, 3), b = Mat::Random(3, 3), c = Mat::Random(3, 3), actual(3, 3);
    const Mat ab = a * b, bc = b * c;
    const Mat triangle = a.template triangularView<Mode>();
    const Mat productTriangle = ab.template triangularView<Mode>();
    actual.noalias() = a.template triangularView<Mode>() * (b * c);
    VERIFY(verifyProduct(actual, triangle.lazyProduct(bc).eval(), triangle, bc));
    actual.noalias() = (b * c) * a.template triangularView<Mode>();
    VERIFY(verifyProduct(actual, bc.lazyProduct(triangle).eval(), bc, triangle));
    actual.noalias() = (a * b).template triangularView<Mode>() * c;
    VERIFY(verifyProduct(actual, productTriangle.lazyProduct(c).eval(), productTriangle, c));
    actual.noalias() = c * (a * b).template triangularView<Mode>();
    VERIFY(verifyProduct(actual, c.lazyProduct(productTriangle).eval(), c, productTriangle));
  }
  const Index panel = internal::product_triangular_matrix_matrix<Scalar, Index, Mode, true, ColMajor, false, ColMajor,
                                                                 false, ColMajor, 1>::SmallPanelWidth;
  // Small-product dispatch, partial diagonal panels, and multiple depth blocks under EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS.
  const Index sizes[] = {2, 3, 5, 6, 7, panel - 1, panel, panel + 1, 2 * panel + 3, 131};
  for (Index n : sizes) {
    for (Index delta : {-1, 0, 1}) {
      const Index rows = n + delta, depth = n - delta, columns = n == 131 ? 531 : n + 2;
      Mat a = Mat::Random(rows, depth), b = Mat::Random(depth, columns);
      const Mat triangle = a.template triangularView<Mode>();
      const Mat expected = triangle.lazyProduct(b);
      // Poison coefficients that the triangular view does not own, including implicit diagonals.
      for (Index j = 0; j < depth; ++j)
        for (Index i = 0; i < rows; ++i)
          if (((Mode & Lower) ? i < j : i > j) || ((Mode & (UnitDiag | ZeroDiag)) && i == j))
            a(i, j) = Scalar(NumTraits<Real>::quiet_NaN());
      Mat actual(rows, columns);
      actual.noalias() = a.template triangularView<Mode>() * b;
      VERIFY(verifyProduct(actual, expected, triangle, b));
      Mat saved = Mat::Random(rows, columns);
      actual = saved;
      actual.noalias() -= a.template triangularView<Mode>() * b;
      VERIFY(verifyProductUpdate(actual, (saved - expected).eval(), Scalar(-1), triangle, b, saved));
      Mat scaledTriangle = Scalar(2) * triangle;
      EIGEN_IF_CONSTEXPR ((Mode & UnitDiag) != 0) scaledTriangle.diagonal().setOnes();
      actual.noalias() = (Scalar(2) * a).template triangularView<Mode>() * b;
      VERIFY(verifyProduct(actual, scaledTriangle.lazyProduct(b).eval(), scaledTriangle, b));
      // A strided destination also checks the untouched interleaved coefficients.
      Mat storage = Mat::Random(2 * rows, 2 * columns), original = storage;
      Map<Mat, 0, Stride<Dynamic, 2>> strided(storage.data(), rows, columns,
                                              Stride<Dynamic, 2>(storage.outerStride(), 2));
      strided.noalias() = a.template triangularView<Mode>() * b;
      VERIFY(verifyProduct(strided, expected, triangle, b));
      for (Index j = 0; j < storage.cols(); ++j)
        for (Index i = 0; i < storage.rows(); ++i) {
          const bool written = Order == ColMajor ? (i % 2 == 0 && j < columns) : (j % 2 == 0 && i < rows);
          if (!written) VERIFY_IS_EQUAL(storage(i, j), original(i, j));
        }
    }
  }
}

// The suite's reduced cache model caps kc at the left kernel's diagonal panel width on wide-register targets, so the
// public products above never slice a diagonal block into horizontal panels there; a hand-set blocking does.
template <typename Scalar, int Mode, int LhsOrder>
void trmm_wide_depth_blocks() {
  struct Blocking : internal::level3_blocking<Scalar, Scalar> {
    Blocking(Index mc, Index nc, Index kc) {
      this->m_mc = mc;
      this->m_nc = nc;
      this->m_kc = kc;
    }
  };
  using Kernel = internal::product_triangular_matrix_matrix<Scalar, Index, Mode, true, LhsOrder, false, ColMajor, false,
                                                            ColMajor, 1>;
  using Lhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Real = typename NumTraits<Scalar>::Real;
  const Index panel = Kernel::HorizontalPanelWidth, nr = Kernel::Traits::nr;
  // Depth blocks of one full horizontal panel plus a row, and of three plus a partial one; an RHS of one column, which
  // only SME slices horizontally, and of one and two micro panels; an lhs block of every row, and of one panel, which
  // packs a depth block one panel at a time.
  for (Index kc : {panel + 1, 3 * panel + 5})
    for (Index cols : {Index(1), nr, 2 * nr + 1})
      for (bool narrowBlock : {false, true}) {
        const Index n = 2 * kc + 3;
        Lhs a = Lhs::Random(n, n);
        const Mat b = Mat::Random(n, cols);
        const Mat triangle = a.template triangularView<Mode>();
        for (Index j = 0; j < n; ++j)
          for (Index i = 0; i < n; ++i)
            if (((Mode & Lower) ? i < j : i > j) || ((Mode & (UnitDiag | ZeroDiag)) && i == j))
              a(i, j) = Scalar(NumTraits<Real>::quiet_NaN());
        Mat actual = Mat::Zero(n, cols);
        Blocking blocking(narrowBlock ? panel + 1 : n, cols, kc);
        Kernel::run(n, cols, n, a.data(), a.outerStride(), b.data(), b.outerStride(), actual.data(), 1,
                    actual.outerStride(), Scalar(1), blocking);
        VERIFY(verifyProduct(actual, triangle.lazyProduct(b).eval(), triangle, b));
      }
}

#define CALL_ALL_ORDERS(NB, SCALAR, MODE)                                             \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, ColMajor, ColMajor, ColMajor>())); \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, ColMajor, ColMajor, RowMajor>())); \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, ColMajor, RowMajor, ColMajor>())); \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, ColMajor, RowMajor, RowMajor>())); \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, RowMajor, ColMajor, ColMajor>())); \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, RowMajor, ColMajor, RowMajor>())); \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, RowMajor, RowMajor, ColMajor>())); \
  EIGEN_CAT(CALL_SUBTEST_, NB)((trmm<SCALAR, MODE, RowMajor, RowMajor, RowMajor>())); \
                                                                                      \
  EIGEN_CAT(CALL_SUBTEST_1, NB)((trmv<SCALAR, MODE, ColMajor>()));                    \
  EIGEN_CAT(CALL_SUBTEST_1, NB)((trmv<SCALAR, MODE, RowMajor>()));

#define CALL_ALL(NB, SCALAR)                               \
  CALL_ALL_ORDERS(EIGEN_CAT(1, NB), SCALAR, Upper)         \
  CALL_ALL_ORDERS(EIGEN_CAT(2, NB), SCALAR, UnitUpper)     \
  CALL_ALL_ORDERS(EIGEN_CAT(3, NB), SCALAR, StrictlyUpper) \
  CALL_ALL_ORDERS(EIGEN_CAT(1, NB), SCALAR, Lower)         \
  CALL_ALL_ORDERS(EIGEN_CAT(2, NB), SCALAR, UnitLower)     \
  CALL_ALL_ORDERS(EIGEN_CAT(3, NB), SCALAR, StrictlyLower)

EIGEN_DECLARE_TEST(product_trmm) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_ALL(1, float);                 //  EIGEN_SUFFIXES;11;111;21;121;31;131
    CALL_ALL(2, double);                //  EIGEN_SUFFIXES;12;112;22;122;32;132
    CALL_ALL(3, std::complex<float>);   //  EIGEN_SUFFIXES;13;113;23;123;33;133
    CALL_ALL(4, std::complex<double>);  //  EIGEN_SUFFIXES;14;114;24;124;34;134
  }
  CALL_SUBTEST_11((trmm_panel_boundaries<float, Lower, ColMajor>()));
  CALL_SUBTEST_12((trmm_panel_boundaries<double, Upper, RowMajor>()));
  CALL_SUBTEST_13((trmm_panel_boundaries<std::complex<float>, Upper, ColMajor>()));
  CALL_SUBTEST_14((trmm_panel_boundaries<std::complex<double>, Lower, RowMajor>()));
  CALL_SUBTEST_21((trmm_panel_boundaries<float, UnitLower, ColMajor>()));
  CALL_SUBTEST_22((trmm_panel_boundaries<double, UnitUpper, RowMajor>()));
  CALL_SUBTEST_31((trmm_panel_boundaries<float, StrictlyLower, ColMajor>()));
  CALL_SUBTEST_32((trmm_panel_boundaries<double, StrictlyUpper, RowMajor>()));
  CALL_SUBTEST_11((trmm_wide_depth_blocks<float, Lower, ColMajor>()));
  CALL_SUBTEST_12((trmm_wide_depth_blocks<double, Upper, ColMajor>()));
  CALL_SUBTEST_13((trmm_wide_depth_blocks<std::complex<float>, Lower, RowMajor>()));
  CALL_SUBTEST_14((trmm_wide_depth_blocks<std::complex<double>, Upper, ColMajor>()));
  CALL_SUBTEST_21((trmm_wide_depth_blocks<float, UnitUpper, RowMajor>()));
  CALL_SUBTEST_22((trmm_wide_depth_blocks<double, UnitLower, ColMajor>()));
  CALL_SUBTEST_31((trmm_wide_depth_blocks<float, StrictlyLower, ColMajor>()));
  CALL_SUBTEST_32((trmm_wide_depth_blocks<double, StrictlyUpper, RowMajor>()));
  // 1x1 and blocking-boundary sizes for the view-times-view products.
  CALL_SUBTEST_11((trmm<float, Lower, ColMajor, ColMajor, ColMajor>(1, 1, 1)));
  CALL_SUBTEST_11((trmm<float, Lower, ColMajor, ColMajor, ColMajor>(6, 6, 6)));
  CALL_SUBTEST_14((trmm<std::complex<double>, Upper, RowMajor, ColMajor, RowMajor>(1, 1, 1)));
  CALL_SUBTEST_12((trmm<double, Upper, ColMajor, ColMajor, ColMajor>(EIGEN_TEST_MAX_SIZE, EIGEN_TEST_MAX_SIZE, 1)));
  // Trapezoids spanning several depth blocks, with partial diagonal panels and packet tails.
  CALL_SUBTEST_11((trmm<float, Lower, ColMajor, ColMajor, ColMajor>(513, 521, 517)));
  CALL_SUBTEST_12((trmm<double, Upper, RowMajor, RowMajor, RowMajor>(521, 513, 517)));
  CALL_SUBTEST_21((trmm<float, UnitLower, ColMajor, RowMajor, RowMajor>(521, 513, 517)));
  CALL_SUBTEST_22((trmm<double, UnitUpper, RowMajor, ColMajor, ColMajor>(513, 521, 517)));
  CALL_SUBTEST_31((trmm<float, StrictlyUpper, ColMajor, ColMajor, ColMajor>(521, 513, 517)));
  CALL_SUBTEST_32((trmm<double, StrictlyLower, RowMajor, RowMajor, RowMajor>(513, 521, 517)));
  CALL_SUBTEST_11((trmm_maps<Lower, Matrix<float, 24, 20>, Matrix<float, 20, 16>, Matrix<float, 24, 16>>(24, 20, 16)));
  // Upper trapezoids filling the inline blockB, with a packed triangle (depth^2 scalars) that ends off a packet
  // boundary, so the general panel after it is moved forward to realign. Large enough (rows + depth + cols >=
  // EIGEN_GEMM_TO_COEFFBASED_THRESHOLD) to be packed rather than computed coefficient-wise.
  CALL_SUBTEST_11((trmm_maps<Upper, Matrix<float, 8, 7>, Matrix<float, 7, 9>, Matrix<float, 8, 9>>(8, 7, 9)));
  CALL_SUBTEST_12((trmm_maps<Upper, Matrix<double, 8, 7>, Matrix<double, 7, 9>, Matrix<double, 8, 9>>(8, 7, 9)));
  CALL_SUBTEST_12(
      (trmm_maps<UnitUpper, Matrix<double, 13, 17, RowMajor>, Matrix<double, 17, 11>, Matrix<double, 13, 11, RowMajor>>(
          13, 17, 11)));
  CALL_SUBTEST_21(
      (trmm_maps<Lower, Matrix<float, 517, Dynamic>, MatrixXf, Matrix<float, 517, Dynamic>>(517, 521, 513)));
  CALL_SUBTEST_22(
      (trmm_maps<StrictlyUpper, Matrix<double, Dynamic, Dynamic, RowMajor, 528, Dynamic>,
                 Matrix<double, Dynamic, Dynamic, RowMajor>, Matrix<double, Dynamic, Dynamic, RowMajor, 528, Dynamic>>(
          521, 513, 517)));
}
