// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#define VERIFY_TRSM(TRI, XB)                             \
  {                                                      \
    (XB).setRandom();                                    \
    ref = (XB);                                          \
    (TRI).solveInPlace(XB);                              \
    VERIFY_IS_APPROX((TRI).toDenseMatrix() * (XB), ref); \
    (XB).setRandom();                                    \
    ref = (XB);                                          \
    (XB) = (TRI).solve(XB);                              \
    VERIFY_IS_APPROX((TRI).toDenseMatrix() * (XB), ref); \
  }

#define VERIFY_TRSM_ONTHERIGHT(TRI, XB)                                                      \
  {                                                                                          \
    (XB).setRandom();                                                                        \
    ref = (XB);                                                                              \
    (TRI).transpose().template solveInPlace<OnTheRight>(XB.transpose());                     \
    VERIFY_IS_APPROX((XB).transpose() * (TRI).transpose().toDenseMatrix(), ref.transpose()); \
    (XB).setRandom();                                                                        \
    ref = (XB);                                                                              \
    (XB).transpose() = (TRI).transpose().template solve<OnTheRight>(XB.transpose());         \
    VERIFY_IS_APPROX((XB).transpose() * (TRI).transpose().toDenseMatrix(), ref.transpose()); \
  }

template <typename Scalar, int Size, int Cols>
void trsolve(int size = Size, int cols = Cols) {
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Matrix<Scalar, Size, Size, ColMajor> cmLhs(size, size);
  Matrix<Scalar, Size, Size, RowMajor> rmLhs(size, size);

  enum { colmajor = Size == 1 ? RowMajor : ColMajor, rowmajor = Cols == 1 ? ColMajor : RowMajor };
  Matrix<Scalar, Size, Cols, colmajor> cmRhs(size, cols);
  Matrix<Scalar, Size, Cols, rowmajor> rmRhs(size, cols);
  Matrix<Scalar, Dynamic, Dynamic, colmajor> ref(size, cols);

  cmLhs.setRandom();
  cmLhs *= static_cast<RealScalar>(0.1);
  cmLhs.diagonal().array() += static_cast<RealScalar>(1);
  rmLhs.setRandom();
  rmLhs *= static_cast<RealScalar>(0.1);
  rmLhs.diagonal().array() += static_cast<RealScalar>(1);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(cmLhs.adjoint().template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(cmLhs.template triangularView<Upper>(), cmRhs);
  VERIFY_TRSM(cmLhs.template triangularView<Lower>(), rmRhs);
  VERIFY_TRSM(cmLhs.conjugate().template triangularView<Upper>(), rmRhs);
  VERIFY_TRSM(cmLhs.adjoint().template triangularView<Upper>(), rmRhs);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<UnitLower>(), cmRhs);
  VERIFY_TRSM(cmLhs.template triangularView<UnitUpper>(), rmRhs);

  VERIFY_TRSM(rmLhs.template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(rmLhs.conjugate().template triangularView<UnitUpper>(), rmRhs);

  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs.template triangularView<Upper>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs.template triangularView<Lower>(), rmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<Upper>(), rmRhs);

  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<UnitLower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs.template triangularView<UnitUpper>(), rmRhs);

  VERIFY_TRSM_ONTHERIGHT(rmLhs.template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(rmLhs.conjugate().template triangularView<UnitUpper>(), rmRhs);

  int c = internal::random<int>(0, cols - 1);
  VERIFY_TRSM(rmLhs.template triangularView<Lower>(), rmRhs.col(c));
  VERIFY_TRSM(cmLhs.template triangularView<Lower>(), rmRhs.col(c));

  // destination with a non-default inner-stride
  // see bug 1741
  {
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    MatrixX buffer(2 * cmRhs.rows(), 2 * cmRhs.cols());
    Map<Matrix<Scalar, Size, Cols, colmajor>, 0, Stride<Dynamic, 2> > map1(
        buffer.data(), cmRhs.rows(), cmRhs.cols(), Stride<Dynamic, 2>(2 * cmRhs.outerStride(), 2));
    Map<Matrix<Scalar, Size, Cols, rowmajor>, 0, Stride<Dynamic, 2> > map2(
        buffer.data(), rmRhs.rows(), rmRhs.cols(), Stride<Dynamic, 2>(2 * rmRhs.outerStride(), 2));
    buffer.setZero();
    VERIFY_TRSM(cmLhs.conjugate().template triangularView<Lower>(), map1);
    buffer.setZero();
    VERIFY_TRSM(cmLhs.template triangularView<Lower>(), map2);
  }

  if (Size == Dynamic) {
    cmLhs.resize(0, 0);
    cmRhs.resize(0, cmRhs.cols());
    Matrix<Scalar, Size, Cols, colmajor> res = cmLhs.template triangularView<Lower>().solve(cmRhs);
    VERIFY_IS_EQUAL(res.rows(), 0);
    VERIFY_IS_EQUAL(res.cols(), cmRhs.cols());
    res = cmRhs;
    cmLhs.template triangularView<Lower>().solveInPlace(res);
    VERIFY_IS_EQUAL(res.rows(), 0);
    VERIFY_IS_EQUAL(res.cols(), cmRhs.cols());
  }
}

// Test triangular solve with non-unit inner stride at blocking boundary sizes.
// The scalar fallback path in trsmKernelR (TriangularSolverMatrix.h lines 156-166)
// is used when OtherInnerStride != 1. The existing bug 1741 test only uses
// InnerStride=2 at random sizes. This exercises the scalar path at sizes that
// trigger blocking transitions and tests additional configurations.
template <int>
void trsolve_strided_boundary() {
  typedef double Scalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;

  const int sizes[] = {1, 2, 3, 4, 8, 12, 16, 24, 32, 47, 48, 49, 64};
  for (int si = 0; si < 13; ++si) {
    int n = sizes[si];

    MatrixX lhs = MatrixX::Random(n, n);
    lhs *= 0.1;
    lhs.diagonal().array() += 1.0;

    // InnerStride = 2: ColMajor RHS, OnTheLeft, Lower
    {
      int cols = 5;
      MatrixX buffer(2 * n, 2 * cols);
      Map<MatrixX, 0, Stride<Dynamic, 2> > map(buffer.data(), n, cols, Stride<Dynamic, 2>(2 * n, 2));
      MatrixX ref(n, cols);
      buffer.setZero();
      map.setRandom();
      ref = map;
      lhs.triangularView<Lower>().solveInPlace(map);
      VERIFY_IS_APPROX(lhs.triangularView<Lower>().toDenseMatrix() * MatrixX(map), ref);
    }

    // InnerStride = 2: Upper triangular
    {
      int cols = 5;
      MatrixX buffer(2 * n, 2 * cols);
      Map<MatrixX, 0, Stride<Dynamic, 2> > map(buffer.data(), n, cols, Stride<Dynamic, 2>(2 * n, 2));
      MatrixX ref(n, cols);
      buffer.setZero();
      map.setRandom();
      ref = map;
      lhs.triangularView<Upper>().solveInPlace(map);
      VERIFY_IS_APPROX(lhs.triangularView<Upper>().toDenseMatrix() * MatrixX(map), ref);
    }

    // InnerStride = 2: UnitLower (tests the UnitDiag path without diagonal scaling)
    {
      int cols = 3;
      MatrixX buffer(2 * n, 2 * cols);
      Map<MatrixX, 0, Stride<Dynamic, 2> > map(buffer.data(), n, cols, Stride<Dynamic, 2>(2 * n, 2));
      MatrixX ref(n, cols);
      buffer.setZero();
      map.setRandom();
      ref = map;
      lhs.triangularView<UnitLower>().solveInPlace(map);
      VERIFY_IS_APPROX(lhs.triangularView<UnitLower>().toDenseMatrix() * MatrixX(map), ref);
    }

    // InnerStride = 3: Less common stride to exercise the scalar path more thoroughly
    {
      int cols = 4;
      MatrixX buffer(3 * n, 3 * cols);
      Map<MatrixX, 0, Stride<Dynamic, 3> > map(buffer.data(), n, cols, Stride<Dynamic, 3>(3 * n, 3));
      MatrixX ref(n, cols);
      buffer.setZero();
      map.setRandom();
      ref = map;
      lhs.triangularView<Lower>().solveInPlace(map);
      VERIFY_IS_APPROX(lhs.triangularView<Lower>().toDenseMatrix() * MatrixX(map), ref);
    }

    // Vector RHS with InnerStride = 2
    {
      typedef Matrix<Scalar, Dynamic, 1> VecX;
      VecX buffer(2 * n);
      Map<VecX, 0, InnerStride<2> > map(buffer.data(), n, InnerStride<2>(2));
      buffer.setZero();
      map.setRandom();
      VecX ref = map;
      lhs.triangularView<Lower>().solveInPlace(map);
      VERIFY_IS_APPROX(lhs.triangularView<Lower>().toDenseMatrix() * VecX(map), ref);
    }
  }

  // Complex with non-unit stride: tests conjugation in the scalar fallback path.
  {
    typedef std::complex<double> CScalar;
    typedef Matrix<CScalar, Dynamic, Dynamic> CMatrixX;
    int n = 32;
    CMatrixX lhs = CMatrixX::Random(n, n);
    lhs *= CScalar(0.1);
    lhs.diagonal().array() += CScalar(1.0);

    int cols = 4;
    CMatrixX buffer(2 * n, 2 * cols);
    Map<CMatrixX, 0, Stride<Dynamic, 2> > map(buffer.data(), n, cols, Stride<Dynamic, 2>(2 * n, 2));
    CMatrixX ref(n, cols);

    // Conjugate Lower
    buffer.setZero();
    map.setRandom();
    ref = map;
    lhs.conjugate().triangularView<Lower>().solveInPlace(map);
    VERIFY_IS_APPROX(lhs.conjugate().triangularView<Lower>().toDenseMatrix() * CMatrixX(map), ref);

    // Adjoint Upper
    buffer.setZero();
    map.setRandom();
    ref = map;
    lhs.adjoint().triangularView<Lower>().solveInPlace(map);
    VERIFY_IS_APPROX(lhs.adjoint().triangularView<Lower>().toDenseMatrix() * CMatrixX(map), ref);
  }
}

EIGEN_DECLARE_TEST(product_trsolve) {
  for (int i = 0; i < g_repeat; i++) {
    // matrices
    CALL_SUBTEST_1((trsolve<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                     internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2((trsolve<double, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                      internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_3((trsolve<std::complex<float>, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                                                   internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));
    CALL_SUBTEST_4((trsolve<std::complex<double>, Dynamic, Dynamic>(
        internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2), internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));

    // vectors
    CALL_SUBTEST_5((trsolve<float, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_6((trsolve<double, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_7((trsolve<std::complex<float>, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_8((trsolve<std::complex<double>, Dynamic, 1>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));

    // meta-unrollers
    CALL_SUBTEST_9((trsolve<float, 4, 1>()));
    CALL_SUBTEST_10((trsolve<double, 4, 1>()));
    CALL_SUBTEST_11((trsolve<std::complex<float>, 4, 1>()));
    CALL_SUBTEST_12((trsolve<float, 1, 1>()));
    CALL_SUBTEST_13((trsolve<float, 1, 2>()));
    CALL_SUBTEST_14((trsolve<float, 3, 1>()));
  }

  // Strided solve at blocking boundaries (deterministic, outside g_repeat).
  CALL_SUBTEST_15(trsolve_strided_boundary<0>());
}
