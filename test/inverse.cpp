// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/LU>

template <typename MatrixType, std::enable_if_t<MatrixType::SizeAtCompileTime == Dynamic, int> = 0>
void inverse_for_fixed_size(const MatrixType&) {}

template <typename MatrixType, std::enable_if_t<MatrixType::SizeAtCompileTime != Dynamic, int> = 0>
void inverse_for_fixed_size(const MatrixType& m1) {
  using std::abs;

  MatrixType m2, identity = MatrixType::Identity();

  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using VectorType = Matrix<Scalar, MatrixType::ColsAtCompileTime, 1>;

  // computeInverseAndDetWithCheck tests
  // First: an invertible matrix
  bool invertible;
  Scalar det;

  m2.setZero();
  m1.computeInverseAndDetWithCheck(m2, det, invertible);
  VERIFY(invertible);
  VERIFY_IS_APPROX(identity, m1 * m2);
  VERIFY_IS_APPROX(det, m1.determinant());

  m2.setZero();
  m1.computeInverseWithCheck(m2, invertible);
  VERIFY(invertible);
  VERIFY_IS_APPROX(identity, m1 * m2);

  // Second: a rank one matrix (not invertible, except for 1x1 matrices)
  VectorType v3 = VectorType::Random();
  MatrixType m3 = v3 * v3.transpose(), m4;
  m3.computeInverseAndDetWithCheck(m4, det, invertible);
  VERIFY(m1.rows() == 1 ? invertible : !invertible);
  VERIFY_IS_MUCH_SMALLER_THAN(abs(det - m3.determinant()), RealScalar(1));
  m3.computeInverseWithCheck(m4, invertible);
  VERIFY(m1.rows() == 1 ? invertible : !invertible);

  // check with submatrices
  {
    Matrix<Scalar, MatrixType::RowsAtCompileTime + 1, MatrixType::RowsAtCompileTime + 1, MatrixType::Options> m5;
    m5.setRandom();
    m5.topLeftCorner(m1.rows(), m1.rows()) = m1;
    m2 = m5.template topLeftCorner<MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime>().inverse();
    VERIFY_IS_APPROX((m5.template topLeftCorner<MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime>()),
                     m2.inverse());
  }

  // check that computeInverseAndDetWithCheck resizes dynamic result matrix (issue #2917)
  {
    using DynamicMatrix = Matrix<Scalar, Dynamic, Dynamic>;
    DynamicMatrix m_dyn_inv;
    Scalar det_dyn;
    bool inv_dyn;
    m1.computeInverseAndDetWithCheck(m_dyn_inv, det_dyn, inv_dyn);
    VERIFY(inv_dyn);
    VERIFY_IS_EQUAL(m_dyn_inv.rows(), m1.rows());
    VERIFY_IS_EQUAL(m_dyn_inv.cols(), m1.cols());
    VERIFY_IS_APPROX(identity, m1 * m_dyn_inv);
    VERIFY_IS_APPROX(det_dyn, m1.determinant());
  }

  // check that computeInverseWithCheck resizes dynamic result matrix
  {
    using DynamicMatrix = Matrix<Scalar, Dynamic, Dynamic>;
    DynamicMatrix m_dyn_inv;
    bool inv_dyn;
    m1.computeInverseWithCheck(m_dyn_inv, inv_dyn);
    VERIFY(inv_dyn);
    VERIFY_IS_EQUAL(m_dyn_inv.rows(), m1.rows());
    VERIFY_IS_EQUAL(m_dyn_inv.cols(), m1.cols());
    VERIFY_IS_APPROX(identity, m1 * m_dyn_inv);
  }
}

template <typename MatrixType>
void inverse(const MatrixType& m) {
  /* this test covers the following files:
     Inverse.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  using Scalar = typename MatrixType::Scalar;

  MatrixType m1(rows, cols), m2(rows, cols), identity = MatrixType::Identity(rows, rows);
  createRandomPIMatrixOfRank(rows, rows, rows, m1);
  m2 = m1.inverse();
  VERIFY_IS_APPROX(m1, m2.inverse());

  VERIFY_IS_APPROX((Scalar(2) * m2).inverse(), m2.inverse() * Scalar(0.5));

  VERIFY_IS_APPROX(identity, m1.inverse() * m1);
  VERIFY_IS_APPROX(identity, m1 * m1.inverse());

  VERIFY_IS_APPROX(m1, m1.inverse().inverse());

  // since for the general case we implement separately row-major and col-major, test that
  VERIFY_IS_APPROX(MatrixType(m1.transpose().inverse()), MatrixType(m1.inverse().transpose()));

  inverse_for_fixed_size(m1);

  // check in-place inversion
  if (MatrixType::RowsAtCompileTime >= 2 && MatrixType::RowsAtCompileTime <= 4) {
    // in-place is forbidden
    VERIFY_RAISES_ASSERT(m1 = m1.inverse());
  } else {
    m2 = m1.inverse();
    m1 = m1.inverse();
    VERIFY_IS_APPROX(m1, m2);
  }
}

// Products with an inverse of a fixed-size complex matrix. GCC 15.1 and 15.2 with FMA miscompiled v * v.inverse() for
// Matrix2cf at -O2 and -O3 (and v * w, w = v.inverse(), for Matrix3cf at -O3): the inverse's complex products,
// broadcast by pset1<Packet2cf> in the lazy product, came out as garbage (see EIGEN_GCC15_SLP_BUG). Each case is a
// function of its own, as in user code: the miscompile needs the inverse and the product inlined together.
template <int Case, typename MatrixType, typename VectorType>
EIGEN_DONT_INLINE MatrixType inverse_product_case(const MatrixType& v, const MatrixType& b, const VectorType& d) {
  switch (Case) {
    case 0:
      return v * v.inverse();
    case 1:
      return v.inverse() * v;
    case 2:
      return v.lazyProduct(v.inverse());
    case 3:
      return v * v.inverse().eval();
    case 4: {
      const MatrixType w = v.inverse();
      return v * w;
    }
    case 5: {
      const MatrixType w = v.inverse();
      return w * v;
    }
    case 6:
      return b * v.inverse();
    default:
      return v * d.asDiagonal() * v.inverse();
  }
}

template <typename MatrixType>
void inverse_product_fixed_size() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  constexpr Index n = MatrixType::RowsAtCompileTime;
  const MatrixType identity = MatrixType::Identity();
  // ||R||_2 <= sqrt(2) n for |re|, |im| <= 1, so 2 n I + R is well conditioned.
  const MatrixType v = MatrixType::Random() + RealScalar(2 * n) * identity;
  const MatrixType b = MatrixType::Random();
  const Matrix<Scalar, n, 1> d = Matrix<Scalar, n, 1>::Random();
  const RealScalar cond = v.norm() * v.partialPivLu().inverse().norm();
  const RealScalar tol = RealScalar(4 * n) * NumTraits<RealScalar>::epsilon() * cond;
  const auto max_abs_diff = [](const MatrixType& x, const MatrixType& y) { return (x - y).cwiseAbs().maxCoeff(); };
  // x * v by scalar loops, so the check does not go through the product evaluator under test.
  const auto ref_times_v = [&v](const MatrixType& x) {
    MatrixType r = MatrixType::Zero();
    for (Index i = 0; i < x.rows(); ++i)
      for (Index j = 0; j < v.cols(); ++j)
        for (Index k = 0; k < x.cols(); ++k) r.coeffRef(i, j) += x.coeff(i, k) * v.coeff(k, j);
    return r;
  };

  VERIFY(max_abs_diff(inverse_product_case<0>(v, b, d), identity) <= tol);
  VERIFY(max_abs_diff(inverse_product_case<1>(v, b, d), identity) <= tol);
  VERIFY(max_abs_diff(inverse_product_case<2>(v, b, d), identity) <= tol);
  VERIFY(max_abs_diff(inverse_product_case<3>(v, b, d), identity) <= tol);
  VERIFY(max_abs_diff(inverse_product_case<4>(v, b, d), identity) <= tol);
  VERIFY(max_abs_diff(inverse_product_case<5>(v, b, d), identity) <= tol);
  // (b v^-1) v = b and (v diag(d) v^-1) v = v diag(d)
  VERIFY(max_abs_diff(ref_times_v(inverse_product_case<6>(v, b, d)), b) <= tol * b.norm());
  const MatrixType vd = v * d.asDiagonal();
  VERIFY(max_abs_diff(ref_times_v(inverse_product_case<7>(v, b, d)), vd) <= tol * vd.norm());
}

template <typename Scalar>
void inverse_zerosized() {
  Matrix<Scalar, Dynamic, Dynamic> A(0, 0);
  {
    Matrix<Scalar, 0, 1> b, x;
    x = A.inverse() * b;
  }
  {
    Matrix<Scalar, Dynamic, Dynamic> b(0, 1), x;
    x = A.inverse() * b;
    VERIFY_IS_EQUAL(x.rows(), 0);
    VERIFY_IS_EQUAL(x.cols(), 1);
  }
}

// Writes inverse() and computeInverseAndDetWithCheck() through destinations with a non-unit inner stride, an outer
// stride, or both, and checks that no element outside the destination changes.
template <typename Dst, typename Scalar, typename Matrix, typename Maker>
void inverse_check_strided_destination(const Matrix& A, const Matrix& ref, Maker make, Index buffer_size) {
  for (int variant = 0; variant < 2; ++variant) {
    std::vector<Scalar> buffer(buffer_size);
    for (Index i = 0; i < buffer_size; ++i) buffer[i] = Scalar(i + 1);
    std::vector<Scalar> expected(buffer);
    Dst dst(make(buffer.data()));
    Dst dst_expected(make(expected.data()));
    if (variant == 0) {
      dst = A.inverse();
    } else {
      bool invertible = false;
      Scalar det;
      A.computeInverseAndDetWithCheck(dst, det, invertible);
      VERIFY(invertible);
    }
    Matrix got = dst;
    VERIFY_IS_APPROX(got, ref);

    // Elements outside the destination must be untouched: after storing the exact reference into both
    // destinations, the two buffers must agree bit for bit.
    dst = ref;
    dst_expected = ref;
    for (Index i = 0; i < buffer_size; ++i) VERIFY_IS_EQUAL(buffer[i], expected[i]);
  }
}

template <typename Scalar, int Size, int Options>
void inverse_strided_destination_order() {
  using M = Matrix<Scalar, Size, Size, Options>;
  M A;
  createRandomPIMatrixOfRank(Size, Size, Size, A);
  M ref = A.inverse();
  const Index n = Size;
  const Index buffer_size = 4 * n * n + 7;

  inverse_check_strided_destination<Map<M, 0, InnerStride<2>>, Scalar>(
      A, ref, [&](Scalar* p) { return Map<M, 0, InnerStride<2>>(p); }, buffer_size);
  inverse_check_strided_destination<Map<M, 0, InnerStride<3>>, Scalar>(
      A, ref, [&](Scalar* p) { return Map<M, 0, InnerStride<3>>(p); }, buffer_size);
  inverse_check_strided_destination<Map<M, 0, OuterStride<2 * Size + 1>>, Scalar>(
      A, ref, [&](Scalar* p) { return Map<M, 0, OuterStride<2 * Size + 1>>(p); }, buffer_size);
  inverse_check_strided_destination<Map<M, 0, Stride<4 * Size + 1, 2>>, Scalar>(
      A, ref, [&](Scalar* p) { return Map<M, 0, Stride<4 * Size + 1, 2>>(p); }, buffer_size);
  // Inner stride known only at run time, both unit and non-unit.
  for (Index inner = 1; inner <= 3; ++inner) {
    using Dyn = Map<M, 0, Stride<Dynamic, Dynamic>>;
    inverse_check_strided_destination<Dyn, Scalar>(
        A, ref, [&](Scalar* p) { return Dyn(p, n, n, Stride<Dynamic, Dynamic>(inner * n + 1, inner)); }, buffer_size);
    using DynInner = Map<M, 0, InnerStride<Dynamic>>;
    inverse_check_strided_destination<DynInner, Scalar>(
        A, ref, [&](Scalar* p) { return DynInner(p, n, n, InnerStride<Dynamic>(inner)); }, buffer_size);
    using DynRef = Ref<M, 0, Stride<Dynamic, Dynamic>>;
    inverse_check_strided_destination<DynRef, Scalar>(
        A, ref,
        [&](Scalar* p) {
          return Map<M, 0, Stride<Dynamic, Dynamic>>(p, n, n, Stride<Dynamic, Dynamic>(inner * n + 1, inner));
        },
        buffer_size);
  }
}

template <typename Scalar, int Size>
void inverse_strided_destination() {
  inverse_strided_destination_order<Scalar, Size, ColMajor>();
  inverse_strided_destination_order<Scalar, Size, RowMajor>();
}

EIGEN_DECLARE_TEST(inverse) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(inverse(Matrix<double, 1, 1>()));
    CALL_SUBTEST_2(inverse(Matrix2d()));
    CALL_SUBTEST_2((inverse_strided_destination<double, 2>()));
    CALL_SUBTEST_3(inverse(Matrix3f()));
    CALL_SUBTEST_3((inverse_strided_destination<float, 3>()));
    CALL_SUBTEST_4(inverse(Matrix4f()));
    CALL_SUBTEST_4(inverse(Matrix<float, 4, 4, DontAlign>()));
    CALL_SUBTEST_4((inverse_strided_destination<float, 4>()));

    s = internal::random<int>(50, 320);
    CALL_SUBTEST_5(inverse(MatrixXf(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
    CALL_SUBTEST_5(inverse_zerosized<float>());
    CALL_SUBTEST_5(inverse(MatrixXf(0, 0)));
    CALL_SUBTEST_5(inverse(MatrixXf(1, 1)));

    s = internal::random<int>(25, 100);
    CALL_SUBTEST_6(inverse(MatrixXcd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    CALL_SUBTEST_7(inverse(Matrix4d()));
    CALL_SUBTEST_7(inverse(Matrix<double, 4, 4, DontAlign>()));
    CALL_SUBTEST_7((inverse_strided_destination<double, 4>()));

    CALL_SUBTEST_8(inverse(Matrix4cd()));

    CALL_SUBTEST_9(inverse_product_fixed_size<Matrix2cf>());
    CALL_SUBTEST_9(inverse_product_fixed_size<Matrix3cf>());
    CALL_SUBTEST_9(inverse_product_fixed_size<Matrix4cf>());
    CALL_SUBTEST_9((inverse_product_fixed_size<Matrix<std::complex<float>, 2, 2, RowMajor>>()));
    CALL_SUBTEST_9(inverse_product_fixed_size<Matrix2cd>());
    CALL_SUBTEST_9(inverse_product_fixed_size<Matrix3cd>());
    CALL_SUBTEST_9(inverse_product_fixed_size<Matrix4cd>());
  }
}
