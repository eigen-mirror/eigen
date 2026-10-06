// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/QR>
#include "solverbase.h"

template <typename MatrixType>
void qr(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;

  MatrixType a = MatrixType::Random(rows, cols);
  HouseholderQR<MatrixType> qrOfA(a);

  MatrixQType q = qrOfA.householderQ();
  VERIFY_IS_UNITARY(q);

  MatrixType r = qrOfA.matrixQR().template triangularView<Upper>();
  VERIFY_IS_APPROX(a, qrOfA.householderQ() * r);
}

template <typename MatrixType, int Cols2>
void qr_fixedsize() {
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef typename MatrixType::Scalar Scalar;
  Matrix<Scalar, Rows, Cols> m1 = Matrix<Scalar, Rows, Cols>::Random();
  if (Rows < Cols) {
    // Transposed solves depend on R_11, which has the singular values of the leading square block.
    static constexpr int Size = (Rows < Cols) ? Rows : Cols;
    using RealScalar = typename MatrixType::RealScalar;
    using SingularValues = Matrix<RealScalar, Size, 1>;
    const SingularValues svs = setupRangeSvs<SingularValues>(Size, RealScalar(0.5), RealScalar(1));
    auto leading = m1.template topLeftCorner<Size, Size>();
    generateRandomMatrixSvs(svs, Size, Size, leading);
  }
  HouseholderQR<Matrix<Scalar, Rows, Cols> > qr(m1);

  Matrix<Scalar, Rows, Cols> r = qr.matrixQR();
  // FIXME need better way to construct trapezoid
  for (int i = 0; i < Rows; i++)
    for (int j = 0; j < Cols; j++)
      if (i > j) r(i, j) = Scalar(0);

  VERIFY_IS_APPROX(m1, qr.householderQ() * r);

  check_solverbase<Matrix<Scalar, Cols, Cols2>, Matrix<Scalar, Rows, Cols2> >(m1, qr, Rows, Cols, Cols2);
}

template <typename MatrixType>
void qr_invertible() {
  using std::abs;
  using std::log;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  STATIC_CHECK((std::is_same<typename HouseholderQR<MatrixType>::StorageIndex, int>::value));

  int size = internal::random<int>(10, 50);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size, size);

  if (std::is_same<RealScalar, float>::value) {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size, size * 4);
    m1 += a * a.adjoint();
  }

  HouseholderQR<MatrixType> qr(m1);

  check_solverbase<MatrixType, MatrixType>(m1, qr, size, size, size);

  // now construct a matrix with prescribed determinant
  m1.setZero();
  setRandomWellConditionedDiagonal(m1);
  Scalar det = m1.diagonal().prod();
  RealScalar absdet = abs(det);
  m3 = qr.householderQ();  // get a unitary
  m1 = m3 * m1 * m3.adjoint();
  qr.compute(m1);
  VERIFY_IS_APPROX(log(absdet), qr.logAbsDeterminant());
  VERIFY_IS_APPROX(numext::sign(det), qr.signDeterminant());
  VERIFY_IS_APPROX(det, qr.determinant());
  VERIFY_IS_APPROX(absdet, qr.absDeterminant());
}

template <typename MatrixType>
void qr_check_thin_factors(const MatrixType& a) {
  const Index rows = a.rows(), cols = a.cols(), k = (std::min)(rows, cols);
  HouseholderQR<MatrixType> qr(a);
  // The thin factors: the first k columns of Q and the top k rows of R.
  const MatrixType q = qr.householderQ() * MatrixType::Identity(rows, k);
  const MatrixType r = qr.matrixQR().topRows(k).template triangularView<Upper>();
  // Householder QR is backward stable: ||A - QR|| <= c * max(rows, cols) * eps * ||A||; c = 4 has a wide margin here.
  const double eps = NumTraits<double>::epsilon();
  VERIFY((a - q * r).norm() <= 4 * double((std::max)(rows, cols)) * eps * a.norm());
  VERIFY((q.adjoint() * q - MatrixType::Identity(k, k)).norm() <= 4 * double(rows) * eps * std::sqrt(double(k)));
}

// HouseholderQR factors in a single panel when rows * cols * min(rows, cols) <= 64^3 and otherwise picks a panel width
// from {8, 16, 24, 32, 48} (internal::householder_qr_panel_width). The shapes straddle that threshold, reach both
// ends of the width range and leave a partial last panel.
template <int>
void qr_blocking_shapes() {
  using MatrixType = Matrix<double, Dynamic, Dynamic>;
  VERIFY_IS_EQUAL(internal::householder_qr_panel_width<double>(64, 64), Index(64));
  VERIFY_IS_EQUAL(internal::householder_qr_panel_width<double>(1024, 16), Index(16));
  // A packet wider than every candidate width falls back to one packet per panel.
  if (internal::packet_traits<double>::size <= 48) {
    VERIFY(internal::householder_qr_panel_width<double>(65, 64) < 64);
    VERIFY(internal::householder_qr_panel_width<double>(64, 65) < 64);
  }
  // Blocked widths are whole packets: with 16-float packets (AVX-512) that rules out the 8 and 24 these shapes get for
  // double.
  const Index packetShapes[][2] = {{65, 64}, {3000, 40}, {300, 300}};
  for (const auto& shape : packetShapes) {
    const Index width = internal::householder_qr_panel_width<float>(shape[0], shape[1]);
    VERIFY_IS_EQUAL(width % Index(internal::packet_traits<float>::size), Index(0));
  }
  const Index shapes[][2] = {{64, 64}, {65, 64}, {64, 65}, {1024, 16}, {3000, 40}, {300, 300}, {530, 520}, {100, 4000}};
  for (const auto& shape : shapes) qr_check_thin_factors(MatrixType(MatrixType::Random(shape[0], shape[1])));

  // Zero columns at panel edges give tau = 0 reflectors, i.e. zero diagonal entries of T. The repeated column ends the
  // second panel rank-deficient: |R(2b-1, 2b-1)| = O(eps), and its reflector is built from rounding error.
  MatrixType a = MatrixType::Random(300, 300);
  const Index b = internal::householder_qr_panel_width<double>(300, 300);
  a.col(0).setZero();
  a.col(b - 1).setZero();
  a.col(b).setZero();
  a.col(2 * b - 1) = a.col(2 * b - 2);
  qr_check_thin_factors(a);
}

template <typename MatrixType>
void qr_verify_assert() {
  MatrixType tmp;

  HouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp))
  VERIFY_RAISES_ASSERT(qr.transpose().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.adjoint().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.householderQ())
  VERIFY_RAISES_ASSERT(qr.determinant())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.signDeterminant())
}

EIGEN_DECLARE_TEST(qr) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(
        qr(MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2(qr(MatrixXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));
    CALL_SUBTEST_3((qr_fixedsize<Matrix<float, 3, 4>, 2>()));
    CALL_SUBTEST_4((qr_fixedsize<Matrix<double, 6, 2>, 4>()));
    CALL_SUBTEST_5((qr_fixedsize<Matrix<double, 2, 5>, 7>()));
    CALL_SUBTEST_11(qr(Matrix<float, 1, 1>()));
  }

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(qr_invertible<MatrixXf>());
    CALL_SUBTEST_6(qr_invertible<MatrixXd>());
    CALL_SUBTEST_7(qr_invertible<MatrixXcf>());
    CALL_SUBTEST_8(qr_invertible<MatrixXcd>());
  }

  CALL_SUBTEST_9(qr_verify_assert<Matrix3f>());
  CALL_SUBTEST_10(qr_verify_assert<Matrix3d>());
  CALL_SUBTEST_1(qr_verify_assert<MatrixXf>());
  CALL_SUBTEST_6(qr_verify_assert<MatrixXd>());
  CALL_SUBTEST_7(qr_verify_assert<MatrixXcf>());
  CALL_SUBTEST_8(qr_verify_assert<MatrixXcd>());

  // Test problem size constructors
  CALL_SUBTEST_12(HouseholderQR<MatrixXf>(10, 20));

  CALL_SUBTEST_13(qr_blocking_shapes<0>());
}
