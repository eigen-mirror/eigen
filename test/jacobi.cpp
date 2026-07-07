// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/SVD>

template <typename MatrixType, typename JacobiScalar>
void jacobi(const MatrixType& m = MatrixType()) {
  Index rows = m.rows();
  Index cols = m.cols();

  enum { RowsAtCompileTime = MatrixType::RowsAtCompileTime, ColsAtCompileTime = MatrixType::ColsAtCompileTime };

  typedef Matrix<JacobiScalar, 2, 1> JacobiVector;

  const MatrixType a(MatrixType::Random(rows, cols));

  JacobiVector v = JacobiVector::Random().normalized();
  JacobiScalar c = v.x(), s = v.y();
  JacobiRotation<JacobiScalar> rot(c, s);

  {
    Index p = internal::random<Index>(0, rows - 1);
    Index q;
    do {
      q = internal::random<Index>(0, rows - 1);
    } while (q == p);

    MatrixType b = a;
    b.applyOnTheLeft(p, q, rot);
    VERIFY_IS_APPROX(b.row(p), c * a.row(p) + numext::conj(s) * a.row(q));
    VERIFY_IS_APPROX(b.row(q), -s * a.row(p) + numext::conj(c) * a.row(q));
  }

  {
    Index p = internal::random<Index>(0, cols - 1);
    Index q;
    do {
      q = internal::random<Index>(0, cols - 1);
    } while (q == p);

    MatrixType b = a;
    b.applyOnTheRight(p, q, rot);
    VERIFY_IS_APPROX(b.col(p), c * a.col(p) - s * a.col(q));
    VERIFY_IS_APPROX(b.col(q), numext::conj(s) * a.col(p) + numext::conj(c) * a.col(q));
  }
}

// Verify that JacobiRotation::makeGivens(p, q, &r) produces a rotation that
// zeros out q, even when (p, q) straddle the over-/underflow thresholds
// where the direct formula r = p * sqrt(1 + (q/p)^2) would over- or
// underflow.  Eigen's convention is r >= 0 with sign carried in c.
template <typename Scalar>
void verify_makeGivens(const Scalar& p, const Scalar& q) {
  using std::abs;
  Scalar r;
  JacobiRotation<Scalar> rot;
  rot.makeGivens(p, q, &r);

  // Eigen's J^T * [p; q] = [r; 0] with J = [c s; -s c], so:
  //   c*p - s*q = r,  s*p + c*q = 0.
  Scalar rotated0 = rot.c() * p - rot.s() * q;
  Scalar rotated1 = rot.s() * p + rot.c() * q;

  // The check itself performs two rounded products and an addition/subtraction, sometimes at the safe-scaling
  // overflow threshold. Keep the tolerance relative to r, but leave enough room for compiler-specific contraction and
  // reassociation in the verification expression.
  Scalar tol = NumTraits<Scalar>::epsilon() * (abs(r) + (std::numeric_limits<Scalar>::min)()) * Scalar(64);
  VERIFY(abs(rotated0 - r) <= tol);
  VERIFY(abs(rotated1) <= tol);
  VERIFY(r >= Scalar(0));
  VERIFY_IS_APPROX(numext::abs2(rot.c()) + numext::abs2(rot.s()), Scalar(1));
}

template <typename Scalar>
void jacobi_makegivens_safe_scaling() {
  using std::sqrt;
  const Scalar safmin = (std::numeric_limits<Scalar>::min)();
  const Scalar safmax = Scalar(1) / safmin;
  const Scalar rtmin = sqrt(safmin);
  const Scalar rtmax = sqrt(safmax / Scalar(2));
  const Scalar one(1);
  const Scalar two(2);
  const Scalar half(0.5);

  // Safe-range cases (regression — must keep existing fast path working).
  verify_makeGivens<Scalar>(Scalar(3), Scalar(4));
  verify_makeGivens<Scalar>(Scalar(-3), Scalar(4));
  verify_makeGivens<Scalar>(Scalar(3), Scalar(-4));
  verify_makeGivens<Scalar>(Scalar(-3), Scalar(-4));

  // Both inputs near overflow: direct formula r = p * sqrt(1+(q/p)^2) would
  // overflow because sqrt(1+1) > 1.  Prescaling avoids this.
  verify_makeGivens<Scalar>(rtmax * two, rtmax);
  verify_makeGivens<Scalar>(-rtmax * two, rtmax);
  verify_makeGivens<Scalar>(rtmax, rtmax);
  verify_makeGivens<Scalar>(rtmax * Scalar(1.5), rtmax * Scalar(1.5));

  // Both inputs near underflow / subnormal: direct (q/p)^2 underflows to 0.
  verify_makeGivens<Scalar>(rtmin * half, rtmin * half);
  verify_makeGivens<Scalar>(safmin, safmin);
  verify_makeGivens<Scalar>(-safmin, safmin);

  // Mixed: one near overflow, one normal.
  verify_makeGivens<Scalar>(rtmax * Scalar(1.5), one);
  verify_makeGivens<Scalar>(one, rtmax * Scalar(1.5));
  verify_makeGivens<Scalar>(-rtmax * Scalar(1.5), one);

  // Mixed: one near underflow, one normal.
  verify_makeGivens<Scalar>(safmin, one);
  verify_makeGivens<Scalar>(one, safmin);

  // Mixed: subnormal and near-overflow simultaneously.
  verify_makeGivens<Scalar>(safmin, rtmax);
  verify_makeGivens<Scalar>(rtmax, safmin);
}

EIGEN_DECLARE_TEST(jacobi) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_7((jacobi_makegivens_safe_scaling<float>()));
    CALL_SUBTEST_7((jacobi_makegivens_safe_scaling<double>()));

    CALL_SUBTEST_1((jacobi<Matrix3f, float>()));
    CALL_SUBTEST_2((jacobi<Matrix4d, double>()));
    CALL_SUBTEST_3((jacobi<Matrix4cf, float>()));
    CALL_SUBTEST_3((jacobi<Matrix4cf, std::complex<float> >()));

    CALL_SUBTEST_1((jacobi<Matrix<float, 3, 3, RowMajor>, float>()));
    CALL_SUBTEST_2((jacobi<Matrix<double, 4, 4, RowMajor>, double>()));
    CALL_SUBTEST_3((jacobi<Matrix<std::complex<float>, 4, 4, RowMajor>, float>()));
    CALL_SUBTEST_3((jacobi<Matrix<std::complex<float>, 4, 4, RowMajor>, std::complex<float> >()));

    int r = internal::random<int>(2, internal::random<int>(1, EIGEN_TEST_MAX_SIZE) / 2),
        c = internal::random<int>(2, internal::random<int>(1, EIGEN_TEST_MAX_SIZE) / 2);
    CALL_SUBTEST_4((jacobi<MatrixXf, float>(MatrixXf(r, c))));
    CALL_SUBTEST_5((jacobi<MatrixXcd, double>(MatrixXcd(r, c))));
    CALL_SUBTEST_5((jacobi<MatrixXcd, std::complex<double> >(MatrixXcd(r, c))));
    // complex<float> is really important to test as it is the only way to cover conjugation issues in certain unaligned
    // paths
    CALL_SUBTEST_6((jacobi<MatrixXcf, float>(MatrixXcf(r, c))));
    CALL_SUBTEST_6((jacobi<MatrixXcf, std::complex<float> >(MatrixXcf(r, c))));

    TEST_SET_BUT_UNUSED_VARIABLE(r);
    TEST_SET_BUT_UNUSED_VARIABLE(c);
  }
}
