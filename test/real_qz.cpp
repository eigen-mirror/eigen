// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Alexey Korepanov <kaikaikai@yandex.ru>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>

template <typename MatrixType>
void real_qz(const MatrixType& m) {
  /* this test covers the following files:
     RealQZ.h
  */
  using std::abs;

  Index dim = m.cols();

  MatrixType A = MatrixType::Random(dim, dim), B = MatrixType::Random(dim, dim);

  // Regression test for bug 985: Randomly set rows or columns to zero
  Index k = internal::random<Index>(0, dim - 1);
  switch (internal::random<int>(0, 10)) {
    case 0:
      A.row(k).setZero();
      break;
    case 1:
      A.col(k).setZero();
      break;
    case 2:
      B.row(k).setZero();
      break;
    case 3:
      B.col(k).setZero();
      break;
    default:
      break;
  }

  RealQZ<MatrixType> qz(dim);
  qz.compute(A, B);

  VERIFY_IS_EQUAL(qz.info(), Success);
  // check for zeros
  bool all_zeros = true;
  for (Index i = 0; i < A.cols(); i++)
    for (Index j = 0; j < i; j++) {
      if (!numext::is_exactly_zero(abs(qz.matrixT()(i, j)))) {
        std::cerr << "Error: T(" << i << "," << j << ") = " << qz.matrixT()(i, j) << std::endl;
        all_zeros = false;
      }
      if (j < i - 1 && !numext::is_exactly_zero(abs(qz.matrixS()(i, j)))) {
        std::cerr << "Error: S(" << i << "," << j << ") = " << qz.matrixS()(i, j) << std::endl;
        all_zeros = false;
      }
      if (j == i - 1 && j > 0 && !numext::is_exactly_zero(abs(qz.matrixS()(i, j))) &&
          !numext::is_exactly_zero(abs(qz.matrixS()(i - 1, j - 1)))) {
        std::cerr << "Error: S(" << i << "," << j << ") = " << qz.matrixS()(i, j) << " && S(" << i - 1 << "," << j - 1
                  << ") = " << qz.matrixS()(i - 1, j - 1) << std::endl;
        all_zeros = false;
      }
    }
  VERIFY_IS_EQUAL(all_zeros, true);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixS() * qz.matrixZ(), A);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixT() * qz.matrixZ(), B);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixQ().adjoint(), MatrixType::Identity(dim, dim));
  VERIFY_IS_APPROX(qz.matrixZ() * qz.matrixZ().adjoint(), MatrixType::Identity(dim, dim));
}

template <typename MatrixType>
void verify_real_qz(const MatrixType& a, const MatrixType& b, const RealQZ<MatrixType>& qz) {
  using Scalar = typename MatrixType::Scalar;
  // Accumulated rounding in the orthogonal transformations is O(n eps).
  const Scalar tolerance = Scalar(128 * a.rows()) * NumTraits<Scalar>::epsilon();
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY((a - qz.matrixQ() * qz.matrixS() * qz.matrixZ()).norm() <= tolerance * a.norm());
  VERIFY((b - qz.matrixQ() * qz.matrixT() * qz.matrixZ()).norm() <= tolerance * b.norm());
  VERIFY(qz.matrixQ().isUnitary(tolerance));
  VERIFY(qz.matrixZ().isUnitary(tolerance));
  // T is upper triangular and S quasi-triangular, with T diagonal on each 2x2 diagonal block of S.
  VERIFY(qz.matrixT().template triangularView<StrictlyLower>().toDenseMatrix().isZero(0));
  for (Index j = 0; j + 2 < a.cols(); ++j) VERIFY(qz.matrixS().col(j).tail(a.rows() - j - 2).isZero(0));
  for (Index i = 0; i + 1 < a.rows(); ++i) {
    if (qz.matrixS()(i + 1, i) == Scalar(0)) continue;
    VERIFY_IS_EQUAL(qz.matrixT()(i, i + 1), Scalar(0));
    if (i + 2 < a.rows()) VERIFY_IS_EQUAL(qz.matrixS()(i + 2, i + 1), Scalar(0));
  }
}

// On a 3x3 all-zero pencil every pass through the iteration took the pushDownZero branch without progress; when that
// branch did not count towards the iteration limit, compute() never returned. It then reported NoConvergence: an exact
// zero on the subdiagonal of S did not deflate when |S(k-1, k-1)| + |S(k, k)| and ||S|| were both 0. For A = 0 and
// n >= 3 that also failed with a nonzero B, and GeneralizedEigenSolver's eigenvalues with it. Both pencils are already
// in generalized Schur form.
template <typename MatrixType>
void real_qz_zero_pencil() {
  for (Index dim : {1, 2, 3, 5}) {
    const MatrixType zero = MatrixType::Zero(dim, dim), random = MatrixType::Random(dim, dim);
    for (const MatrixType* b : {&zero, &random}) {
      RealQZ<MatrixType> qz(zero, *b);
      verify_real_qz(zero, *b, qz);
    }
    GeneralizedEigenSolver<MatrixType> eig(zero, random, false);
    VERIFY_IS_EQUAL(eig.info(), Success);
    VERIFY((eig.alphas().array() == typename MatrixType::Scalar(0)).all());
  }
}

// A pencil that never deflates must still return: every pass through the iteration either deflates or counts towards
// the limit, chases of a zero on the diagonal of T included. With B = 0 every pass takes that branch.
template <typename MatrixType>
void real_qz_nan_pencil_terminates() {
  using Scalar = typename MatrixType::Scalar;
  for (Index n : {3, 5}) {
    MatrixType a = MatrixType::Random(n, n);
    a(n - 1, 0) = NumTraits<Scalar>::quiet_NaN();
    const MatrixType b = MatrixType::Zero(n, n);
    RealQZ<MatrixType> qz(a, b);
    VERIFY(qz.info() != Success);
  }
}

// setMaxIterations() bounds the steps between deflations. A 1x1 or 2x2 pencil needs no step, so a limit of 0 reduces
// it. A 3x3 pencil with T(0, 0) = 0 needs one chase of that zero, which deflates the infinite eigenvalue, and no
// QR-like step: a limit of 1 reduces it. The limit used to be checked before that deflation was found.
template <typename MatrixType>
void real_qz_iteration_limit() {
  for (Index n : {1, 2}) {
    const MatrixType a = MatrixType::Random(n, n), b = MatrixType::Random(n, n);
    RealQZ<MatrixType> qz(n);
    qz.setMaxIterations(0);
    qz.compute(a, b);
    verify_real_qz(a, b, qz);
  }
  // Already Hessenberg-triangular, so the reduction keeps T(0, 0) exactly 0.
  MatrixType a = MatrixType::Random(3, 3), b = MatrixType::Random(3, 3).template triangularView<Upper>();
  a(2, 0) = 0;
  b(0, 0) = 0;
  RealQZ<MatrixType> qz(3);
  qz.setMaxIterations(1);
  qz.compute(a, b);
  verify_real_qz(a, b, qz);
  VERIFY_IS_EQUAL(qz.iterations(), 0);
}

// A and B are scaled by powers of two near the ends of the exponent range, which made the iteration overflow or
// underflow (#3212). The scaling must leave Q and Z, and S and T up to the same powers of two, unchanged.
template <typename MatrixType>
void real_qz_extreme_scaling() {
  using Scalar = typename MatrixType::Scalar;
  const Index dim = 6;
  // Entries of magnitude in [1/3, 1], so that no scaled entry becomes subnormal.
  const auto awayFromZero = [](const Scalar& x) { return (x < 0 ? x - Scalar(0.5) : x + Scalar(0.5)) / Scalar(1.5); };
  const MatrixType a0 = MatrixType::Random(dim, dim).unaryExpr(awayFromZero);
  const MatrixType b0 = MatrixType::Random(dim, dim).unaryExpr(awayFromZero);
  const int big = std::numeric_limits<Scalar>::max_exponent - 16;
  const Scalar tolerance = Scalar(128 * dim) * NumTraits<Scalar>::epsilon();
  for (int ea : {0, big, -big}) {
    for (int eb : {0, big, -big}) {
      const MatrixType A = a0 * numext::ldexp(Scalar(1), ea), B = b0 * numext::ldexp(Scalar(1), eb);
      RealQZ<MatrixType> qz(A, B);
      VERIFY_IS_EQUAL(qz.info(), Success);
      const MatrixType S = qz.matrixS() * numext::ldexp(Scalar(1), -ea);
      const MatrixType T = qz.matrixT() * numext::ldexp(Scalar(1), -eb);
      VERIFY(S.allFinite() && T.allFinite());
      VERIFY((qz.matrixQ() * S * qz.matrixZ() - a0).norm() <= tolerance * a0.norm());
      VERIFY((qz.matrixQ() * T * qz.matrixZ() - b0).norm() <= tolerance * b0.norm());
    }
  }
}

// RealQZ returned Success with inf in S or T where Q^T A Z or Q^T B Z is not representable. For A or B = c 1 with
// c = 0.9 highest, that triangular factor has rank one and Frobenius norm c n, so its nonzeros form a p x q block with
// p + q <= n + 1, one of them at least 2 c n / (n + 1) > highest: info() must report NumericalIssue.
template <typename MatrixType>
void real_qz_unrepresentable() {
  using Scalar = typename MatrixType::Scalar;
  const Scalar c = Scalar(0.9) * NumTraits<Scalar>::highest();
  for (Index n : {2, 10}) {
    const MatrixType ones = MatrixType::Constant(n, n, c), random = MatrixType::Random(n, n);
    RealQZ<MatrixType> qzA(ones, random), qzB(random, ones);
    VERIFY_IS_EQUAL(qzA.info(), NumericalIssue);
    VERIFY_IS_EQUAL(qzB.info(), NumericalIssue);
    const MatrixType big = random * c;
    RealQZ<MatrixType> qzBig(big, random);
    VERIFY(qzBig.info() != Success || (qzBig.matrixS().allFinite() && qzBig.matrixT().allFinite()));
  }
}

template <typename MatrixType>
void verify_real_qz_backward_error(const MatrixType& a, const MatrixType& b) {
  using Scalar = typename MatrixType::Scalar;
  // Accumulated rounding in the orthogonal transformations is O(n eps).
  const Scalar tolerance = Scalar(128 * a.rows()) * NumTraits<Scalar>::epsilon();
  const RealQZ<MatrixType> qz(a, b);
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY((a - qz.matrixQ() * qz.matrixS() * qz.matrixZ()).norm() <= tolerance * a.norm());
  VERIFY((b - qz.matrixQ() * qz.matrixT() * qz.matrixZ()).norm() <= tolerance * b.norm());
  VERIFY(qz.matrixQ().isUnitary(tolerance));
  VERIFY(qz.matrixZ().isUnitary(tolerance));
}

// RealQZ split a 2x2 block with real eigenvalues through S T^-1: with a diagonal entry delta of T in the block, the
// backward error grew like eps / delta, to 1e-3 relative for double at delta = 1e-12. The 3x3 pencils first move
// T(0, 0) = 0 to the bottom. A nearly scalar block a = lambda0 b + O(g) has a close pair of eigenvalues, which a
// discriminant formed without a shift resolves only to O(eps / g): the split must keep both its accuracy and its
// classification as real or complex. The backward error must stay O(n eps).
template <typename MatrixType>
void real_qz_ill_conditioned_block() {
  using Scalar = typename MatrixType::Scalar;
  using Complex = std::complex<Scalar>;
  const Scalar eps = NumTraits<Scalar>::epsilon();
  for (const Scalar delta : {numext::sqrt(eps), Scalar(1024) * eps}) {
    for (int kind = 0; kind < 3; ++kind) {
      const Index n = kind < 2 ? 2 : 3;
      for (int repeat = 0; repeat < 20; ++repeat) {
        MatrixType a = MatrixType::Random(n, n), b = MatrixType::Random(n, n).template triangularView<Upper>();
        if (kind == 0) b(1, 1) = delta;
        if (kind == 1) b(0, 0) = delta;
        if (kind == 2) {
          a(2, 0) = Scalar(0);
          b(0, 0) = Scalar(0);
          b(2, 2) = delta;
        }
        verify_real_qz_backward_error(a, b);
      }
    }
  }

  // b = I: a = [1 t; s t 1] is already Hessenberg-triangular, with the eigenvalues 1 +- t (s = 1, split) or 1 +- i t
  // (s = -1). The pencil is normal, so the eigenvalue error is bounded by the backward error: a few eps.
  for (const Scalar t : {Scalar(1e-2), numext::sqrt(eps), Scalar(16) * eps}) {
    for (const int s : {1, -1}) {
      MatrixType a(2, 2);
      a << Scalar(1), t, Scalar(s) * t, Scalar(1);
      const MatrixType b = MatrixType::Identity(2, 2);
      verify_real_qz_backward_error(a, b);
      VERIFY_IS_EQUAL(numext::is_exactly_zero(RealQZ<MatrixType>(a, b).matrixS()(1, 0)), s > 0);
      const Complex shift = s > 0 ? Complex(t, 0) : Complex(0, t);
      const Complex plus = Scalar(1) + shift, minus = Scalar(1) - shift;
      const GeneralizedEigenSolver<MatrixType> solver(a, b, false);
      const auto lambda = solver.eigenvalues().eval();
      const Scalar error = numext::mini(numext::maxi(numext::abs(lambda(0) - plus), numext::abs(lambda(1) - minus)),
                                        numext::maxi(numext::abs(lambda(0) - minus), numext::abs(lambda(1) - plus)));
      VERIFY(error <= Scalar(16) * eps);
    }
  }

  // a = lambda0 b + g E for a random E and a triangular b whose diagonal entries have random signs and moduli in
  // delta [1, 2]: the largest entry of b is on its diagonal for delta = 1, often off it for delta = 1/4, and off it for
  // delta = sqrt(eps).
  for (const Scalar delta : {Scalar(1), Scalar(0.25), numext::sqrt(eps)}) {
    for (const Scalar g : {numext::sqrt(eps), Scalar(16) * eps}) {
      for (int repeat = 0; repeat < 20; ++repeat) {
        MatrixType b = MatrixType::Random(2, 2).template triangularView<Upper>();
        for (Index k = 0; k < 2; ++k)
          b(k, k) = delta * (internal::random<bool>() ? Scalar(1) : Scalar(-1)) *
                    internal::random<Scalar>(Scalar(1), Scalar(2));
        const Scalar lambda0 = internal::random<Scalar>(Scalar(0.5), Scalar(1.5));
        verify_real_qz_backward_error(MatrixType(lambda0 * b + g * MatrixType::Random(2, 2)), b);
      }
    }
  }

  // a = [1 0; 1/2 1], b = I: a double eigenvalue 1, which the split leaves exact.
  MatrixType a(2, 2);
  a << Scalar(1), Scalar(0), Scalar(0.5), Scalar(1);
  const RealQZ<MatrixType> qz(a, MatrixType::Identity(2, 2));
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY_IS_EQUAL(qz.matrixS()(1, 0), Scalar(0));
  VERIFY_IS_EQUAL(qz.matrixS()(0, 0), qz.matrixT()(0, 0));
  VERIFY_IS_EQUAL(qz.matrixS()(1, 1), qz.matrixT()(1, 1));
}

// The split of a block with real eigenvalues puts the eigenvalue near (S T^-1)(0, 0) on top: here the eigenvalues are
// about top = a(0, 0) / b(0, 0) and 1, for every sign of the diagonal of b and of b(0, 1), with the largest entry of b
// on or off its diagonal.
template <typename MatrixType>
void real_qz_split_order() {
  using Scalar = typename MatrixType::Scalar;
  for (int k = 0; k < 16; ++k) {
    const Scalar s0 = (k & 1) ? Scalar(-1) : Scalar(1), s1 = (k & 2) ? Scalar(-1) : Scalar(1);
    const Scalar b01 = ((k & 4) ? Scalar(-1) : Scalar(1)) * ((k & 8) ? Scalar(1) : Scalar(0.25));
    for (const Scalar top : {Scalar(2), Scalar(-2)}) {
      MatrixType a(2, 2), b(2, 2);
      b << s0 * Scalar(0.5), b01, Scalar(0), s1 * Scalar(0.5);
      a << top * b(0, 0), Scalar(0.3), Scalar(1e-3), b(1, 1);
      const RealQZ<MatrixType> qz(a, b);
      VERIFY_IS_EQUAL(qz.info(), Success);
      VERIFY_IS_EQUAL(qz.matrixS()(1, 0), Scalar(0));
      const Scalar lambda = qz.matrixS()(0, 0) / qz.matrixT()(0, 0);
      VERIFY(numext::abs(lambda - top) < numext::abs(lambda - Scalar(1)));
    }
  }
}

// Block Householder operations still allocate at larger sizes (issue #3163).
template <typename MatrixType>
void real_qz_no_malloc() {
  const Index dim = 24;
  MatrixType A = MatrixType::Random(dim, dim), B = MatrixType::Random(dim, dim);
  RealQZ<MatrixType> qz(dim);
  internal::set_is_malloc_allowed(false);
  qz.compute(A, B, false);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(qz.info(), Success);
  const MatrixType S = qz.matrixS(), T = qz.matrixT();

  for (int i = 0; i < 2; ++i) {
    internal::set_is_malloc_allowed(false);
    qz.compute(A, B, true);
    internal::set_is_malloc_allowed(true);
    VERIFY_IS_EQUAL(qz.info(), Success);
    VERIFY_IS_APPROX(qz.matrixS(), S);
    VERIFY_IS_APPROX(qz.matrixT(), T);
    VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixS() * qz.matrixZ(), A);
    VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixT() * qz.matrixZ(), B);
    VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixQ().adjoint(), MatrixType::Identity(dim, dim));
    VERIFY_IS_APPROX(qz.matrixZ() * qz.matrixZ().adjoint(), MatrixType::Identity(dim, dim));
  }
}

EIGEN_DECLARE_TEST(real_qz) {
  CALL_SUBTEST_2((real_qz_no_malloc<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_extreme_scaling<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_extreme_scaling<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_unrepresentable<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_unrepresentable<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_ill_conditioned_block<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_ill_conditioned_block<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_split_order<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_split_order<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_zero_pencil<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_zero_pencil<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_nan_pencil_terminates<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_nan_pencil_terminates<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_iteration_limit<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_iteration_limit<MatrixXd>()));
  CALL_SUBTEST_5((real_qz_no_malloc<Matrix<float, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_6((real_qz_no_malloc<Matrix<double, Dynamic, Dynamic, RowMajor>>()));

  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(real_qz(Matrix4f()));
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    CALL_SUBTEST_2(real_qz(MatrixXd(s, s)));

    // some trivial but implementation-wise tricky cases
    CALL_SUBTEST_2(real_qz(MatrixXd(1, 1)));
    CALL_SUBTEST_2(real_qz(MatrixXd(2, 2)));
    CALL_SUBTEST_3(real_qz(Matrix<double, 1, 1>()));
    CALL_SUBTEST_4(real_qz(Matrix2d()));
    CALL_SUBTEST_5((real_qz(Matrix<float, Dynamic, Dynamic, RowMajor>(17, 17))));
    CALL_SUBTEST_6((real_qz(Matrix<double, Dynamic, Dynamic, RowMajor>(17, 17))));
  }

  TEST_SET_BUT_UNUSED_VARIABLE(s);
}
