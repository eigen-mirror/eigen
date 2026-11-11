// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
// Copyright (C) 2012 desire Nuentsa <desire.nuentsa_wakam@inria.fr
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// FIXME: These tests all check for hard-coded values. Ideally, parameters and start estimates should be randomized.

#include <stdio.h>

#include "main.h"
#include <contrib/Eigen/LevenbergMarquardt>

// This disables some useless Warnings on MSVC.
// It is intended to be done for this test only.
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

// tolerance for checking number of iterations
#define LM_EVAL_COUNT_TOL 2

#define EIGEN_TEST_LM_DENSE_FUNCTOR_BASE
#include "lm_test_functors.h"

struct NistAdapter {
  template <typename Solver>
  static const VectorXd &fvec(Solver &lm) {
    return lm.fvec();
  }
  template <typename Solver>
  static void setFtol(Solver &lm, double value) {
    lm.setFtol(value);
  }
  template <typename Solver>
  static void setXtol(Solver &lm, double value) {
    lm.setXtol(value);
  }
  template <typename Solver>
  static void setFactor(Solver &lm, double value) {
    lm.setFactor(value);
  }
  template <typename Solver>
  static void setMaxfev(Solver &lm, int value) {
    lm.setMaxfev(value);
  }
  template <typename Solver>
  static void check_counts(const Solver &, int, int) {}
  template <typename Solver>
  static void check_counts_lm(const Solver &lm, int nfev, int njev) {
    VERIFY(lm.nfev() < nfev * LM_EVAL_COUNT_TOL);
    VERIFY(lm.njev() < njev * LM_EVAL_COUNT_TOL);
  }
  // The exit must pass the xtol test, which on a rejected last step needs xnorm carried over from the last
  // accepted one. Whether the ftol test passes too depends on actred / prered with both at roundoff level
  // (~1e-14): x86 exits with 3, ARMv7 NEON and SVE with 2.
  static void check_xtol_exit(LevenbergMarquardtSpace::Status info) {
    VERIFY(info == LevenbergMarquardtSpace::RelativeErrorTooSmall ||
           info == LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall);
  }
};

#include "lm_nist_tests.h"

void testLmder1() {
  int n = 3, info;

  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.lmder1(x);
  EIGEN_UNUSED_VARIABLE(info);  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 6);
  // VERIFY_IS_EQUAL(lm.njev(), 5);

  // check norm
  VERIFY_IS_APPROX(lm.fvec().blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmder() {
  const int m = 15, n = 3;
  int info;
  double fnorm, covfac;
  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return values
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 6);
  // VERIFY_IS_EQUAL(lm.njev(), 5);

  // check norm
  fnorm = lm.fvec().blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covfac = fnorm * fnorm / (m - n);
  internal::covar(lm.matrixR(), lm.permutation().indices());  // TODO : move this as a function of lm

  MatrixXd cov_ref(n, n);
  cov_ref << 0.0001531202, 0.002869941, -0.002656662, 0.002869941, 0.09480935, -0.09098995, -0.002656662, -0.09098995,
      0.08778727;

  //  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov = covfac * lm.matrixR().topLeftCorner<n, n>();
  VERIFY_IS_APPROX(cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

void testLmdif1() {
  const int n = 3;
  int info;

  VectorXd x(n), fvec(15);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmdif_functor functor;
  DenseIndex nfev;
  info = LevenbergMarquardt<lmdif_functor>::lmdif1(functor, x, &nfev);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  //   VERIFY_IS_EQUAL(nfev, 26);

  // check norm
  functor(x, fvec);
  VERIFY_IS_APPROX(fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.0824106, 1.1330366, 2.3436947;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmdif() {
  const int m = 15, n = 3;
  int info;
  double fnorm, covfac;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmdif_functor functor;
  NumericalDiff<lmdif_functor> numDiff(functor);
  LevenbergMarquardt<NumericalDiff<lmdif_functor> > lm(numDiff);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return values
  // VERIFY_IS_EQUAL(info, 1);
  //   VERIFY_IS_EQUAL(lm.nfev(), 26);

  // check norm
  fnorm = lm.fvec().blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covfac = fnorm * fnorm / (m - n);
  internal::covar(lm.matrixR(), lm.permutation().indices());  // TODO : move this as a function of lm

  MatrixXd cov_ref(n, n);
  cov_ref << 0.0001531202, 0.002869942, -0.002656662, 0.002869942, 0.09480937, -0.09098997, -0.002656662, -0.09098997,
      0.08778729;

  //  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov = covfac * lm.matrixR().topLeftCorner<n, n>();
  VERIFY_IS_APPROX(cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

// covar() takes the rank to end at the first pivot below tol * |r(0,0)| (MINPACK) and zeroes the rest;
// a NaN pivot is not negligible, so it reaches the result instead of reading as zero variance.
void testCovarRankDeficient() {
  MatrixXd r(3, 3);
  r << 2, 1, 1, 0, 1e-20, 1, 0, 0, 3;
  VectorXi ipvt(3);
  ipvt << 0, 1, 2;
  internal::covar(r, ipvt);
  MatrixXd expected = MatrixXd::Zero(3, 3);
  expected(0, 0) = 0.25;
  VERIFY_IS_APPROX(r, expected);

  r << 2, 1, 1, 0, NumTraits<double>::quiet_NaN(), 1, 0, 0, 3;
  internal::covar(r, ipvt);
  VERIFY(r.hasNaN());
}

// lmqrsolv() must minimize ||[R; D] z - [qtb; 0]|| and leave the eliminated
// factor in the strict lower triangle of s for lmpar2(). Issue #657: the QR is
// rank-revealing, so R carries zeros above the diagonal, and the Givens
// rotation that eliminates a row of D acts on the whole row: it rotates sdiag
// against those zeros and fills them in.
void testLmqrsolv() {
  const Index n = 6;

  // An upper triangular factor with zeros above the diagonal.
  MatrixXd r = MatrixXd::Zero(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = i; j < n; ++j)
      r(i, j) = ((i == 0 && j == 3) || (i == 1 && j == 4)) ? 0.0 : 1.0 + 0.5 * double(i + 1) * double(j + 1);

  VectorXd diag(n), qtb(n);
  for (Index i = 0; i < n; ++i) {
    diag(i) = 0.5 + 0.25 * double(i);
    qtb(i) = 1.0 - 0.3 * double(i);
  }

  PermutationMatrix<Dynamic, Dynamic, int> perm(n);
  perm.setIdentity();

  MatrixXd s = r;
  VectorXd x(n), sdiag(n);
  internal::lmqrsolv(s, perm, diag, qtb, x, sdiag);

  MatrixXd augmented(2 * n, n);
  augmented.topRows(n) = r;
  augmented.bottomRows(n) = diag.asDiagonal();
  VectorXd rhs = VectorXd::Zero(2 * n);
  rhs.head(n) = qtb;
  VERIFY_IS_APPROX(x, augmented.colPivHouseholderQr().solve(rhs));

  // The upper triangle and the diagonal of s are restored. lmpar2() reads the
  // eliminated factor S out of the strict lower triangle of s, transposed, with
  // its diagonal in sdiag; S is the triangular factor of [R; D].
  VERIFY_IS_APPROX(MatrixXd(s.triangularView<Upper>()), r);
  MatrixXd eliminated = MatrixXd(s.triangularView<StrictlyLower>()).transpose();
  eliminated.diagonal() = sdiag;
  VERIFY_IS_APPROX(MatrixXd(eliminated.transpose() * eliminated),
                   MatrixXd(r.transpose() * r + MatrixXd(diag.cwiseAbs2().asDiagonal())));
}

// Exponential decay with two exactly collinear offsets x2 and x3, so the
// Jacobian is rank deficient and the triangular factor of its sparse QR loses
// its last diagonal entry. Issue #657: the sparse solver then asserted inside
// Eigen::internal::sparse_solve_triangular_selector.
struct collinear_offsets_functor : SparseFunctor<double, int> {
  collinear_offsets_functor(int m) : SparseFunctor<double, int>(4, m), m_t(m), m_y(m) {
    for (int i = 0; i < m; ++i) {
      m_t(i) = 0.05 * i;
      m_y(i) = 2.5 * std::exp(-0.7 * m_t(i)) + 0.3;
    }
  }
  int operator()(const VectorXd &x, VectorXd &fvec) {
    for (int i = 0; i < values(); ++i) fvec(i) = x(0) * std::exp(-x(1) * m_t(i)) + x(2) + x(3) - m_y(i);
    return 0;
  }
  int df(const VectorXd &x, JacobianType &jac) {
    std::vector<Triplet<double> > triplets;
    for (int i = 0; i < values(); ++i) {
      const double e = std::exp(-x(1) * m_t(i));
      triplets.push_back(Triplet<double>(i, 0, e));
      triplets.push_back(Triplet<double>(i, 1, -x(0) * m_t(i) * e));
      triplets.push_back(Triplet<double>(i, 2, 1.0));
      triplets.push_back(Triplet<double>(i, 3, 1.0));
    }
    jac.resize(values(), inputs());
    jac.setFromTriplets(triplets.begin(), triplets.end());
    return 0;
  }
  VectorXd m_t, m_y;
};

void testSparseFunctor() {
  collinear_offsets_functor functor(30);
  LevenbergMarquardt<collinear_offsets_functor> lm(functor);
  VectorXd x(4);
  x << 5.0, 5.0, 5.0, 5.0;
  const LevenbergMarquardtSpace::Status info = lm.minimize(x);
  VERIFY(info > LevenbergMarquardtSpace::ImproperInputParameters &&
         info < LevenbergMarquardtSpace::TooManyFunctionEvaluation);

  // x2 and x3 are determined only through their sum.
  VERIFY_IS_APPROX(x(0), 2.5);
  VERIFY_IS_APPROX(x(1), 0.7);
  VERIFY_IS_APPROX(x(2) + x(3), 0.3);
  VERIFY(lm.fnorm() <= 1e3 * NumTraits<double>::epsilon() * functor.m_y.norm());
}

EIGEN_DECLARE_TEST(levenberg_marquardt) {
  CALL_SUBTEST(testLmqrsolv());
  CALL_SUBTEST(testCovarRankDeficient());
  CALL_SUBTEST(testSparseFunctor());

  // Tests using the examples provided by (c)minpack
  CALL_SUBTEST(testLmder1());
  CALL_SUBTEST(testLmder());
  CALL_SUBTEST(testLmdif1());
  //     CALL_SUBTEST(testLmstr1());
  //     CALL_SUBTEST(testLmstr());
  CALL_SUBTEST(testLmdif());

  // NIST tests, level of difficulty = "Lower"
  CALL_SUBTEST(testNistMisra1a());
  CALL_SUBTEST(testNistChwirut2());

  // NIST tests, level of difficulty = "Average"
  CALL_SUBTEST(testNistHahn1());
  CALL_SUBTEST(testNistMisra1d());
  CALL_SUBTEST(testNistMGH17());
  CALL_SUBTEST(testNistLanczos1());

  //     // NIST tests, level of difficulty = "Higher"
  CALL_SUBTEST(testNistRat42());
  CALL_SUBTEST(testNistMGH10());
  CALL_SUBTEST(testNistBoxBOD());
  //     CALL_SUBTEST(testNistMGH09());
  CALL_SUBTEST(testNistBennett5());
  CALL_SUBTEST(testNistThurber());
  CALL_SUBTEST(testNistRat43());
  CALL_SUBTEST(testNistEckerle4());
}
