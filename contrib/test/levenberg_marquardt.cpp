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

using std::sqrt;

using LmFunctorBase = DenseFunctor<double>;
#include "lm_test_functors.h"

// tolerance for checking number of iterations
#define LM_EVAL_COUNT_TOL 2

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

// http://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml
void testNistChwirut2(void) {
  const int n = 3;
  LevenbergMarquardtSpace::Status info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 0.1, 0.01, 0.02;
  // do the computation
  chwirut2_functor functor;
  LevenbergMarquardt<chwirut2_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  //   VERIFY_IS_EQUAL(lm.nfev(), 10);
  // VERIFY_IS_EQUAL(lm.njev(), 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.1304802941E+02);
  // check x
  VERIFY_IS_APPROX(x[0], 1.6657666537E-01);
  VERIFY_IS_APPROX(x[1], 5.1653291286E-03);
  VERIFY_IS_APPROX(x[2], 1.2150007096E-02);

  /*
   * Second try
   */
  x << 0.15, 0.008, 0.010;
  // do the computation
  lm.resetParameters();
  lm.setFtol(1.E6 * NumTraits<double>::epsilon());
  lm.setXtol(1.E6 * NumTraits<double>::epsilon());
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  //   VERIFY_IS_EQUAL(lm.nfev(), 7);
  // VERIFY_IS_EQUAL(lm.njev(), 6);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.1304802941E+02);
  // check x
  VERIFY_IS_APPROX(x[0], 1.6657666537E-01);
  VERIFY_IS_APPROX(x[1], 5.1653291286E-03);
  VERIFY_IS_APPROX(x[2], 1.2150007096E-02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml
void testNistMisra1a(void) {
  const int n = 2;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 500., 0.0001;
  // do the computation
  misra1a_functor functor;
  LevenbergMarquardt<misra1a_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 19);
  // VERIFY_IS_EQUAL(lm.njev(), 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.2455138894E-01);
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);

  /*
   * Second try
   */
  x << 250., 0.0005;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 5);
  // VERIFY_IS_EQUAL(lm.njev(), 4);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.2455138894E-01);
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);
}

// http://www.itl.nist.gov/div898/strd/nls/data/hahn1.shtml
void testNistHahn1(void) {
  const int n = 7;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 10., -1., .05, -.00001, -.05, .001, -.000001;
  // do the computation
  hahn1_functor functor;
  LevenbergMarquardt<hahn1_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 11);
  // VERIFY_IS_EQUAL(lm.njev(), 10);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.5324382854E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 1.0776351733E+00);
  VERIFY_IS_APPROX(x[1], -1.2269296921E-01);
  VERIFY_IS_APPROX(x[2], 4.0863750610E-03);
  VERIFY_IS_APPROX(x[3], -1.426264e-06);  // should be : -1.4262662514E-06
  VERIFY_IS_APPROX(x[4], -5.7609940901E-03);
  VERIFY_IS_APPROX(x[5], 2.4053735503E-04);
  VERIFY_IS_APPROX(x[6], -1.2314450199E-07);

  /*
   * Second try
   */
  x << .1, -.1, .005, -.000001, -.005, .0001, -.0000001;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  //   VERIFY_IS_EQUAL(lm.nfev(), 11);
  // VERIFY_IS_EQUAL(lm.njev(), 10);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.5324382854E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 1.077640);       // should be :  1.0776351733E+00
  VERIFY_IS_APPROX(x[1], -0.1226933);     // should be : -1.2269296921E-01
  VERIFY_IS_APPROX(x[2], 0.004086383);    // should be : 4.0863750610E-03
  VERIFY_IS_APPROX(x[3], -1.426277e-06);  // should be : -1.4262662514E-06
  VERIFY_IS_APPROX(x[4], -5.7609940901E-03);
  VERIFY_IS_APPROX(x[5], 0.00024053772);  // should be : 2.4053735503E-04
  VERIFY_IS_APPROX(x[6], -1.231450e-07);  // should be : -1.2314450199E-07
}

// http://www.itl.nist.gov/div898/strd/nls/data/misra1d.shtml
void testNistMisra1d(void) {
  const int n = 2;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 500., 0.0001;
  // do the computation
  misra1d_functor functor;
  LevenbergMarquardt<misra1d_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 9);
  // VERIFY_IS_EQUAL(lm.njev(), 7);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.6419295283E-02);
  // check x
  VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  VERIFY_IS_APPROX(x[1], 3.0227324449E-04);

  /*
   * Second try
   */
  x << 450., 0.0003;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 4);
  // VERIFY_IS_EQUAL(lm.njev(), 3);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.6419295283E-02);
  // check x
  VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  VERIFY_IS_APPROX(x[1], 3.0227324449E-04);
}

// http://www.itl.nist.gov/div898/strd/nls/data/lanczos1.shtml
void testNistLanczos1(void) {
  const int n = 6;
  LevenbergMarquardtSpace::Status info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1.2, 0.3, 5.6, 5.5, 6.5, 7.6;
  // do the computation
  lanczos1_functor functor;
  LevenbergMarquardt<lanczos1_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, LevenbergMarquardtSpace::RelativeErrorTooSmall);
  // VERIFY_IS_EQUAL(lm.nfev(), 79);
  // VERIFY_IS_EQUAL(lm.njev(), 72);
  // check norm^2
  VERIFY(lm.fvec().squaredNorm() <= 1.44E-25);
  // check x
  VERIFY_IS_APPROX(x[0], 9.5100000027E-02);
  VERIFY_IS_APPROX(x[1], 1.0000000001E+00);
  VERIFY_IS_APPROX(x[2], 8.6070000013E-01);
  VERIFY_IS_APPROX(x[3], 3.0000000002E+00);
  VERIFY_IS_APPROX(x[4], 1.5575999998E+00);
  VERIFY_IS_APPROX(x[5], 5.0000000001E+00);

  /*
   * Second try
   */
  x << 0.5, 0.7, 3.6, 4.2, 4., 6.3;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, LevenbergMarquardtSpace::RelativeErrorTooSmall);
  // VERIFY_IS_EQUAL(lm.nfev(), 9);
  // VERIFY_IS_EQUAL(lm.njev(), 8);
  // check norm^2
  VERIFY(lm.fvec().squaredNorm() <= 1.44E-25);
  // check x
  VERIFY_IS_APPROX(x[0], 9.5100000027E-02);
  VERIFY_IS_APPROX(x[1], 1.0000000001E+00);
  VERIFY_IS_APPROX(x[2], 8.6070000013E-01);
  VERIFY_IS_APPROX(x[3], 3.0000000002E+00);
  VERIFY_IS_APPROX(x[4], 1.5575999998E+00);
  VERIFY_IS_APPROX(x[5], 5.0000000001E+00);
}

// http://www.itl.nist.gov/div898/strd/nls/data/ratkowsky2.shtml
void testNistRat42(void) {
  const int n = 3;
  LevenbergMarquardtSpace::Status info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 100., 1., 0.1;
  // do the computation
  rat42_functor functor;
  LevenbergMarquardt<rat42_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, LevenbergMarquardtSpace::RelativeReductionTooSmall);
  // VERIFY_IS_EQUAL(lm.nfev(), 10);
  // VERIFY_IS_EQUAL(lm.njev(), 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 8.0565229338E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 7.2462237576E+01);
  VERIFY_IS_APPROX(x[1], 2.6180768402E+00);
  VERIFY_IS_APPROX(x[2], 6.7359200066E-02);

  /*
   * Second try
   */
  x << 75., 2.5, 0.07;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, LevenbergMarquardtSpace::RelativeReductionTooSmall);
  // VERIFY_IS_EQUAL(lm.nfev(), 6);
  // VERIFY_IS_EQUAL(lm.njev(), 5);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 8.0565229338E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 7.2462237576E+01);
  VERIFY_IS_APPROX(x[1], 2.6180768402E+00);
  VERIFY_IS_APPROX(x[2], 6.7359200066E-02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml
void testNistMGH10(void) {
  const int n = 3;
  LevenbergMarquardtSpace::Status info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 2., 400000., 25000.;
  // do the computation
  MGH10_functor functor;
  LevenbergMarquardt<MGH10_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);  // ++g_test_level;
  // VERIFY_IS_EQUAL(info, LevenbergMarquardtSpace::RelativeReductionTooSmall);
  // --g_test_level;
  // was: VERIFY_IS_EQUAL(info, 1);

  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);

  // check return value

  // ++g_test_level;
  // VERIFY_IS_EQUAL(lm.nfev(), 284 );
  // VERIFY_IS_EQUAL(lm.njev(), 249 );
  // --g_test_level;
  VERIFY(lm.nfev() < 284 * LM_EVAL_COUNT_TOL);
  VERIFY(lm.njev() < 249 * LM_EVAL_COUNT_TOL);

  /*
   * Second try
   */
  x << 0.02, 4000., 250.;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);  // ++g_test_level;
  // VERIFY_IS_EQUAL(info, LevenbergMarquardtSpace::RelativeReductionTooSmall);
  // // was: VERIFY_IS_EQUAL(info, 1);
  // --g_test_level;

  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);

  // check return value
  // ++g_test_level;
  // VERIFY_IS_EQUAL(lm.nfev(), 126);
  // VERIFY_IS_EQUAL(lm.njev(), 116);
  // --g_test_level;
  VERIFY(lm.nfev() < 126 * LM_EVAL_COUNT_TOL);
  VERIFY(lm.njev() < 116 * LM_EVAL_COUNT_TOL);
}

// http://www.itl.nist.gov/div898/strd/nls/data/boxbod.shtml
void testNistBoxBOD(void) {
  const int n = 2;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1., 1.;
  // do the computation
  BoxBOD_functor functor;
  LevenbergMarquardt<BoxBOD_functor> lm(functor);
  lm.setFtol(1.E6 * NumTraits<double>::epsilon());
  lm.setXtol(1.E6 * NumTraits<double>::epsilon());
  lm.setFactor(10);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.1680088766E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 2.1380940889E+02);
  VERIFY_IS_APPROX(x[1], 5.4723748542E-01);

  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY(lm.nfev() < 31); // 31
  // VERIFY(lm.njev() < 25); // 25

  /*
   * Second try
   */
  x << 100., 0.75;
  // do the computation
  lm.resetParameters();
  lm.setFtol(NumTraits<double>::epsilon());
  lm.setXtol(NumTraits<double>::epsilon());
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // ++g_test_level;
  // VERIFY_IS_EQUAL(lm.nfev(), 16 );
  // VERIFY_IS_EQUAL(lm.njev(), 15 );
  // --g_test_level;
  VERIFY(lm.nfev() < 16 * LM_EVAL_COUNT_TOL);
  VERIFY(lm.njev() < 15 * LM_EVAL_COUNT_TOL);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.1680088766E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 2.1380940889E+02);
  VERIFY_IS_APPROX(x[1], 5.4723748542E-01);
}

// http://www.itl.nist.gov/div898/strd/nls/data/mgh17.shtml
void testNistMGH17(void) {
  const int n = 5;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 50., 150., -100., 1., 2.;
  // do the computation
  MGH17_functor functor;
  LevenbergMarquardt<MGH17_functor> lm(functor);
  lm.setFtol(NumTraits<double>::epsilon());
  lm.setXtol(NumTraits<double>::epsilon());
  lm.setMaxfev(1000);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.4648946975E-05);
  // check x
  VERIFY_IS_APPROX(x[0], 3.7541005211E-01);
  VERIFY_IS_APPROX(x[1], 1.9358469127E+00);
  VERIFY_IS_APPROX(x[2], -1.4646871366E+00);
  VERIFY_IS_APPROX(x[3], 1.2867534640E-02);
  VERIFY_IS_APPROX(x[4], 2.2122699662E-02);

  // check return value
  //   VERIFY_IS_EQUAL(info, 2);  //FIXME Use (lm.info() == Success)
  // VERIFY(lm.nfev() < 700 ); // 602
  // VERIFY(lm.njev() < 600 ); // 545

  /*
   * Second try
   */
  x << 0.5, 1.5, -1, 0.01, 0.02;
  // do the computation
  lm.resetParameters();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 18);
  // VERIFY_IS_EQUAL(lm.njev(), 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.4648946975E-05);
  // check x
  VERIFY_IS_APPROX(x[0], 3.7541005211E-01);
  VERIFY_IS_APPROX(x[1], 1.9358469127E+00);
  VERIFY_IS_APPROX(x[2], -1.4646871366E+00);
  VERIFY_IS_APPROX(x[3], 1.2867534640E-02);
  VERIFY_IS_APPROX(x[4], 2.2122699662E-02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/mgh09.shtml
void testNistMGH09(void) {
  const int n = 4;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 25., 39, 41.5, 39.;
  // do the computation
  MGH09_functor functor;
  LevenbergMarquardt<MGH09_functor> lm(functor);
  lm.setMaxfev(1000);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 3.0750560385E-04);
  // check x
  VERIFY_IS_APPROX(x[0], 0.1928077089);   // should be 1.9280693458E-01
  VERIFY_IS_APPROX(x[1], 0.19126423573);  // should be 1.9128232873E-01
  VERIFY_IS_APPROX(x[2], 0.12305309914);  // should be 1.2305650693E-01
  VERIFY_IS_APPROX(x[3], 0.13605395375);  // should be 1.3606233068E-01
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY(lm.nfev() < 510 ); // 490
  // VERIFY(lm.njev() < 400 ); // 376

  /*
   * Second try
   */
  x << 0.25, 0.39, 0.415, 0.39;
  // do the computation
  lm.resetParameters();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 18);
  // VERIFY_IS_EQUAL(lm.njev(), 16);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 3.0750560385E-04);
  // check x
  VERIFY_IS_APPROX(x[0], 0.19280781);  // should be 1.9280693458E-01
  VERIFY_IS_APPROX(x[1], 0.19126265);  // should be 1.9128232873E-01
  VERIFY_IS_APPROX(x[2], 0.12305280);  // should be 1.2305650693E-01
  VERIFY_IS_APPROX(x[3], 0.13605322);  // should be 1.3606233068E-01
}

// http://www.itl.nist.gov/div898/strd/nls/data/bennett5.shtml
void testNistBennett5(void) {
  const int n = 3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << -2000., 50., 0.8;
  // do the computation
  Bennett5_functor functor;
  LevenbergMarquardt<Bennett5_functor> lm(functor);
  lm.setMaxfev(1000);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 758);
  // VERIFY_IS_EQUAL(lm.njev(), 744);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.2404744073E-04);
  // check x
  VERIFY_IS_APPROX(x[0], -2.5235058043E+03);
  VERIFY_IS_APPROX(x[1], 4.6736564644E+01);
  VERIFY_IS_APPROX(x[2], 9.3218483193E-01);
  /*
   * Second try
   */
  x << -1500., 45., 0.85;
  // do the computation
  lm.resetParameters();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 203);
  // VERIFY_IS_EQUAL(lm.njev(), 192);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.2404744073E-04);
  // check x
  VERIFY_IS_APPROX(x[0], -2523.3007865);  // should be -2.5235058043E+03
  VERIFY_IS_APPROX(x[1], 46.735705771);   // should be 4.6736564644E+01);
  VERIFY_IS_APPROX(x[2], 0.93219881891);  // should be 9.3218483193E-01);
}

// http://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
void testNistThurber(void) {
  const int n = 7;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1000, 1000, 400, 40, 0.7, 0.3, 0.0;
  // do the computation
  thurber_functor functor;
  LevenbergMarquardt<thurber_functor> lm(functor);
  lm.setFtol(1.E4 * NumTraits<double>::epsilon());
  lm.setXtol(1.E4 * NumTraits<double>::epsilon());
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 39);
  // VERIFY_IS_EQUAL(lm.njev(), 36);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.6427082397E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.2881396800E+03);
  VERIFY_IS_APPROX(x[1], 1.4910792535E+03);
  VERIFY_IS_APPROX(x[2], 5.8323836877E+02);
  VERIFY_IS_APPROX(x[3], 7.5416644291E+01);
  VERIFY_IS_APPROX(x[4], 9.6629502864E-01);
  VERIFY_IS_APPROX(x[5], 3.9797285797E-01);
  VERIFY_IS_APPROX(x[6], 4.9727297349E-02);

  /*
   * Second try
   */
  x << 1300, 1500, 500, 75, 1, 0.4, 0.05;
  // do the computation
  lm.resetParameters();
  lm.setFtol(1.E4 * NumTraits<double>::epsilon());
  lm.setXtol(1.E4 * NumTraits<double>::epsilon());
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 29);
  // VERIFY_IS_EQUAL(lm.njev(), 28);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.6427082397E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.2881396800E+03);
  VERIFY_IS_APPROX(x[1], 1.4910792535E+03);
  VERIFY_IS_APPROX(x[2], 5.8323836877E+02);
  VERIFY_IS_APPROX(x[3], 7.5416644291E+01);
  VERIFY_IS_APPROX(x[4], 9.6629502864E-01);
  VERIFY_IS_APPROX(x[5], 3.9797285797E-01);
  VERIFY_IS_APPROX(x[6], 4.9727297349E-02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/ratkowsky3.shtml
void testNistRat43(void) {
  const int n = 4;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 100., 10., 1., 1.;
  // do the computation
  rat43_functor functor;
  LevenbergMarquardt<rat43_functor> lm(functor);
  lm.setFtol(1.E6 * NumTraits<double>::epsilon());
  lm.setXtol(1.E6 * NumTraits<double>::epsilon());
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 27);
  // VERIFY_IS_EQUAL(lm.njev(), 20);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 8.7864049080E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 6.9964151270E+02);
  VERIFY_IS_APPROX(x[1], 5.2771253025E+00);
  VERIFY_IS_APPROX(x[2], 7.5962938329E-01);
  VERIFY_IS_APPROX(x[3], 1.2792483859E+00);

  /*
   * Second try
   */
  x << 700., 5., 0.75, 1.3;
  // do the computation
  lm.resetParameters();
  lm.setFtol(1.E5 * NumTraits<double>::epsilon());
  lm.setXtol(1.E5 * NumTraits<double>::epsilon());
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 9);
  // VERIFY_IS_EQUAL(lm.njev(), 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 8.7864049080E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 6.9964151270E+02);
  VERIFY_IS_APPROX(x[1], 5.2771253025E+00);
  VERIFY_IS_APPROX(x[2], 7.5962938329E-01);
  VERIFY_IS_APPROX(x[3], 1.2792483859E+00);
}

// http://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml
void testNistEckerle4(void) {
  const int n = 3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1., 10., 500.;
  // do the computation
  eckerle4_functor functor;
  LevenbergMarquardt<eckerle4_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 18);
  // VERIFY_IS_EQUAL(lm.njev(), 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.4635887487E-03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.5543827178);
  VERIFY_IS_APPROX(x[1], 4.0888321754);
  VERIFY_IS_APPROX(x[2], 4.5154121844E+02);

  /*
   * Second try
   */
  x << 1.5, 5., 450.;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  // VERIFY_IS_EQUAL(lm.nfev(), 7);
  // VERIFY_IS_EQUAL(lm.njev(), 6);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 1.4635887487E-03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.5543827178);
  VERIFY_IS_APPROX(x[1], 4.0888321754);
  VERIFY_IS_APPROX(x[2], 4.5154121844E+02);
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
