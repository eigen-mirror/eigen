// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_LM_NIST_TESTS_H
#define EIGEN_TEST_LM_NIST_TESTS_H

// The NIST StRD problems for LevenbergMarquardt, shared by NonLinearOptimization.cpp and levenberg_marquardt.cpp.
// Both modules define Eigen::LevenbergMarquardt, so the including file includes its module and lm_test_functors.h,
// then defines NistAdapter with the static members used below; an adapter omits the checks its module does not make.
// check_counts holds the evaluation-count bounds NonLinearOptimization asserts, check_counts_lm and check_xtol_exit
// the extra bounds and exit status levenberg_marquardt asserts.

// http://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml
void testNistChwirut2(void) {
  const int n = 3;

  VectorXd x(n);

  /*
   * First try
   */
  x << 0.1, 0.01, 0.02;
  // do the computation
  chwirut2_functor functor;
  LevenbergMarquardt<chwirut2_functor> lm(functor);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 10, 8);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.1304802941E+02);
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
  NistAdapter::setFtol(lm, 1.E6 * NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, 1.E6 * NumTraits<double>::epsilon());
  lm.minimize(x);
  NistAdapter::check_counts(lm, 7, 6);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.1304802941E+02);
  // check x
  VERIFY_IS_APPROX(x[0], 1.6657666537E-01);
  VERIFY_IS_APPROX(x[1], 5.1653291286E-03);
  VERIFY_IS_APPROX(x[2], 1.2150007096E-02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml
void testNistMisra1a(void) {
  const int n = 2;

  VectorXd x(n);

  /*
   * First try
   */
  x << 500., 0.0001;
  // do the computation
  misra1a_functor functor;
  LevenbergMarquardt<misra1a_functor> lm(functor);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 19, 15);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.2455138894E-01);
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);

  /*
   * Second try
   */
  x << 250., 0.0005;
  // do the computation
  lm.minimize(x);
  NistAdapter::check_counts(lm, 5, 4);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.2455138894E-01);
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);
}

// http://www.itl.nist.gov/div898/strd/nls/data/hahn1.shtml
void testNistHahn1(void) {
  const int n = 7;

  VectorXd x(n);

  /*
   * First try
   */
  x << 10., -1., .05, -.00001, -.05, .001, -.000001;
  // do the computation
  hahn1_functor functor;
  LevenbergMarquardt<hahn1_functor> lm(functor);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 11, 10);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.5324382854E+00);
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
  lm.minimize(x);
  NistAdapter::check_counts(lm, 11, 10);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.5324382854E+00);
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

  VectorXd x(n);

  /*
   * First try
   */
  x << 500., 0.0001;
  // do the computation
  misra1d_functor functor;
  LevenbergMarquardt<misra1d_functor> lm(functor);
  NistAdapter::check_xtol_exit(lm.minimize(x));
  NistAdapter::check_counts(lm, 9, 7);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.6419295283E-02);
  // check x
  VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  VERIFY_IS_APPROX(x[1], 3.0227324449E-04);

  /*
   * Second try
   */
  x << 450., 0.0003;
  // do the computation
  lm.minimize(x);
  NistAdapter::check_counts(lm, 4, 3);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.6419295283E-02);
  // check x
  VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  VERIFY_IS_APPROX(x[1], 3.0227324449E-04);
}

// http://www.itl.nist.gov/div898/strd/nls/data/lanczos1.shtml
void testNistLanczos1(void) {
  const int n = 6;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1.2, 0.3, 5.6, 5.5, 6.5, 7.6;
  // do the computation
  lanczos1_functor functor;
  LevenbergMarquardt<lanczos1_functor> lm(functor);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 79, 72);
  // check norm^2
  VERIFY(NistAdapter::fvec(lm).squaredNorm() <= 1.44E-25);
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
  lm.minimize(x);
  NistAdapter::check_counts(lm, 9, 8);
  // check norm^2
  VERIFY(NistAdapter::fvec(lm).squaredNorm() <= 1.44E-25);
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

  VectorXd x(n);

  /*
   * First try
   */
  x << 100., 1., 0.1;
  // do the computation
  rat42_functor functor;
  LevenbergMarquardt<rat42_functor> lm(functor);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 10, 8);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 8.0565229338E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 7.2462237576E+01);
  VERIFY_IS_APPROX(x[1], 2.6180768402E+00);
  VERIFY_IS_APPROX(x[2], 6.7359200066E-02);

  /*
   * Second try
   */
  x << 75., 2.5, 0.07;
  // do the computation
  lm.minimize(x);
  NistAdapter::check_counts(lm, 6, 5);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 8.0565229338E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 7.2462237576E+01);
  VERIFY_IS_APPROX(x[1], 2.6180768402E+00);
  VERIFY_IS_APPROX(x[2], 6.7359200066E-02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml
void testNistMGH10(void) {
  const int n = 3;

  VectorXd x(n);

  /*
   * First try
   */
  x << 2., 400000., 25000.;
  // do the computation
  MGH10_functor functor;
  LevenbergMarquardt<MGH10_functor> lm(functor);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 284, 249);
  NistAdapter::check_counts_lm(lm, 284, 249);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);

  /*
   * Second try
   */
  x << 0.02, 4000., 250.;
  // do the computation
  lm.minimize(x);
  NistAdapter::check_counts(lm, 126, 116);
  NistAdapter::check_counts_lm(lm, 126, 116);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/boxbod.shtml
void testNistBoxBOD(void) {
  const int n = 2;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1., 1.;
  // do the computation
  BoxBOD_functor functor;
  LevenbergMarquardt<BoxBOD_functor> lm(functor);
  NistAdapter::setFtol(lm, 1.E6 * NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, 1.E6 * NumTraits<double>::epsilon());
  NistAdapter::setFactor(lm, 10.);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 31, 25);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.1680088766E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 2.1380940889E+02);
  VERIFY_IS_APPROX(x[1], 5.4723748542E-01);

  /*
   * Second try
   */
  x << 100., 0.75;
  // do the computation
  lm.resetParameters();
  NistAdapter::setFtol(lm, NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, NumTraits<double>::epsilon());
  lm.minimize(x);
  NistAdapter::check_counts(lm, 20, 14);
  NistAdapter::check_counts_lm(lm, 16, 15);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.1680088766E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 2.1380940889E+02);
  VERIFY_IS_APPROX(x[1], 5.4723748542E-01);
}

// http://www.itl.nist.gov/div898/strd/nls/data/mgh17.shtml
void testNistMGH17(void) {
  const int n = 5;

  VectorXd x(n);

  /*
   * First try
   */
  x << 50., 150., -100., 1., 2.;
  // do the computation
  MGH17_functor functor;
  LevenbergMarquardt<MGH17_functor> lm(functor);
  NistAdapter::setFtol(lm, NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, NumTraits<double>::epsilon());
  NistAdapter::setMaxfev(lm, 1000);
  lm.minimize(x);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.4648946975E-05);
  // check x
  VERIFY_IS_APPROX(x[0], 3.7541005211E-01);
  VERIFY_IS_APPROX(x[1], 1.9358469127E+00);
  VERIFY_IS_APPROX(x[2], -1.4646871366E+00);
  VERIFY_IS_APPROX(x[3], 1.2867534640E-02);
  VERIFY_IS_APPROX(x[4], 2.2122699662E-02);

  NistAdapter::check_counts(lm, 602, 545);

  /*
   * Second try
   */
  x << 0.5, 1.5, -1, 0.01, 0.02;
  // do the computation
  lm.resetParameters();
  lm.minimize(x);
  NistAdapter::check_counts(lm, 18, 15);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.4648946975E-05);
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

  VectorXd x(n);

  /*
   * First try
   */
  x << 25., 39, 41.5, 39.;
  // do the computation
  MGH09_functor functor;
  LevenbergMarquardt<MGH09_functor> lm(functor);
  NistAdapter::setMaxfev(lm, 1000);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 490, 376);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 3.0750560385E-04);
  // check x
  VERIFY_IS_APPROX(x[0], 0.1928077089);   // should be 1.9280693458E-01
  VERIFY_IS_APPROX(x[1], 0.19126423573);  // should be 1.9128232873E-01
  VERIFY_IS_APPROX(x[2], 0.12305309914);  // should be 1.2305650693E-01
  VERIFY_IS_APPROX(x[3], 0.13605395375);  // should be 1.3606233068E-01

  /*
   * Second try
   */
  x << 0.25, 0.39, 0.415, 0.39;
  // do the computation
  lm.resetParameters();
  lm.minimize(x);
  NistAdapter::check_counts(lm, 18, 16);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 3.0750560385E-04);
  // check x
  VERIFY_IS_APPROX(x[0], 0.19280781);  // should be 1.9280693458E-01
  VERIFY_IS_APPROX(x[1], 0.19126265);  // should be 1.9128232873E-01
  VERIFY_IS_APPROX(x[2], 0.12305280);  // should be 1.2305650693E-01
  VERIFY_IS_APPROX(x[3], 0.13605322);  // should be 1.3606233068E-01
}

// http://www.itl.nist.gov/div898/strd/nls/data/bennett5.shtml
void testNistBennett5(void) {
  const int n = 3;

  VectorXd x(n);

  /*
   * First try
   */
  x << -2000., 50., 0.8;
  // do the computation
  Bennett5_functor functor;
  LevenbergMarquardt<Bennett5_functor> lm(functor);
  NistAdapter::setMaxfev(lm, 1000);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 758, 744);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.2404744073E-04);
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
  lm.minimize(x);
  NistAdapter::check_counts(lm, 203, 192);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.2404744073E-04);
  // check x
  VERIFY_IS_APPROX(x[0], -2523.3007865);  // should be -2.5235058043E+03
  VERIFY_IS_APPROX(x[1], 46.735705771);   // should be 4.6736564644E+01);
  VERIFY_IS_APPROX(x[2], 0.93219881891);  // should be 9.3218483193E-01);
}

// http://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
void testNistThurber(void) {
  const int n = 7;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1000, 1000, 400, 40, 0.7, 0.3, 0.0;
  // do the computation
  thurber_functor functor;
  LevenbergMarquardt<thurber_functor> lm(functor);
  NistAdapter::setFtol(lm, 1.E4 * NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, 1.E4 * NumTraits<double>::epsilon());
  lm.minimize(x);
  NistAdapter::check_counts(lm, 39, 36);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.6427082397E+03);
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
  NistAdapter::setFtol(lm, 1.E4 * NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, 1.E4 * NumTraits<double>::epsilon());
  lm.minimize(x);
  NistAdapter::check_counts(lm, 29, 28);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 5.6427082397E+03);
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

  VectorXd x(n);

  /*
   * First try
   */
  x << 100., 10., 1., 1.;
  // do the computation
  rat43_functor functor;
  LevenbergMarquardt<rat43_functor> lm(functor);
  NistAdapter::setFtol(lm, 1.E6 * NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, 1.E6 * NumTraits<double>::epsilon());
  lm.minimize(x);
  NistAdapter::check_counts(lm, 27, 20);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 8.7864049080E+03);
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
  NistAdapter::setFtol(lm, 1.E5 * NumTraits<double>::epsilon());
  NistAdapter::setXtol(lm, 1.E5 * NumTraits<double>::epsilon());
  lm.minimize(x);
  NistAdapter::check_counts(lm, 9, 8);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 8.7864049080E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 6.9964151270E+02);
  VERIFY_IS_APPROX(x[1], 5.2771253025E+00);
  VERIFY_IS_APPROX(x[2], 7.5962938329E-01);
  VERIFY_IS_APPROX(x[3], 1.2792483859E+00);
}

// http://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml
void testNistEckerle4(void) {
  const int n = 3;

  VectorXd x(n);

  /*
   * First try
   */
  x << 1., 10., 500.;
  // do the computation
  eckerle4_functor functor;
  LevenbergMarquardt<eckerle4_functor> lm(functor);
  lm.minimize(x);
  NistAdapter::check_counts(lm, 18, 15);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.4635887487E-03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.5543827178);
  VERIFY_IS_APPROX(x[1], 4.0888321754);
  VERIFY_IS_APPROX(x[2], 4.5154121844E+02);

  /*
   * Second try
   */
  x << 1.5, 5., 450.;
  // do the computation
  lm.minimize(x);
  NistAdapter::check_counts(lm, 7, 6);
  // check norm^2
  VERIFY_IS_APPROX(NistAdapter::fvec(lm).squaredNorm(), 1.4635887487E-03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.5543827178);
  VERIFY_IS_APPROX(x[1], 4.0888321754);
  VERIFY_IS_APPROX(x[2], 4.5154121844E+02);
}

#endif  // EIGEN_TEST_LM_NIST_TESTS_H
