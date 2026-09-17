// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
// SPDX-License-Identifier: MPL-2.0

#include <stdio.h>

#include "main.h"
#include <contrib/Eigen/NonLinearOptimization>

// This disables some useless Warnings on MSVC.
// It is intended to be done for this test only.
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

// tolerance for checking number of iterations
#define LM_EVAL_COUNT_TOL 2

#define LM_CHECK_N_ITERS(SOLVER, NFEV, NJEV)         \
  {                                                  \
    VERIFY(SOLVER.nfev <= NFEV * LM_EVAL_COUNT_TOL); \
    VERIFY(SOLVER.njev <= NJEV * LM_EVAL_COUNT_TOL); \
  }

int fcn_chkder(const VectorXd &x, VectorXd &fvec, MatrixXd &fjac, int iflag) {
  /*      subroutine fcn for chkder example. */

  int i;
  assert(15 == fvec.size());
  assert(3 == x.size());
  double tmp1, tmp2, tmp3, tmp4;
  static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1,
                               3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34,   2.1,    4.39};

  if (iflag == 0) return 0;

  if (iflag != 2)
    for (i = 0; i < 15; i++) {
      tmp1 = i + 1;
      tmp2 = 16 - i - 1;
      tmp3 = tmp1;
      if (i >= 8) tmp3 = tmp2;
      fvec[i] = y[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3));
    }
  else {
    for (i = 0; i < 15; i++) {
      tmp1 = i + 1;
      tmp2 = 16 - i - 1;

      /* error introduced into next statement for illustration. */
      /* corrected statement should read    tmp3 = tmp1 . */

      tmp3 = tmp2;
      if (i >= 8) tmp3 = tmp2;
      tmp4 = (x[1] * tmp2 + x[2] * tmp3);
      tmp4 = tmp4 * tmp4;
      fjac(i, 0) = -1.;
      fjac(i, 1) = tmp1 * tmp2 / tmp4;
      fjac(i, 2) = tmp1 * tmp3 / tmp4;
    }
  }
  return 0;
}

void testChkder() {
  const int m = 15, n = 3;
  VectorXd x(n), fvec(m), xp, fvecp(m), err;
  MatrixXd fjac(m, n);
  VectorXi ipvt;

  /*      the following values should be suitable for */
  /*      checking the jacobian matrix. */
  x << 9.2e-1, 1.3e-1, 5.4e-1;

  internal::chkder(x, fvec, fjac, xp, fvecp, 1, err);
  fcn_chkder(x, fvec, fjac, 1);
  fcn_chkder(x, fvec, fjac, 2);
  fcn_chkder(xp, fvecp, fjac, 1);
  internal::chkder(x, fvec, fjac, xp, fvecp, 2, err);

  fvecp -= fvec;

  // check those
  VectorXd fvec_ref(m), fvecp_ref(m), err_ref(m);
  fvec_ref << -1.181606, -1.429655, -1.606344, -1.745269, -1.840654, -1.921586, -1.984141, -2.022537, -2.468977,
      -2.827562, -3.473582, -4.437612, -6.047662, -9.267761, -18.91806;
  fvecp_ref << -7.724666e-09, -3.432406e-09, -2.034843e-10, 2.313685e-09, 4.331078e-09, 5.984096e-09, 7.363281e-09,
      8.53147e-09, 1.488591e-08, 2.33585e-08, 3.522012e-08, 5.301255e-08, 8.26666e-08, 1.419747e-07, 3.19899e-07;
  err_ref << 0.1141397, 0.09943516, 0.09674474, 0.09980447, 0.1073116, 0.1220445, 0.1526814, 1, 1, 1, 1, 1, 1, 1, 1;

  VERIFY_IS_APPROX(fvec, fvec_ref);
  VERIFY_IS_APPROX(fvecp, fvecp_ref);
  VERIFY_IS_APPROX(err, err_ref);
}

// Generic functor
template <typename Scalar_, int NX = Dynamic, int NY = Dynamic>
struct Functor {
  typedef Scalar_ Scalar;
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
  typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  const int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  // you should define that in the subclass :
  //  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

using LmFunctorBase = Functor<double>;
#include "lm_test_functors.h"

void testLmder1() {
  int n = 3, info;

  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.lmder1(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 6, 5);

  // check norm
  VERIFY_IS_APPROX(lm.fvec.blueNorm(), 0.09063596);

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
  LM_CHECK_N_ITERS(lm, 6, 5);

  // check norm
  fnorm = lm.fvec.blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covfac = fnorm * fnorm / (m - n);
  internal::covar(lm.fjac, lm.permutation.indices());  // TODO : move this as a function of lm

  MatrixXd cov_ref(n, n);
  cov_ref << 0.0001531202, 0.002869941, -0.002656662, 0.002869941, 0.09480935, -0.09098995, -0.002656662, -0.09098995,
      0.08778727;

  //  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov = covfac * lm.fjac.topLeftCorner<n, n>();
  VERIFY_IS_APPROX(cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

struct hybrj_functor : Functor<double> {
  hybrj_functor(void) : Functor<double>(9, 9) {}

  int operator()(const VectorXd &x, VectorXd &fvec) {
    double temp, temp1, temp2;
    const VectorXd::Index n = x.size();
    assert(fvec.size() == n);
    for (VectorXd::Index k = 0; k < n; k++) {
      temp = (3. - 2. * x[k]) * x[k];
      temp1 = 0.;
      if (k) temp1 = x[k - 1];
      temp2 = 0.;
      if (k != n - 1) temp2 = x[k + 1];
      fvec[k] = temp - temp1 - 2. * temp2 + 1.;
    }
    return 0;
  }
  int df(const VectorXd &x, MatrixXd &fjac) {
    const VectorXd::Index n = x.size();
    assert(fjac.rows() == n);
    assert(fjac.cols() == n);
    for (VectorXd::Index k = 0; k < n; k++) {
      for (VectorXd::Index j = 0; j < n; j++) fjac(k, j) = 0.;
      fjac(k, k) = 3. - 4. * x[k];
      if (k) fjac(k, k - 1) = -1.;
      if (k != n - 1) fjac(k, k + 1) = -2.;
    }
    return 0;
  }
};

void testHybrj1() {
  const int n = 9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  info = solver.hybrj1(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(solver, 11, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrj() {
  const int n = 9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solve(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(solver, 11, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

struct hybrd_functor : Functor<double> {
  hybrd_functor(void) : Functor<double>(9, 9) {}
  int operator()(const VectorXd &x, VectorXd &fvec) const {
    double temp, temp1, temp2;
    const VectorXd::Index n = x.size();

    assert(fvec.size() == n);
    for (VectorXd::Index k = 0; k < n; k++) {
      temp = (3. - 2. * x[k]) * x[k];
      temp1 = 0.;
      if (k) temp1 = x[k - 1];
      temp2 = 0.;
      if (k != n - 1) temp2 = x[k + 1];
      fvec[k] = temp - temp1 - 2. * temp2 + 1.;
    }
    return 0;
  }
};

void testHybrd1() {
  int n = 9, info;
  VectorXd x(n);

  /* the following starting values provide a rough solution. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  info = solver.hybrd1(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  VERIFY(solver.nfev <= 20 * LM_EVAL_COUNT_TOL);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrd() {
  const int n = 9;
  int info;
  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  solver.parameters.nb_of_subdiagonals = 1;
  solver.parameters.nb_of_superdiagonals = 1;
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solveNumericalDiff(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  VERIFY(solver.nfev <= 14 * LM_EVAL_COUNT_TOL);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

struct lmstr_functor : Functor<double> {
  lmstr_functor(void) : Functor<double>(3, 15) {}
  int operator()(const VectorXd &x, VectorXd &fvec) {
    /*  subroutine fcn for lmstr1 example. */
    double tmp1, tmp2, tmp3;
    static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1,
                                 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34,   2.1,    4.39};

    assert(15 == fvec.size());
    assert(3 == x.size());

    for (int i = 0; i < 15; i++) {
      tmp1 = i + 1;
      tmp2 = 16 - i - 1;
      tmp3 = (i >= 8) ? tmp2 : tmp1;
      fvec[i] = y[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3));
    }
    return 0;
  }
  int df(const VectorXd &x, VectorXd &jac_row, VectorXd::Index rownb) {
    assert(x.size() == 3);
    assert(jac_row.size() == x.size());
    double tmp1, tmp2, tmp3, tmp4;

    VectorXd::Index i = rownb - 2;
    tmp1 = i + 1;
    tmp2 = 16 - i - 1;
    tmp3 = (i >= 8) ? tmp2 : tmp1;
    tmp4 = (x[1] * tmp2 + x[2] * tmp3);
    tmp4 = tmp4 * tmp4;
    jac_row[0] = -1;
    jac_row[1] = tmp1 * tmp2 / tmp4;
    jac_row[2] = tmp1 * tmp3 / tmp4;
    return 0;
  }
};

void testLmstr1() {
  const int n = 3;
  int info;

  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmstr_functor functor;
  LevenbergMarquardt<lmstr_functor> lm(functor);
  info = lm.lmstr1(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 6, 5);

  // check norm
  VERIFY_IS_APPROX(lm.fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmstr() {
  const int n = 3;
  int info;
  double fnorm;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmstr_functor functor;
  LevenbergMarquardt<lmstr_functor> lm(functor);
  info = lm.minimizeOptimumStorage(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return values
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 6, 5);

  // check norm
  fnorm = lm.fvec.blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmdif1() {
  const int n = 3;
  int info;

  VectorXd x(n), fvec(15);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmdif_functor functor;
  DenseIndex nfev = -1;  // initialize to avoid maybe-uninitialized warning
  info = LevenbergMarquardt<lmdif_functor>::lmdif1(functor, x, &nfev);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  VERIFY(nfev <= 26 * LM_EVAL_COUNT_TOL);

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
  VERIFY(lm.nfev <= 26 * LM_EVAL_COUNT_TOL);

  // check norm
  fnorm = lm.fvec.blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covfac = fnorm * fnorm / (m - n);
  internal::covar(lm.fjac, lm.permutation.indices());  // TODO : move this as a function of lm

  MatrixXd cov_ref(n, n);
  cov_ref << 0.0001531202, 0.002869942, -0.002656662, 0.002869942, 0.09480937, -0.09098997, -0.002656662, -0.09098997,
      0.08778729;

  //  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov = covfac * lm.fjac.topLeftCorner<n, n>();
  VERIFY_IS_APPROX(cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

// http://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml
void testNistChwirut2(void) {
  const int n = 3;
  int info;

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
  LM_CHECK_N_ITERS(lm, 10, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.1304802941E+02);
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
  lm.parameters.ftol = 1.E6 * NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E6 * NumTraits<double>::epsilon();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 7, 6);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.1304802941E+02);
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
  LM_CHECK_N_ITERS(lm, 19, 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.2455138894E-01);
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
  LM_CHECK_N_ITERS(lm, 5, 4);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.2455138894E-01);
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
  LM_CHECK_N_ITERS(lm, 11, 10);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.5324382854E+00);
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
  LM_CHECK_N_ITERS(lm, 11, 10);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.5324382854E+00);
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
  // VERIFY_IS_EQUAL(info, 3);
  LM_CHECK_N_ITERS(lm, 9, 7);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6419295283E-02);
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
  LM_CHECK_N_ITERS(lm, 4, 3);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6419295283E-02);
  // check x
  VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  VERIFY_IS_APPROX(x[1], 3.0227324449E-04);
}

// http://www.itl.nist.gov/div898/strd/nls/data/lanczos1.shtml
void testNistLanczos1(void) {
  const int n = 6;
  int info;

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
  // VERIFY_IS_EQUAL(info, 2);
  LM_CHECK_N_ITERS(lm, 79, 72);
  // check norm^2
  // std::cout.precision(30);
  // std::cout << lm.fvec.squaredNorm() << "\n";
  VERIFY(lm.fvec.squaredNorm() <= 1.44E-25);
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
  // VERIFY_IS_EQUAL(info, 2);
  LM_CHECK_N_ITERS(lm, 9, 8);
  // check norm^2
  VERIFY(lm.fvec.squaredNorm() <= 1.44E-25);
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
  int info;

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
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 10, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.0565229338E+00);
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
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 6, 5);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.0565229338E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 7.2462237576E+01);
  VERIFY_IS_APPROX(x[1], 2.6180768402E+00);
  VERIFY_IS_APPROX(x[2], 6.7359200066E-02);
}

// http://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml
void testNistMGH10(void) {
  const int n = 3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 2., 400000., 25000.;
  // do the computation
  MGH10_functor functor;
  LevenbergMarquardt<MGH10_functor> lm(functor);
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 2);
  LM_CHECK_N_ITERS(lm, 284, 249);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);

  /*
   * Second try
   */
  x << 0.02, 4000., 250.;
  // do the computation
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 3);
  LM_CHECK_N_ITERS(lm, 126, 116);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);
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
  lm.parameters.ftol = 1.E6 * NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E6 * NumTraits<double>::epsilon();
  lm.parameters.factor = 10.;
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 31, 25);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.1680088766E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 2.1380940889E+02);
  VERIFY_IS_APPROX(x[1], 5.4723748542E-01);

  /*
   * Second try
   */
  x << 100., 0.75;
  // do the computation
  lm.resetParameters();
  lm.parameters.ftol = NumTraits<double>::epsilon();
  lm.parameters.xtol = NumTraits<double>::epsilon();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 20, 14);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.1680088766E+03);
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
  lm.parameters.ftol = NumTraits<double>::epsilon();
  lm.parameters.xtol = NumTraits<double>::epsilon();
  lm.parameters.maxfev = 1000;
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.4648946975E-05);
  // check x
  VERIFY_IS_APPROX(x[0], 3.7541005211E-01);
  VERIFY_IS_APPROX(x[1], 1.9358469127E+00);
  VERIFY_IS_APPROX(x[2], -1.4646871366E+00);
  VERIFY_IS_APPROX(x[3], 1.2867534640E-02);
  VERIFY_IS_APPROX(x[4], 2.2122699662E-02);

  // check return value
  // VERIFY_IS_EQUAL(info, 2);
  LM_CHECK_N_ITERS(lm, 602, 545);

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
  LM_CHECK_N_ITERS(lm, 18, 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.4648946975E-05);
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
  lm.parameters.maxfev = 1000;
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 490, 376);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 3.0750560385E-04);
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
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 18, 16);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 3.0750560385E-04);
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
  lm.parameters.maxfev = 1000;
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 758, 744);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.2404744073E-04);
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
  LM_CHECK_N_ITERS(lm, 203, 192);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.2404744073E-04);
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
  lm.parameters.ftol = 1.E4 * NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E4 * NumTraits<double>::epsilon();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 39, 36);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6427082397E+03);
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
  lm.parameters.ftol = 1.E4 * NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E4 * NumTraits<double>::epsilon();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 29, 28);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6427082397E+03);
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
  lm.parameters.ftol = 1.E6 * NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E6 * NumTraits<double>::epsilon();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 27, 20);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7864049080E+03);
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
  lm.parameters.ftol = 1.E5 * NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E5 * NumTraits<double>::epsilon();
  info = lm.minimize(x);
  EIGEN_UNUSED_VARIABLE(info);
  // check return value
  // VERIFY_IS_EQUAL(info, 1);
  LM_CHECK_N_ITERS(lm, 9, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7864049080E+03);
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
  LM_CHECK_N_ITERS(lm, 18, 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.4635887487E-03);
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
  LM_CHECK_N_ITERS(lm, 7, 6);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.4635887487E-03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.5543827178);
  VERIFY_IS_APPROX(x[1], 4.0888321754);
  VERIFY_IS_APPROX(x[2], 4.5154121844E+02);
}

// r1mpyq() indexed its argument through a raw pointer with a hard-coded
// column-major layout, so it corrupted the accumulated orthogonal factor
// whenever Matrix<Scalar, Dynamic, Dynamic> is row major, which is what
// EIGEN_DEFAULT_TO_ROW_MAJOR makes it.
template <typename StorageType>
void checkR1mpyq(const MatrixXd &a, const MatrixXd &expected, const std::vector<JacobiRotation<double> > &v_givens,
                 const std::vector<JacobiRotation<double> > &w_givens) {
  StorageType stored = a;
  internal::r1mpyq<double>(stored, v_givens, w_givens);
  VERIFY_IS_APPROX(MatrixXd(stored), expected);
}

void testR1mpyq() {
  const Index n = 5;

  MatrixXd a(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) a(i, j) = 1. + double(i) + 10. * double(j);

  std::vector<JacobiRotation<double> > v_givens(n), w_givens(n);
  for (Index j = 0; j + 1 < n; ++j) {
    v_givens[j].makeGivens(1. + double(j), 2. - .5 * double(j));
    w_givens[j].makeGivens(.5 * double(j) - 1., 3. + double(j));
  }

  // The MINPACK recurrence, spelled out independently of Eigen's rotation
  // conventions and of any storage order.
  MatrixXd expected = a;
  for (Index j = n - 2; j >= 0; --j)
    for (Index i = 0; i < n; ++i) {
      const double temp = v_givens[j].c() * expected(i, j) - v_givens[j].s() * expected(i, n - 1);
      expected(i, n - 1) = v_givens[j].s() * expected(i, j) + v_givens[j].c() * expected(i, n - 1);
      expected(i, j) = temp;
    }
  for (Index j = 0; j + 1 < n; ++j)
    for (Index i = 0; i < n; ++i) {
      const double temp = w_givens[j].c() * expected(i, j) + w_givens[j].s() * expected(i, n - 1);
      expected(i, n - 1) = -w_givens[j].s() * expected(i, j) + w_givens[j].c() * expected(i, n - 1);
      expected(i, j) = temp;
    }

  checkR1mpyq<Matrix<double, Dynamic, Dynamic, ColMajor> >(a, expected, v_givens, w_givens);
  checkR1mpyq<Matrix<double, Dynamic, Dynamic, RowMajor> >(a, expected, v_givens, w_givens);

  // HybridNonLinearSolver also rotates qtf, which it passes as a single row.
  VectorXd qtf = a.row(0).transpose();
  Transpose<VectorXd> qtf_row = qtf.transpose();
  internal::r1mpyq<double>(qtf_row, v_givens, w_givens);
  VERIFY_IS_APPROX(qtf.transpose(), expected.row(0));
}

EIGEN_DECLARE_TEST(NonLinearOptimization) {
  CALL_SUBTEST /*_2*/ (testR1mpyq());

  // Tests using the examples provided by (c)minpack
  CALL_SUBTEST /*_1*/ (testChkder());
  CALL_SUBTEST /*_1*/ (testLmder1());
  CALL_SUBTEST /*_1*/ (testLmder());
  CALL_SUBTEST /*_2*/ (testHybrj1());
  CALL_SUBTEST /*_2*/ (testHybrj());
  CALL_SUBTEST /*_2*/ (testHybrd1());
  CALL_SUBTEST /*_2*/ (testHybrd());
  CALL_SUBTEST /*_3*/ (testLmstr1());
  CALL_SUBTEST /*_3*/ (testLmstr());
  CALL_SUBTEST /*_3*/ (testLmdif1());
  CALL_SUBTEST /*_3*/ (testLmdif());

  // NIST tests, level of difficulty = "Lower"
  CALL_SUBTEST /*_4*/ (testNistMisra1a());
  CALL_SUBTEST /*_4*/ (testNistChwirut2());

  // NIST tests, level of difficulty = "Average"
  CALL_SUBTEST /*_5*/ (testNistHahn1());
  CALL_SUBTEST /*_6*/ (testNistMisra1d());
  CALL_SUBTEST /*_7*/ (testNistMGH17());
  CALL_SUBTEST /*_8*/ (testNistLanczos1());

  //     // NIST tests, level of difficulty = "Higher"
  CALL_SUBTEST /*_9*/ (testNistRat42());
  //     CALL_SUBTEST/*_10*/(testNistMGH10());
  CALL_SUBTEST /*_11*/ (testNistBoxBOD());
  //     CALL_SUBTEST/*_12*/(testNistMGH09());
  CALL_SUBTEST /*_13*/ (testNistBennett5());
  CALL_SUBTEST /*_14*/ (testNistThurber());
  CALL_SUBTEST /*_15*/ (testNistRat43());
  CALL_SUBTEST /*_16*/ (testNistEckerle4());
}

/*
 * Can be useful for debugging...
  printf("info, nfev : %d, %d\n", info, lm.nfev);
  printf("info, nfev, njev : %d, %d, %d\n", info, solver.nfev, solver.njev);
  printf("info, nfev : %d, %d\n", info, solver.nfev);
  printf("x[0] : %.32g\n", x[0]);
  printf("x[1] : %.32g\n", x[1]);
  printf("x[2] : %.32g\n", x[2]);
  printf("x[3] : %.32g\n", x[3]);
  printf("fvec.blueNorm() : %.32g\n", solver.fvec.blueNorm());
  printf("fvec.blueNorm() : %.32g\n", lm.fvec.blueNorm());

  printf("info, nfev, njev : %d, %d, %d\n", info, lm.nfev, lm.njev);
  printf("fvec.squaredNorm() : %.13g\n", lm.fvec.squaredNorm());
  std::cout << x << std::endl;
  std::cout.precision(9);
  std::cout << x[0] << std::endl;
  std::cout << x[1] << std::endl;
  std::cout << x[2] << std::endl;
  std::cout << x[3] << std::endl;
*/
