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

#include "lm_test_functors.h"

struct NistAdapter {
  template <typename Solver>
  static const VectorXd &fvec(const Solver &lm) {
    return lm.fvec;
  }
  template <typename Solver>
  static void setFtol(Solver &lm, double value) {
    lm.parameters.ftol = value;
  }
  template <typename Solver>
  static void setXtol(Solver &lm, double value) {
    lm.parameters.xtol = value;
  }
  template <typename Solver>
  static void setFactor(Solver &lm, double value) {
    lm.parameters.factor = value;
  }
  template <typename Solver>
  static void setMaxfev(Solver &lm, int value) {
    lm.parameters.maxfev = value;
  }
  template <typename Solver>
  static void check_counts(const Solver &lm, int nfev, int njev) {
    LM_CHECK_N_ITERS(lm, nfev, njev);
  }
  template <typename Solver>
  static void check_counts_lm(const Solver &, int, int) {}
  template <typename Status>
  static void check_xtol_exit(Status) {}
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

// The first lmstr step scales by the column norms of R, which equal those of the Jacobian (MINPACK).
void testLmstrScaling() {
  lmstr_functor functor;
  VectorXd x = VectorXd::Ones(3), row(3);
  MatrixXd jac(15, 3);
  for (int i = 0; i < 15; ++i) {
    functor.df(x, row, i + 2);
    jac.row(i) = row.transpose();
  }
  LevenbergMarquardt<lmstr_functor> lm(functor);
  lm.minimizeOptimumStorageInit(x);
  lm.minimizeOptimumStorageOneStep(x);
  VERIFY_IS_APPROX(lm.diag, VectorXd(jac.colwise().norm().transpose()));
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

// dogleg() replaces a zero pivot r(j,j) by epsmch * max_i |r(i,j)| (MINPACK), not by the signed maximum.
void testDoglegZeroPivot() {
  MatrixXd r(2, 2);
  r << -4, -2, 0, 0;
  VectorXd diag = VectorXd::Ones(2), qtb = VectorXd::Ones(2), x(2);
  internal::dogleg<double>(r, diag, qtb, 1e300, x);
  const double x1 = 1 / (2 * NumTraits<double>::epsilon());
  VERIFY_IS_APPROX(x, Vector2d(-(1 + 2 * x1) / 4, x1));
}

EIGEN_DECLARE_TEST(NonLinearOptimization) {
  CALL_SUBTEST /*_2*/ (testR1mpyq());
  CALL_SUBTEST /*_2*/ (testDoglegZeroPivot());

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
  CALL_SUBTEST /*_3*/ (testLmstrScaling());
  CALL_SUBTEST /*_3*/ (testCovarRankDeficient());
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
