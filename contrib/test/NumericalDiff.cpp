// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
// SPDX-License-Identifier: MPL-2.0

#include <stdio.h>

#include "main.h"
#include <contrib/Eigen/NumericalDiff>

#include "lm_test_functors.h"

// Scaling by a power of two is exact, so the difference quotients of this map equal the scales
// iff each divisor is the representable step actually applied to x.
struct pow2_scaling_functor : Functor<double> {
  pow2_scaling_functor() : Functor<double>(7, 7), scale(7) { scale << 0.25, 1.0, 8.0, -2.0, 0.5, 4.0, -0.125; }
  int operator()(const VectorXd &x, VectorXd &fvec) const {
    fvec = scale.cwiseProduct(x);
    return 0;
  }
  VectorXd scale;
};

// When used with forward or central differences about x=1000, this will just return the relative step size used *
// 1000.0.
struct step_size_functor : Functor<double> {
  step_size_functor() : Functor<double>(1, 1) {}
  int operator()(const VectorXd &x, VectorXd &fvec) const {
    fvec[0] = (x[0] - 1000.0) * (x[0] - 1000.0);
    if (x[0] < 1000.0) {
      fvec[0] *= -1.0;
    }
    return 0;
  }
};

// When used with forward or central difference about x=0, this will return the absolute step size used.
struct step_size_functor_abs : Functor<double> {
  step_size_functor_abs() : Functor<double>(1, 1) {}
  int operator()(const VectorXd &x, VectorXd &fvec) const {
    fvec[0] = x[0] * x[0];
    if (x[0] < 0.0) {
      fvec[0] *= -1.0;
    }
    return 0;
  }
};

// Checks the numerical Jacobian of lmder_functor at x against the analytic one and returns the largest error.
template <NumericalDiffMode Mode>
double check_lmder_jacobian(const VectorXd &x) {
  MatrixXd jac(15, 3);
  MatrixXd actual_jac(15, 3);
  lmder_functor functor;

  functor.df(x, actual_jac);

  NumericalDiff<lmder_functor, Mode> numDiff(functor);
  numDiff.df(x, jac);

  VERIFY_IS_APPROX(jac, actual_jac);
  return (jac - actual_jac).cwiseAbs().maxCoeff();
}

void test_forward() {
  VectorXd x(3);
  x << 0.082, 1.13, 2.35;
  check_lmder_jacobian<Forward>(x);
}

void test_central() {
  VectorXd x(3);
  x << 0.082, 1.13, 2.35;
  // With the default changed to cbrt(epsilon), the maximum error has become lower
  // This verifies we get the new, lower maximum error.
  VERIFY_LE(check_lmder_jacobian<Central>(x), 1e-10);
}

void test_small() {
  VectorXd x(3);
  x << 0.082, 1.13, 1e-8;
  check_lmder_jacobian<Central>(x);
}

template <NumericalDiffMode Mode>
void test_step_alignment(double epsfcn) {
  pow2_scaling_functor functor;
  NumericalDiff<pow2_scaling_functor, Mode> numDiff(functor, epsfcn);

  // Non-dyadic coordinates, so that x + h rounds; the first two take the absolute step eps, and the
  // last two are negative with |x| at or just above a power of two, where x - h is a rounding tie.
  VectorXd x(7);
  x << 0.0, 0.082, 1.13, -2350.7, 7.7e8 + 0.3, -1.0 - std::ldexp(1.0, -27), -1.0;
  MatrixXd jac(7, 7);
  numDiff.df(x, jac);

  MatrixXd expected = functor.scale.asDiagonal();
  VERIFY_IS_EQUAL(jac, expected);
}

// The one-by-one Jacobian of `diff` at x0 must equal `expected`.
template <typename Functor_, NumericalDiffMode Mode>
void check_step(NumericalDiff<Functor_, Mode> &diff, double x0, double expected) {
  VectorXd x(1);
  x << x0;
  MatrixXd jac(1, 1);
  diff.df(x, jac);
  MatrixXd expected_jac(1, 1);
  expected_jac << expected;
  VERIFY_IS_APPROX(jac, expected_jac);
}

// step_size_functor probes the relative step about x = 1000 and step_size_functor_abs the absolute step about x = 0.
// default_tol is the mode's default for both; rel_only_eps is a relative tolerance passed to the constructor alone.
template <NumericalDiffMode Mode>
void check_step_size(double rel_only_eps, double default_tol) {
  step_size_functor functor;
  step_size_functor_abs functor_abs;

  // Setting relative step size and absolute step sizes gives the expected answer.
  NumericalDiff<step_size_functor, Mode> both_set(functor);
  both_set.setRelTol(1e-4);
  both_set.setAbsTol(1e-6);
  check_step(both_set, 1000.0, 1e-1);  // rel_tol * 1000.0
  NumericalDiff<step_size_functor_abs, Mode> both_set_abs(functor_abs);
  both_set_abs.setRelTol(1e-4);
  both_set_abs.setAbsTol(1e-6);
  check_step(both_set_abs, 0.0, 1e-6);  // abs_tol

  // Setting the relative step size only gives abs_tol = rel_tol.
  NumericalDiff<step_size_functor, Mode> rel_only(functor, rel_only_eps);
  check_step(rel_only, 1000.0, 1e-1);  // rel_tol * 1000.0
  NumericalDiff<step_size_functor_abs, Mode> rel_only_abs(functor_abs, rel_only_eps);
  check_step(rel_only_abs, 0.0, 1e-4);  // abs_tol = rel_tol

  // Setting neither gives rel_tol = abs_tol = default_tol.
  NumericalDiff<step_size_functor, Mode> defaults(functor);
  check_step(defaults, 1000.0, 1000.0 * default_tol);
  NumericalDiff<step_size_functor_abs, Mode> defaults_abs(functor_abs);
  check_step(defaults_abs, 0.0, default_tol);
}

void test_step_size() {
  check_step_size<Forward>(1e-8, std::sqrt(std::numeric_limits<double>::epsilon()));
  check_step_size<Central>(1e-12, std::cbrt(std::numeric_limits<double>::epsilon()));
}

// Test functor to ensure constructors work. The operator returns a1 at
// x = 0.0 and 0.0 otherwise.
struct G : Functor<double> {
  G() : Functor<double>(1, 1) {}
  G(double _a0, double _a1) : Functor<double>(1, 1), a0(_a0), a1(_a1) {}
  int operator()(const VectorXd &x, VectorXd &fvec) const {
    if (x[0] == 0.0) {
      fvec[0] = a1;
    } else {
      fvec[0] = 0.0;
    }
    return 0;
  }
  double a0 = 0.0;
  double a1 = 0.0;
};

struct float_functor : Functor<float> {
  float_functor() : Functor<float>(1, 1) {}
  int operator()(const VectorXf &x, VectorXf &fvec) const {
    if (x[0] == 0.0f) {
      fvec[0] = a0;
    } else {
      fvec[0] = 0.0;
    }
    return 0;
  }
  float a0 = 5.0;
};

void test_constructors() {
  VectorXd x(1);
  MatrixXd jac(1, 1);
  MatrixXd expected(1, 1);
  x << -1.0;

  NumericalDiff<G> g(2.0, 3.0);
  g.setRelTol(1.0);
  g.df(x, jac);
  expected << 3.0;  // Should be a1 if the constructor worked
  VERIFY_IS_APPROX(jac, expected);

  VectorXf fx(1);
  MatrixXf fjac(1, 1);
  MatrixXf fexpected(1, 1);
  float_functor ff;
  fx << -1.0;

  NumericalDiff<float_functor> nd(ff, 1.0);
  nd.df(fx, fjac);
  fexpected << 5.0;  // Should be 5.0
  VERIFY_IS_APPROX(fjac, fexpected);
}

EIGEN_DECLARE_TEST(NumericalDiff) {
  CALL_SUBTEST(test_forward());
  CALL_SUBTEST(test_central());
  CALL_SUBTEST(test_small());
  CALL_SUBTEST(test_step_size());
  CALL_SUBTEST(test_constructors());
  CALL_SUBTEST(test_step_alignment<Forward>(0.0));
  CALL_SUBTEST(test_step_alignment<Central>(0.0));
  // A non-dyadic eps also misaligns the absolute step taken for |x| < 1.
  CALL_SUBTEST(test_step_alignment<Forward>(1e-10));
  CALL_SUBTEST(test_step_alignment<Central>(1e-10));
}
