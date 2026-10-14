// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
// SPDX-License-Identifier: MPL-2.0

#include <stdio.h>

#include "main.h"
#include <contrib/Eigen/NumericalDiff>

// Generic functor
template <typename Scalar_, int NX = Dynamic, int NY = Dynamic>
struct Functor {
  typedef Scalar_ Scalar;
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
  typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs_, int values_) : m_inputs(inputs_), m_values(values_) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }
};

struct my_functor : Functor<double> {
  my_functor(void) : Functor<double>(3, 15) {}
  int operator()(const VectorXd &x, VectorXd &fvec) const {
    double tmp1, tmp2, tmp3;
    double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1,
                    3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34,   2.1,    4.39};

    for (int i = 0; i < values(); i++) {
      tmp1 = i + 1;
      tmp2 = 16 - i - 1;
      tmp3 = (i >= 8) ? tmp2 : tmp1;
      fvec[i] = y[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3));
    }
    return 0;
  }

  int actual_df(const VectorXd &x, MatrixXd &fjac) const {
    double tmp1, tmp2, tmp3, tmp4;
    for (int i = 0; i < values(); i++) {
      tmp1 = i + 1;
      tmp2 = 16 - i - 1;
      tmp3 = (i >= 8) ? tmp2 : tmp1;
      tmp4 = (x[1] * tmp2 + x[2] * tmp3);
      tmp4 = tmp4 * tmp4;
      fjac(i, 0) = -1;
      fjac(i, 1) = tmp1 * tmp2 / tmp4;
      fjac(i, 2) = tmp1 * tmp3 / tmp4;
    }
    return 0;
  }
};

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

void test_forward() {
  VectorXd x(3);
  MatrixXd jac(15, 3);
  MatrixXd actual_jac(15, 3);
  my_functor functor;

  x << 0.082, 1.13, 2.35;

  // real one
  functor.actual_df(x, actual_jac);
  //    std::cout << actual_jac << std::endl << std::endl;

  // using NumericalDiff
  NumericalDiff<my_functor> numDiff(functor);
  numDiff.df(x, jac);
  //    std::cout << jac << std::endl;

  VERIFY_IS_APPROX(jac, actual_jac);
}

void test_central() {
  VectorXd x(3);
  MatrixXd jac(15, 3);
  MatrixXd actual_jac(15, 3);
  my_functor functor;

  x << 0.082, 1.13, 2.35;

  // real one
  functor.actual_df(x, actual_jac);

  // using NumericalDiff
  NumericalDiff<my_functor, Central> numDiff(functor);
  numDiff.df(x, jac);

  VERIFY_IS_APPROX(jac, actual_jac);

  // With the default changed to cbrt(epsilon), the maximum error has become lower
  // This verifies we get the new, lower maximum error.
  VERIFY_LE((jac - actual_jac).cwiseAbs().maxCoeff(), 1e-10);
}

void test_small() {
  VectorXd x(3);
  MatrixXd jac(15, 3);
  MatrixXd actual_jac(15, 3);
  my_functor functor;

  x << 0.082, 1.13, 1e-8;

  // real one
  functor.actual_df(x, actual_jac);

  // using NumericalDiff
  NumericalDiff<my_functor, Central> numDiff(functor);
  numDiff.df(x, jac);

  VERIFY_IS_APPROX(jac, actual_jac);
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

void test_step_size() {
  step_size_functor functor;
  step_size_functor_abs functor_abs;
  VectorXd x(1);
  MatrixXd jac(1, 1);
  MatrixXd expected(1, 1);

  // Test that setting relative step size and absolute step sizes gives the expected answer
  NumericalDiff<step_size_functor> numDiffBothSet(functor);
  numDiffBothSet.setRelTol(1e-4);
  numDiffBothSet.setAbsTol(1e-6);
  x << 1000.0;
  numDiffBothSet.df(x, jac);
  expected << 1e-1;  // rel_tol * 1000.0
  VERIFY_IS_APPROX(jac, expected);

  NumericalDiff<step_size_functor_abs> numDiffBothSetAbs(functor_abs);
  numDiffBothSetAbs.setRelTol(1e-4);
  numDiffBothSetAbs.setAbsTol(1e-6);
  x << 0.0;
  numDiffBothSetAbs.df(x, jac);
  expected << 1e-6;  // abs_tol
  VERIFY_IS_APPROX(jac, expected);

  // Test that setting relative step size only gives abs_tol = rel_tol
  NumericalDiff<step_size_functor> numDiffRelOnly(functor, 1e-8);
  x << 1000.0;
  expected << 1e-1;  // rel_tol * 1000.0
  numDiffRelOnly.df(x, jac);
  VERIFY_IS_APPROX(jac, expected);

  NumericalDiff<step_size_functor_abs> numDiffRelOnlyAbs(functor_abs, 1e-8);
  x << 0.0;
  numDiffRelOnlyAbs.df(x, jac);
  expected << 1e-4;  // abs_tol = rel_tol
  VERIFY_IS_APPROX(jac, expected);

  // Test that setting neither gives rel_tol = abs_tol = sqrt(eps)
  NumericalDiff<step_size_functor> numDiffDefaults(functor);
  x << 1000.0;
  expected << 1000.0 * std::sqrt(std::numeric_limits<double>::epsilon());  // 1000.0 * rel_tol
  numDiffDefaults.df(x, jac);
  VERIFY_IS_APPROX(jac, expected);

  NumericalDiff<step_size_functor_abs> numDiffDefaultsAbs(functor_abs);
  x << 0.0;
  numDiffDefaultsAbs.df(x, jac);
  expected << std::sqrt(std::numeric_limits<double>::epsilon());  // abs_tol = rel_tol
  VERIFY_IS_APPROX(jac, expected);

  // Run the same tests, but for central differences

  // Test that setting relative step size and absolute step sizes gives the expected answer
  NumericalDiff<step_size_functor, Central> numDiffCentralBothSet(functor);
  numDiffCentralBothSet.setRelTol(1e-4);
  numDiffCentralBothSet.setAbsTol(1e-6);
  x << 1000.0;
  numDiffCentralBothSet.df(x, jac);
  expected << 1e-1;  // rel_tol * 1000.0
  VERIFY_IS_APPROX(jac, expected);

  NumericalDiff<step_size_functor_abs, Central> numDiffCentralBothSetAbs(functor_abs);
  numDiffCentralBothSetAbs.setRelTol(1e-4);
  numDiffCentralBothSetAbs.setAbsTol(1e-6);
  x << 0.0;
  numDiffCentralBothSetAbs.df(x, jac);
  expected << 1e-6;  // abs_tol
  VERIFY_IS_APPROX(jac, expected);

  // Test that setting relative step size only gives abs_tol = rel_tol
  NumericalDiff<step_size_functor, Central> numDiffCentralRelOnly(functor, 1e-12);
  x << 1000.0;
  expected << 1e-1;  // rel_tol * 1000.0
  numDiffCentralRelOnly.df(x, jac);
  VERIFY_IS_APPROX(jac, expected);

  NumericalDiff<step_size_functor_abs, Central> numDiffCentralRelOnlyAbs(functor_abs, 1e-12);
  x << 0.0;
  numDiffCentralRelOnlyAbs.df(x, jac);
  expected << 1e-4;  // abs_tol = rel_tol
  VERIFY_IS_APPROX(jac, expected);

  // Test that setting neither gives rel_tol = abs_tol = cbrt(eps)
  NumericalDiff<step_size_functor, Central> numDiffCentralDefaults(functor);
  x << 1000.0;
  expected << 1000.0 * std::cbrt(std::numeric_limits<double>::epsilon());  // 1000.0 * rel_tol
  numDiffCentralDefaults.df(x, jac);
  VERIFY_IS_APPROX(jac, expected);

  NumericalDiff<step_size_functor_abs, Central> numDiffCentralDefaultsAbs(functor_abs);
  x << 0.0;
  numDiffCentralDefaultsAbs.df(x, jac);
  expected << std::cbrt(std::numeric_limits<double>::epsilon());  // abs_tol = rel_tol
  VERIFY_IS_APPROX(jac, expected);
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
