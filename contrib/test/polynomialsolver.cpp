// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <contrib/Eigen/Polynomials>
#include <iostream>
#include <algorithm>

using namespace std;

namespace Eigen {
namespace internal {
template <int Size>
struct increment_if_fixed_size : std::integral_constant<int, (Size == Dynamic) ? Dynamic : Size + 1> {};
}  // namespace internal
}  // namespace Eigen

template <typename PolynomialType>
PolynomialType polyder(const PolynomialType& p) {
  typedef typename PolynomialType::Scalar Scalar;
  PolynomialType res(p.size());
  for (Index i = 1; i < p.size(); ++i) res[i - 1] = p[i] * Scalar(i);
  res[p.size() - 1] = 0.;
  return res;
}

template <int Deg, typename POLYNOMIAL, typename SOLVER>
bool aux_evalSolver(const POLYNOMIAL& pols, SOLVER& psolve) {
  typedef typename POLYNOMIAL::Scalar Scalar;
  typedef typename POLYNOMIAL::RealScalar RealScalar;

  typedef typename SOLVER::RootsType RootsType;
  typedef Matrix<RealScalar, Deg, 1> EvalRootsType;

  const Index deg = pols.size() - 1;

  // Test template constructor from coefficient vector
  SOLVER solve_constr(pols);

  psolve.compute(pols);
  const RootsType& roots(psolve.roots());
  EvalRootsType evr(deg);
  POLYNOMIAL pols_der = polyder(pols);
  EvalRootsType der(deg);
  for (int i = 0; i < roots.size(); ++i) {
    evr[i] = std::abs(poly_eval(pols, roots[i]));
    der[i] = numext::maxi(RealScalar(1.), std::abs(poly_eval(pols_der, roots[i])));
  }

  // we need to divide by the magnitude of the derivative because
  // with a high derivative is very small error in the value of the root
  // yiels a very large error in the polynomial evaluation.
  bool evalToZero = (evr.cwiseQuotient(der)).isZero(test_precision<Scalar>());
  if (!evalToZero) {
    cerr << "WRONG root: " << endl;
    cerr << "Polynomial: " << pols.transpose() << endl;
    cerr << "Roots found: " << roots.transpose() << endl;
    cerr << "Abs value of the polynomial at the roots: " << evr.transpose() << endl;
    cerr << endl;
  }

  std::vector<RealScalar> rootModuli(roots.size());
  Map<EvalRootsType> aux(&rootModuli[0], roots.size());
  aux = roots.array().abs();
  std::sort(rootModuli.begin(), rootModuli.end());
  bool distinctModuli = true;
  for (size_t i = 1; i < rootModuli.size() && distinctModuli; ++i) {
    if (internal::isApprox(rootModuli[i], rootModuli[i - 1])) {
      distinctModuli = false;
    }
  }
  VERIFY(evalToZero || !distinctModuli);

  return distinctModuli;
}

template <int Deg, typename POLYNOMIAL>
void evalSolver(const POLYNOMIAL& pols) {
  typedef typename POLYNOMIAL::Scalar Scalar;

  typedef PolynomialSolver<Scalar, Deg> PolynomialSolverType;

  PolynomialSolverType psolve;
  aux_evalSolver<Deg, POLYNOMIAL, PolynomialSolverType>(pols, psolve);
}

template <typename Solver>
void verify_polynomialsolver_sugar(const Solver& solver, typename Solver::RealScalar threshold) {
  using Real = typename Solver::RealScalar;
  const auto& computed = solver.roots();
  Index greatest, smallest;
  computed.cwiseAbs2().maxCoeff(&greatest);
  computed.cwiseAbs2().minCoeff(&smallest);
  VERIFY_IS_EQUAL(solver.greatestRoot(), computed[greatest]);
  VERIFY_IS_EQUAL(solver.smallestRoot(), computed[smallest]);

  std::vector<Real> expectedRealRoots, expectedRealExtrema;
  for (Index i = 0; i < computed.size(); ++i) {
    // realRoots uses a strict threshold; the extremal queries include its boundary.
    if (numext::abs(computed[i].imag()) < threshold) expectedRealRoots.push_back(computed[i].real());
    if (numext::abs(computed[i].imag()) <= threshold) expectedRealExtrema.push_back(computed[i].real());
  }
  std::vector<Real> actualRealRoots;
  solver.realRoots(actualRealRoots, threshold);
  VERIFY_IS_EQUAL(actualRealRoots.size(), expectedRealRoots.size());
  for (size_t i = 0; i < expectedRealRoots.size(); ++i) VERIFY_IS_EQUAL(actualRealRoots[i], expectedRealRoots[i]);

  const bool expectedHasRealRoot = !expectedRealExtrema.empty();
  const auto absLess = [](Real a, Real b) { return numext::abs(a) < numext::abs(b); };
  bool hasRealRoot;
  Real result = solver.absGreatestRealRoot(hasRealRoot, threshold);
  VERIFY_IS_EQUAL(hasRealRoot, expectedHasRealRoot);
  if (hasRealRoot)
    VERIFY_IS_EQUAL(result, *std::max_element(expectedRealExtrema.begin(), expectedRealExtrema.end(), absLess));
  result = solver.absSmallestRealRoot(hasRealRoot, threshold);
  VERIFY_IS_EQUAL(hasRealRoot, expectedHasRealRoot);
  if (hasRealRoot)
    VERIFY_IS_EQUAL(result, *std::min_element(expectedRealExtrema.begin(), expectedRealExtrema.end(), absLess));
  result = solver.greatestRealRoot(hasRealRoot, threshold);
  VERIFY_IS_EQUAL(hasRealRoot, expectedHasRealRoot);
  if (hasRealRoot) VERIFY_IS_EQUAL(result, *std::max_element(expectedRealExtrema.begin(), expectedRealExtrema.end()));
  result = solver.smallestRealRoot(hasRealRoot, threshold);
  VERIFY_IS_EQUAL(hasRealRoot, expectedHasRealRoot);
  if (hasRealRoot) VERIFY_IS_EQUAL(result, *std::min_element(expectedRealExtrema.begin(), expectedRealExtrema.end()));
}

template <int Deg, typename POLYNOMIAL, typename REAL_ROOTS>
void evalSolverSugarFunction(const POLYNOMIAL& pols, const REAL_ROOTS& real_roots) {
  using Scalar = typename POLYNOMIAL::Scalar;
  using RealScalar = typename POLYNOMIAL::RealScalar;
  using PolynomialSolverType = PolynomialSolver<Scalar, Deg>;

  PolynomialSolverType psolve;
  if (aux_evalSolver<Deg, POLYNOMIAL, PolynomialSolverType>(pols, psolve)) {
    // First-order root displacement estimate: delta * sum_k |r_j|^k / |p'(r_j)| for a monic polynomial,
    // p'(r_j) = prod_{k != j} (r_j - r_k). Use delta = 32 eps max_k |a_k| for the companion eigenvalue error.
    const RealScalar coefficientError = RealScalar(32) * NumTraits<RealScalar>::epsilon() * pols.cwiseAbs().maxCoeff();
    Matrix<RealScalar, Dynamic, 1> tolerance(real_roots.size());
    for (Index j = 0; j < real_roots.size(); ++j) {
      const RealScalar root = real_roots[j];
      RealScalar powerSum = RealScalar(0), power = RealScalar(1), derivative = RealScalar(1);
      for (Index k = 0; k < pols.size(); ++k) {
        powerSum += power;
        power *= numext::abs(root);
      }
      for (Index k = 0; k < real_roots.size(); ++k) {
        if (k != j) derivative *= numext::abs(root - real_roots[k]);
      }
      tolerance[j] = coefficientError * powerSum / derivative;
    }
    // A broad cluster tolerance must not hide a missing, well-conditioned root.
    for (Index j = 0; j < real_roots.size(); ++j) {
      VERIFY((numext::isfinite)(tolerance[j]));
      VERIFY((psolve.roots().array() - real_roots[j]).abs().minCoeff() <= tolerance[j]);
    }
    for (Index i = 0; i < psolve.roots().size(); ++i) {
      bool found = false;
      for (Index j = 0; j < real_roots.size() && !found; ++j) {
        VERIFY((numext::isfinite)(tolerance[j]));
        if (numext::abs(psolve.roots()[i] - real_roots[j]) <= tolerance[j]) found = true;
      }
      VERIFY(found);
    }
  }
  verify_polynomialsolver_sugar(psolve, numext::sqrt(test_precision<RealScalar>()));
}

void polynomialsolver_sugar_cluster() {
  Matrix<float, 7, 1> roots;
  roots << -0.8f, 0.2f, 0.2001f, 0.5f, 0.7f, 0.9f, 1.0f;
  Matrix<float, 8, 1> poly;
  roots_to_monicPolynomial(roots, poly);
  evalSolverSugarFunction<7>(poly, roots);
}

void polynomialsolver_sugar_filtering() {
  Vector4d poly;
  poly << 0, 1, 0, 1;
  PolynomialSolver<double, 3> solver(poly);
  verify_polynomialsolver_sugar(solver, 0.0);
  verify_polynomialsolver_sugar(solver, 0.5);
  verify_polynomialsolver_sugar(solver, 1.0);
  verify_polynomialsolver_sugar(solver, 2.0);
  Vector3d noRealRoots;
  noRealRoots << 1, 0, 1;
  PolynomialSolver<double, 2> complexSolver(noRealRoots);
  verify_polynomialsolver_sugar(complexSolver, 0.5);
}

template <typename Scalar_, int Deg_>
void polynomialsolver(int deg) {
  typedef typename NumTraits<Scalar_>::Real RealScalar;
  typedef internal::increment_if_fixed_size<Deg_> Dim;
  typedef Matrix<Scalar_, Dim::value, 1> PolynomialType;
  typedef Matrix<Scalar_, Deg_, 1> EvalRootsType;
  typedef Matrix<RealScalar, Deg_, 1> RealRootsType;

  cout << "Standard cases" << endl;
  PolynomialType pols = PolynomialType::Random(deg + 1);
  evalSolver<Deg_, PolynomialType>(pols);

  cout << "Hard cases" << endl;
  Scalar_ multipleRoot = internal::random<Scalar_>();
  EvalRootsType allRoots = EvalRootsType::Constant(deg, multipleRoot);
  roots_to_monicPolynomial(allRoots, pols);
  evalSolver<Deg_, PolynomialType>(pols);

  // The companion matrix eigenvalue approach has limited accuracy for float at
  // high degrees. The PolynomialSolver documentation itself warns: "With 32bit
  // (float) floating types this problem shows up frequently." Skip the sugar
  // function test (which requires exact root matching) for float beyond degree 8.
  if (deg <= 8 || sizeof(RealScalar) > sizeof(float)) {
    cout << "Test sugar" << endl;
    RealRootsType realRoots = RealRootsType::Random(deg);
    // sort by ascending absolute value to mitigate precision lost during polynomial expansion
    std::sort(realRoots.begin(), realRoots.end(),
              [](RealScalar a, RealScalar b) { return numext::abs(a) < numext::abs(b); });
    roots_to_monicPolynomial(realRoots, pols);
    evalSolverSugarFunction<Deg_>(pols, realRoots);
  }
}

// Componentwise backward error |p(z)| / sum_k |a_k| |z|^k of a computed root, evaluated in WideReal so that the check
// does not share the solver's rounding.
template <typename WideReal, typename PolynomialType, typename RootType>
WideReal root_backward_error(const PolynomialType& pols, const RootType& root) {
  const std::complex<WideReal> z(WideReal(numext::real(root)), WideReal(numext::imag(root)));
  std::complex<WideReal> value(0);
  WideReal magnitude(0);
  for (Index k = pols.size() - 1; k >= 0; --k) {
    const std::complex<WideReal> coefficient(WideReal(numext::real(pols[k])), WideReal(numext::imag(pols[k])));
    value = value * z + coefficient;
    magnitude = magnitude * numext::abs(z) + numext::abs(coefficient);
  }
  return numext::abs(value) / magnitude;
}

// Every refined root must be an exact root of a nearby polynomial, and, where first-order perturbation theory applies,
// within its condition bound of the same polynomial's roots solved in WideScalar. The roots of a real polynomial must
// be real or exact conjugate pairs.
template <typename Scalar, typename WideScalar, int Deg>
void polynomialsolver_refinement_accuracy(int deg) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using WideReal = typename NumTraits<WideScalar>::Real;
  using PolynomialType = Matrix<Scalar, internal::increment_if_fixed_size<Deg>::value, 1>;
  using RootsType = Matrix<Scalar, Deg, 1>;
  const RootsType roots = RootsType::Random(deg);
  PolynomialType pols;
  roots_to_monicPolynomial(roots, pols);
  PolynomialSolver<Scalar, Deg> solver(pols);
  const WideReal eps = WideReal(NumTraits<RealScalar>::epsilon());

  // A Newton correction below one ulp bounds |p(z)| by eps |z p'(z)| <= deg eps sum_k |a_k| |z|^k; sampled roots of
  // float and double polynomials up to degree 50 stay below 6 eps.
  for (Index i = 0; i < deg; ++i)
    VERIFY(root_backward_error<WideReal>(pols, solver.roots()[i]) <= WideReal(4 * deg) * eps);

  if (!NumTraits<Scalar>::IsComplex) {
    for (Index i = 0; i < deg; ++i) {
      if (numext::imag(solver.roots()[i]) == RealScalar(0)) continue;
      bool paired = false;
      for (Index j = 0; j < deg && !paired; ++j)
        paired = j != i && solver.roots()[j] == numext::conj(solver.roots()[i]);
      VERIFY(paired);
    }
  }

  // A coefficient perturbation of size delta moves a simple root r_j by at most
  // delta * sum_k |r_j|^k / prod_{k != j} |r_j - r_k| to first order, valid while that is small against the gap to the
  // nearest other root. Each such root must have a computed root within 16 times the bound at delta = eps max_k |a_k|.
  const PolynomialSolver<WideScalar, Deg> reference(pols.template cast<WideScalar>().eval());
  const WideReal delta = eps * WideReal(pols.cwiseAbs().maxCoeff());
  for (Index j = 0; j < deg; ++j) {
    const std::complex<WideReal> r = reference.roots()[j];
    WideReal powerSum(0), power(1), derivative(1), gap = NumTraits<WideReal>::highest();
    for (Index k = 0; k <= deg; ++k) {
      powerSum += power;
      power *= numext::abs(r);
    }
    for (Index k = 0; k < deg; ++k) {
      if (k == j) continue;
      const WideReal d = numext::abs(r - std::complex<WideReal>(reference.roots()[k]));
      derivative *= d;
      gap = numext::mini(gap, d);
    }
    const WideReal bound = WideReal(16) * delta * powerSum / derivative;
    if (!(WideReal(4) * bound < gap)) continue;
    WideReal distance = NumTraits<WideReal>::highest();
    for (Index i = 0; i < deg; ++i) {
      const std::complex<WideReal> z(WideReal(numext::real(solver.roots()[i])),
                                     WideReal(numext::imag(solver.roots()[i])));
      distance = numext::mini(distance, numext::abs(z - r));
    }
    VERIFY(distance <= bound);
  }
}

// At |z| = 3 rounding dominates the residual of the pair 0.1 +- 3i, which is still an accurate root; comparing that
// residual with the one at the real part 0.1 once reported the pair as a double real root.
void polynomialsolver_complex_pair_kept() {
  Matrix<std::complex<float>, 12, 1> roots;
  roots << std::complex<float>(0.1f, 3.0f), std::complex<float>(0.1f, -3.0f), 0.2f, -0.2f, 0.4f, -0.4f, 0.6f, -0.6f,
      0.8f, -0.8f, 1.0f, -1.0f;
  Matrix<std::complex<float>, 13, 1> complexPolynomial;
  roots_to_monicPolynomial(roots, complexPolynomial);
  const Matrix<float, 13, 1> pols = complexPolynomial.real();
  PolynomialSolver<float, 12> solver(pols);
  Index farFromAxis = 0;
  for (Index i = 0; i < 12; ++i) {
    VERIFY(root_backward_error<double>(pols, solver.roots()[i]) <= 48 * double(NumTraits<float>::epsilon()));
    if (numext::abs(numext::imag(solver.roots()[i])) > 1.0f) ++farFromAxis;
  }
  VERIFY_IS_EQUAL(farFromAxis, Index(2));
}

// The companion eigenvalue of a root at 1e-12 carries an absolute error of order eps, a backward error of about
// 3e3 eps; refinement restores the root's full relative accuracy.
void polynomialsolver_tiny_root() {
  Matrix<double, 5, 1> roots;
  roots << 1e-12, 0.25, -0.5, 0.75, -1.0;
  Matrix<double, 6, 1> pols;
  roots_to_monicPolynomial(roots, pols);
  PolynomialSolver<double, 5> solver(pols);
  const double eps = NumTraits<double>::epsilon();
  bool found = false;
  for (Index i = 0; i < 5; ++i) {
    VERIFY(root_backward_error<long double>(pols, solver.roots()[i]) <= static_cast<long double>(20 * eps));
    found = found || numext::abs(solver.roots()[i] - std::complex<double>(1e-12)) <= 16 * eps * 1e-12;
  }
  VERIFY(found);
}

template <typename Polynomial, typename Roots>
void polynomialsolver_verify_root_set(const Polynomial& poly, const Roots& expected) {
  using Scalar = typename Polynomial::Scalar;
  using Real = typename NumTraits<Scalar>::Real;
  const PolynomialSolver<Scalar, Roots::RowsAtCompileTime> solver(poly);
  Array<bool, Dynamic, 1> matched = Array<bool, Dynamic, 1>::Constant(expected.size(), false);
  VERIFY_IS_EQUAL(solver.roots().size(), expected.size());
  // These exactly represented polynomials have well-separated simple roots or exact roots at zero.
  const Real tolerance = Real(64) * NumTraits<Real>::epsilon();
  for (Index i = 0; i < expected.size(); ++i) {
    Index match = expected.size();
    for (Index j = 0; j < expected.size(); ++j) {
      if (!matched[j] && numext::abs(solver.roots()[i] - expected[j]) <= tolerance) {
        match = j;
        break;
      }
    }
    VERIFY(match < expected.size());
    matched[match] = true;
  }
}

template <typename Scalar>
void polynomialsolver_real_part_is_another_root() {
  using Real = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<Real>;
  Matrix<Scalar, 4, 1> poly;
  Matrix<Complex, 3, 1> expected;
  poly << 0, 1, 0, 1;
  expected << Complex(0), Complex(0, 1), Complex(0, -1);
  polynomialsolver_verify_root_set(poly, expected);
  poly << -2, 4, -3, 1;
  expected << Complex(1), Complex(1, 1), Complex(1, -1);
  polynomialsolver_verify_root_set(poly, expected);

  Matrix<Scalar, Dynamic, 1> repeated(5);
  Matrix<Complex, Dynamic, 1> repeatedExpected(4);
  repeated << 0, 0, 1, 0, 1;
  repeatedExpected << Complex(0), Complex(0), Complex(0, 1), Complex(0, -1);
  polynomialsolver_verify_root_set(repeated, repeatedExpected);
}

template <typename Scalar>
void polynomialsolver_scaled_quadratic() {
  using Real = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<Real>;
  Matrix<Scalar, 3, 1> poly;
  Matrix<Complex, 2, 1> expected;
  expected << Complex(0, 1), Complex(0, -1);
  const Real scales[] = {Real(1), NumTraits<Real>::highest() * Real(0.75), (std::numeric_limits<Real>::min)()};
  for (Real scale : scales) {
    poly << Scalar(scale), Scalar(0), Scalar(scale);
    polynomialsolver_verify_root_set(poly, expected);
  }
}

template <int Degree>
void polynomialsolver_real_starts_for_complex_pair() {
  Matrix<double, 7, 1> poly;
  poly << 0.015754773452117808, -0.2921305061016638, 1.8654828811488313, -4.9912618133859823, 6.4755674799802918,
      -4.0734106361481288, 1;
  PolynomialSolver<double, Degree> solver(poly);
  // A nearly double nonreal pair initially appears as two real eigenvalues. Unconverged real-axis refinement
  // used to increase the maximum componentwise backward error from 2.2 eps to 3.1e5 eps.
  const long double tolerance = 32 * static_cast<long double>(NumTraits<double>::epsilon());
  for (Index i = 0; i < solver.roots().size(); ++i)
    VERIFY(root_backward_error<long double>(poly, solver.roots()[i]) <= tolerance);
}

EIGEN_DECLARE_TEST(polynomialsolver) {
  CALL_SUBTEST_7(polynomialsolver_sugar_cluster());
  CALL_SUBTEST_13(polynomialsolver_sugar_filtering());
  CALL_SUBTEST_18(polynomialsolver_real_starts_for_complex_pair<6>());
  CALL_SUBTEST_18(polynomialsolver_real_starts_for_complex_pair<Dynamic>());
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((polynomialsolver<float, 1>(1)));
    CALL_SUBTEST_2((polynomialsolver<double, 2>(2)));
    CALL_SUBTEST_3((polynomialsolver<double, 3>(3)));
    CALL_SUBTEST_4((polynomialsolver<float, 4>(4)));
    CALL_SUBTEST_5((polynomialsolver<double, 5>(5)));
    CALL_SUBTEST_6((polynomialsolver<float, 6>(6)));
    CALL_SUBTEST_7((polynomialsolver<float, 7>(7)));
    CALL_SUBTEST_8((polynomialsolver<double, 8>(8)));

    CALL_SUBTEST_9((polynomialsolver<float, Dynamic>(internal::random<int>(9, 13))));
    CALL_SUBTEST_10((polynomialsolver<double, Dynamic>(internal::random<int>(9, 13))));
    CALL_SUBTEST_11((polynomialsolver<float, Dynamic>(1)));
    CALL_SUBTEST_12((polynomialsolver<std::complex<double>, Dynamic>(internal::random<int>(2, 13))));

    CALL_SUBTEST_13((polynomialsolver_refinement_accuracy<float, double, 4>(4)));
    CALL_SUBTEST_19((polynomialsolver_refinement_accuracy<float, double, 6>(6)));
    CALL_SUBTEST_20((polynomialsolver_refinement_accuracy<float, double, 7>(7)));
    CALL_SUBTEST_14((polynomialsolver_refinement_accuracy<float, double, Dynamic>(internal::random<int>(8, 13))));
    CALL_SUBTEST_21((polynomialsolver_refinement_accuracy<std::complex<float>, std::complex<double>, Dynamic>(
        internal::random<int>(2, 13))));
    CALL_SUBTEST_22((polynomialsolver_refinement_accuracy<double, long double, Dynamic>(internal::random<int>(2, 20))));
  }
  CALL_SUBTEST_15(polynomialsolver_complex_pair_kept());
  CALL_SUBTEST_15(polynomialsolver_tiny_root());
  CALL_SUBTEST_16(polynomialsolver_real_part_is_another_root<float>());
  CALL_SUBTEST_23(polynomialsolver_real_part_is_another_root<double>());
  CALL_SUBTEST_24(polynomialsolver_real_part_is_another_root<std::complex<double>>());
  CALL_SUBTEST_17(polynomialsolver_scaled_quadratic<float>());
  CALL_SUBTEST_17(polynomialsolver_scaled_quadratic<double>());
  CALL_SUBTEST_25(polynomialsolver_scaled_quadratic<std::complex<double>>());
}
