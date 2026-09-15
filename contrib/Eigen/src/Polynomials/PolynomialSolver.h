// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_POLYNOMIAL_SOLVER_H
#define EIGEN_POLYNOMIAL_SOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup Polynomials_Module
 *  \class PolynomialSolverBase.
 *
 * \brief Defined to be inherited by polynomial solvers: it provides
 * convenient methods such as
 *  - real roots,
 *  - greatest, smallest complex roots,
 *  - real roots with greatest, smallest absolute real value,
 *  - greatest, smallest real roots.
 *
 * It stores the set of roots as a vector of complexes.
 *
 */
template <typename Scalar_, int Deg_>
class PolynomialSolverBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(Scalar_, Deg_ == Dynamic ? Dynamic : Deg_)

  typedef Scalar_ Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef internal::make_complex_t<Scalar> RootType;
  typedef Matrix<RootType, Deg_, 1> RootsType;

  typedef DenseIndex Index;

 protected:
  template <typename OtherPolynomial>
  inline void setPolynomial(const OtherPolynomial& poly) {
    m_roots.resize(poly.size() - 1);
  }

 public:
  template <typename OtherPolynomial>
  inline PolynomialSolverBase(const OtherPolynomial& poly) {
    setPolynomial(poly());
  }

  inline PolynomialSolverBase() {}

 public:
  /** \returns the complex roots of the polynomial */
  inline const RootsType& roots() const { return m_roots; }

 public:
  /** Clear and fills the back insertion sequence with the real roots of the polynomial
   * i.e. the real part of the complex roots that have an imaginary part which
   * absolute value is smaller than absImaginaryThreshold.
   * absImaginaryThreshold takes the dummy_precision associated
   * with the Scalar_ template parameter of the PolynomialSolver class as the default value.
   *
   * \param[out] bi_seq : the back insertion sequence (stl concept)
   * \param[in]  absImaginaryThreshold : the maximum bound of the imaginary part of a complex
   *  number that is considered as real.
   * */
  template <typename Stl_back_insertion_sequence>
  inline void realRoots(Stl_back_insertion_sequence& bi_seq,
                        const RealScalar& absImaginaryThreshold = NumTraits<Scalar>::dummy_precision()) const {
    using std::abs;
    bi_seq.clear();
    for (Index i = 0; i < m_roots.size(); ++i) {
      if (abs(m_roots[i].imag()) < absImaginaryThreshold) {
        bi_seq.push_back(m_roots[i].real());
      }
    }
  }

 protected:
  template <typename Predicate>
  inline const RootType& selectComplexRoot_withRespectToNorm(Predicate& pred) const {
    Index res = 0;
    RealScalar norm2 = numext::abs2(m_roots[0]);
    for (Index i = 1; i < m_roots.size(); ++i) {
      const RealScalar currNorm2 = numext::abs2(m_roots[i]);
      if (pred(currNorm2, norm2)) {
        res = i;
        norm2 = currNorm2;
      }
    }
    return m_roots[res];
  }

 public:
  /**
   * \returns the complex root with greatest norm.
   */
  inline const RootType& greatestRoot() const {
    std::greater<RealScalar> greater;
    return selectComplexRoot_withRespectToNorm(greater);
  }

  /**
   * \returns the complex root with smallest norm.
   */
  inline const RootType& smallestRoot() const {
    std::less<RealScalar> less;
    return selectComplexRoot_withRespectToNorm(less);
  }

 protected:
  template <typename Predicate>
  inline const RealScalar& selectRealRoot_withRespectToAbsRealPart(
      Predicate& pred, bool& hasArealRoot,
      const RealScalar& absImaginaryThreshold = NumTraits<Scalar>::dummy_precision()) const {
    using std::abs;
    hasArealRoot = false;
    Index res = 0;
    RealScalar val(0);

    for (Index i = 0; i < m_roots.size(); ++i) {
      if (abs(m_roots[i].imag()) <= absImaginaryThreshold) {
        if (!hasArealRoot) {
          hasArealRoot = true;
          res = i;
          val = abs(m_roots[i].real());
        } else {
          const RealScalar curr = abs(m_roots[i].real());
          if (pred(curr, val)) {
            val = curr;
            res = i;
          }
        }
      } else if (!hasArealRoot) {
        if (abs(m_roots[i].imag()) < abs(m_roots[res].imag())) {
          res = i;
        }
      }
    }
    return numext::real_ref(m_roots[res]);
  }

  template <typename Predicate>
  inline const RealScalar& selectRealRoot_withRespectToRealPart(
      Predicate& pred, bool& hasArealRoot,
      const RealScalar& absImaginaryThreshold = NumTraits<Scalar>::dummy_precision()) const {
    using std::abs;
    hasArealRoot = false;
    Index res = 0;
    RealScalar val(0);

    for (Index i = 0; i < m_roots.size(); ++i) {
      if (abs(m_roots[i].imag()) <= absImaginaryThreshold) {
        if (!hasArealRoot) {
          hasArealRoot = true;
          res = i;
          val = m_roots[i].real();
        } else {
          const RealScalar curr = m_roots[i].real();
          if (pred(curr, val)) {
            val = curr;
            res = i;
          }
        }
      } else {
        if (abs(m_roots[i].imag()) < abs(m_roots[res].imag())) {
          res = i;
        }
      }
    }
    return numext::real_ref(m_roots[res]);
  }

 public:
  /**
   * \returns a real root with greatest absolute magnitude.
   * A real root is defined as the real part of a complex root with absolute imaginary
   * part smaller than absImaginaryThreshold.
   * absImaginaryThreshold takes the dummy_precision associated
   * with the Scalar_ template parameter of the PolynomialSolver class as the default value.
   * If no real root is found the boolean hasArealRoot is set to false and the real part of
   * the root with smallest absolute imaginary part is returned instead.
   *
   * \param[out] hasArealRoot : boolean true if a real root is found according to the
   *  absImaginaryThreshold criterion, false otherwise.
   * \param[in] absImaginaryThreshold : threshold on the absolute imaginary part to decide
   *  whether or not a root is real.
   */
  inline const RealScalar& absGreatestRealRoot(
      bool& hasArealRoot, const RealScalar& absImaginaryThreshold = NumTraits<Scalar>::dummy_precision()) const {
    std::greater<RealScalar> greater;
    return selectRealRoot_withRespectToAbsRealPart(greater, hasArealRoot, absImaginaryThreshold);
  }

  /**
   * \returns a real root with smallest absolute magnitude.
   * A real root is defined as the real part of a complex root with absolute imaginary
   * part smaller than absImaginaryThreshold.
   * absImaginaryThreshold takes the dummy_precision associated
   * with the Scalar_ template parameter of the PolynomialSolver class as the default value.
   * If no real root is found the boolean hasArealRoot is set to false and the real part of
   * the root with smallest absolute imaginary part is returned instead.
   *
   * \param[out] hasArealRoot : boolean true if a real root is found according to the
   *  absImaginaryThreshold criterion, false otherwise.
   * \param[in] absImaginaryThreshold : threshold on the absolute imaginary part to decide
   *  whether or not a root is real.
   */
  inline const RealScalar& absSmallestRealRoot(
      bool& hasArealRoot, const RealScalar& absImaginaryThreshold = NumTraits<Scalar>::dummy_precision()) const {
    std::less<RealScalar> less;
    return selectRealRoot_withRespectToAbsRealPart(less, hasArealRoot, absImaginaryThreshold);
  }

  /**
   * \returns the real root with greatest value.
   * A real root is defined as the real part of a complex root with absolute imaginary
   * part smaller than absImaginaryThreshold.
   * absImaginaryThreshold takes the dummy_precision associated
   * with the Scalar_ template parameter of the PolynomialSolver class as the default value.
   * If no real root is found the boolean hasArealRoot is set to false and the real part of
   * the root with smallest absolute imaginary part is returned instead.
   *
   * \param[out] hasArealRoot : boolean true if a real root is found according to the
   *  absImaginaryThreshold criterion, false otherwise.
   * \param[in] absImaginaryThreshold : threshold on the absolute imaginary part to decide
   *  whether or not a root is real.
   */
  inline const RealScalar& greatestRealRoot(
      bool& hasArealRoot, const RealScalar& absImaginaryThreshold = NumTraits<Scalar>::dummy_precision()) const {
    std::greater<RealScalar> greater;
    return selectRealRoot_withRespectToRealPart(greater, hasArealRoot, absImaginaryThreshold);
  }

  /**
   * \returns the real root with smallest value.
   * A real root is defined as the real part of a complex root with absolute imaginary
   * part smaller than absImaginaryThreshold.
   * absImaginaryThreshold takes the dummy_precision associated
   * with the Scalar_ template parameter of the PolynomialSolver class as the default value.
   * If no real root is found the boolean hasArealRoot is set to false and the real part of
   * the root with smallest absolute imaginary part is returned instead.
   *
   * \param[out] hasArealRoot : boolean true if a real root is found according to the
   *  absImaginaryThreshold criterion, false otherwise.
   * \param[in] absImaginaryThreshold : threshold on the absolute imaginary part to decide
   *  whether or not a root is real.
   */
  inline const RealScalar& smallestRealRoot(
      bool& hasArealRoot, const RealScalar& absImaginaryThreshold = NumTraits<Scalar>::dummy_precision()) const {
    std::less<RealScalar> less;
    return selectRealRoot_withRespectToRealPart(less, hasArealRoot, absImaginaryThreshold);
  }

 protected:
  RootsType m_roots;
};

#define EIGEN_POLYNOMIAL_SOLVER_BASE_INHERITED_TYPES(BASE) \
  typedef typename BASE::Scalar Scalar;                    \
  typedef typename BASE::RealScalar RealScalar;            \
  typedef typename BASE::RootType RootType;                \
  typedef typename BASE::RootsType RootsType;

/** \ingroup Polynomials_Module
 *
 * \class PolynomialSolver
 *
 * \brief A polynomial solver
 *
 * Computes the complex roots of a real polynomial.
 *
 * \param Scalar_ the scalar type, i.e., the type of the polynomial coefficients
 * \param Deg_ the degree of the polynomial, can be a compile time value or Dynamic.
 *             Notice that the number of polynomial coefficients is Deg_+1.
 *
 * This class implements a polynomial solver and provides convenient methods such as
 * - real roots,
 * - greatest, smallest complex roots,
 * - real roots with greatest, smallest absolute real value.
 * - greatest, smallest real roots.
 *
 * WARNING: this polynomial solver is experimental, part of the contrib Eigen modules.
 *
 *
 * The eigenvalues of the balanced companion matrix of the polynomial give first approximations of
 * the roots. Ehrlich-Aberth iterations on the polynomial itself target a residual within the rounding
 * error of evaluation, with a finite sweep limit and an initial-estimate fallback for unfinished
 * iterates whose residual increased. The roots of a real polynomial are returned as real numbers or exact
 * conjugate pairs. A root of multiplicity \f$ m \f$ can only be located to about
 * \f$ \varepsilon^{1/m} \f$.
 */
template <typename Scalar_, int Deg_>
class PolynomialSolver : public PolynomialSolverBase<Scalar_, Deg_> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(Scalar_, Deg_ == Dynamic ? Dynamic : Deg_)

  typedef PolynomialSolverBase<Scalar_, Deg_> PS_Base;
  EIGEN_POLYNOMIAL_SOLVER_BASE_INHERITED_TYPES(PS_Base)

  typedef Matrix<Scalar, Deg_, Deg_> CompanionMatrixType;
  typedef std::conditional_t<NumTraits<Scalar>::IsComplex, ComplexEigenSolver<CompanionMatrixType>,
                             EigenSolver<CompanionMatrixType> >
      EigenSolverType;
  typedef internal::make_complex_t<Scalar_> ComplexScalar;

 public:
  /** Computes the complex roots of a new polynomial. */
  template <typename OtherPolynomial>
  void compute(const OtherPolynomial& poly) {
    eigen_assert(Scalar(0) != poly[poly.size() - 1]);
    eigen_assert(poly.size() > 1);
    if (poly.size() > 2) {
      internal::companion<Scalar, Deg_> companion(poly);
      companion.balance();
      m_eigenSolver.compute(companion.denseMatrix());
      eigen_assert(m_eigenSolver.info() == Eigen::Success);
      m_roots = m_eigenSolver.eigenvalues();
      // Uniform coefficient scaling must not overflow or underflow the refinement's Horner recurrences.
      const RealScalar scale = poly.realView().cwiseAbs().maxCoeff();
      const auto scaledPoly = (poly / scale).eval();
      refineRoots(scaledPoly);
      cleanUpRoots(scaledPoly);
    } else if (poly.size() == 2) {
      m_roots.resize(1);
      m_roots[0] = -poly[0] / poly[1];
    }
  }

 public:
  template <typename OtherPolynomial>
  inline PolynomialSolver(const OtherPolynomial& poly) {
    compute(poly);
  }

  inline PolynomialSolver() {}

 protected:
  // Evaluates p(z) and p'(z) by Horner's rule. With withBound, returns Higham's running bound on the rounding error
  // of the value (Higham 2002, Algorithm 5.1), otherwise zero: the moduli it needs cost more than the recurrence.
  // The real bound is u (2 mu - |p(z)|), u = eps / 2; a complex product rounds by at most 2 sqrt(2) u |z| |y| and a
  // complex sum by u |y|, so the complex bound is sqrt(2) eps (2 mu - |p(z)|).
  template <typename OtherPolynomial>
  static RealScalar evaluate(const OtherPolynomial& poly, const RootType& z, RootType& value, RootType& derivative,
                             bool withBound) {
    const Index degree = poly.size() - 1;
    value = RootType(poly[degree]);
    derivative = RootType(0);
    if (!withBound) {
      for (Index k = degree - 1; k >= 0; --k) {
        derivative = derivative * z + value;
        value = value * z + RootType(poly[k]);
      }
      return RealScalar(0);
    }
    const RealScalar absz = numext::abs(z);
    RealScalar mu = numext::abs(value) / RealScalar(2);
    for (Index k = degree - 1; k >= 0; --k) {
      derivative = derivative * z + value;
      value = value * z + RootType(poly[k]);
      mu = mu * absz + numext::abs(value);
    }
    const RealScalar errorScale = numext::sqrt(RealScalar(2)) * NumTraits<RealScalar>::epsilon();
    return (RealScalar(2) * errorScale) * mu - errorScale * numext::abs(value);
  }

  /** Refines the eigenvalue estimates in m_roots by Ehrlich-Aberth iterations (Ehrlich 1967; Aberth 1973),
   * \f$ z_i \leftarrow z_i - w_i / (1 - w_i \sum_{j \ne i} (z_i - z_j)^{-1}) \f$ with \f$ w_i = p(z_i) / p'(z_i) \f$.
   * The iteration is cubic for simple roots, and the sum keeps the iterates apart, so every root is refined at
   * once without deflation. Updates use the latest estimates; real starting values receive an epsilon-sized imaginary
   * perturbation so they can reach nonreal roots. At the sweep limit, unfinished estimates are compared with
   * their initial polynomial residuals. A root is final once \f$ |p(z_i)| \f$ is
   * within the rounding bound of its evaluation, so that \f$ z_i \f$ is an exact root of a polynomial that close to
   * \f$ p \f$ (the stopping rule of Bini 1996), or once its Newton correction is below one ulp. */
  template <typename OtherPolynomial>
  void refineRoots(const OtherPolynomial& poly) {
    const Index n = m_roots.size();
    const RealScalar eps = NumTraits<RealScalar>::epsilon();
    // Convergence is not guaranteed for clustered roots; retain the eigenvalue estimates as a fallback.
    const int maxSweeps = NumTraits<RealScalar>::digits();
    EIGEN_IF_CONSTEXPR (!NumTraits<Scalar>::IsComplex) {
      for (Index i = 0; i < n; ++i) {
        // Real iterates cannot reach a nonreal root; perturb by +/- i eps |z|.
        if (numext::imag(m_roots[i]) == RealScalar(0)) {
          const RealScalar perturbation = eps * numext::abs(m_roots[i]);
          m_roots[i] += RootType(0, i % 2 == 0 ? perturbation : -perturbation);
        }
      }
    }
    Array<bool, Deg_, 1> active = Array<bool, Deg_, 1>::Constant(n, true);
    RootType value, derivative;
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
      bool moving = false;
      for (Index i = 0; i < n; ++i) {
        if (!active[i]) continue;
        const RootType z = m_roots[i];
        // The eigenvalues can pass the rounding test with a backward error of ~10 eps, while one step from them
        // lands near eps, so the test applies from the second sweep on.
        const bool testResidual = sweep > 0;
        const RealScalar bound = evaluate(poly, z, value, derivative, testResidual);
        if ((testResidual && !(numext::abs(value) > bound)) || !(numext::isfinite)(bound) ||
            derivative == RootType(0)) {
          active[i] = false;
          continue;
        }
        const RootType newton = value / derivative;
        if (!(numext::abs(newton) > eps * numext::abs(z))) {
          active[i] = false;
          continue;
        }
        RootType repulsion(0);
        for (Index j = 0; j < n; ++j) {
          const RootType gap = z - m_roots[j];
          if (j != i && gap != RootType(0)) repulsion += RootType(1) / gap;
        }
        const RootType denominator = RootType(1) - newton * repulsion;
        const RootType refined = z - (denominator == RootType(0) ? newton : RootType(newton / denominator));
        if ((numext::isfinite)(numext::abs(refined))) {
          m_roots[i] = refined;
          moving = true;
        } else {
          active[i] = false;
        }
      }
      if (!moving) return;
    }
    // Keep improvements to slowly converging multiple roots, but reject a larger residual at the sweep cap.
    for (Index i = 0; i < n; ++i) {
      if (!active[i]) continue;
      RootType initialValue;
      evaluate(poly, m_roots[i], value, derivative, false);
      evaluate(poly, m_eigenSolver.eigenvalues()[i], initialValue, derivative, false);
      if (!(numext::abs(value) <= numext::abs(initialValue))) m_roots[i] = m_eigenSolver.eigenvalues()[i];
    }
  }

  /** Restores the structure the in-place iteration keeps only to within rounding. The roots of a real polynomial
   * are real or conjugate pairs: two iterates closer to each other's conjugate than to the real axis become exactly
   * conjugate, and an iterate left without a partner becomes real. Residual-based snapping of the remaining
   * roots requires an imaginary part at most sqrt(eps) times the real part and a finite rounding bound. */
  template <typename OtherPolynomial>
  void cleanUpRoots(const OtherPolynomial& poly) {
    const Index n = m_roots.size();
    constexpr bool realPolynomial = !NumTraits<Scalar>::IsComplex;
    Array<bool, Deg_, 1> paired = Array<bool, Deg_, 1>::Constant(n, false);
    EIGEN_IF_CONSTEXPR (realPolynomial) {
      for (Index i = 0; i < n; ++i) {
        if (!(numext::imag(m_roots[i]) > RealScalar(0))) continue;
        Index partner = n;
        RealScalar distance = numext::imag(m_roots[i]);
        for (Index j = 0; j < n; ++j) {
          if (paired[j] || !(numext::imag(m_roots[j]) < RealScalar(0))) continue;
          const RealScalar d = numext::abs(m_roots[j] - numext::conj(m_roots[i]));
          if (d < distance) {
            distance = d;
            partner = j;
          }
        }
        if (partner < n) {
          paired[i] = paired[partner] = true;
          const RootType mean = (m_roots[i] + numext::conj(m_roots[partner])) / RealScalar(2);
          m_roots[i] = mean;
          m_roots[partner] = numext::conj(mean);
        }
      }
    }
    RootType value, derivative;
    for (Index i = 0; i < n; ++i) {
      if (numext::imag(m_roots[i]) == RealScalar(0)) continue;
      const RootType realPart(numext::real(m_roots[i]));
      if (realPolynomial && !paired[i]) {
        m_roots[i] = realPart;
        continue;
      }
      // A small residual at realPart can belong to a different root.
      if (!(numext::abs(numext::imag(m_roots[i])) <=
            numext::sqrt(NumTraits<RealScalar>::epsilon()) * numext::abs(realPart)))
        continue;
      const RealScalar bound = evaluate(poly, realPart, value, derivative, true);
      if ((numext::isfinite)(bound) && numext::abs(value) <= bound) m_roots[i] = realPart;
    }
  }

  using PS_Base::m_roots;
  EigenSolverType m_eigenSolver;
};

template <typename Scalar_>
class PolynomialSolver<Scalar_, 1> : public PolynomialSolverBase<Scalar_, 1> {
 public:
  typedef PolynomialSolverBase<Scalar_, 1> PS_Base;
  EIGEN_POLYNOMIAL_SOLVER_BASE_INHERITED_TYPES(PS_Base)

 public:
  /** Computes the complex roots of a new polynomial. */
  template <typename OtherPolynomial>
  void compute(const OtherPolynomial& poly) {
    eigen_assert(poly.size() == 2);
    eigen_assert(Scalar(0) != poly[1]);
    m_roots[0] = -poly[0] / poly[1];
  }

 public:
  template <typename OtherPolynomial>
  inline PolynomialSolver(const OtherPolynomial& poly) {
    compute(poly);
  }

  inline PolynomialSolver() {}

 protected:
  using PS_Base::m_roots;
};

}  // end namespace Eigen

#endif  // EIGEN_POLYNOMIAL_SOLVER_H
