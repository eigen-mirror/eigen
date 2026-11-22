// -*- coding: utf-8
// vim: set fileencoding=utf-8
// SPDX-License-Identifier: MPL-2.0

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NUMERICAL_DIFF_H
#define EIGEN_NUMERICAL_DIFF_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

enum NumericalDiffMode { Forward, Central };

/**
 * This class allows you to add a method df() to your functor, which will
 * use numerical differentiation to compute an approximate of the
 * derivative for the functor. Of course, if you have an analytical form
 * for the derivative, you should rather implement df() by yourself.
 *
 * More information on
 * http://en.wikipedia.org/wiki/Numerical_differentiation
 *
 * Currently only "Forward" and "Central" scheme are implemented.
 *
 * The step along coordinate \c j is
 * \f[
 *   h_j = \begin{cases} \max(|x_j|\,\mathrm{rel\_tol},\ \mathrm{abs\_tol}) & \text{if this is nonzero},\\
 *   \mathrm{rel\_tol} & \text{otherwise.}
 *   \end{cases}
 * \f]
 * The forward difference is \f$(f(x + h_j e_j) - f(x))/h_j\f$ and the central difference is
 * \f$(f(x + h_j e_j) - f(x - h_j e_j))/(2 h_j)\f$, each with the divisor replaced by the representable step
 * actually applied. By default \f$\mathrm{abs\_tol} = \mathrm{rel\_tol}\f$ (set by the constructors), so
 * \f$h_j \ge \mathrm{rel\_tol}\f$: this suits residuals with an offset of order 1 even when some \f$|x_j|\f$ is
 * tiny, where a step proportional to \f$|x_j|\f$ would be lost in the rounding error of \f$f\f$.
 * For a function whose natural scale is \f$|x_j|\f$ itself (parameters of tiny magnitude throughout), call
 * setAbsTol(0): then \f$h_j = \mathrm{rel\_tol}\,|x_j|\f$ as in MINPACK's \c fdjac2, and the second case applies only
 * when this product is zero (\f$x_j = 0\f$, or \f$|x_j|\,\mathrm{rel\_tol}\f$ underflows, so that \f$x_j + h\f$ would
 * equal \f$x_j\f$).
 */
template <typename Functor_, NumericalDiffMode mode = Forward>
class NumericalDiff : public Functor_ {
 public:
  using Functor = Functor_;
  using Scalar = typename Functor::Scalar;
  using InputType = typename Functor::InputType;
  using ValueType = typename Functor::ValueType;
  using JacobianType = typename Functor::JacobianType;

 private:
  // Fallback/Forward mode helper
  static Scalar defaultRelTol() {
    return mode == Central ? numext::cbrt(NumTraits<Scalar>::epsilon()) : numext::sqrt(NumTraits<Scalar>::epsilon());
  }

 public:
  /** Constructs a default-initialized functor with \c rel_tol the square root (forward) or cube root (central) of
   * <tt>max(epsfcn, NumTraits<Scalar>::epsilon())</tt> and \c abs_tol = \c rel_tol, see the class documentation. */
  NumericalDiff(Scalar epsfcn = 0.)
      : Functor(),
        rel_tol(mode == Central ? numext::cbrt(numext::maxi(epsfcn, NumTraits<Scalar>::epsilon()))
                                : numext::sqrt(numext::maxi(epsfcn, NumTraits<Scalar>::epsilon()))) {
    setAbsTol(rel_tol);
  }
  /** Same as above, copying the functor \a f. */
  NumericalDiff(const Functor& f, Scalar epsfcn = 0.)
      : Functor(f),
        rel_tol(mode == Central ? numext::cbrt(numext::maxi(epsfcn, NumTraits<Scalar>::epsilon()))
                                : numext::sqrt(numext::maxi(epsfcn, NumTraits<Scalar>::epsilon()))) {
    setAbsTol(rel_tol);
  }

  // forward constructors
  template <typename T0, std::enable_if_t<std::is_constructible<Functor, const T0&>::value, int> = 0>
  NumericalDiff(const T0& a0) : Functor(a0), rel_tol(defaultRelTol()), abs_tol(defaultRelTol()) {}

  template <typename T0, typename T1,
            std::enable_if_t<std::is_constructible<Functor, const T0&, const T1&>::value, int> = 0>
  NumericalDiff(const T0& a0, const T1& a1) : Functor(a0, a1), rel_tol(defaultRelTol()), abs_tol(defaultRelTol()) {}

  template <typename T0, typename T1, typename T2,
            std::enable_if_t<std::is_constructible<Functor, const T0&, const T1&, const T2&>::value, int> = 0>
  NumericalDiff(const T0& a0, const T1& a1, const T2& a2)
      : Functor(a0, a1, a2), rel_tol(defaultRelTol()), abs_tol(defaultRelTol()) {}

  /** Sets the relative step \c rel_tol, which must be at least NumTraits<Scalar>::epsilon(). */
  void setRelTol(Scalar _rel_tol) {
    eigen_assert(_rel_tol >= NumTraits<Scalar>::epsilon());
    rel_tol = _rel_tol;
  }
  /** Sets the lower bound \c abs_tol >= 0 of the step: \f$h_j = \max(|x_j|\,\mathrm{rel\_tol}, \mathrm{abs\_tol})\f$.
   * The default \c abs_tol = \c rel_tol suits residuals with an offset of order 1 even when some \f$|x_j|\f$ is tiny.
   * With 0 the step is purely relative, \f$h_j = \mathrm{rel\_tol}\,|x_j|\f$ (\c rel_tol at \f$x_j = 0\f$), which suits
   * a function whose natural scale is \f$|x_j|\f$ itself. */
  void setAbsTol(Scalar _abs_tol) {
    eigen_assert(_abs_tol >= 0);
    abs_tol = _abs_tol;
  }
  /** \returns the relative step \c rel_tol. */
  Scalar relTol() const { return rel_tol; }
  /** \returns the lower bound \c abs_tol of the step. */
  Scalar absTol() const { return abs_tol; }

  enum { InputsAtCompileTime = Functor::InputsAtCompileTime, ValuesAtCompileTime = Functor::ValuesAtCompileTime };

  /**
   * Computes the Jacobian of the functor at \a _x into \a jac and returns the number of functor evaluations.
   *
   * The step along coordinate \c j is <tt>h = max(|x[j]| * rel_tol, abs_tol)</tt>, or <tt>h = rel_tol</tt> when that
   * is zero (possible only for <tt>abs_tol == 0</tt>), with rel_tol and abs_tol selected by the user. The rel_tol
   * selected by the user must be greater than or equal to epsilon, and abs_tol must be nonnegative, where \c epsilon
   * is the machine precision NumTraits<Scalar>::epsilon(); the difference quotient divides by the representable step
   * <tt>fl(x[j] + h) - x[j]</tt> actually applied.
   *
   * In forward mode, the default rel_tol is <tt> sqrt(NumTraits<Scalar>::epsilon()) </tt> and the default abs_tol
   * is abs_tol = rel_tol. In central mode, the default rel_tol is <tt> cbrt(NumTraits<Scalar>::epsilon()) </tt>
   * and the default abs_tol = rel_tol. Call setAbsTol(0) for a purely relative step, see the class documentation.
   */
  int df(const InputType& _x, JacobianType& jac) const {
    using std::abs;
    /* Local variables */
    Scalar h;
    int nfev = 0;
    const typename InputType::Index n = _x.size();
    ValueType val1, val2;
    InputType x = _x;
    // TODO: We should do this only if the size is not already known.
    val1.resize(Functor::values());
    val2.resize(Functor::values());

    // initialization
    switch (mode) {
      case Forward:
        // compute f(x)
        Functor::operator()(x, val1);
        nfev++;
        break;
      case Central:
        // do nothing
        break;
      default:
        eigen_assert(false);
    }

    // Function Body
    for (int j = 0; j < n; ++j) {
      const Scalar x_abs = abs(x[j]);
      h = numext::maxi(x_abs * rel_tol, abs_tol);
      if (h == Scalar(0)) h = rel_tol;
      // The functor is evaluated at fl(x[j] + h), so divide by that representable step: the rounding
      // of x[j] + h perturbs h by up to ulp(x[j]) <= epsilon |x[j]| <= (epsilon/rel_tol) * h <= h, comparable
      // to the error of the difference quotient itself.
      Scalar x_plus = _x[j] + h;
      // Keeps -ffast-math from folding (x[j] + h) - x[j] back into h.
      internal::pbarrier(x_plus);
      h = x_plus - _x[j];
      switch (mode) {
        case Forward:
          x[j] = x_plus;
          Functor::operator()(x, val2);
          nfev++;
          x[j] = _x[j];
          jac.col(j) = (val2 - val1) / h;
          break;
        case Central: {
          x[j] = x_plus;
          Functor::operator()(x, val2);
          nfev++;
          // x[j] - h can round (a tie when x[j] < 0 has |x[j]| within h above a power of two), so
          // divide by the separation of the two evaluation points rather than by 2*h.
          Scalar x_minus = _x[j] - h;
          internal::pbarrier(x_minus);
          x[j] = x_minus;
          Functor::operator()(x, val1);
          nfev++;
          x[j] = _x[j];
          jac.col(j) = (val2 - val1) / (x_plus - x_minus);
          break;
        }
        default:
          eigen_assert(false);
      }
    }
    return nfev;
  }

 private:
  Scalar rel_tol;
  Scalar abs_tol;

  NumericalDiff& operator=(const NumericalDiff&) = delete;
};

}  // end namespace Eigen

// vim: ai ts=4 sts=4 et sw=4
#endif  // EIGEN_NUMERICAL_DIFF_H
