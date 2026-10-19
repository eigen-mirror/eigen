// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_NUMERICALDIFF_RIDDERS_H
#define EIGEN_NUMERICALDIFF_RIDDERS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename F, typename = void>
struct ridders_solver_traits {};

template <typename F>
struct ridders_solver_traits<F, void_t<typename F::QRSolver>> {
  using QRSolver = typename F::QRSolver;
};

}  // namespace internal

/** \class Ridders
 * \ingroup NumericalDiff_Module
 * \brief Adaptive numerical Jacobians using Ridders' extrapolation method.
 *
 * Stores a functor by value and computes its Jacobian from central differences at successively
 * smaller steps. Richardson extrapolation in the squared step size reduces truncation error;
 * the Jacobian column with the smallest estimated error is retained. The stopping test uses
 * the maximum absolute componentwise difference over the entire column.
 *
 * The functor supplies \c Scalar, \c InputType, \c ValueType, \c JacobianType,
 * \c InputsAtCompileTime, \c ValuesAtCompileTime, \c values(), and \c operator(),
 * as for NumericalDiff. The scalar type must be real and non-integer.
 * The functor's \c inputs() method is required only when calling inputs().
 *
 * \warning The functor must be smooth and defined throughout the initial step interval.
 *
 * \code
 * Eigen::Ridders<MyFunctor> derivative(functor);
 * derivative.df(x, jacobian);       // Returns the number of function evaluations.
 * derivative.df(x, jacobian, error); // Also returns an absolute error estimate per input coordinate.
 * \endcode
 *
 * \tparam Functor_ The vector-valued functor to differentiate.
 */
template <typename Functor_>
class Ridders : public internal::ridders_solver_traits<Functor_> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Functor = Functor_;
  using Scalar = typename Functor::Scalar;
  using InputType = typename Functor::InputType;
  using ValueType = typename Functor::ValueType;
  using JacobianType = typename Functor::JacobianType;

  static constexpr int InputsAtCompileTime = Functor::InputsAtCompileTime;
  static constexpr int ValuesAtCompileTime = Functor::ValuesAtCompileTime;

  /** \brief Parameters controlling the initial interval and the maximum refinement cost. */
  struct Parameters {
    /** \brief Initial step scale, giving \f$h_j = \mathrm{initialStep}\,\max(|x_j|, 1)\f$.
     * Must be positive and finite. */
    Scalar initialStep = Scalar(0.01);
    /** \brief Step reduction factor applied at each refinement.
     * The factor must be greater than one and its square must be finite.
     * Values close to one amplify rounding errors in extrapolation and may repeat sampling points.
     * Refinement stops when both sampling points repeat; this does not detect all ill-conditioning. */
    Scalar stepReduction = Scalar(2);
    /** \brief Maximum number of central differences per input coordinate.
     * Must be at least two and at most half the maximum representable Index. */
    int maxSteps = 10;
  };

  Ridders() = default;

  /** \brief Constructs from a copy of a functor and differentiation parameters.
   * \param functor The functor to differentiate.
   * \param parameters The differentiation parameters. */
  explicit Ridders(const Functor& functor, const Parameters& parameters = Parameters()) : m_functor(functor) {
    setParameters(parameters);
  }

  /** \brief Sets the differentiation parameters.
   * \param parameters The differentiation parameters. */
  void setParameters(const Parameters& parameters) {
    eigen_assert(parameters.initialStep > Scalar(0) && (numext::isfinite)(parameters.initialStep));
    eigen_assert(parameters.stepReduction > Scalar(1) &&
                 (numext::isfinite)(parameters.stepReduction * parameters.stepReduction));
    eigen_assert(parameters.maxSteps >= 2);
    eigen_assert(parameters.maxSteps <= NumTraits<Index>::highest() / 2);
    m_parameters = parameters;
  }

  /** \returns the differentiation parameters. */
  const Parameters& parameters() const { return m_parameters; }
  /** \returns the stored functor. */
  Functor& functor() { return m_functor; }
  /** \returns the stored functor. */
  const Functor& functor() const { return m_functor; }

  /** \brief Evaluates the stored functor.
   * \param[in] x The evaluation point.
   * \param[out] values The function values.
   * \returns the value returned by the functor. */
  int operator()(const InputType& x, ValueType& values) const { return m_functor(x, values); }
  /** \returns the stored functor's number of inputs. */
  int inputs() const { return m_functor.inputs(); }
  /** \returns the stored functor's number of values. */
  int values() const { return m_functor.values(); }

  /**
   * \brief Computes the Jacobian using adaptive central differences.
   *
   * The caller sizes \a jac to values() by x.size(). At most \c 2*maxSteps evaluations
   * are made per coordinate. Difference quotients divide by the separation of the representable
   * evaluation points. Refinement stops when successive tableau diagonals differ by at least
   * twice the best estimated error, when evaluation points coincide or repeat the previous pair,
   * or when a non-finite value is encountered. The best finite estimate is retained;
   * if none exists, the column contains NaNs.
   *
   * \param[in] x The evaluation point.
   * \param[out] jac The Jacobian at \a x.
   * \returns the number of functor evaluations.
   * \sa Parameters
   */
  int df(const InputType& x, JacobianType& jac) const { return dfImpl(x, jac, nullptr); }

  /**
   * \brief Computes the Jacobian and an absolute error estimate per input coordinate.
   *
   * The estimate is the maximum componentwise difference from the two parent tableau entries;
   * it is a heuristic, not an error bound. An infinite estimate means no extrapolation was available.
   *
   * \param[in] x The evaluation point.
   * \param[out] jac The Jacobian at \a x, sized by the caller to values() by x.size().
   * \param[out] error The error estimates, resized to x.size().
   * \returns the number of functor evaluations.
   * \sa Parameters
   */
  int df(const InputType& x, JacobianType& jac, InputType& error) const {
    error.resize(x.size());
    return dfImpl(x, jac, &error);
  }

 private:
  Functor m_functor;
  Parameters m_parameters;

  int dfImpl(const InputType& input, JacobianType& jac, InputType* error) const {
    EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsComplex && !NumTraits<Scalar>::IsInteger,
                        NUMERICAL_DIFF_REQUIRES_A_REAL_FLOATING_POINT_SCALAR);
    // C. J. F. Ridders, "Accurate computation of F'(x) and F'(x)F''(x)",
    // Advances in Engineering Software 4(2), 75-76 (1982). Extrapolate central differences in h^2.
    using Tableau = Matrix<Scalar, Functor::ValuesAtCompileTime, Dynamic>;
    const Index values = m_functor.values();
    eigen_assert(jac.rows() == values && jac.cols() == input.size());
    if (values == 0 || input.size() == 0) {
      if (error) error->setZero();
      return 0;
    }

    Tableau workspace(values, 2 * Index(m_parameters.maxSteps));
    auto first = workspace.leftCols(m_parameters.maxSteps);
    auto second = workspace.rightCols(m_parameters.maxSteps);
    auto current = &first;
    auto previous = &second;

    // Functors may require ValueType& outputs, which cannot bind to column views.
    ValueType plus(values), minus(values);
    InputType x = input;
    const Scalar reductionSquared = m_parameters.stepReduction * m_parameters.stepReduction;
    int evaluations = 0;

    for (Index j = 0; j < input.size(); ++j) {
      Scalar h = m_parameters.initialStep * numext::maxi(numext::abs(input[j]), Scalar(1));
      Scalar previousPlus = input[j], previousMinus = input[j];
      Scalar bestError = NumTraits<Scalar>::infinity();
      jac.col(j).setConstant(NumTraits<Scalar>::quiet_NaN());
      for (int step = 0; step < m_parameters.maxSteps; ++step) {
        Scalar xPlus = input[j] + h;
        Scalar xMinus = input[j] - h;
        internal::numerical_diff_barrier(xPlus);
        internal::numerical_diff_barrier(xMinus);
        if (!(numext::isfinite)(xPlus) || !(numext::isfinite)(xMinus)) break;
        if (step > 0 && xPlus == previousPlus && xMinus == previousMinus) break;

        const Scalar separation = xPlus - xMinus;
        if (!(separation > Scalar(0)) || !(numext::isfinite)(separation)) break;
        previousPlus = xPlus;
        previousMinus = xMinus;

        x[j] = xPlus;
        m_functor(x, plus);
        x[j] = xMinus;
        m_functor(x, minus);
        x[j] = input[j];
        evaluations += 2;

        current->col(0) = (plus - minus) / separation;
        if (!current->col(0).allFinite()) {
          if (separation <= Scalar(1)) break;
          // Divide before subtracting to avoid overflow, preserving already finite quotients.
          auto quotient = current->col(0).array();
          quotient = quotient.isFiniteTyped().select(quotient, plus.array() / separation - minus.array() / separation);
          if (!current->col(0).allFinite()) break;
        }
        Index bestOrder = step == 0 ? 0 : -1;

        Scalar factor = reductionSquared;
        bool finite = true;
        for (int order = 1; order <= step; ++order) {
          // A(k,i) = (t^(2k) * A(k-1,i) - A(k-1,i-1)) / (t^(2k) - 1).
          // Avoid forming `t^(2k) * A(k-1,i)` directly, since it can overflow.
          current->col(order) =
              current->col(order - 1) + (current->col(order - 1) - previous->col(order - 1)) / (factor - Scalar(1));
          if (!current->col(order).allFinite()) {
            finite = false;
            break;
          }

          const Scalar estimate =
              numext::maxi((current->col(order) - current->col(order - 1)).template lpNorm<Infinity>(),
                           (current->col(order) - previous->col(order - 1)).template lpNorm<Infinity>());
          if ((numext::isfinite)(estimate) && estimate <= bestError) {
            bestError = estimate;
            bestOrder = order;
          }

          factor *= reductionSquared;
        }
        if (bestOrder >= 0) jac.col(j) = current->col(bestOrder);
        if (!finite) break;
        if (step > 0 &&
            (current->col(step) - previous->col(step - 1)).template lpNorm<Infinity>() >= Scalar(2) * bestError)
          break;

        std::swap(current, previous);
        h /= m_parameters.stepReduction;
      }
      if (error) (*error)[j] = bestError;
    }
    return evaluations;
  }
};

#if EIGEN_COMP_CXXVER < 17
template <typename Functor>
constexpr int Ridders<Functor>::InputsAtCompileTime;
template <typename Functor>
constexpr int Ridders<Functor>::ValuesAtCompileTime;
#endif

}  // namespace Eigen

#endif  // EIGEN_NUMERICALDIFF_RIDDERS_H
