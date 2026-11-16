// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_ITERATIVE_SOLVER_BASE_H
#define EIGEN_ITERATIVE_SOLVER_BASE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename T>
auto is_ref_compatible_test(T& matrix) -> decltype(Ref<const T>(matrix), std::true_type());
std::false_type is_ref_compatible_test(...);

template <typename MatrixType>
using is_ref_compatible = decltype(is_ref_compatible_test(std::declval<remove_all_t<MatrixType>&>()));

// Returns a \a rows x \a cols matrix whose columns are an orthonormal basis of a random subspace,
// obtained by QR-orthonormalizing a random matrix. The IDR(s)-type solvers use this to build the
// shadow space; the basis only has to be (almost surely) non-degenerate, so any random seed is fine.
template <typename MatrixType>
MatrixType random_orthonormal_basis(Index rows, Index cols) {
  HouseholderQR<MatrixType> qr(MatrixType::Random(rows, cols));
  return qr.householderQ() * MatrixType::Identity(rows, cols);
}

// The more severe of two solver statuses: NumericalIssue, then NoConvergence, then Success. The enumerators are not
// ordered by severity.
inline ComputationInfo most_severe_info(ComputationInfo a, ComputationInfo b) {
  if (a == NumericalIssue || b == NumericalIssue) return NumericalIssue;
  if (a == NoConvergence || b == NoConvergence) return NoConvergence;
  return a == Success ? b : a;
}

template <typename RealScalar>
EIGEN_DEVICE_FUNC RealScalar iterative_solver_scaling_factor(RealScalar norm) {
  // Scale only when forming a quadratic recurrence term from norm, or from the eps norm that a solver can reduce it
  // to, may underflow or overflow. A square below the smallest normal number is subnormal, which loses precision, and
  // is flushed to 0 where subnormals are not supported (float NEON on ARMv7).
  const RealScalar sqrtMin = numext::sqrt((std::numeric_limits<RealScalar>::min)());
  const RealScalar sqrtMax = numext::sqrt((std::numeric_limits<RealScalar>::max)());
  return norm < sqrtMin / NumTraits<RealScalar>::epsilon() || norm > sqrtMax ? norm : RealScalar(1);
}

/** \internal Calls \a solve(b, x) on the problem scaled to a right-hand side of unit norm, \a solve(b / s, x / s), and
 * scales x back, when iterative_solver_scaling_factor() of |b| is not 1: for a small |b| the products and squares of
 * residual-sized quantities that a Krylov method forms (inner products, squared norms) are subnormal, which loses
 * precision, and are flushed to 0 by float NEON on ARMv7. A x = b is equivalent to A (x / s) = b / s, and the
 * tolerances of the solvers are relative. The problem is left unscaled when b is zero or not finite, or when the scaled
 * guess is not finite. \a solve must accept a dense vector for b and x, and returns the result of the solver. */
template <typename Rhs, typename Dest, typename Solve>
auto iterative_solver_on_scaled_problem(const Rhs& b, Dest& x, Solve solve) -> decltype(solve(b, x)) {
  using RealScalar = typename Dest::RealScalar;
  using VectorType = Matrix<typename Dest::Scalar, Dynamic, 1>;
  const RealScalar normb = b.stableNorm();
  if (normb > RealScalar(0) && (numext::isfinite)(normb)) {
    const RealScalar scale = iterative_solver_scaling_factor(normb);
    if (scale != RealScalar(1)) {
      VectorType scaledGuess = x / scale;
      if (scaledGuess.allFinite()) {
        const VectorType scaledRhs = b / scale;
        const auto ret = solve(scaledRhs, scaledGuess);
        x = scaledGuess * scale;
        return ret;
      }
    }
  }
  return solve(b, x);
}

// The least residual norm that a stopping rule can ask for. Recurrences that form squares of residual norms underflow
// below it, so a tolerance of 0 ("iterate until the residual vanishes") must stop here, not at the first denormal.
template <typename RealScalar>
EIGEN_DEVICE_FUNC RealScalar iterative_solver_residual_floor() {
  return numext::sqrt((std::numeric_limits<RealScalar>::min)());
}

// The stopping threshold tol |b| for the residual, raised to min(floor, eps |b|). The raise exists only so that a
// tolerance of 0, or below what the recurrences can resolve, stops where the residual underflows instead of dividing by
// zero. It never exceeds eps |b|, which a solver cannot reach anyway, so it leaves every tolerance tol >= eps unchanged
// whatever the size of b, where an absolute floor would exceed tol |b| for small |b| (|b| < 1e-12 in float at the
// default tolerance).
template <typename RealScalar>
EIGEN_DEVICE_FUNC RealScalar iterative_solver_threshold(RealScalar tol, RealScalar norm, RealScalar floor) {
  return numext::maxi(RealScalar(tol * norm), numext::mini(floor, RealScalar(NumTraits<RealScalar>::epsilon() * norm)));
}

template <typename MatrixType, bool MatrixFree = !internal::is_ref_compatible<MatrixType>::value>
class generic_matrix_wrapper;

// We have an explicit matrix at hand, compatible with Ref<>
template <typename MatrixType>
class generic_matrix_wrapper<MatrixType, false> {
 public:
  using ActualMatrixType = Ref<const MatrixType>;
  template <int UpLo>
  struct ConstSelfAdjointViewReturnType {
    using Type = typename ActualMatrixType::template ConstSelfAdjointViewReturnType<UpLo>::Type;
  };

  enum { MatrixFree = false };

  // Default-construct: passing (0,0) trips the size assertion for fixed-size MatrixType (#1704).
  generic_matrix_wrapper() : m_dummy(), m_matrix(m_dummy) {}

  template <typename InputType>
  generic_matrix_wrapper(const InputType& mat) : m_matrix(mat) {}

  const ActualMatrixType& matrix() const { return m_matrix; }

  bool hasMatrix() const { return true; }

  template <typename MatrixDerived>
  void grab(const EigenBase<MatrixDerived>& mat) {
    internal::destroy_at(&m_matrix);
    internal::construct_at(&m_matrix, mat.derived());
  }

  void grab(const Ref<const MatrixType>& mat) {
    if (&(mat.derived()) != &m_matrix) {
      internal::destroy_at(&m_matrix);
      internal::construct_at(&m_matrix, mat);
    }
  }

 protected:
  MatrixType m_dummy;  // used to default initialize the Ref<> object
  ActualMatrixType m_matrix;
};

// MatrixType is not compatible with Ref<> -> matrix-free wrapper
template <typename MatrixType>
class generic_matrix_wrapper<MatrixType, true> {
 public:
  using ActualMatrixType = MatrixType;
  template <int UpLo>
  struct ConstSelfAdjointViewReturnType {
    using Type = ActualMatrixType;
  };

  enum { MatrixFree = true };

  generic_matrix_wrapper() = default;

  generic_matrix_wrapper(const MatrixType& mat) : mp_matrix(&mat) {}

  const ActualMatrixType& matrix() const { return *mp_matrix; }

  // False until compute(): a matrix-free operator has nothing to dereference before it is attached.
  bool hasMatrix() const { return mp_matrix != nullptr; }

  void grab(const MatrixType& mat) { mp_matrix = &mat; }

 protected:
  const ActualMatrixType* mp_matrix = nullptr;
};

}  // namespace internal

/** \ingroup IterativeLinearSolvers_Module
 * \brief Base class for linear iterative solvers
 *
 * \sa class SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
 */
template <typename Derived>
class IterativeSolverBase : public SparseSolverBase<Derived> {
 protected:
  using Base = SparseSolverBase<Derived>;
  using Base::m_isInitialized;

 public:
  using MatrixType = typename internal::traits<Derived>::MatrixType;
  using Preconditioner = typename internal::traits<Derived>::Preconditioner;
  using Scalar = typename MatrixType::Scalar;
  using StorageIndex = typename MatrixType::StorageIndex;
  using RealScalar = typename MatrixType::RealScalar;

  enum { ColsAtCompileTime = MatrixType::ColsAtCompileTime, MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime };

 public:
  using Base::derived;

  /** Default constructor. */
  IterativeSolverBase() { init(); }

  /** Initialize the solver with matrix \a A for further \c Ax=b solving.
   *
   * This constructor is a shortcut for the default constructor followed
   * by a call to compute().
   *
   * \warning this class stores a reference to the matrix A as well as some
   * precomputed values that depend on it. Therefore, if \a A is changed
   * this class becomes invalid. Call compute() to update it with the new
   * matrix A, or modify a copy of A.
   */
  template <typename MatrixDerived>
  explicit IterativeSolverBase(const EigenBase<MatrixDerived>& A) : m_matrixWrapper(A.derived()) {
    init();
    compute(matrix());
  }

  IterativeSolverBase(IterativeSolverBase&&) = default;

  /** Initializes the iterative solver for the sparsity pattern of the matrix \a A for further solving \c Ax=b problems.
   *
   * Currently, this function mostly calls analyzePattern on the preconditioner. In the future
   * we might, for instance, implement column reordering for faster matrix vector products.
   */
  template <typename MatrixDerived>
  Derived& analyzePattern(const EigenBase<MatrixDerived>& A) {
    grab(A.derived());
    m_preconditioner.analyzePattern(matrix());
    m_isInitialized = true;
    m_analysisIsOk = true;
    m_info = m_preconditioner.info();
    return derived();
  }

  /** Initializes the iterative solver with the numerical values of the matrix \a A for further solving \c Ax=b
   * problems.
   *
   * Currently, this function mostly calls factorize on the preconditioner.
   *
   * \warning this class stores a reference to the matrix A as well as some
   * precomputed values that depend on it. Therefore, if \a A is changed
   * this class becomes invalid. Call compute() to update it with the new
   * matrix A, or modify a copy of A.
   */
  template <typename MatrixDerived>
  Derived& factorize(const EigenBase<MatrixDerived>& A) {
    eigen_assert(m_analysisIsOk && "You must first call analyzePattern()");
    grab(A.derived());
    m_preconditioner.factorize(matrix());
    m_factorizationIsOk = true;
    m_info = m_preconditioner.info();
    return derived();
  }

  /** Initializes the iterative solver with the matrix \a A for further solving \c Ax=b problems.
   *
   * Currently, this function mostly initializes/computes the preconditioner. In the future
   * we might, for instance, implement column reordering for faster matrix vector products.
   *
   * \warning this class stores a reference to the matrix A as well as some
   * precomputed values that depend on it. Therefore, if \a A is changed
   * this class becomes invalid. Call compute() to update it with the new
   * matrix A, or modify a copy of A.
   */
  template <typename MatrixDerived>
  Derived& compute(const EigenBase<MatrixDerived>& A) {
    grab(A.derived());
    m_preconditioner.compute(matrix());
    m_isInitialized = true;
    m_analysisIsOk = true;
    m_factorizationIsOk = true;
    m_info = m_preconditioner.info();
    return derived();
  }

  /** \internal */
  constexpr Index rows() const noexcept { return matrix().rows(); }

  /** \internal */
  constexpr Index cols() const noexcept { return matrix().cols(); }

  /** \returns the tolerance threshold used by the stopping criteria.
   * \sa setTolerance()
   */
  RealScalar tolerance() const { return m_tolerance; }

  /** Sets the tolerance threshold used by the stopping criteria.
   *
   * This value is used as an upper bound to the relative residual error: |Ax-b|/|b|, or to the measure the solver
   * documents instead; LeastSquaresConjugateGradient bounds the normal-equation residual |A'(Ax-b)|/|A'b|, and
   * LSMR bounds |A'(Ax-b)|/(|A| |Ax-b|).
   * The default value is the machine precision given by NumTraits<Scalar>::epsilon(), unless the solver documents
   * another; with this default, IDRS and MINRES also report Success up to \f$ 8 \sqrt{n}\, \epsilon \f$ for an n x n
   * matrix, see MINRES::tolerance().
   * A tolerance of 0 asks for iterating until the residual vanishes or maxIterations() is reached. The solvers stop
   * earlier when it underflows or the Krylov space is exhausted, and return a finite solution.
   */
  Derived& setTolerance(const RealScalar& tolerance) {
    m_tolerance = tolerance;
    return derived();
  }

  /** \returns a read-write reference to the preconditioner for custom configuration. */
  Preconditioner& preconditioner() { return m_preconditioner; }

  /** \returns a read-only reference to the preconditioner. */
  const Preconditioner& preconditioner() const { return m_preconditioner; }

  /** \returns the max number of iterations.
   * It is either the value set by setMaxIterations or, by default,
   * twice the number of columns of the matrix, which is 0 for a matrix-free operator not yet attached by compute().
   */
  Index maxIterations() const { return (m_maxIterations < 0) ? 2 * attachedCols() : m_maxIterations; }

  /** Sets the max number of iterations.
   * Default is twice the number of columns of the matrix.
   */
  Derived& setMaxIterations(Index maxIters) {
    m_maxIterations = maxIters;
    return derived();
  }

  /** \returns the number of iterations performed during the last solve */
  Index iterations() const {
    eigen_assert(m_isInitialized && "IterativeSolverBase is not initialized.");
    return m_iterations;
  }

  /** \returns the tolerance error reached during the last solve.
   * It is a close approximation of the true relative residual error |Ax-b|/|b|, unless the solver documents a
   * different measure: LeastSquaresConjugateGradient reports the normal-equation residual |A'(Ax-b)|/|A'b|, and
   * LSMR reports the normal-equation residual estimate |A'(Ax-b)|/(|A| |Ax-b|).
   */
  RealScalar error() const {
    eigen_assert(m_isInitialized && "IterativeSolverBase is not initialized.");
    return m_error;
  }

  /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A
   * and \a x0 as an initial solution.
   *
   * \sa solve(), compute()
   */
  template <typename Rhs, typename Guess>
  inline const SolveWithGuess<Derived, Rhs, Guess> solveWithGuess(const MatrixBase<Rhs>& b, const Guess& x0) const {
    eigen_assert(m_isInitialized && "Solver is not initialized.");
    EIGEN_INTERNAL_ASSERT_SOLVER_RHS_MSG(derived(), b, false,
                                         "solve(): invalid number of rows of the right hand side matrix b");
    return SolveWithGuess<Derived, Rhs, Guess>(derived(), b.derived(), x0);
  }

  /** Solves \f$ A x = b \f$ in place: \a x holds the initial guess on entry and the solution on
   * exit, and info(), iterations() and error() report on the run.
   *
   * solve() and solveWithGuess() return expressions and therefore require Eigen dense or sparse
   * operands. This entry point accepts any single-column right-hand side and solution types that
   * provide the vector operations the algorithm uses, for instance the device-resident vectors of
   * the GPU module (\c contrib/Eigen/GPU) together with a matrix-free matrix type.
   *
   * \sa solve(), solveWithGuess()
   */
  template <typename Rhs, typename Dest>
  void solveWithGuessInPlace(const Rhs& b, Dest& x) const {
    eigen_assert(m_isInitialized && "Solver is not initialized.");
    EIGEN_INTERNAL_ASSERT_SOLVER_RHS_MSG(derived(), b, false,
                                         "solveWithGuessInPlace(): invalid number of rows of the right hand side b");
    derived()._solve_vector_with_guess_impl(b, x);
  }

  /** \returns Success if the iterations converged, NumericalIssue if the solver broke down, and NoConvergence
   * otherwise. With several right-hand sides, it is the most severe status over the columns: NumericalIssue if any
   * column broke down, else NoConvergence if any did not converge, else Success. iterations() and error() then
   * describe the last column. */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "IterativeSolverBase is not initialized.");
    return m_info;
  }

  /** \internal */
  template <typename Rhs, typename DestDerived>
  void _solve_with_guess_impl(const Rhs& b, SparseMatrixBase<DestDerived>& aDest) const {
    EIGEN_INTERNAL_ASSERT_SOLVER_RHS(derived(), b, false);

    Index rhsCols = b.cols();
    Index size = b.rows();
    DestDerived& dest(aDest.derived());
    using DestScalar = typename DestDerived::Scalar;
    Eigen::Matrix<DestScalar, Dynamic, 1> tb(size);
    Eigen::Matrix<DestScalar, Dynamic, 1> tx(cols());
    // We do not directly fill dest because sparse expressions have to be free of aliasing issue.
    // For non square least-square problems, b and dest might not have the same size whereas they might alias
    // each-other.
    typename DestDerived::PlainObject tmp(cols(), rhsCols);
    ComputationInfo global_info = Success;
    for (Index k = 0; k < rhsCols; ++k) {
      tb = b.col(k);
      tx = dest.col(k);
      derived()._solve_vector_with_guess_impl(tb, tx);
      tmp.col(k) = tx.sparseView(0);

      // The call to _solve_vector_with_guess_impl updates m_info, so keep the most severe status over the columns.
      global_info = internal::most_severe_info(global_info, m_info);
    }
    m_info = global_info;
    dest.swap(tmp);
  }

  template <typename Rhs, typename DestDerived>
  std::enable_if_t<Rhs::ColsAtCompileTime != 1 && DestDerived::ColsAtCompileTime != 1> _solve_with_guess_impl(
      const Rhs& b, MatrixBase<DestDerived>& aDest) const {
    EIGEN_INTERNAL_ASSERT_SOLVER_RHS(derived(), b, false);

    Index rhsCols = b.cols();
    DestDerived& dest(aDest.derived());
    ComputationInfo global_info = Success;
    for (Index k = 0; k < rhsCols; ++k) {
      typename DestDerived::ColXpr xk(dest, k);
      typename Rhs::ConstColXpr bk(b, k);
      derived()._solve_vector_with_guess_impl(bk, xk);

      // The call to _solve_vector_with_guess_impl updates m_info, so keep the most severe status over the columns.
      global_info = internal::most_severe_info(global_info, m_info);
    }
    m_info = global_info;
  }

  template <typename Rhs, typename DestDerived>
  std::enable_if_t<Rhs::ColsAtCompileTime == 1 || DestDerived::ColsAtCompileTime == 1> _solve_with_guess_impl(
      const Rhs& b, MatrixBase<DestDerived>& dest) const {
    derived()._solve_vector_with_guess_impl(b, dest.derived());
  }

  /** \internal default initial guess = 0 */
  template <typename Rhs, typename Dest>
  void _solve_impl(const Rhs& b, Dest& x) const {
    x.setZero();
    derived()._solve_with_guess_impl(b, x);
  }

 protected:
  void init() {
    m_isInitialized = false;
    m_analysisIsOk = false;
    m_factorizationIsOk = false;
    m_maxIterations = -1;
    m_tolerance = NumTraits<Scalar>::epsilon();
  }

  using MatrixWrapper = internal::generic_matrix_wrapper<MatrixType>;
  using ActualMatrixType = typename MatrixWrapper::ActualMatrixType;

  const ActualMatrixType& matrix() const { return m_matrixWrapper.matrix(); }

  /** \internal The number of columns of the matrix, or 0 while a matrix-free operator is not attached yet. */
  Index attachedCols() const { return m_matrixWrapper.hasMatrix() ? matrix().cols() : Index(0); }

  /** \internal 8 sqrt(max(n, 1)) eps with n = attachedCols(): the residual up to which IDRS and MINRES report Success
   * at their default tolerance. */
  RealScalar sizeScaledTolerance() const {
    return RealScalar(8) * numext::sqrt(RealScalar(numext::maxi(attachedCols(), Index(1)))) *
           NumTraits<RealScalar>::epsilon();
  }

  /** \internal The tolerance of a solver whose default is selected by a negative stored tolerance: the stored one, or
   * eps by default. The solver sets m_tolerance = -1 in its constructors. */
  RealScalar toleranceOrEps() const {
    return m_tolerance >= RealScalar(0) ? m_tolerance : NumTraits<RealScalar>::epsilon();
  }

  /** \internal The status of a solve that ended with the computed relative residual \a error for the tolerance \a tol
   * of toleranceOrEps(). The residual of the computed x stagnates above eps by an amount that depends on the matrix,
   * so at the default tolerance only, Success also covers sizeScaledTolerance(). A tolerance set by setTolerance() is
   * never relaxed. \a error must be the computed |b - A x| / |b|. */
  ComputationInfo infoAtDefaultOrSetTolerance(RealScalar error, RealScalar tol) const {
    const bool confirmed = error <= tol || (m_tolerance < RealScalar(0) && error <= sizeScaledTolerance());
    return confirmed ? Success : NoConvergence;
  }

  template <typename InputType>
  void grab(const InputType& A) {
    m_matrixWrapper.grab(A);
  }

  MatrixWrapper m_matrixWrapper;
  Preconditioner m_preconditioner;

  Index m_maxIterations;
  RealScalar m_tolerance;

  mutable RealScalar m_error;
  mutable Index m_iterations;
  mutable ComputationInfo m_info;
  mutable bool m_analysisIsOk, m_factorizationIsOk;
};

}  // end namespace Eigen

#endif  // EIGEN_ITERATIVE_SOLVER_BASE_H
