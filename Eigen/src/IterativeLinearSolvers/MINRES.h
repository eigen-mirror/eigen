// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Giacomo Po <gpo@ucla.edu>
// Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2018 David Hyde <dabh@stanford.edu>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MINRES_H
#define EIGEN_MINRES_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal Low-level MINRES algorithm
 * \param mat The matrix A
 * \param rhs The right hand side vector b
 * \param x On input and initial solution, on output the computed solution.
 * \param precond A right preconditioner being able to efficiently solve for an
 *                approximation of Ax=b (regardless of b)
 * \param iters On input the max number of iteration, on output the number of performed iterations.
 * \param tol_error On input the tolerance error, on output an estimation of the relative error.
 */
template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
EIGEN_DONT_INLINE void minres(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond,
                              Index& iters, typename Dest::RealScalar& tol_error) {
  using RealScalar = typename Dest::RealScalar;
  using Scalar = typename Dest::Scalar;
  using VectorType = Matrix<Scalar, Dynamic, 1>;

  // Check for zero rhs
  const RealScalar rhsNorm(rhs.stableNorm());
  if (rhsNorm == 0) {
    x.setZero();
    iters = 0;
    tol_error = 0;
    return;
  }

  // initialize
  const Index maxIters(iters);                      // initialize maxIters to iters
  const Index N(mat.cols());                        // the size of the matrix
  // Convergence threshold (compared to residualNorm), raised to at most the smallest normal number so that
  // tol_error = 0 stops.
  const RealScalar threshold(
      internal::iterative_solver_threshold(tol_error, rhsNorm, RealScalar((std::numeric_limits<RealScalar>::min)())));

  // Initialize preconditioned Lanczos
  VectorType v_old(N);                // will be initialized inside loop
  VectorType v(VectorType::Zero(N));  // initialize v
  VectorType v_new(rhs - mat * x);    // initialize v_new
  RealScalar residualNorm(v_new.stableNorm());
  if (residualNorm == 0 || residualNorm < threshold) {
    iters = 0;
    tol_error = residualNorm / rhsNorm;
    return;
  }

  // Keep the quadratic Lanczos terms representable for very small or large residuals.
  const RealScalar residualScale = internal::iterative_solver_scaling_factor(residualNorm);
  v_new /= residualScale;
  VectorType w(N);                         // will be initialized inside loop
  VectorType w_new(precond.solve(v_new));  // initialize w_new
  RealScalar beta_new2(v_new.dot(w_new));
  eigen_assert(beta_new2 >= RealScalar(0) && "PRECONDITIONER IS NOT POSITIVE DEFINITE");
  RealScalar beta_new(numext::sqrt(beta_new2));
  const RealScalar beta_one(beta_new);
  // Initialize other variables
  RealScalar c(1.0);  // the cosine of the Givens rotation
  RealScalar c_old(1.0);
  RealScalar s(0.0);                      // the sine of the Givens rotation
  RealScalar s_old(0.0);                  // the sine of the Givens rotation
  VectorType p_oold(N);                   // will be initialized in loop
  VectorType p_old(VectorType::Zero(N));  // initialize p_old=0
  VectorType p(p_old);                    // initialize p=0
  RealScalar eta(1.0);

  // The recurrence below estimates the residual in the M^{-1}-norm, which MINRES minimizes. With a preconditioner it is
  // not the 2-norm |b - A x| that error() reports, so convergence is accepted only when b - A x confirms it, and the
  // estimate restarts from b - A x otherwise. residualNorm is a computed |b - A x| exactly when residualIsTrue is set.
  // Stagnation exit: a failed confirmation at least 5 iterations after |b - A x| last dropped to half of its reference
  // value, without such a drop, ends the iteration. Otherwise a tolerance below the rounding error of b - A x would
  // fail confirmations until maxIterations().
  bool residualIsTrue = true;
  RealScalar referenceResidualNorm = (std::numeric_limits<RealScalar>::max)();
  Index referenceIters = 0;

  iters = 0;  // reset iters
  while (iters < maxIters) {
    // Preconditioned Lanczos
    /* Note that there are 4 variants on the Lanczos algorithm. These are
     * described in Paige, C. C. (1972). Computational variants of
     * the Lanczos method for the eigenproblem. IMA Journal of Applied
     * Mathematics, 10(3), 373-381. The current implementation corresponds
     * to the case A(2,7) in the paper. It also corresponds to
     * algorithm 6.14 in Y. Saad, Iterative Methods for Sparse Linear
     * Systems, 2003 p.173. For the preconditioned version see
     * A. Greenbaum, Iterative Methods for Solving Linear Systems, SIAM (1987).
     */
    const RealScalar beta(beta_new);
    v_old = v;          // update: at first time step, this makes v_old = 0 so value of beta doesn't matter
    v_new /= beta_new;  // overwrite v_new for next iteration
    w_new /= beta_new;  // overwrite w_new for next iteration
    v = v_new;          // update
    w = w_new;          // update
    v_new.noalias() = mat * w - beta * v_old;  // compute v_new
    const RealScalar alpha = v_new.dot(w);
    v_new -= alpha * v;            // overwrite v_new
    w_new = precond.solve(v_new);  // overwrite w_new
    beta_new2 = v_new.dot(w_new);  // compute beta_new
    eigen_assert(beta_new2 >= RealScalar(0) && "PRECONDITIONER IS NOT POSITIVE DEFINITE");
    beta_new = numext::sqrt(beta_new2);  // compute beta_new

    // Givens rotation
    const RealScalar r2 = s * alpha + c * c_old * beta;  // s, s_old, c and c_old are still from previous iteration
    const RealScalar r3 = s_old * beta;                  // s, s_old, c and c_old are still from previous iteration
    const RealScalar r1_hat = c * alpha - c_old * s * beta;
    const RealScalar r1 = numext::hypot(r1_hat, beta_new);
    c_old = c;          // store for next iteration
    s_old = s;          // store for next iteration
    c = r1_hat / r1;    // new cosine
    s = beta_new / r1;  // new sine

    // Update solution
    p_oold = p_old;
    p_old = p;
    p = (w - r2 * p_old - r3 * p_oold) / r1;
    x += (residualScale * beta_one * c * eta) * p;
    iters++;  // increment iteration number (for output purposes)

    // Update the estimated residual norm, then confirm convergence with b - A x. beta_new == 0 means the Krylov space
    // is exhausted, and the Lanczos recurrence cannot continue.
    residualNorm *= numext::abs(s);
    residualIsTrue = false;
    const bool krylovExhausted = beta_new == RealScalar(0);
    if (residualNorm < threshold || krylovExhausted) {
      residualNorm = (rhs - mat * x).stableNorm();
      residualIsTrue = true;
      if (residualNorm < threshold || krylovExhausted) {
        break;
      }
      if (residualNorm <= RealScalar(0.5) * referenceResidualNorm) {
        referenceResidualNorm = residualNorm;
        referenceIters = iters;
      } else if (iters - referenceIters >= 5) {
        break;
      }
    }

    eta = -s * eta;  // update eta
  }

  // error() reports |b - A x| / |b|
  if (!residualIsTrue) {
    residualNorm = (rhs - mat * x).stableNorm();
  }
  tol_error = residualNorm / rhsNorm;
}

}  // namespace internal

template <typename MatrixType_, int UpLo_ = Lower, typename Preconditioner_ = IdentityPreconditioner>
class MINRES;

namespace internal {

template <typename MatrixType_, int UpLo_, typename Preconditioner_>
struct traits<MINRES<MatrixType_, UpLo_, Preconditioner_> > {
  using MatrixType = MatrixType_;
  using Preconditioner = Preconditioner_;
};

}  // namespace internal

/** \ingroup IterativeLinearSolvers_Module
 * \brief A minimal residual solver for sparse symmetric problems
 *
 * This class allows to solve for A.x = b sparse linear problems using the MINRES algorithm
 * of Paige and Saunders (1975). The sparse matrix A must be symmetric (possibly indefinite).
 * The vectors x and b can be either dense or sparse.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, can be a dense or a sparse matrix.
 * \tparam UpLo_ the triangular part that will be used for the computations. It can be Lower,
 *               Upper, or Lower|Upper in which the full matrix entries will be considered. Default is Lower.
 * \tparam Preconditioner_ the type of the preconditioner. Default is IdentityPreconditioner
 *
 * The maximal number of iterations and tolerance value can be controlled via the setMaxIterations()
 * and setTolerance() methods. The defaults are twice the number of columns of the matrix for the maximal
 * number of iterations and NumTraits<Scalar>::epsilon() for the tolerance, see tolerance().
 *
 * The tolerance corresponds to the relative residual error: |Ax-b|/|b|. The solver reports Success only when
 * \f$ \| b - A x \| / \| b \| \f$, computed from the returned x, meets it, or, with the default tolerance, the
 * confirmable bound \f$ 8 \sqrt{n}\, \epsilon \f$, see tolerance().
 *
 * This class can be used as the direct solver classes. Here is a typical usage example:
 * \code
 * int n = 10000;
 * VectorXd x(n), b(n);
 * SparseMatrix<double> A(n,n);
 * // fill A and b
 * MINRES<SparseMatrix<double> > mr;
 * mr.compute(A);
 * x = mr.solve(b);
 * std::cout << "#iterations:     " << mr.iterations() << std::endl;
 * std::cout << "estimated error: " << mr.error()      << std::endl;
 * // update b, and solve again
 * x = mr.solve(b);
 * \endcode
 *
 * By default the iterations start with x=0 as an initial guess of the solution.
 * One can control the start using the solveWithGuess() method.
 *
 * MINRES can also be used in a matrix-free context, see the following \link MatrixfreeSolverExample example \endlink.
 *
 * \sa class ConjugateGradient, BiCGSTAB, SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
 */
template <typename MatrixType_, int UpLo_, typename Preconditioner_>
class MINRES : public IterativeSolverBase<MINRES<MatrixType_, UpLo_, Preconditioner_> > {
 protected:
  using Base = IterativeSolverBase<MINRES>;
  using Base::m_error;
  using Base::m_info;
  using Base::m_isInitialized;
  using Base::m_iterations;
  using Base::matrix;

 public:
  using Base::_solve_impl;
  using MatrixType = MatrixType_;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using Preconditioner = Preconditioner_;

  enum { UpLo = UpLo_ };

 public:
  /** Default constructor. */
  MINRES() : Base() { Base::m_tolerance = RealScalar(-1); }

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
  explicit MINRES(const EigenBase<MatrixDerived>& A) : Base(A.derived()) {
    Base::m_tolerance = RealScalar(-1);
  }

  /** \returns the tolerance set by setTolerance(), or by default NumTraits<Scalar>::epsilon().
   *
   * MINRES reports Success only when \f$ \| b - A x \| / \| b \| \f$, computed from the returned x, meets the
   * tolerance. The rounding error of b - A x is of order \f$ \sqrt{m+1}\, \epsilon\, (|b_i| + (|A| |x|)_i) \f$ per
   * component for m nonzeros in a row of A (Higham and Mary, A new approach to probabilistic rounding error analysis,
   * SIAM J. Sci. Comput. 41(5), 2019), and the recurrence for x adds to it, so the residual of the computed x
   * stagnates at a level that depends on the matrix and often exceeds \f$ \epsilon \f$. With the default tolerance
   * the iteration therefore aims at \f$ \epsilon \f$, and whichever way it ends (stagnation of the computed residual,
   * maxIterations(), or an exhausted Krylov space) it reports Success if the computed \f$ \| b - A x \| / \| b \| \f$
   * is at most \f$ 8 \sqrt{n}\, \epsilon \f$ for an n x n matrix, the default tolerance of IDRS. A tolerance set
   * by setTolerance() is never relaxed, so a tolerance below what the arithmetic can confirm ends in NoConvergence
   * with the best iterate and its residual in error().
   *
   * The default is selected by a negative stored tolerance, which no residual can meet, so setTolerance() with a
   * negative value restores the default behavior instead of asking for the impossible.
   */
  RealScalar tolerance() const { return Base::toleranceOrEps(); }

  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const {
    using MatrixWrapper = typename Base::MatrixWrapper;
    using ActualMatrixType = typename Base::ActualMatrixType;
    enum {
      TransposeInput = (!MatrixWrapper::MatrixFree) && (UpLo == (Lower | Upper)) && (!MatrixType::IsRowMajor) &&
                       (!NumTraits<Scalar>::IsComplex)
    };
    using RowMajorWrapper =
        std::conditional_t<TransposeInput, Transpose<const ActualMatrixType>, ActualMatrixType const&>;
    EIGEN_STATIC_ASSERT(internal::check_implication(MatrixWrapper::MatrixFree, UpLo == (Lower | Upper)),
                        MATRIX_FREE_CONJUGATE_GRADIENT_IS_COMPATIBLE_WITH_UPPER_UNION_LOWER_MODE_ONLY);
    using SelfAdjointWrapper =
        std::conditional_t<UpLo == (Lower | Upper), RowMajorWrapper,
                           typename MatrixWrapper::template ConstSelfAdjointViewReturnType<UpLo>::Type>;

    m_iterations = Base::maxIterations();
    const RealScalar tol = tolerance();
    m_error = tol;
    RowMajorWrapper row_mat(matrix());
    internal::minres(SelfAdjointWrapper(row_mat), b, x, Base::m_preconditioner, m_iterations, m_error);
    m_info = Base::infoAtDefaultOrSetTolerance(m_error, tol);
  }

 protected:
};

}  // end namespace Eigen

#endif  // EIGEN_MINRES_H
