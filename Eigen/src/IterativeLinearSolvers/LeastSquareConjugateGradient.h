// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_LEAST_SQUARE_CONJUGATE_GRADIENT_H
#define EIGEN_LEAST_SQUARE_CONJUGATE_GRADIENT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal The largest power of two that is at most \a v, for 0 < v < 1, so that s <= v < 2 s. It is found in
 * O(log log (1 / v)) products: the squares 2^-1, 2^-2, 2^-4, ... that exceed v, then a binary descent over them for the
 * largest exponent e with 2^-e > v, and the result is 2^-(e+1). Every product is a power of two, so it is exact, and a
 * product below the least positive number rounds to 0, which does not exceed v and is rejected: nothing is taken from
 * an underflowed value. Only comparison and multiplication of RealScalar are used. */
template <typename RealScalar>
RealScalar least_square_power_of_two_at_most(const RealScalar& v) {
  // 2^-2^k for k < 32, which covers the exponents up to 2^32 - 1. For a scalar with a wider range the result is the
  // power of two 2^-2^32 if that is above v: still an exact scale of b, but it raises b by less than s <= v < 2 s does.
  RealScalar factors[32];
  int count = 0;
  for (RealScalar f = RealScalar(0.5); count < 32 && f > v; f = RealScalar(f * f)) factors[count++] = f;
  RealScalar s(1);  // 2^-e with s > v
  for (int k = count - 1; k >= 0; --k) {
    const RealScalar t = RealScalar(s * factors[k]);
    if (t > v) s = t;
  }
  return RealScalar(s * RealScalar(0.5));
}

/** \internal The number of terms of a dot product of A' r: the average number of nonzeros per column of A when A
 * exposes nonZeros() (for a dense matrix that is its number of rows), and else the number of rows. */
template <typename RealScalar, typename MatrixType>
auto least_square_dot_length(const MatrixType& mat, int) -> decltype(RealScalar(mat.nonZeros())) {
  return numext::maxi(RealScalar(1),
                      RealScalar(RealScalar(mat.nonZeros()) / RealScalar(numext::maxi(mat.cols(), Index(1)))));
}
template <typename RealScalar, typename MatrixType>
RealScalar least_square_dot_length(const MatrixType& mat, long) {
  return RealScalar(mat.rows());
}

/** \internal Low-level conjugate gradient algorithm for least-square problems
 * \param mat The matrix A
 * \param rhs The right hand side vector b
 * \param x On input and initial solution, on output the computed solution.
 * \param precond A preconditioner being able to efficiently solve for an
 *                approximation of A'Ax=b (regardless of b)
 * \param iters On input the max number of iteration, on output the number of performed iterations.
 * \param tol_error On input the tolerance error, on output an estimation of the relative error.
 * \return true if \a tol_error meets the tolerance, or if the normal-equation residual has reached its rounding floor
 *         \f$ \sqrt{L}\, \epsilon \|A\| \|r\| \f$, L the number of nonzeros per column of A, below which it
 *         cannot be resolved.
 */
template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
EIGEN_DONT_INLINE bool least_square_conjugate_gradient(const MatrixType& mat, const Rhs& rhs, Dest& x,
                                                       const Preconditioner& precond, Index& iters,
                                                       typename Dest::RealScalar& tol_error) {
  using RealScalar = typename Dest::RealScalar;
  using Scalar = typename Dest::Scalar;
  using VectorType = Matrix<Scalar, Dynamic, 1>;

  RealScalar tol = tol_error;
  Index maxIters = iters;

  Index m = mat.rows(), n = mat.cols();

  VectorType residual = rhs - mat * x;
  VectorType normal_residual = mat.adjoint() * residual;
  VectorType normal_rhs = mat.adjoint() * rhs;

  RealScalar rhsNorm = normal_rhs.stableNorm();
  // A'b below the smallest normal number has lost its precision or underflowed to 0, and x = 0 would be returned with
  // Success. When |b| < 1, solve for b / s and x / s instead, with s the power of two at most |b| so that the scaling
  // is exact: that only raises A'b, unlike a division by a |b| above 1. The guess is left alone if x / s overflows.
  RealScalar rhsScale(1);
  if (rhsNorm < (std::numeric_limits<RealScalar>::min)()) {
    const RealScalar bNorm = rhs.stableNorm();
    if (bNorm > RealScalar(0) && bNorm < RealScalar(1)) {
      const RealScalar scale = least_square_power_of_two_at_most(bNorm);
      const VectorType scaledGuess = x / scale;
      if (scaledGuess.allFinite()) {
        rhsScale = scale;
        x = scaledGuess;
        residual = rhs / rhsScale - mat * x;
        normal_residual = mat.adjoint() * residual;
        normal_rhs = mat.adjoint() * (rhs / rhsScale);
        rhsNorm = normal_rhs.stableNorm();
      }
    }
  }
  const auto restoreScale = [&]() {
    if (rhsScale != RealScalar(1)) x *= rhsScale;
  };
  if (rhsNorm == 0) {
    x.setZero();
    iters = 0;
    tol_error = 0;
    return true;
  }
  RealScalar threshold = tol * rhsNorm;
  RealScalar residualNorm = normal_residual.stableNorm();
  if (residualNorm == 0 || residualNorm < threshold) {
    restoreScale();
    iters = 0;
    tol_error = residualNorm / rhsNorm;
    return true;
  }

  // Keep the quadratic recurrence terms representable for very small or large residuals.
  const RealScalar residualScale = internal::iterative_solver_scaling_factor(residualNorm);
  normal_residual /= residualScale;
  residualNorm /= residualScale;
  // The threshold and the floor are in units of the scaled residual; the floor makes tol_error = 0 stop where the
  // squares of the residual would underflow.
  threshold = internal::iterative_solver_threshold(tol, RealScalar(rhsNorm / residualScale),
                                                   internal::iterative_solver_residual_floor<RealScalar>());

  VectorType p(n);
  p = precond.solve(normal_residual);  // initial search direction

  VectorType z(n), tmp(m);
  RealScalar absNew = numext::real(normal_residual.dot(p));  // the square of the absolute value of r scaled by invM
  Index i = 0;
  // The gradient A' r cannot be resolved below about sqrt(L) eps |A| |r|, the rounding error of its dot products of L
  // terms, the nonzeros of a column of A (Higham and Mary, SIAM J. Sci. Comput. 41(5), 2019), and CG iterates on that
  // noise past it and can diverge (Hestenes and Stiefel 1952; Bjorck, Numerical Methods for Least Squares Problems,
  // SIAM 1996). |A| is bounded below by |A p| / |p|. The floor is in units of the scaled normal residual. The squares
  // of p and r / scale cost one pass each, and are not used if they overflow.
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar sqrtLength = numext::sqrt(internal::least_square_dot_length<RealScalar>(mat, 0));
  RealScalar normA(0);
  const auto gradientFloor = [&](RealScalar apNorm) {
    const RealScalar pNorm2 = p.squaredNorm();
    if (pNorm2 > RealScalar(0) && (numext::isfinite)(pNorm2))
      normA = numext::maxi(normA, RealScalar(apNorm / numext::sqrt(pNorm2)));
    const RealScalar rNorm2 = (residual / residualScale).squaredNorm();
    const RealScalar floor = RealScalar(sqrtLength * eps * normA * numext::sqrt(rNorm2));
    return (numext::isfinite)(floor) ? floor : RealScalar(0);
  };
  bool atFloor = false;
  while (i < maxIters) {
    tmp.noalias() = mat * p;

    // The step along p, in units of the unscaled residual: residualScale |g|^2 / |A p|^2.
    const RealScalar tmpNorm2 = tmp.squaredNorm();
    RealScalar apNorm = numext::sqrt(tmpNorm2);
    RealScalar step = RealScalar(residualScale * (absNew / tmpNorm2));
    if (!(tmpNorm2 >= (std::numeric_limits<RealScalar>::min)() && (numext::isfinite)(tmpNorm2)) ||
        !(numext::isfinite)(step)) {
      // |A p|^2 or the step over- or underflows: use the scaled norm, as the product (|g|^2 / |A p|) (scale / |A p|)
      // whose factors are applied one after the other.
      apNorm = tmp.stableNorm();
      step = RealScalar(absNew / apNorm);
      const RealScalar tail = RealScalar(residualScale / apNorm);
      if (!(numext::isfinite)(step) || !(numext::isfinite)(tail) || !(numext::isfinite)(apNorm)) {
        // A p vanishes or is not finite, so the step is not defined: p is numerically in the null space of A. Only a
        // gradient below its finite floor counts as converged.
        atFloor = residualNorm <= gradientFloor(apNorm);
        break;
      }
      x += step * (tail * p);           // update solution
      residual -= step * (tail * tmp);  // update residual
    } else {
      x += step * p;           // update solution
      residual -= step * tmp;  // update residual
    }
    normal_residual.noalias() = mat.adjoint() * residual;  // update residual of the normal equation
    normal_residual /= residualScale;

    residualNorm = normal_residual.stableNorm();
    if (residualNorm < threshold) break;

    // Past the floor the gradient is noise: stop with the current iterate, which is as good as any later one.
    if (residualNorm <= gradientFloor(apNorm)) {
      atFloor = true;
      break;
    }

    z = precond.solve(normal_residual);  // approximately solve for "A'A z = normal_residual"

    RealScalar absOld = absNew;
    absNew = numext::real(normal_residual.dot(z));  // update the absolute value of r
    RealScalar beta = absNew / absOld;  // calculate the Gram-Schmidt value used to create the new search direction
    p = z + beta * p;                   // update search direction
    i++;
  }
  restoreScale();
  tol_error = residualNorm / (rhsNorm / residualScale);
  iters = i;
  return tol_error <= tol || atFloor;
}

}  // namespace internal

template <typename MatrixType_,
          typename Preconditioner_ = LeastSquareDiagonalPreconditioner<typename MatrixType_::Scalar> >
class LeastSquaresConjugateGradient;

namespace internal {

template <typename MatrixType_, typename Preconditioner_>
struct traits<LeastSquaresConjugateGradient<MatrixType_, Preconditioner_> > {
  using MatrixType = MatrixType_;
  using Preconditioner = Preconditioner_;
};

}  // namespace internal

/** \ingroup IterativeLinearSolvers_Module
  * \brief A conjugate gradient solver for sparse (or dense) least-square problems
  *
  * This class solves for the least-squares solution to A x = b using an iterative conjugate gradient algorithm.
  * The matrix A can be non symmetric and rectangular, but the matrix A' A should be positive-definite to guaranty
  stability.
  * Otherwise, the SparseLU or SparseQR classes might be preferable.
  * The matrix A and the vectors x and b can be either dense or sparse.
  *
  * \tparam MatrixType_ the type of the matrix A, can be a dense or a sparse matrix.
  * \tparam Preconditioner_ the type of the preconditioner. Default is LeastSquareDiagonalPreconditioner
  *
  * \implsparsesolverconcept
  *
  * The maximal number of iterations and tolerance value can be controlled via the setMaxIterations()
  * and setTolerance() methods. The defaults are twice the number of columns of the matrix for the maximal
  * number of iterations and NumTraits<Scalar>::epsilon() for the tolerance.
  *
  * The tolerance bounds the normal-equation residual \f$ |A'(Ax-b)|/|A'b| \f$, which cannot be evaluated below its
  * rounding error, about \f$ \sqrt{L}\, \epsilon |A| |Ax-b| \f$ for a matrix A with L nonzeros per column (m for a
  * dense m x n matrix). The iteration stops with Success when the residual reaches that floor, as it can only iterate
  * on rounding noise past it, so a tolerance below the floor, such as the default, is met whenever the solution is as
  * accurate as the arithmetic permits.
  *
  * This class can be used as the direct solver classes. Here is a typical usage example:
    \code
    int m=1000000, n = 10000;
    VectorXd x(n), b(m);
    SparseMatrix<double> A(m,n);
    // fill A and b
    LeastSquaresConjugateGradient<SparseMatrix<double> > lscg;
    lscg.compute(A);
    x = lscg.solve(b);
    std::cout << "#iterations:     " << lscg.iterations() << std::endl;
    std::cout << "estimated error: " << lscg.error()      << std::endl;
    // update b, and solve again
    x = lscg.solve(b);
    \endcode
  *
  * By default the iterations start with x=0 as an initial guess of the solution.
  * One can control the start using the solveWithGuess() method.
  *
  * \sa class ConjugateGradient, SparseLU, SparseQR
  */
template <typename MatrixType_, typename Preconditioner_>
class LeastSquaresConjugateGradient
    : public IterativeSolverBase<LeastSquaresConjugateGradient<MatrixType_, Preconditioner_> > {
 protected:
  using Base = IterativeSolverBase<LeastSquaresConjugateGradient>;
  using Base::m_error;
  using Base::m_info;
  using Base::m_isInitialized;
  using Base::m_iterations;
  using Base::matrix;

 public:
  using MatrixType = MatrixType_;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using Preconditioner = Preconditioner_;

 public:
  /** Default constructor. */
  LeastSquaresConjugateGradient() : Base() {}

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
  explicit LeastSquaresConjugateGradient(const EigenBase<MatrixDerived>& A) : Base(A.derived()) {}

  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const {
    m_iterations = Base::maxIterations();
    m_error = Base::m_tolerance;

    const bool converged =
        internal::least_square_conjugate_gradient(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error);
    m_info = converged ? Success : NoConvergence;
  }
};

}  // end namespace Eigen

#endif  // EIGEN_LEAST_SQUARE_CONJUGATE_GRADIENT_H
