// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MATRIXBASEEIGENVALUES_H
#define EIGEN_MATRIXBASEEIGENVALUES_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Derived, bool IsComplex>
struct eigenvalues_selector {
  // this is the implementation for the case IsComplex = true
  static inline typename MatrixBase<Derived>::EigenvaluesReturnType const run(const MatrixBase<Derived>& m) {
    using PlainObject = typename Derived::PlainObject;
    PlainObject m_eval(m);
    return ComplexEigenSolver<PlainObject>(m_eval, false).eigenvalues();
  }
};

template <typename Derived>
struct eigenvalues_selector<Derived, false> {
  static inline typename MatrixBase<Derived>::EigenvaluesReturnType const run(const MatrixBase<Derived>& m) {
    using PlainObject = typename Derived::PlainObject;
    PlainObject m_eval(m);
    return EigenSolver<PlainObject>(m_eval, false).eigenvalues();
  }
};

}  // end namespace internal

/** \brief Computes the eigenvalues of a matrix
 * \returns Column vector containing the eigenvalues.
 *
 * \eigenvalues_module
 * This function computes the eigenvalues with the help of the EigenSolver
 * class (for real matrices) or the ComplexEigenSolver class (for complex
 * matrices).
 *
 * The eigenvalues are repeated according to their algebraic multiplicity,
 * so there are as many eigenvalues as rows in the matrix.
 *
 * The SelfAdjointView class provides a better algorithm for selfadjoint
 * matrices.
 *
 * Example: \include MatrixBase_eigenvalues.cpp
 * Output: \verbinclude MatrixBase_eigenvalues.out
 *
 * \sa EigenSolver::eigenvalues(), ComplexEigenSolver::eigenvalues(),
 *     SelfAdjointView::eigenvalues()
 */
template <typename Derived>
inline typename MatrixBase<Derived>::EigenvaluesReturnType MatrixBase<Derived>::eigenvalues() const {
  return internal::eigenvalues_selector<Derived, NumTraits<Scalar>::IsComplex>::run(derived());
}

/** \brief Computes the eigenvalues of a matrix
 * \returns Column vector containing the eigenvalues.
 *
 * \eigenvalues_module
 * This function computes the eigenvalues with the help of the
 * SelfAdjointEigenSolver class.  The eigenvalues are repeated according to
 * their algebraic multiplicity, so there are as many eigenvalues as rows in
 * the matrix.
 *
 * Example: \include SelfAdjointView_eigenvalues.cpp
 * Output: \verbinclude SelfAdjointView_eigenvalues.out
 *
 * \sa SelfAdjointEigenSolver::eigenvalues(), MatrixBase::eigenvalues()
 */
template <typename MatrixType, unsigned int UpLo>
EIGEN_DEVICE_FUNC inline typename SelfAdjointView<MatrixType, UpLo>::EigenvaluesReturnType
SelfAdjointView<MatrixType, UpLo>::eigenvalues() const {
  PlainObject thisAsMatrix(*this);
  return SelfAdjointEigenSolver<PlainObject>(thisAsMatrix, false).eigenvalues();
}

/** \brief Computes the L2 operator norm
 * \returns Operator norm of the matrix.
 *
 * \eigenvalues_module
 * This function computes the L2 operator norm of a matrix, which is also
 * known as the spectral norm. The norm of a matrix \f$ A \f$ is defined to be
 * \f[ \|A\|_2 = \max_x \frac{\|Ax\|_2}{\|x\|_2} \f]
 * where the maximum is over all vectors and the norm on the right is the
 * Euclidean vector norm. The norm equals the largest singular value, which is
 * the square root of the largest eigenvalue of the positive semi-definite
 * matrix \f$ A^*A \f$.
 *
 * The current implementation uses the eigenvalues of \f$ A^*A \f$, as computed
 * by SelfAdjointView::eigenvalues(), to compute the operator norm of a
 * matrix.  The SelfAdjointView class provides a better algorithm for
 * selfadjoint matrices.
 *
 * Example: \include MatrixBase_operatorNorm.cpp
 * Output: \verbinclude MatrixBase_operatorNorm.out
 *
 * \sa SelfAdjointView::eigenvalues(), SelfAdjointView::operatorNorm()
 */
template <typename Derived>
inline typename MatrixBase<Derived>::RealScalar MatrixBase<Derived>::operatorNorm() const {
  typename Derived::PlainObject m_eval(derived());
  if (m_eval.size() == 0) return RealScalar(0);
  // Scale by a power of two first, so that forming A*A^* can neither overflow nor underflow: ||A||_2 = s*||A/s||_2.
  // A NaN or an infinity in A makes the norm NaN or infinite, as it does for the other norms.
  const RealScalar maxCoeff = internal::safe_scaling<RealScalar>::max_abs_component(m_eval);
  if (!(numext::isfinite)(maxCoeff) || numext::is_exactly_zero(maxCoeff)) return maxCoeff;
  const auto factors = internal::safe_scaling<RealScalar>::scale_in_place(m_eval, maxCoeff);
  const auto gram = (m_eval * m_eval.adjoint()).eval();
  SelfAdjointEigenSolver<std::decay_t<decltype(gram)>> solver(gram, EigenvaluesOnly);
  if (solver.info() != Success) return NumTraits<RealScalar>::quiet_NaN();
  RealScalar result = numext::sqrt(numext::maxi(solver.eigenvalues().maxCoeff(), RealScalar(0)));
  internal::safe_scaling<RealScalar>::unscale_in_place(result, factors);
  return result;
}

/** \brief Computes the L2 operator norm
 * \returns Operator norm of the matrix.
 *
 * \eigenvalues_module
 * This function computes the L2 operator norm of a self-adjoint matrix. For a
 * self-adjoint matrix, the operator norm is the largest eigenvalue.
 *
 * The current implementation uses the eigenvalues of the matrix, as computed
 * by eigenvalues(), to compute the operator norm of the matrix.
 *
 * Example: \include SelfAdjointView_operatorNorm.cpp
 * Output: \verbinclude SelfAdjointView_operatorNorm.out
 *
 * \sa eigenvalues(), MatrixBase::operatorNorm()
 */
template <typename MatrixType, unsigned int UpLo>
EIGEN_DEVICE_FUNC inline typename SelfAdjointView<MatrixType, UpLo>::RealScalar
SelfAdjointView<MatrixType, UpLo>::operatorNorm() const {
  PlainObject thisAsMatrix(*this);
  if (thisAsMatrix.size() == 0) return RealScalar(0);
  // The solver ignores the imaginary part of the diagonal, so does the check below.
  EIGEN_IF_CONSTEXPR (NumTraits<Scalar>::IsComplex) {
    thisAsMatrix.diagonal() = thisAsMatrix.diagonal().real();
  }
  // A NaN gives NaN and an infinity +infinity, as for MatrixBase::operatorNorm(): the solver rejects both, and the
  // eigenvalues are unspecified after it fails.
  const RealScalar maxCoeff = internal::safe_scaling<RealScalar>::max_abs_component(thisAsMatrix);
  if (!(numext::isfinite)(maxCoeff)) return maxCoeff;
  SelfAdjointEigenSolver<PlainObject> solver(thisAsMatrix, EigenvaluesOnly);
  if (solver.info() != Success) return NumTraits<RealScalar>::quiet_NaN();
  return solver.eigenvalues().cwiseAbs().maxCoeff();
}

}  // end namespace Eigen

#endif
