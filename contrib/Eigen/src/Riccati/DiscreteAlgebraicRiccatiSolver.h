// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-FileCopyrightText: 2001-2002 Enthought, Inc.
// SPDX-FileCopyrightText: 2003 SciPy Developers
// SPDX-FileCopyrightText: 1992-2025 The University of Tennessee and The University of Tennessee Research Foundation
// SPDX-FileCopyrightText: 2000-2025 The University of California Berkeley
// SPDX-FileCopyrightText: 2006-2025 The University of Colorado Denver
// SPDX-License-Identifier: MPL-2.0 AND BSD-3-Clause AND BSD-3-Clause-Open-MPI
//
// This file incorporates work covered by the following copyright and
// permission notices:
//
//   DiscreteAlgebraicRiccatiSolver is adapted from scipy.linalg.solve_discrete_are
//   (scipy/linalg/_solvers.py, SciPy 1.18.0):
//
//    Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
//    All rights reserved.
//
//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions
//    are met:
//
//    1. Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//    3. Neither the name of the copyright holder nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//   internal::riccati_balance_exponents() is adapted from LAPACK's xGEBAL
//   (https://github.com/Reference-LAPACK/lapack):
//
//    Copyright (c) 1992-2025 The University of Tennessee and The University
//                            of Tennessee Research Foundation.  All rights
//                            reserved.
//    Copyright (c) 2000-2025 The University of California Berkeley. All
//                            rights reserved.
//    Copyright (c) 2006-2025 The University of Colorado Denver.  All rights
//                            reserved.
//
//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//
//    - Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    - Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer listed
//      in this license in the documentation and/or other materials
//      provided with the distribution.
//
//    - Neither the name of the copyright holders nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
//    The copyright holders provide no reassurances that the source code
//    provided does not infringe any patent, copyright, or any other
//    intellectual property rights of third parties.  The copyright holders
//    disclaim any liability to any recipient for claims brought against
//    recipient by any third party for infringement of that parties
//    intellectual property rights.
//
//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef EIGEN_DISCRETE_ALGEBRAIC_RICCATI_SOLVER_H
#define EIGEN_DISCRETE_ALGEBRAIC_RICCATI_SOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal The QZ decomposition used for a given scalar type. */
template <typename MatrixType, bool IsComplex = NumTraits<typename MatrixType::Scalar>::IsComplex>
struct riccati_qz {
  using type = RealQZ<MatrixType>;
};
template <typename MatrixType>
struct riccati_qz<MatrixType, true> {
  using type = ComplexQZ<MatrixType>;
};

/** \internal \returns the 1-norm (maximum absolute column sum) of \p m. */
template <typename Derived>
typename NumTraits<typename Derived::Scalar>::Real riccati_norm1(const MatrixBase<Derived>& m) {
  return m.size() == 0 ? typename NumTraits<typename Derived::Scalar>::Real(0)
                       : m.cwiseAbs().colwise().sum().maxCoeff();
}

/** \internal
 * Computes the power-of-two scaling of the nonnegative square matrix \p M that balances the 2-norms of its rows and
 * columns, as LAPACK's xGEBAL with JOB='S'. \p M is overwritten by the balanced matrix and the exponents of the
 * scaling factors are returned in \p exponents, such that the balanced matrix is \f$ D^{-1} M D \f$ with
 * \f$ D = \mathrm{diag}(2^{e_i}) \f$.
 */
template <typename RealMatrix>
void riccati_balance_exponents(RealMatrix& M, VectorXi& exponents) {
  using RealScalar = typename RealMatrix::Scalar;
  const Index n = M.rows();
  const RealScalar radix(2);
  const RealScalar factor(0.95);
  const RealScalar sfmin1 = (std::numeric_limits<RealScalar>::min)() / NumTraits<RealScalar>::epsilon();
  const RealScalar sfmax1 = RealScalar(1) / sfmin1;
  const RealScalar sfmin2 = sfmin1 * radix;
  const RealScalar sfmax2 = RealScalar(1) / sfmin2;

  exponents.setZero(n);
  Matrix<RealScalar, Dynamic, 1> scale = Matrix<RealScalar, Dynamic, 1>::Ones(n);
  bool noconv = true;
  while (noconv) {
    noconv = false;
    for (Index i = 0; i < n; ++i) {
      RealScalar c = M.col(i).stableNorm();
      RealScalar r = M.row(i).stableNorm();
      if (numext::is_exactly_zero(c) || numext::is_exactly_zero(r)) continue;

      RealScalar ca = M.col(i).maxCoeff();
      RealScalar ra = M.row(i).maxCoeff();

      RealScalar g = r / radix;
      RealScalar f(1);
      const RealScalar s = c + r;
      int e = 0;
      while (c < g && numext::maxi(f, numext::maxi(c, ca)) < sfmax2 && numext::mini(r, numext::mini(g, ra)) > sfmin2) {
        f *= radix;
        c *= radix;
        ca *= radix;
        r /= radix;
        g /= radix;
        ra /= radix;
        ++e;
      }
      g = c / radix;
      while (g >= r && numext::maxi(r, ra) < sfmax2 && numext::mini(numext::mini(f, c), numext::mini(g, ca)) > sfmin2) {
        f /= radix;
        c /= radix;
        g /= radix;
        ca /= radix;
        r *= radix;
        ra *= radix;
        --e;
      }

      if (c + r >= factor * s) continue;
      if (f < RealScalar(1) && scale(i) < RealScalar(1) && f * scale(i) <= sfmin1) continue;
      if (f > RealScalar(1) && scale(i) > RealScalar(1) && scale(i) >= sfmax1 / f) continue;
      scale(i) *= f;
      exponents(i) += e;
      noconv = true;
      M.row(i) /= f;
      M.col(i) *= f;
    }
  }
}

/** \internal Rounds \p d / 2 to the nearest integer, ties to even (as NumPy's round()). */
inline int riccati_half_round_even(int d) {
  const int q = (d >= 0) ? d / 2 : -((-d + 1) / 2);  // floor(d / 2)
  if (d % 2 == 0) return q;
  return (q % 2 == 0) ? q : q + 1;
}

}  // namespace internal

/** \ingroup Riccati_Module
 *
 * \class DiscreteAlgebraicRiccatiSolver
 *
 * \brief Solves the discrete-time algebraic Riccati equation (DARE)
 *
 * \tparam MatrixType_ the type of the solution matrix \f$ X \f$, a square matrix of real or complex scalars;
 * this is expected to be an instantiation of the Matrix class template.
 *
 * This class computes the stabilizing solution \f$ X \f$ of the discrete-time algebraic Riccati equation
 * \f[ A^* X A - X - (A^* X B) (R + B^* X B)^{-1} (B^* X A) + Q = 0 \f]
 * or of its generalized version
 * \f[ A^* X A - E^* X E - (A^* X B + S) (R + B^* X B)^{-1} (B^* X A + S^*) + Q = 0, \f]
 * where \f$ A \f$ and \f$ E \f$ are \f$ m \times m \f$, \f$ B \f$ and \f$ S \f$ are \f$ m \times n \f$,
 * \f$ Q \f$ is \f$ m \times m \f$ selfadjoint, \f$ R \f$ is \f$ n \times n \f$ selfadjoint and \f$ E \f$ is
 * nonsingular. \f$ R \f$ itself does not need to be invertible.
 *
 * For a solution to exist, the eigenvalues of \f$ A \f$ outside the unit disc have to be controllable and the
 * associated symplectic pencil (see below) must not have eigenvalues on, or too close to, the unit circle.
 *
 * The solver follows the generalized eigenvalue approach of [1]: it forms the extended symplectic pencil
 * \f[
 * H - \lambda J =
 * \begin{bmatrix} A & 0 & B \\ -Q & E^* & -S \\ S^* & 0 & R \end{bmatrix}
 * - \lambda
 * \begin{bmatrix} E & 0 & 0 \\ 0 & A^* & 0 \\ 0 & -B^* & 0 \end{bmatrix},
 * \f]
 * optionally balances it as in [3], deflates it by the columns of \f$ R \f$, computes its generalized Schur form
 * (RealQZ or ComplexQZ) and reorders it such that the eigenvalues inside the unit circle come first. The
 * leading \f$ m \f$ Schur vectors \f$ [U_1; U_2] \f$ span the stable deflating subspace and
 * \f$ X = U_2 U_1^{-1} \f$ (for the generalized equation, \f$ U_1 \f$ is first replaced by \f$ E U_1 \f$).
 * The computation is reported as failed if the pencil does not have exactly \f$ m \f$ eigenvalues inside the unit
 * circle, if \f$ U_1 \f$ is numerically singular or so small that \f$ \|X\| \f$ (of the balanced problem) would
 * exceed about \f$ 0.2 / \varepsilon \f$, or if \f$ U_1^* U_2 \f$ is far from selfadjoint, which happens when the
 * pencil has eigenvalues too close to the unit circle [2]. A stabilizing solution that large cannot be told apart
 * from an unstable mode that \f$ B \f$ does not reach, for which none exists.
 *
 * Balancing is essential for some badly scaled problems, but it can cost accuracy for others, notably when \f$ B \f$
 * is very small compared to the other matrices. With balancing enabled (the default), the solver therefore checks
 * the backward error of the balanced solution whenever balancing changed the pencil; if it is large, or the balanced
 * computation fails, the unbalanced pencil is solved as well and the more accurate solution is returned.
 * compute(..., false) never balances.
 *
 * The algorithm follows SciPy's \c scipy.linalg.solve_discrete_are, with these differences: the generalized equation
 * combines the stable subspace with the balanced \f$ E \f$ (SciPy uses the original one, which is wrong when
 * balancing scales the pencil), the unbalanced pencil serves as a fallback (see above), and the number of stable
 * eigenvalues is checked.
 *
 * Example: a discrete-time LQR controller for a sampled double integrator.
 * \include DiscreteAlgebraicRiccatiSolver.cpp
 * Output: \verbinclude DiscreteAlgebraicRiccatiSolver.out
 *
 * References:
 * - [1] P. Van Dooren, "A Generalized Eigenvalue Approach For Solving Riccati Equations", SIAM J. Sci. Stat. Comput.,
 *   2(2), 1981.
 * - [2] A. J. Laub, "A Schur Method for Solving Algebraic Riccati Equations", MIT LIDS-R-859, 1978.
 * - [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", SIAM J. Sci. Comput., 22(5), 2001.
 *
 * \sa class RealQZ, class ComplexQZ
 */
template <typename MatrixType_>
class DiscreteAlgebraicRiccatiSolver {
 public:
  using MatrixType = MatrixType_;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  /** \brief Default constructor; the solver is initialized by calling compute(). */
  DiscreteAlgebraicRiccatiSolver() = default;

  /** \brief Constructor; solves the discrete-time algebraic Riccati equation.
   *
   * This is equivalent to calling compute(const EigenBase<InputTypeA>&, const EigenBase<InputTypeB>&,
   * const EigenBase<InputTypeQ>&, const EigenBase<InputTypeR>&, bool).
   */
  template <typename InputTypeA, typename InputTypeB, typename InputTypeQ, typename InputTypeR>
  DiscreteAlgebraicRiccatiSolver(const EigenBase<InputTypeA>& A, const EigenBase<InputTypeB>& B,
                                 const EigenBase<InputTypeQ>& Q, const EigenBase<InputTypeR>& R, bool balance = true) {
    compute(A, B, Q, R, balance);
  }

  /** \brief Constructor; solves the generalized discrete-time algebraic Riccati equation.
   *
   * This is equivalent to calling compute(const EigenBase<InputTypeA>&, const EigenBase<InputTypeB>&,
   * const EigenBase<InputTypeQ>&, const EigenBase<InputTypeR>&, const EigenBase<InputTypeE>&,
   * const EigenBase<InputTypeS>&, bool).
   */
  template <typename InputTypeA, typename InputTypeB, typename InputTypeQ, typename InputTypeR, typename InputTypeE,
            typename InputTypeS>
  DiscreteAlgebraicRiccatiSolver(const EigenBase<InputTypeA>& A, const EigenBase<InputTypeB>& B,
                                 const EigenBase<InputTypeQ>& Q, const EigenBase<InputTypeR>& R,
                                 const EigenBase<InputTypeE>& E, const EigenBase<InputTypeS>& S, bool balance = true) {
    compute(A, B, Q, R, E, S, balance);
  }

  /** \brief Solves \f$ A^* X A - X - (A^* X B) (R + B^* X B)^{-1} (B^* X A) + Q = 0 \f$.
   *
   * \param[in] A        square \f$ m \times m \f$ matrix
   * \param[in] B        \f$ m \times n \f$ matrix
   * \param[in] Q        selfadjoint \f$ m \times m \f$ matrix
   * \param[in] R        selfadjoint \f$ n \times n \f$ matrix
   * \param[in] balance  whether the symplectic pencil is balanced before the QZ decomposition, see the class
   *                     documentation
   *
   * \returns a reference to \c *this
   *
   * The result of the computation is reported by info().
   */
  template <typename InputTypeA, typename InputTypeB, typename InputTypeQ, typename InputTypeR>
  DiscreteAlgebraicRiccatiSolver& compute(const EigenBase<InputTypeA>& A, const EigenBase<InputTypeB>& B,
                                          const EigenBase<InputTypeQ>& Q, const EigenBase<InputTypeR>& R,
                                          bool balance = true) {
    return computeImpl(A.derived(), B.derived(), Q.derived(), R.derived(), nullptr, nullptr, balance);
  }

  /** \brief Solves \f$ A^* X A - E^* X E - (A^* X B + S) (R + B^* X B)^{-1} (B^* X A + S^*) + Q = 0 \f$.
   *
   * \param[in] A        square \f$ m \times m \f$ matrix
   * \param[in] B        \f$ m \times n \f$ matrix
   * \param[in] Q        selfadjoint \f$ m \times m \f$ matrix
   * \param[in] R        selfadjoint \f$ n \times n \f$ matrix
   * \param[in] E        nonsingular \f$ m \times m \f$ matrix
   * \param[in] S        \f$ m \times n \f$ matrix
   * \param[in] balance  whether the symplectic pencil is balanced before the QZ decomposition, see the class
   *                     documentation
   *
   * \returns a reference to \c *this
   *
   * Pass the identity for \p E or zero for \p S to leave out one of the generalized terms.
   * The result of the computation is reported by info().
   */
  template <typename InputTypeA, typename InputTypeB, typename InputTypeQ, typename InputTypeR, typename InputTypeE,
            typename InputTypeS>
  DiscreteAlgebraicRiccatiSolver& compute(const EigenBase<InputTypeA>& A, const EigenBase<InputTypeB>& B,
                                          const EigenBase<InputTypeQ>& Q, const EigenBase<InputTypeR>& R,
                                          const EigenBase<InputTypeE>& E, const EigenBase<InputTypeS>& S,
                                          bool balance = true) {
    const DynamicMatrix e = E.derived();
    const DynamicMatrix s = S.derived();
    return computeImpl(A.derived(), B.derived(), Q.derived(), R.derived(), &e, &s, balance);
  }

  /** \returns the stabilizing solution \f$ X \f$ of the Riccati equation, which is selfadjoint.
   *
   * \pre The solver has been initialized and info() returns \c Success.
   */
  const MatrixType& solution() const {
    eigen_assert(m_isInitialized && "DiscreteAlgebraicRiccatiSolver is not initialized.");
    eigen_assert(m_info == Success && "The Riccati equation could not be solved.");
    return m_X;
  }

  /** \brief Reports whether the last computation was successful.
   *
   * \returns \c Success if the computation was successful,
   *          \c InvalidInput if an input matrix is not finite, \p Q or \p R is not selfadjoint, or \p E is
   *          numerically singular,
   *          \c NoConvergence if the QZ iteration did not converge, and
   *          \c NumericalIssue if the stable deflating subspace of the symplectic pencil could not be isolated
   *          (no stabilizing solution exists, the pencil has eigenvalues too close to the unit circle, or the
   *          solution would be too large to compute, see the class documentation).
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "DiscreteAlgebraicRiccatiSolver is not initialized.");
    return m_info;
  }

 protected:
  using DynamicMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using RealDynamicMatrix = Matrix<RealScalar, Dynamic, Dynamic>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;

  static bool isSelfAdjoint(const DynamicMatrix& M) {
    return internal::riccati_norm1(M - M.adjoint()) <=
           RealScalar(100) * NumTraits<RealScalar>::epsilon() * internal::riccati_norm1(M);
  }

  DiscreteAlgebraicRiccatiSolver& computeImpl(const DynamicMatrix& A, const DynamicMatrix& B, const DynamicMatrix& Q,
                                              const DynamicMatrix& R, const DynamicMatrix* E, const DynamicMatrix* S,
                                              bool balance);

  static ComputationInfo solvePencil(const DynamicMatrix& A, const DynamicMatrix& B, const DynamicMatrix& Q,
                                     const DynamicMatrix& R, const DynamicMatrix* E, const DynamicMatrix* S,
                                     bool balance, RealScalar rcondThreshold, DynamicMatrix& X, RealVector& balancing);

  // Backward error ratios of a solution on the original and on the balanced equation, see backwardErrorRatios().
  struct BackwardErrorRatios {
    RealScalar original;
    RealScalar balanced;
  };

  static BackwardErrorRatios backwardErrorRatios(const DynamicMatrix& A, const DynamicMatrix& B, const DynamicMatrix& Q,
                                                 const DynamicMatrix& R, const DynamicMatrix* E, const DynamicMatrix* S,
                                                 const DynamicMatrix& X, const RealVector& balancing);

  MatrixType m_X;
  ComputationInfo m_info = InvalidInput;
  bool m_isInitialized = false;
};

template <typename MatrixType_>
DiscreteAlgebraicRiccatiSolver<MatrixType_>& DiscreteAlgebraicRiccatiSolver<MatrixType_>::computeImpl(
    const DynamicMatrix& A, const DynamicMatrix& B, const DynamicMatrix& Q, const DynamicMatrix& R,
    const DynamicMatrix* E, const DynamicMatrix* S, bool balance) {
  const Index m = B.rows();
  const Index n = B.cols();
  EIGEN_ONLY_USED_FOR_DEBUG(n);
  eigen_assert(A.rows() == m && A.cols() == m && "A must be square with as many rows as B.");
  eigen_assert(Q.rows() == m && Q.cols() == m && "Q must have the same size as A.");
  eigen_assert(R.rows() == n && R.cols() == n && "R must be square with as many columns as B.");
  eigen_assert((E == nullptr || (E->rows() == m && E->cols() == m)) && "E must have the same size as A.");
  eigen_assert((S == nullptr || (S->rows() == m && S->cols() == n)) && "S must have the same size as B.");

  m_isInitialized = true;
  m_info = InvalidInput;

  if (!A.allFinite() || !B.allFinite() || !Q.allFinite() || !R.allFinite()) return *this;
  if ((E != nullptr && !E->allFinite()) || (S != nullptr && !S->allFinite())) return *this;
  if (!isSelfAdjoint(Q) || !isSelfAdjoint(R)) return *this;
  if (m == 0) {
    m_X.resize(0, 0);
    m_info = Success;
    return *this;
  }
  // rcond() bounds 1 / kappa_1(E) from above and kappa_1 <= m kappa_2, so this rejects only kappa_2(E) > 1 / eps.
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  if (E != nullptr && !(RealScalar(m) * PartialPivLU<DynamicMatrix>(*E).rcond() >= eps)) return *this;
  // U1 is held to the stricter rcond(U1) >= eps: the test above guarantees no false rejection but can accept
  // kappa_2 > 1 / eps, and for U1 it accepts the non-stabilizable problem of dare_no_stabilizing_solution() with a
  // meaningless ||X|| ~ 1e16.
  const RealScalar rcondThreshold = eps;

  DynamicMatrix X;
  RealVector balancing;
  m_info = solvePencil(A, B, Q, R, E, S, balance, rcondThreshold, X, balancing);
  if (balancing.size() > 0) {
    // Balancing is essential for some badly scaled problems (e.g. DAREX 13 and 14), but costs accuracy for others,
    // notably when B is small. The unbalanced pencil is solved as well if the balanced solve fails or looks
    // inaccurate by either ratio of backwardErrorRatios(): above 10 on the original equation (balancing lost
    // accuracy) or above 100 on the balanced equation (a poor solution that the kappa factor of the first ratio
    // hides, e.g. 1e3 for the unbalanced solutions of DAREX 14, against 0.2 to 3 for good ones). Of the two
    // solutions, the one with the smaller ratio on the original equation is kept. Against a long double reference,
    // this kept the more accurate solution in 40000 random problems like those of dare_random() whenever the
    // difference exceeded a factor 2, and triggered the second solve only for problems with B scaled by 10^-1 or
    // less. If balancing left the pencil unchanged, there is nothing to fall back to.
    const RealScalar inf = NumTraits<RealScalar>::infinity();
    const BackwardErrorRatios ratios =
        m_info == Success ? backwardErrorRatios(A, B, Q, R, E, S, X, balancing) : BackwardErrorRatios{inf, inf};
    if (!(ratios.original <= RealScalar(10) && ratios.balanced <= RealScalar(100))) {
      DynamicMatrix Xu;
      RealVector none;
      if (solvePencil(A, B, Q, R, E, S, false, rcondThreshold, Xu, none) == Success &&
          (m_info != Success || backwardErrorRatios(A, B, Q, R, E, S, Xu, balancing).original < ratios.original)) {
        X.swap(Xu);
        m_info = Success;
      }
    }
  }
  if (m_info == Success) m_X = X;
  return *this;
}

/** \internal
 * \returns the backward error ratios of the solution \p X on the original equation,
 * \f[ \rho = \frac{\|\mathrm{res}\|_F}{m\,\varepsilon\,(\|A^*XA\|_F + \|E^*XE\|_F + \|Q\|_F)\,\kappa},\qquad
 *     \kappa = \max(1, \|E^*XE\|_F / \|Q\|_F), \f]
 * and on the balanced equation,
 * \f[ \rho_b = \frac{\|D^{-1} \mathrm{res}\, D^{-1}\|_F}{\varepsilon\,(\|D^{-1} A^*XA\, D^{-1}\|_F
 *     + \|D^{-1} E^*XE\, D^{-1}\|_F + \|D^{-1} Q D^{-1}\|_F + \|D^{-1} G D^{-1}\|_F)}. \f]
 * Here res \f$ = A^*XA - E^*XE - G + Q \f$ is the residual, \f$ G = (A^*XB + S)(R + B^*XB)^{-1}(B^*XA + S^*) \f$, and
 * \f$ D = \f$ diag(\p balancing) relates \p X to the solution of the balanced equation by \f$ X = D X_b D \f$; its
 * residual \f$ D^{-1} \mathrm{res}\, D^{-1} \f$ is exact in floating point, since \f$ D \f$ consists of powers of two.
 *
 * \f$ \kappa \f$ accounts for the residual that a large solution leaves even when computed accurately: for random
 * problems with B scaled by \f$ 10^{-k} \f$, the residual without \f$ \kappa \f$ grows in proportion to
 * \f$ \|X\|_F / \|Q\|_F \f$, for SciPy (LAPACK) as for this solver. When \f$ X \f$ is large only because the problem
 * is badly scaled, \f$ \kappa \f$ also hides poor solutions, which \f$ \rho_b \f$ exposes. Non-finite ratios are
 * reported as infinity.
 */
template <typename MatrixType_>
typename DiscreteAlgebraicRiccatiSolver<MatrixType_>::BackwardErrorRatios
DiscreteAlgebraicRiccatiSolver<MatrixType_>::backwardErrorRatios(const DynamicMatrix& A, const DynamicMatrix& B,
                                                                 const DynamicMatrix& Q, const DynamicMatrix& R,
                                                                 const DynamicMatrix* E, const DynamicMatrix* S,
                                                                 const DynamicMatrix& X, const RealVector& balancing) {
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const auto finite = [](RealScalar r) { return (numext::isfinite)(r) ? r : NumTraits<RealScalar>::infinity(); };
  const DynamicMatrix XA = X * A;
  DynamicMatrix AXA = A.adjoint() * XA;
  DynamicMatrix EXE = (E != nullptr) ? DynamicMatrix(E->adjoint() * X * *E) : X;
  DynamicMatrix G = DynamicMatrix::Zero(A.rows(), A.rows());
  if (B.cols() > 0) {
    DynamicMatrix AXBS = XA.adjoint() * B;
    if (S != nullptr) AXBS += *S;
    G = AXBS * (R + B.adjoint() * X * B).partialPivLu().solve(AXBS.adjoint());
  }
  DynamicMatrix res = AXA - EXE - G + Q;
  if (numext::is_exactly_zero(res.norm())) return BackwardErrorRatios{RealScalar(0), RealScalar(0)};

  BackwardErrorRatios ratios;
  const RealScalar normQ = Q.norm(), normEXE = EXE.norm();
  const RealScalar kappa = (normQ > RealScalar(0) && normEXE > normQ) ? normEXE / normQ : RealScalar(1);
  ratios.original = finite(res.norm() / (RealScalar(A.rows()) * eps * (AXA.norm() + normEXE + normQ) * kappa));
  DynamicMatrix Qb = Q;
  if (balancing.size() > 0) {
    const RealVector inv = balancing.cwiseInverse();
    for (DynamicMatrix* M : {&res, &AXA, &EXE, &G, &Qb}) *M = inv.asDiagonal() * *M * inv.asDiagonal();
  }
  ratios.balanced = finite(res.norm() / (eps * (AXA.norm() + EXE.norm() + Qb.norm() + G.norm())));
  return ratios;
}

/** \internal
 * Solves the equation through the extended symplectic pencil, balanced if \p balance is true; on success, \p X holds
 * the selfadjoint solution.
 */
template <typename MatrixType_>
ComputationInfo DiscreteAlgebraicRiccatiSolver<MatrixType_>::solvePencil(const DynamicMatrix& A, const DynamicMatrix& B,
                                                                         const DynamicMatrix& Q, const DynamicMatrix& R,
                                                                         const DynamicMatrix* E, const DynamicMatrix* S,
                                                                         bool balance, RealScalar rcondThreshold,
                                                                         DynamicMatrix& X, RealVector& balancing) {
  const Index m = B.rows();
  const Index n = B.cols();

  // Form the extended symplectic pencil H - lambda J.
  const Index p = 2 * m + n;
  DynamicMatrix H = DynamicMatrix::Zero(p, p);
  DynamicMatrix J = DynamicMatrix::Zero(p, p);
  H.topLeftCorner(m, m) = A;
  H.topRightCorner(m, n) = B;
  H.block(m, 0, m, m) = -Q;
  if (E != nullptr)
    H.block(m, m, m, m) = E->adjoint();
  else
    H.block(m, m, m, m).setIdentity();
  if (S != nullptr) {
    H.block(m, 2 * m, m, n) = -*S;
    H.bottomLeftCorner(n, m) = S->adjoint();
  }
  H.bottomRightCorner(n, n) = R;

  if (E != nullptr)
    J.topLeftCorner(m, m) = *E;
  else
    J.topLeftCorner(m, m).setIdentity();
  J.block(m, m, m, m) = A.adjoint();
  J.block(2 * m, m, n, m) = -B.adjoint();

  // Balance the pencil without destroying its symplectic structure: the scaling is diag(D, D^-1, D_R),
  // with D derived from the xGEBAL scaling of |H| + |J| with zeroed diagonal, see [3].
  RealVector scaling;
  bool balanced = false;
  balancing.resize(0);
  if (balance) {
    RealDynamicMatrix M = H.cwiseAbs() + J.cwiseAbs();
    M.diagonal().setZero();
    VectorXi exponents;
    internal::riccati_balance_exponents(M, exponents);
    if (!(exponents.array() == 0).all()) {
      balanced = true;
      scaling.resize(p);
      for (Index i = 0; i < m; ++i) {
        const int d = internal::riccati_half_round_even(exponents(m + i) - exponents(i));
        scaling(i) = numext::ldexp(RealScalar(1), d);
        scaling(m + i) = numext::ldexp(RealScalar(1), -d);
      }
      // As in SciPy, the R block is scaled by D_R = diag(2^e), the inverse of the xGEBAL scaling D^-1 M D. The
      // xGEBAL-consistent 2^-e loses accuracy: the DAREX 13 and 14 residuals grow from 7.7e-7 and 1.3e-7 to 1.6e-5
      // and 1.8e-6.
      for (Index i = 2 * m; i < p; ++i) scaling(i) = numext::ldexp(RealScalar(1), exponents(i));
      // H <- D H D^-1, J <- D J D^-1 with D = diag(scaling), exact in floating point.
      H = scaling.asDiagonal() * H * scaling.cwiseInverse().asDiagonal();
      J = scaling.asDiagonal() * J * scaling.cwiseInverse().asDiagonal();
      // Also reported when the solve fails, so that computeImpl() falls back to the unbalanced pencil.
      balancing = scaling.head(m);
    }
  }

  // Deflate the pencil by the columns belonging to R, see [1].
  DynamicMatrix Hd, Jd;
  if (n > 0) {
    const HouseholderQR<DynamicMatrix> qr(H.rightCols(n));
    Hd = (qr.householderQ().adjoint() * H.leftCols(2 * m)).bottomRows(2 * m);
    Jd = (qr.householderQ().adjoint() * J.leftCols(2 * m)).bottomRows(2 * m);
  } else {
    Hd = H;
    Jd = J;
  }

  // Generalized Schur form with the eigenvalues inside the unit circle first.
  typename internal::riccati_qz<DynamicMatrix>::type qz(Hd, Jd, true);
  if (qz.info() == NoConvergence) return NoConvergence;
  DynamicMatrix SS = qz.matrixS();
  DynamicMatrix TT = qz.matrixT();
  DynamicMatrix ZZ = qz.matrixZ().adjoint();
  if (qz.info() != Success) {
    const DynamicMatrix QQ = qz.matrixQ();
    // ComplexQZ reports NumericalIssue when its direct reduction of a 2x2 block leaves a subdiagonal entry of S above
    // eps ||S||, although the reduction is only backward stable to a few eps ||S||. Accept the decomposition if
    // ||Q (S, T) Z^* - (Hd, Jd)||_F <= 10 (2m) eps ||(Hd, Jd)||_F once that entry is dropped.
    internal::clean_generalized_schur_form(SS, TT);
    const RealScalar bound = RealScalar(10 * 2 * m) * NumTraits<RealScalar>::epsilon();
    if (!((QQ * SS * ZZ.adjoint() - Hd).norm() <= bound * Hd.norm() &&
          (QQ * TT * ZZ.adjoint() - Jd).norm() <= bound * Jd.norm()))
      return NumericalIssue;
  }
  // The eigenvalues of the symplectic pencil come in pairs (lambda, 1 / conj(lambda)), so exactly m of them lie
  // inside the unit circle if a stabilizing solution exists. (SciPy does not check this.)
  if (internal::reorder_generalized_schur_inside_unit_circle<DynamicMatrix>(SS, TT, nullptr, ZZ) != m)
    return NumericalIssue;

  // The leading m right Schur vectors span the stable deflating subspace.
  DynamicMatrix U = ZZ.leftCols(m);
  if (E != nullptr) {
    // U belongs to the balanced pencil, so it is combined with the balanced E held in J. (SciPy combines it with the
    // original E, which yields wrong solutions or spurious failures whenever balancing scales the pencil.)
    U.topRows(m) = (J.topLeftCorner(m, m) * U.topRows(m)).eval();
    U = HouseholderQR<DynamicMatrix>(U).householderQ() * DynamicMatrix::Identity(2 * m, m);
  }
  const auto U1 = U.topRows(m);
  const auto U2 = U.bottomRows(m);

  // [U1; U2] has orthonormal columns, so ||X||_2 = sqrt(1 - sigma_min(U1)^2) / sigma_min(U1). rcond(U1) is scale
  // invariant and misses a uniformly small U1, which an unstable mode that B does not reach produces from rounding
  // errors. ||U1||_1 rcond(U1) estimates 1 / ||U1^-1||_1 <= sqrt(m) sigma_min(U1), so the second test bounds the
  // solution of the (balanced) pencil by about sqrt(m) / (5 rcondThreshold). Stabilizing solutions beyond that are
  // numerically indistinguishable from the absence of one. The factor 5 rejected all of 14400 random problems with
  // unstable modes that B does not reach; with 1, 10% of them returned a meaningless X.
  const PartialPivLU<DynamicMatrix> lu(U1);
  const RealScalar rcondU1 = lu.rcond();
  if (!(rcondU1 >= rcondThreshold && internal::riccati_norm1(U1) * rcondU1 >= RealScalar(5) * rcondThreshold))
    return NumericalIssue;

  // A large deviation of U1^* U2 from selfadjointness indicates failure, see Theorem 5 in [2].
  const DynamicMatrix uSym = U1.adjoint() * U2;
  // SciPy's threshold max(spacing(1000), 0.1 ||U1^* U2||_1), with spacing(1000) = 512 eps.
  const RealScalar symThreshold =
      numext::maxi(RealScalar(512) * NumTraits<RealScalar>::epsilon(), RealScalar(0.1) * internal::riccati_norm1(uSym));
  if (!(internal::riccati_norm1(uSym - uSym.adjoint()) <= symThreshold)) return NumericalIssue;

  // X = U2 U1^-1 is computed as (U1^-* U2^*)^*.
  X = lu.adjoint().solve(U2.adjoint());
  X.adjointInPlace();
  if (balanced) X = balancing.asDiagonal() * X * balancing.asDiagonal();
  X = (X + X.adjoint()).eval() / RealScalar(2);
  // The solve, the unscaling and the symmetrization can overflow.
  if (!X.allFinite()) return NumericalIssue;
  return Success;
}

}  // namespace Eigen

#endif  // EIGEN_DISCRETE_ALGEBRAIC_RICCATI_SOLVER_H
