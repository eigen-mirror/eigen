// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TRIDIAGONAL_EIGENSOLVER_H
#define EIGEN_TRIDIAGONAL_EIGENSOLVER_H

#include "./TridiagonalBisection.h"

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \eigenvalues_module \ingroup Eigenvalues_Module
 *
 * \class TridiagonalEigenSolver
 *
 * \brief Computes eigenvalues of a real symmetric tridiagonal matrix
 *
 * \tparam Scalar_ the (real floating-point) scalar type of the matrix, e.g. \c float or \c double.
 *
 * This solver computes the eigenvalues of a real symmetric tridiagonal matrix \f$ T \f$, given by
 * its diagonal and sub-diagonal, using SIMD-accelerated, multi-threaded Sturm-sequence spectral
 * bisection (cf. LAPACK's \c xSTEBZ). Unlike the implicit-QR algorithm used by
 * SelfAdjointEigenSolver::computeFromTridiagonal(), it can compute an arbitrary contiguous subset
 * of the spectrum -- by index range or by value range -- selected with an EigenvalueRange:
 *
 * \code
 * TridiagonalEigenSolver<double> es;
 * es.computeEigenvalues(diag, subdiag);                                   // the full spectrum
 * es.computeEigenvalues(diag, subdiag, EigenvalueRange::indices(0, 10));  // the 10 smallest
 * es.computeEigenvalues(diag, subdiag, EigenvalueRange::values(vl, vu));  // those in [vl, vu)
 * \endcode
 *
 * In every mode the computed eigenvalues are returned by eigenvalues() in non-decreasing order,
 * one entry per selected eigenvalue.
 *
 * Besides subset selection, bisection is typically more accurate than the QR algorithm: it
 * resolves each eigenvalue to about one unit in the last place of \f$ \|T\| \f$ independently of
 * the others, whereas the QR forward error accumulates through its O(n) sweeps and grows with the
 * matrix size. (Both are backward stable; this is absolute accuracy relative to \f$ \|T\| \f$, not
 * high relative accuracy for eigenvalues much smaller than \f$ \|T\| \f$.)
 *
 * \note The eigenvalues of a complex Hermitian tridiagonal matrix depend only on the moduli of its
 * off-diagonal entries: it is unitarily similar (via a diagonal phase matrix) to the real symmetric
 * tridiagonal with the same diagonal and off-diagonals \f$ |\beta_k| \f$. So to compute them, pass
 * \c subdiag.cwiseAbs() as the real sub-diagonal.
 *
 * \sa EigenvalueRange, SelfAdjointEigenSolver::computeFromTridiagonal(), class Tridiagonalization
 */
template <typename Scalar_>
class TridiagonalEigenSolver {
 public:
  /** \brief Scalar type of the matrix; must be real. */
  typedef Scalar_ Scalar;
  typedef Scalar RealScalar;
  static_assert(NumTraits<Scalar>::IsComplex == 0 && NumTraits<Scalar>::IsInteger == 0,
                "TridiagonalEigenSolver requires a real floating-point scalar type; for the eigenvalues of a complex "
                "Hermitian tridiagonal matrix, pass subdiag.cwiseAbs() as the real sub-diagonal (see the class "
                "documentation).");

  /** \brief Type for the eigenvalues and the input diagonals: a dynamic-size column vector. */
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  /** \brief Default constructor. Call computeEigenvalues() before querying any result. */
  TridiagonalEigenSolver() = default;

  /** \brief Constructor pre-allocating room for the eigenvalues of a matrix of dimension \a size. */
  explicit TridiagonalEigenSolver(Index size) : m_eivalues(size) {}

  /** \brief Computes the selected eigenvalues of a real symmetric tridiagonal matrix.
   *
   * \param[in] diag    The diagonal of the matrix \f$ T \f$ (length \c n).
   * \param[in] subdiag The sub-diagonal of \f$ T \f$ (length \c n-1).
   * \param[in] range   Which eigenvalues to compute (see EigenvalueRange); defaults to the whole
   *            spectrum.
   * \returns Reference to \c *this
   *
   * After the call, eigenvalues() returns the selected eigenvalues in non-decreasing order (one
   * entry per selected eigenvalue), and info() reports \c Success, or \c NoConvergence if the
   * input contains a non-finite entry.
   */
  template <typename DiagType, typename SubdiagType>
  TridiagonalEigenSolver& computeEigenvalues(const MatrixBase<DiagType>& diag, const MatrixBase<SubdiagType>& subdiag,
                                             const EigenvalueRange& range = EigenvalueRange::all());

  /** \brief Returns the computed eigenvalues, in non-decreasing order.
   *
   * \pre computeEigenvalues() has been called.
   *
   * The returned vector has one entry per \e selected eigenvalue, so a subset request yields a
   * vector shorter than the matrix dimension. Eigenvalues are repeated according to their
   * algebraic multiplicity.
   */
  const VectorType& eigenvalues() const {
    eigen_assert(m_isInitialized && "TridiagonalEigenSolver is not initialized.");
    return m_eivalues;
  }

  /** \brief Reports whether the computation was successful.
   *
   * \returns \c Success if the computation was successful, \c NoConvergence otherwise.
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "TridiagonalEigenSolver is not initialized.");
    return m_info;
  }

 protected:
  VectorType m_eivalues;
  ComputationInfo m_info = InvalidInput;
  bool m_isInitialized = false;
};

template <typename Scalar_>
template <typename DiagType, typename SubdiagType>
TridiagonalEigenSolver<Scalar_>& TridiagonalEigenSolver<Scalar_>::computeEigenvalues(
    const MatrixBase<DiagType>& diag, const MatrixBase<SubdiagType>& subdiag, const EigenvalueRange& range) {
  static_assert(internal::is_same<typename DiagType::Scalar, Scalar>::value &&
                    internal::is_same<typename SubdiagType::Scalar, Scalar>::value,
                "diag and subdiag must have the solver's scalar type");
  const Index n = diag.size();
  eigen_assert(subdiag.size() == (n > 0 ? n - 1 : 0) && "sub-diagonal must have one fewer entry than the diagonal");

  // Reject non-finite input up front (cf. SelfAdjointEigenSolver::computeFromTridiagonal()): the
  // Gershgorin bracketing below would otherwise iterate on NaN brackets and return garbage with
  // info() == Success. allFinite() rather than a max-reduction, whose default PropagateFast
  // semantics may not surface a NaN.
  if (!(diag.allFinite() && subdiag.allFinite())) {
    m_eivalues.resize(0);
    m_info = NoConvergence;
    m_isInitialized = true;
    return *this;
  }

  internal::tridiagonal_bisection(diag.derived(), subdiag.derived(), range, RealScalar(0), m_eivalues);
  m_info = Success;
  m_isInitialized = true;
  return *this;
}

}  // namespace Eigen

#endif  // EIGEN_TRIDIAGONAL_EIGENSOLVER_H
