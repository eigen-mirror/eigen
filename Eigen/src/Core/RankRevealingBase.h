// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_RANK_REVEALING_BASE_H
#define EIGEN_RANK_REVEALING_BASE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal Stores the column norms of \a block in \a norms, using the fast norm() unless the largest of them is
 * outside the range where it is accurate for the rank decisions that follow: the sum of squares overflows when the
 * norm exceeds ~sqrt(max), and rank decisions compare residual column norms N with eps * maxnorm. The squares of
 * the entries of a column of norm N below 2*sqrt(rows*min) can be lost (flushed to zero) to the point of losing a
 * constant fraction of N^2, so the fast norm must not be used once eps * maxnorm falls below that. Then the norms are
 * recomputed with stableNorm().
 * \returns whether stableNorm() was used, so that later recomputations of partial column norms use it as well.
 */
template <typename BlockType, typename NormsType>
bool column_norms_with_fallback(const BlockType& block, NormsType& norms) {
  using RealScalar = typename NumTraits<typename BlockType::Scalar>::Real;
  norms = block.colwise().norm();
  if (block.size() == 0) return false;
  const RealScalar maxnorm = norms.maxCoeff();
  const RealScalar smallest_accurate =
      RealScalar(2) * numext::sqrt((std::numeric_limits<RealScalar>::min)() * RealScalar(block.rows())) /
      NumTraits<RealScalar>::epsilon();
  // Comparisons against finite bounds, which survive -ffinite-math-only, unlike isfinite().
  if (EIGEN_PREDICT_FALSE(!(maxnorm >= smallest_accurate && maxnorm <= NumTraits<RealScalar>::highest()))) {
    norms = block.colwise().stableNorm();
    return true;
  }
  return false;
}

}  // end namespace internal

/** \brief CRTP mixin providing threshold management, rank computation, and rank-derived queries
 *         for rank-revealing decompositions (FullPivLU, ColPivHouseholderQR, FullPivHouseholderQR).
 *
 * \tparam Derived the concrete decomposition class (CRTP parameter)
 *
 * The derived class must provide:
 *   - rows(), cols() (inherited from SolverBase)
 *   - m_isInitialized (bool member, also used by SolverBase)
 *   - pivotCoeff(Index i) returning the absolute value of the i-th pivot
 */
template <typename Derived>
class RankRevealingBase {
 public:
  using Scalar = typename internal::traits<Derived>::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  RankRevealingBase()
      : m_usePrescribedThreshold(false),
        m_prescribedThreshold(RealScalar(0)),
        m_maxpivot(RealScalar(0)),
        m_nonzero_pivots(0) {}

  /** Allows to prescribe a threshold to be used by certain methods, such as rank(),
   * which need to determine when pivots are to be considered nonzero. This is not used for the
   * decomposition itself, except by CompleteOrthogonalDecomposition, whose rank is determined by compute().
   *
   * When it needs to get the threshold value, Eigen calls threshold(). By default, this is
   * \f$ 4 \, \min(rows, cols) \, \epsilon \f$, where \f$ \epsilon \f$ is \c NumTraits<Scalar>::epsilon() and
   * \a rows and \a cols are the dimensions of the decomposed matrix.
   * Once you have called the present method setThreshold(const RealScalar&),
   * your value is used instead.
   *
   * \param threshold The new value to use as the threshold.
   *
   * A pivot will be considered nonzero if its absolute value is strictly greater than
   *  \f$ threshold \times \vert maxpivot \vert \f$
   * where maxpivot is the biggest pivot.
   *
   * If you want to come back to the default behavior, call setThreshold(Default_t)
   */
  Derived& setThreshold(const RealScalar& threshold) {
    m_usePrescribedThreshold = true;
    m_prescribedThreshold = threshold;
    return self();
  }

  /** Allows to come back to the default behavior, letting Eigen use its default formula for
   * determining the threshold.
   *
   * You should pass the special object Eigen::Default as parameter here.
   * \code dec.setThreshold(Eigen::Default); \endcode
   *
   * See the documentation of setThreshold(const RealScalar&).
   */
  Derived& setThreshold(Default_t) {
    m_usePrescribedThreshold = false;
    return self();
  }

  /** Returns the threshold that will be used by certain methods such as rank(): the value given to
   * setThreshold(const RealScalar&), or by default \f$ 4 \, \min(rows, cols) \, \epsilon \f$ with
   * \f$ \epsilon \f$ = \c NumTraits<Scalar>::epsilon(). It is relative to the biggest pivot.
   */
  RealScalar threshold() const {
    eigen_assert(self().m_isInitialized || m_usePrescribedThreshold);
    // Higham's backward error bound: ||ΔA||₂ ≤ c·min(m,n)·u·||A||₂.
    // The factor of 4 covers the constant c.
    return m_usePrescribedThreshold
               ? m_prescribedThreshold
               : NumTraits<Scalar>::epsilon() * RealScalar(4 * (std::min)(self().rows(), self().cols()));
  }

  /** \returns the rank of the matrix of which *this is the decomposition.
   *
   * \note This method has to determine which pivots should be considered nonzero.
   *       For that, it uses the threshold value that you can control by calling
   *       setThreshold(const RealScalar&).
   */
  inline Index rank() const {
    eigen_assert(self().m_isInitialized && "Decomposition is not initialized.");
    RealScalar premultiplied_threshold = numext::abs(m_maxpivot) * threshold();
    Index result = 0;
    for (Index i = 0; i < m_nonzero_pivots; ++i) result += (self().pivotCoeff(i) > premultiplied_threshold);
    return result;
  }

  /** \returns the dimension of the kernel of the matrix of which *this is the decomposition.
   *
   * \note This method has to determine which pivots should be considered nonzero.
   *       For that, it uses the threshold value that you can control by calling
   *       setThreshold(const RealScalar&).
   */
  inline Index dimensionOfKernel() const {
    eigen_assert(self().m_isInitialized && "Decomposition is not initialized.");
    return self().cols() - rank();
  }

  /** \returns true if the matrix of which *this is the decomposition represents an injective
   *          linear map, i.e. has trivial kernel; false otherwise.
   *
   * \note This method has to determine which pivots should be considered nonzero.
   *       For that, it uses the threshold value that you can control by calling
   *       setThreshold(const RealScalar&).
   */
  inline bool isInjective() const {
    eigen_assert(self().m_isInitialized && "Decomposition is not initialized.");
    return rank() == self().cols();
  }

  /** \returns true if the matrix of which *this is the decomposition represents a surjective
   *          linear map; false otherwise.
   *
   * \note This method has to determine which pivots should be considered nonzero.
   *       For that, it uses the threshold value that you can control by calling
   *       setThreshold(const RealScalar&).
   */
  inline bool isSurjective() const {
    eigen_assert(self().m_isInitialized && "Decomposition is not initialized.");
    return rank() == self().rows();
  }

  /** \returns true if the matrix of which *this is the decomposition is invertible.
   *
   * \note This method has to determine which pivots should be considered nonzero.
   *       For that, it uses the threshold value that you can control by calling
   *       setThreshold(const RealScalar&).
   */
  inline bool isInvertible() const {
    eigen_assert(self().m_isInitialized && "Decomposition is not initialized.");
    return isInjective() && isSurjective();
  }

  /** \returns the number of nonzero pivots in the decomposition.
   * Here nonzero is meant in the exact sense, not in a fuzzy sense.
   * So that notion isn't really intrinsically interesting, but it is
   * still useful when implementing algorithms.
   *
   * \sa rank()
   */
  inline Index nonzeroPivots() const {
    eigen_assert(self().m_isInitialized && "Decomposition is not initialized.");
    return m_nonzero_pivots;
  }

  /** \returns the absolute value of the biggest pivot, i.e. the biggest
   *          diagonal coefficient of U (or R).
   */
  RealScalar maxPivot() const { return m_maxpivot; }

 protected:
  bool m_usePrescribedThreshold;
  RealScalar m_prescribedThreshold;
  RealScalar m_maxpivot;
  Index m_nonzero_pivots;

 private:
  Derived& self() { return static_cast<Derived&>(*this); }
  const Derived& self() const { return static_cast<const Derived&>(*this); }
};

}  // end namespace Eigen

#endif  // EIGEN_RANK_REVEALING_BASE_H
