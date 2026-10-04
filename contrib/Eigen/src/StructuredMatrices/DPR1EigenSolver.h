// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_DPR1_EIGEN_SOLVER_H
#define EIGEN_STRUCTURED_DPR1_EIGEN_SOLVER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup StructuredMatrices_Module
 * \class DPR1EigenSolver
 * \brief Direct O(n^2) eigensolver for real symmetric diagonal-plus-rank-one
 * matrices \f$ A = D + \rho\, z z^T \f$, via the secular equation.
 *
 * This is the standalone version of the kernel at the heart of the
 * divide-and-conquer symmetric eigensolvers (LAPACK's xLAED2/3/4): after
 * \em deflation -- entries with negligible \f$ |z_i| \f$ are eigenpairs of the
 * diagonal already, and (nearly) equal diagonal entries are combined by Givens
 * rotations whose dropped coupling is below a backward-stability threshold --
 * the surviving eigenvalues are the roots of the secular equation
 * \f[ f(\lambda) = 1 + \rho \sum_i \frac{z_i^2}{d_i - \lambda} = 0, \f]
 * one in each interval between consecutive poles. Each root is bracketed and
 * bisected in coordinates \em shifted to its nearest pole, so every distance
 * \f$ \lambda - d_i \f$ is retained as an exact data difference plus a small
 * offset instead of a cancellation-prone subtraction of close numbers.
 * Eigenvectors are then built not from the original \c z but from the
 * Gu-Eisenstat vector \f$ \hat z \f$ -- the one for which the computed roots
 * are \em exact secular eigenvalues -- which is what makes the computed
 * eigenvector matrix numerically orthogonal without any reorthogonalization.
 *
 * The total cost is O(n^2): O(n log(1/eps)) per bisected root in the common
 * case (the iteration cap is sized to the scalar's full exponent range, so
 * even roots subnormally close to their pole resolve), O(n) per Gu-Eisenstat
 * weight, and O(n) per eigenvector (the deflation rotations are replayed
 * instead of accumulated into a dense matrix).
 *
 * Both signs of \f$ \rho \f$ are supported (negative \f$ \rho \f$ is handled by
 * negating the matrix), as are \f$ \rho = 0 \f$, zero \c z, repeated diagonal
 * entries and any ordering of \c d. The problem is rescaled internally by the
 * exact power of two that brings \f$ \max(\|D\|_\infty, |\rho| \|z\|^2) \f$ into
 * [1/2, 1), which changes no bits of the result when \f$ \rho \|z\|^2 \f$ and
 * the scaled poles are normal. \c InvalidInput is reported for non-finite
 * input, for an update \f$ \rho \|z\|^2 \f$ that overflows, and for a computed
 * eigenvalue that is not finite.
 *
 * \code
 *   DPR1EigenSolver<double> es(d, rho, z);
 *   VectorXd  lambda = es.eigenvalues();   // ascending
 *   MatrixXd  V      = es.eigenvectors();  // orthogonal
 * \endcode
 *
 * \tparam RealScalar_ one of \c float, \c double, or \c long \c double.
 *
 * References:
 *  - M. Gu and S. C. Eisenstat, "A stable and efficient algorithm for the
 *    rank-one modification of the symmetric eigenproblem," SIAM J. Matrix Anal.
 *    Appl., 15(4):1266-1276, 1994.
 *  - P. H. Sterbenz, "Floating-Point Computation", Prentice-Hall, 1974.
 *    Scaling by a power of two is exact, the property the problem scaling
 *    relies on.
 *
 * \sa class SelfAdjointEigenSolver
 */
template <typename RealScalar_>
class DPR1EigenSolver {
 public:
  using RealScalar = RealScalar_;
  using Scalar = RealScalar;
  using Index = Eigen::Index;
  using VectorType = Matrix<RealScalar, Dynamic, 1>;
  using MatrixType = Matrix<RealScalar, Dynamic, Dynamic>;

  static_assert(std::is_same<RealScalar, float>::value || std::is_same<RealScalar, double>::value ||
                    std::is_same<RealScalar, long double>::value,
                "DPR1EigenSolver supports only float, double, and long double scalar types.");

  /** Default constructor; call \ref compute before querying results. */
  DPR1EigenSolver() = default;

  /** Computes the eigendecomposition of \c diag(d) + \c rho*z*z^T.
   * \a options is \c ComputeEigenvectors (the default) or \c EigenvaluesOnly. */
  DPR1EigenSolver(const VectorType& d, RealScalar rho, const VectorType& z, int options = ComputeEigenvectors) {
    compute(d, rho, z, options);
  }

  /** Computes the eigendecomposition of \c diag(d) + \c rho*z*z^T. \sa DPR1EigenSolver() */
  DPR1EigenSolver& compute(const VectorType& d, RealScalar rho, const VectorType& z, int options = ComputeEigenvectors);

  /** \returns the eigenvalues, sorted in increasing order. */
  const VectorType& eigenvalues() const {
    eigen_assert(m_isInitialized && "DPR1EigenSolver is not initialized.");
    return m_eivalues;
  }

  /** \returns the orthogonal matrix of eigenvectors; column \c k matches
   * \c eigenvalues()[k]. \pre \ref compute was called with \c ComputeEigenvectors. */
  const MatrixType& eigenvectors() const {
    eigen_assert(m_isInitialized && "DPR1EigenSolver is not initialized.");
    eigen_assert(m_vectorsComputed && "eigenvectors were not computed");
    return m_eivec;
  }

  /** \returns \c Success if the decomposition succeeded, \c NoConvergence if a
   * secular root could not be fully resolved, \c InvalidInput if the input was
   * non-finite, \f$ \rho \|z\|^2 \f$ overflows, or a computed eigenvalue is
   * not finite (the eigenvalues are then NaN). */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "DPR1EigenSolver is not initialized.");
    return m_info;
  }

 private:
  // A deflation-stage Givens rotation acting on working rows (i, j).
  struct Rotation {
    Index i, j;
    RealScalar c, s;
  };

  /** \internal Evaluates the shifted secular function
   * g(tau) = 1 + rho * sum_i zeta_i^2 / (delta_i - tau), with delta_i the pole
   * offsets relative to the chosen shift. */
  static RealScalar secular(const VectorType& delta, const VectorType& zeta2, RealScalar rho, RealScalar tau) {
    return RealScalar(1) + rho * (zeta2.array() / (delta.array() - tau)).sum();
  }

  VectorType m_eivalues;
  MatrixType m_eivec;
  bool m_isInitialized = false;
  bool m_vectorsComputed = false;
  ComputationInfo m_info = InvalidInput;
};

template <typename RealScalar_>
DPR1EigenSolver<RealScalar_>& DPR1EigenSolver<RealScalar_>::compute(const VectorType& d, RealScalar rho,
                                                                    const VectorType& z, int options) {
  const Index n = d.size();
  eigen_assert(z.size() == n && "d and z must have the same size");
  eigen_assert((options & ~EigVecMask) == 0 && (options & EigVecMask) != EigVecMask && "invalid option parameter");
  const bool computeVectors = (options & ComputeEigenvectors) == ComputeEigenvectors;
  m_vectorsComputed = false;
  m_info = Success;

  m_eivalues.resize(n);
  if (computeVectors) m_eivec.setIdentity(n, n);

  // Non-finite input would break the sorting comparator (not a strict weak
  // order under NaN) and silently deflate everything; reject it up front.
  if (!(d.allFinite() && z.allFinite() && (numext::isfinite)(rho))) {
    m_eivalues.setConstant(NumTraits<RealScalar>::quiet_NaN());
    m_info = InvalidInput;
    m_isInitialized = true;
    return *this;
  }
  if (n == 0) {
    m_vectorsComputed = computeVectors;
    m_isInitialized = true;
    return *this;
  }

  const bool negated = rho < RealScalar(0);
  const VectorType dW = negated ? VectorType(-d) : d;

  // pi maps each sorted working index to its input row.
  std::vector<Index> pi;
  pi.reserve(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) pi.push_back(i);
  std::stable_sort(pi.begin(), pi.end(), [&dW](Index a, Index b) { return dW[a] < dW[b]; });
  VectorType ds(n), zs(n);
  for (Index i = 0; i < n; ++i) {
    ds[i] = dW[pi[static_cast<std::size_t>(i)]];
    zs[i] = z[pi[static_cast<std::size_t>(i)]];
  }

  // Normalize z and absorb ||z||^2 into rho, then scale the problem by the exact
  // power of two s = 2^-scaleExp that brings max(||D||_inf, rho ||z||^2) into
  // [1/2, 1): eig(sD + s rho zz^T) = s eig(D + rho zz^T).
  const RealScalar znorm = zs.stableNorm();
  if (znorm > RealScalar(0)) zs /= znorm;
  RealScalar rhoW = rho == RealScalar(0) || znorm == RealScalar(0) ? RealScalar(0) : (numext::abs(rho) * znorm) * znorm;
  if (!(numext::isfinite)(rhoW)) {
    // An overflowing update would make the deflation tolerance infinite and
    // silently deflate it away.
    m_eivalues.setConstant(NumTraits<RealScalar>::quiet_NaN());
    m_info = InvalidInput;
    m_isInitialized = true;
    return *this;
  }
  EIGEN_USING_STD(frexp)
  EIGEN_USING_STD(ldexp)
  RealScalar scaledNorm = numext::maxi(ds.cwiseAbs().maxCoeff(), rhoW);  // in [1/2, 1) once scaled
  int scaleExp = 0;
  if (scaledNorm > RealScalar(0)) scaledNorm = frexp(scaledNorm, &scaleExp);
  ds = ds.array().ldexp(-scaleExp).matrix();
  rhoW = ldexp(rhoW, -scaleExp);

  // Backward-error budget: dropping a coupling of size <= tol perturbs the
  // matrix by O(tol), like LAPACK's xLAED2. Using max(|d|_inf, rho*||z||^2) as
  // the scale is the unscaled-problem generalization of xLAED2's criterion: the
  // perturbation stays O(eps * (||D|| + rho ||z||^2)), i.e. backward stable in
  // the data.
  const RealScalar tol = RealScalar(8) * NumTraits<RealScalar>::epsilon() * scaledNorm;

  std::vector<Rotation> rotations;
  std::vector<bool> deflated;

  if (rhoW <= tol) {
    deflated.assign(static_cast<std::size_t>(n), true);
  } else {
    // Negligible z_i leaves d_i as an eigenvalue.
    deflated.reserve(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) deflated.push_back(rhoW * numext::abs(zs[i]) <= tol);
    // The dropped off-diagonal coupling is |c*s*(ds[i]-ds[p])| <= tol.
    Index p = -1;
    for (Index i = 0; i < n; ++i) {
      if (deflated[static_cast<std::size_t>(i)]) continue;
      if (p >= 0) {
        const RealScalar r = numext::hypot(zs[p], zs[i]);
        const RealScalar c = zs[i] / r, s = zs[p] / r;  // zeroes the earlier entry
        const RealScalar gap = ds[i] - ds[p];
        if (numext::abs(c * s * gap) <= tol) {
          // Rotated diagonal (c^2 a + s^2 b, s^2 a + c^2 b), a = ds[p], b = ds[i], with c^2 + s^2 = 1
          // applied exactly and the smaller weight w = min(c^2, s^2) <= 1/2 on the gap:
          //   |s| <= |c|: (a + s^2 gap, b - s^2 gap),  otherwise (b - c^2 gap, a + c^2 gap).
          // Equal poles (gap = 0) keep their value, which the weighted sums lose to the rounding of
          // c^2 + s^2, and the rounding of a wide gap enters scaled by w.
          const RealScalar left = ds[p], right = ds[i];
          if (numext::abs(s) <= numext::abs(c)) {
            const RealScalar shift = s * s * gap;
            ds[p] = left + shift;
            ds[i] = right - shift;
          } else {
            const RealScalar shift = c * c * gap;
            ds[p] = right - shift;
            ds[i] = left + shift;
          }
          zs[i] = r;
          zs[p] = RealScalar(0);
          deflated[static_cast<std::size_t>(p)] = true;
          // Recorded for both options, keeping computeVectors out of the loops that produce the
          // eigenvalues: GCC unswitches this loop on it, and IBM double-double operations are not
          // commutative, so the two copies' operand orders can round differently.
          rotations.push_back(Rotation{p, i, c, s});
        }
      }
      if (!deflated[static_cast<std::size_t>(i)]) p = i;
    }
  }

  std::vector<Index> sub;  // working positions of the surviving poles
  for (Index i = 0; i < n; ++i)
    if (!deflated[static_cast<std::size_t>(i)]) sub.push_back(i);
  const Index m = static_cast<Index>(sub.size());

  VectorType lambdaW = ds;

  MatrixType subVectors;  // m x m secular eigenvectors (in subproblem coordinates)

  if (m > 0) {
    VectorType delta(m), zeta(m), zeta2(m);
    std::vector<Index> shiftIndex(static_cast<std::size_t>(m));
    VectorType tau(m);
    for (Index a = 0; a < m; ++a) {
      delta[a] = ds[sub[static_cast<std::size_t>(a)]];
      zeta[a] = zs[sub[static_cast<std::size_t>(a)]];
    }
    zeta2.array() = zeta.array() * zeta.array();
    const RealScalar zeta2sum = zeta2.sum();

    // Bisection stops when the bracket has collapsed to adjacent floating-point
    // numbers. The backstop covers exponent_range + 2*digits halvings, enough
    // to shrink a unit-width bracket to a subnormal root offset and resolve it.
    const int digits =
        (std::numeric_limits<RealScalar>::digits > 0) ? static_cast<int>(std::numeric_limits<RealScalar>::digits) : 128;
    const int expRange = (std::numeric_limits<RealScalar>::max_exponent > std::numeric_limits<RealScalar>::min_exponent)
                             ? static_cast<int>(std::numeric_limits<RealScalar>::max_exponent) -
                                   static_cast<int>(std::numeric_limits<RealScalar>::min_exponent)
                             : 16 * digits;
    const int maxBisect = expRange + 2 * digits + 32;
    VectorType lam(m), dsh(m);  // dsh is refilled from scratch each root
    for (Index k = 0; k < m; ++k) {
      // Choose the shift pole and the bracket, entirely in shifted coordinates.
      RealScalar lo, hi;
      Index shift;
      if (k + 1 == m) {
        // Last root: it lies in (delta_m, delta_m + rho*|zeta|^2]; never form
        // the unshifted right end (it can round to the pole itself).
        shift = k;
        lo = RealScalar(0);
        hi = rhoW * zeta2sum;
      } else {
        // Interior root: the secular function is increasing between the poles,
        // so its sign at the midpoint picks the nearer pole as the shift.
        const RealScalar left = delta[k], right = delta[k + 1];
        const RealScalar mid = left + (right - left) / RealScalar(2);
        if (secular(delta, zeta2, rhoW, mid) > RealScalar(0)) {
          shift = k;
          lo = RealScalar(0);
          hi = mid - left;
        } else {
          shift = k + 1;
          lo = mid - right;  // negative
          hi = RealScalar(0);
        }
      }
      const RealScalar shiftVal = delta[shift];
      dsh.array() = delta.array() - shiftVal;

      // Bisection: g(lo) < 0 < g(hi) by the pole signs (for shift = k the
      // function tends to -inf as tau -> 0+, for shift = k+1 to +inf as
      // tau -> 0-). The endpoints are never evaluated.
      RealScalar a0 = lo, b0 = hi;
      bool converged = false;
      for (int iter = 0; iter < maxBisect; ++iter) {
        const RealScalar t = a0 + (b0 - a0) / RealScalar(2);
        if (t == a0 || t == b0) {
          converged = true;  // interval fully resolved
          break;
        }
        if (secular(dsh, zeta2, rhoW, t) > RealScalar(0))
          b0 = t;
        else
          a0 = t;
      }
      if (!converged) m_info = NoConvergence;
      const RealScalar t = a0 + (b0 - a0) / RealScalar(2);
      shiftIndex[static_cast<std::size_t>(k)] = shift;
      tau[k] = t;
      lam[k] = shiftVal + t;
    }

    // ---- Gu-Eisenstat weights: the z-vector for which lam are exact roots ----
    // zhat_i^2 = prod_j (lam_j - delta_i) / (rho * prod_{j != i} (delta_j - delta_i)),
    // with every distance lam_j - delta_i formed as (delta_shift(j) - delta_i) + tau_j.
    // Numerator and denominator factors are paired so the running product stays O(1).
    VectorType zhat(m);
    for (Index i = 0; i < m; ++i) {
      RealScalar acc = ((delta[shiftIndex[static_cast<std::size_t>(i)]] - delta[i]) + tau[i]) / rhoW;
      for (Index j = 0; j < m; ++j) {
        if (j == i) continue;
        const RealScalar num = (delta[shiftIndex[static_cast<std::size_t>(j)]] - delta[i]) + tau[j];
        acc *= num / (delta[j] - delta[i]);
      }
      zhat[i] = numext::abs(acc) > RealScalar(0) ? RealScalar(numext::sqrt(numext::abs(acc))) : RealScalar(0);
      if (zeta[i] < RealScalar(0)) zhat[i] = -zhat[i];
    }

    // ---- secular eigenvectors from the Gu-Eisenstat weights ----
    if (computeVectors) {
      subVectors.resize(m, m);
      for (Index j = 0; j < m; ++j) {
        const Index sj = shiftIndex[static_cast<std::size_t>(j)];
        if (tau[j] == RealScalar(0)) {
          // The root coincides with its shift pole to working precision (a
          // fully underflowed bracket): the eigenvector is that pole's axis.
          subVectors.col(j).setZero();
          subVectors(sj, j) = RealScalar(1);
          continue;
        }
        // Divide by the pole distances delta_i - lam_j, each formed as
        // (delta_i - delta_sj) - tau_j.
        subVectors.col(j).array() = zhat.array() / ((delta.array() - delta[sj]) - tau[j]);
        subVectors.col(j).stableNormalize();
      }
    }
    for (Index k = 0; k < m; ++k) lambdaW[sub[static_cast<std::size_t>(k)]] = lam[k];
  }

  std::vector<Index> order;
  order.reserve(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&lambdaW](Index a, Index b) { return lambdaW[a] < lambdaW[b]; });

  std::vector<Index> subSlot(static_cast<std::size_t>(n), -1);
  for (Index a = 0; a < m; ++a) subSlot[static_cast<std::size_t>(sub[static_cast<std::size_t>(a)])] = a;

  for (Index t = 0; t < n; ++t) {
    const Index w = order[static_cast<std::size_t>(t)];
    const Index outCol = negated ? n - 1 - t : t;
    m_eivalues[outCol] = negated ? -lambdaW[w] : lambdaW[w];
    if (!computeVectors) continue;

    VectorType wvec = VectorType::Zero(n);
    const Index slot = subSlot[static_cast<std::size_t>(w)];
    if (slot < 0) {
      wvec[w] = RealScalar(1);
    } else {
      for (Index a = 0; a < m; ++a) wvec[sub[static_cast<std::size_t>(a)]] = subVectors(a, slot);
    }
    // w <- G*w with G = [[c, s], [-s, c]]: for exactly equal poles this maps the
    // deflated unit vector e_p to (c, -s) = (z_i, -z_p)/r, the exact eigenvector
    // of the 2x2 block orthogonal to the weight vector.
    for (auto it = rotations.rbegin(); it != rotations.rend(); ++it) {
      const RealScalar wi = wvec[it->i], wj = wvec[it->j];
      wvec[it->i] = it->c * wi + it->s * wj;
      wvec[it->j] = -it->s * wi + it->c * wj;
    }
    for (Index i = 0; i < n; ++i) m_eivec(pi[static_cast<std::size_t>(i)], outCol) = wvec[i];
  }

  m_eivalues = m_eivalues.array().ldexp(scaleExp).matrix();
  if (!m_eivalues.allFinite()) {
    m_eivalues.setConstant(NumTraits<RealScalar>::quiet_NaN());
    m_info = InvalidInput;
  }

  m_vectorsComputed = computeVectors && m_info != InvalidInput;
  m_isInitialized = true;
  return *this;
}

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_DPR1_EIGEN_SOLVER_H
