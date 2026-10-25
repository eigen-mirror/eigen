// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_GENERALIZED_SCHUR_REORDERING_H
#define EIGEN_GENERALIZED_SCHUR_REORDERING_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal
 * Size of the diagonal block of the generalized (quasi-)triangular matrix \p S starting at row/column \p k.
 * Complex Schur forms only have 1x1 blocks; real ones also have 2x2 blocks for complex conjugate eigenvalue pairs,
 * which are recognized by a nonzero subdiagonal entry of \p S (block boundaries are exact zeros).
 */
template <typename MatrixType>
Index generalized_schur_block_size(const MatrixType& S, Index k) {
  if (NumTraits<typename MatrixType::Scalar>::IsComplex) return 1;
  return (k + 1 < S.rows() && !numext::is_exactly_zero(S.coeff(k + 1, k))) ? 2 : 1;
}

/** \internal
 * Swaps the adjacent 1x1 diagonal blocks at row/column \p j of the generalized Schur form (\p S, \p T).
 *
 * As in LAPACK's xTGEX2, the rotations are formed directly from the entries rather than from a Sylvester solve, which
 * preserves the relative accuracy of small components: with a = S(j:j+1, j:j+1) and b = T(j:j+1, j:j+1), the right
 * eigenvector of the lower eigenvalue a22 / b22 is v = [g; -f] with f = a22 b11 - b22 a11 and g = a22 b12 - b22 a12,
 * and the first column of the left rotation is a v or b v, whichever is computed from the larger diagonal products.
 */
template <typename MatrixType>
bool swap_generalized_schur_1x1(MatrixType& S, MatrixType& T, MatrixType* Q, MatrixType& Z, Index j) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Matrix2 = Matrix<Scalar, 2, 2>;
  const Matrix2 a = S.template block<2, 2>(j, j);
  const Matrix2 b = T.template block<2, 2>(j, j);
  const Scalar f = a(1, 1) * b(0, 0) - b(1, 1) * a(0, 0);
  const Scalar g = a(1, 1) * b(0, 1) - b(1, 1) * a(0, 1);
  // f = g = 0 only for equal eigenvalues, which never need swapping, or a singular pencil.
  if (numext::is_exactly_zero(f) && numext::is_exactly_zero(g)) return false;
  JacobiRotation<Scalar> gz, gq;
  gz.makeGivens(g, -f);
  Matrix2 an = a, bn = b;
  an.applyOnTheRight(0, 1, gz);
  bn.applyOnTheRight(0, 1, gz);
  if (numext::abs(a(1, 1)) * numext::abs(b(0, 0)) >= numext::abs(a(0, 0)) * numext::abs(b(1, 1)))
    gq.makeGivens(an(0, 0), an(1, 0));
  else
    gq.makeGivens(bn(0, 0), bn(1, 0));
  an.applyOnTheLeft(0, 1, gq.adjoint());
  bn.applyOnTheLeft(0, 1, gq.adjoint());

  // Weak and strong stability tests of xTGEX2, with tau_a = max(20 eps ||a||_F, smlnum) and likewise tau_b.
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar smlnum = (std::numeric_limits<RealScalar>::min)() / eps;
  const RealScalar threshA = numext::maxi(RealScalar(20) * eps * a.stableNorm(), smlnum);
  const RealScalar threshB = numext::maxi(RealScalar(20) * eps * b.stableNorm(), smlnum);
  if (!(numext::abs(an(1, 0)) <= threshA && numext::abs(bn(1, 0)) <= threshB)) return false;
  an(1, 0) = Scalar(0);
  bn(1, 0) = Scalar(0);
  Matrix2 ra = an, rb = bn;
  ra.applyOnTheLeft(0, 1, gq);
  rb.applyOnTheLeft(0, 1, gq);
  ra.applyOnTheRight(0, 1, gz.adjoint());
  rb.applyOnTheRight(0, 1, gz.adjoint());
  if (!((ra - a).stableNorm() <= threshA && (rb - b).stableNorm() <= threshB)) return false;

  S.template block<2, 2>(j, j) = an;
  T.template block<2, 2>(j, j) = bn;
  const Index right = S.cols() - j - 2;
  if (right > 0) {
    S.rightCols(right).applyOnTheLeft(j, j + 1, gq.adjoint());
    T.rightCols(right).applyOnTheLeft(j, j + 1, gq.adjoint());
  }
  if (j > 0) {
    S.topRows(j).applyOnTheRight(j, j + 1, gz);
    T.topRows(j).applyOnTheRight(j, j + 1, gz);
  }
  if (Q != nullptr) Q->applyOnTheRight(j, j + 1, gq);
  Z.applyOnTheRight(j, j + 1, gz);
  return true;
}

/** \internal
 * Swaps the adjacent diagonal blocks of sizes \p n1 (upper) and \p n2 (lower), one of them 2x2, at row/column \p j of
 * the real generalized Schur form (\p S, \p T), following B. Kagstrom and P. Poromaa, "Computing eigenspaces with
 * specified eigenvalues of a regular matrix pair (A, B) and condition estimation: theory, algorithms and software",
 * Numer. Algorithms 12 (1996), and LAPACK's xTGEX2.
 *
 * The solution of the generalized Sylvester equation
 *   A11 X - Y A22 = -A12,  B11 X - Y B22 = -B12
 * gives the deflating subspaces of the lower block, spanned by [X; I] (right) and [Y; I] (left); the leading columns
 * of Zl and Ql are orthonormal bases of these subspaces. The swapped B is then made upper triangular by a QR (from the
 * left) or an RQ (from the right) factorization, whichever leaves the smaller lower left block of A.
 */
template <typename MatrixType>
bool swap_generalized_schur_sylvester(MatrixType& S, MatrixType& T, MatrixType* Q, MatrixType& Z, Index j, Index n1,
                                      Index n2) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using LocalMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor, 4, 4>;
  using SylvesterMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor, 8, 8>;
  using SylvesterVector = Matrix<Scalar, Dynamic, 1, ColMajor, 8, 1>;
  const Index m = n1 + n2;
  const Index k = n1 * n2;
  const LocalMatrix A = S.block(j, j, m, m);
  const LocalMatrix B = T.block(j, j, m, m);

  // K [vec(X); vec(Y)] = -[vec(A12); vec(B12)] with
  // K = [I (x) A11, -A22^T (x) I; I (x) B11, -B22^T (x) I].
  SylvesterMatrix K = SylvesterMatrix::Zero(2 * k, 2 * k);
  for (Index c = 0; c < n2; ++c) {
    K.block(c * n1, c * n1, n1, n1) = A.topLeftCorner(n1, n1);
    K.block(k + c * n1, c * n1, n1, n1) = B.topLeftCorner(n1, n1);
    for (Index i = 0; i < n2; ++i) {
      K.block(c * n1, k + i * n1, n1, n1).diagonal().setConstant(-A(n1 + i, n1 + c));
      K.block(k + c * n1, k + i * n1, n1, n1).diagonal().setConstant(-B(n1 + i, n1 + c));
    }
  }
  SylvesterVector rhs(2 * k);
  rhs << -A.topRightCorner(n1, n2).reshaped(), -B.topRightCorner(n1, n2).reshaped();
  const FullPivLU<SylvesterMatrix> sylvester(K);
  if (!sylvester.isInvertible()) return false;
  const SylvesterVector sol = sylvester.solve(rhs);

  LocalMatrix basis(m, n2);
  basis.topRows(n1) = sol.head(k).reshaped(n1, n2);
  basis.bottomRows(n2).setIdentity();
  const LocalMatrix Zl = HouseholderQR<LocalMatrix>(basis).householderQ();
  basis.topRows(n1) = sol.tail(k).reshaped(n1, n2);
  const LocalMatrix Ql = HouseholderQR<LocalMatrix>(basis).householderQ();
  const LocalMatrix An = Ql.adjoint() * A * Zl;
  const LocalMatrix Bn = Ql.adjoint() * B * Zl;

  // QR: Bn = H R, so H^* (An, Bn) with Ql <- Ql H.
  const LocalMatrix H = HouseholderQR<LocalMatrix>(Bn).householderQ();
  LocalMatrix Aq = H.adjoint() * An;
  // RQ: Bn V is upper triangular for V = W P, where P reverses the order and M^* P = W R' is a QR factorization of
  // the columns of M^* in reverse order; then (An, Bn) V with Zl <- Zl V.
  const LocalMatrix W = HouseholderQR<LocalMatrix>(Bn.adjoint().rowwise().reverse()).householderQ();
  const LocalMatrix V = W.rowwise().reverse();
  LocalMatrix Ar = An * V;

  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar smlnum = (std::numeric_limits<RealScalar>::min)() / eps;
  const RealScalar threshA = numext::maxi(RealScalar(20) * eps * A.stableNorm(), smlnum);
  const RealScalar threshB = numext::maxi(RealScalar(20) * eps * B.stableNorm(), smlnum);
  const RealScalar qrA21 = Aq.bottomLeftCorner(n1, n2).stableNorm();
  const RealScalar rqA21 = Ar.bottomLeftCorner(n1, n2).stableNorm();

  // Weak stability test: ||A21||_F <= tau_a after the swap (B21 vanishes by construction).
  LocalMatrix Qn, Zn, Sn, Tn;
  if (qrA21 <= rqA21 && qrA21 <= threshA) {
    Qn.noalias() = Ql * H;
    Zn = Zl;
    Sn = Aq;
    Tn.noalias() = H.adjoint() * Bn;
  } else if (rqA21 <= threshA) {
    Qn = Ql;
    Zn.noalias() = Zl * V;
    Sn = Ar;
    Tn.noalias() = Bn * V;
  } else {
    return false;
  }
  Sn.bottomLeftCorner(n1, n2).setZero();
  Tn.template triangularView<StrictlyLower>().setZero();

  // Strong stability test: ||Qn Sn Zn^* - A||_F <= tau_a and ||Qn Tn Zn^* - B||_F <= tau_b.
  if (!((Qn * Sn * Zn.adjoint() - A).stableNorm() <= threshA && (Qn * Tn * Zn.adjoint() - B).stableNorm() <= threshB))
    return false;

  S.block(j, j, m, m) = Sn;
  T.block(j, j, m, m) = Tn;
  const Index right = S.cols() - j - m;
  if (right > 0) {
    S.block(j, j + m, m, right) = Qn.adjoint() * S.block(j, j + m, m, right);
    T.block(j, j + m, m, right) = Qn.adjoint() * T.block(j, j + m, m, right);
  }
  if (j > 0) {
    S.block(0, j, j, m) = S.block(0, j, j, m) * Zn;
    T.block(0, j, j, m) = T.block(0, j, j, m) * Zn;
  }
  if (Q != nullptr) Q->middleCols(j, m) = Q->middleCols(j, m) * Qn;
  Z.middleCols(j, m) = Z.middleCols(j, m) * Zn;
  return true;
}

/** \internal
 * Swaps the adjacent diagonal blocks of sizes \p n1 (upper) and \p n2 (lower) starting at row/column \p j of the
 * generalized Schur form (\p S, \p T), where \p S is upper (quasi-)triangular and \p T is upper triangular.
 *
 * \p Q, unless null, and \p Z are updated such that A = Q S Z^* and B = Q T Z^* keep holding (as in LAPACK, \p Z is the
 * adjoint of the matrix returned by RealQZ::matrixZ()).
 *
 * \returns false, leaving all matrices untouched, if the swap fails the weak or strong stability test of Kagstrom and
 * Poromaa (1996), typically because the eigenvalues of both blocks are too close.
 */
template <typename MatrixType>
bool swap_generalized_schur_blocks(MatrixType& S, MatrixType& T, MatrixType* Q, MatrixType& Z, Index j, Index n1,
                                   Index n2) {
  eigen_assert(n1 >= 1 && n1 <= 2 && n2 >= 1 && n2 <= 2 && j >= 0 && j + n1 + n2 <= S.rows());
  if (n1 == 1 && n2 == 1) return swap_generalized_schur_1x1(S, T, Q, Z, j);
  return swap_generalized_schur_sylvester(S, T, Q, Z, j, n1, n2);
}

/** \internal
 * Sets the entries below the structure of the generalized Schur form (\p S, \p T) to zero: below the diagonal of \p T,
 * and below the diagonal (complex) or the first subdiagonal (real) of \p S. These are rounding errors left by the QZ
 * iteration.
 */
template <typename MatrixType>
void clean_generalized_schur_form(MatrixType& S, MatrixType& T) {
  const Index n = S.rows();
  T.template triangularView<StrictlyLower>().setZero();
  if (NumTraits<typename MatrixType::Scalar>::IsComplex)
    S.template triangularView<StrictlyLower>().setZero();
  else if (n > 2)
    S.bottomLeftCorner(n - 2, n - 2).template triangularView<Lower>().setZero();
}

/** \internal
 * Reorders the generalized Schur form (\p S, \p T) with transformations \p Q (optional) and \p Z (see
 * swap_generalized_schur_blocks()) such that all eigenvalues strictly inside the unit circle lead the diagonal, keeping
 * their relative order. This corresponds to SciPy's `ordqz(..., sort='iuc')`; infinite eigenvalues are outside the unit
 * circle.
 *
 * \returns the number of eigenvalues inside the unit circle, or -1 if a swap was rejected.
 */
template <typename MatrixType>
Index reorder_generalized_schur_inside_unit_circle(MatrixType& S, MatrixType& T, MatrixType* Q, MatrixType& Z) {
  using Scalar = typename MatrixType::Scalar;
  const Index n = S.rows();
  clean_generalized_schur_form(S, T);

  Index selected = 0;
  for (Index k = 0; k < n;) {
    const Index bs = generalized_schur_block_size(S, k);
    bool inside;
    if (bs == 1) {
      inside = numext::abs(S(k, k)) < numext::abs(T(k, k));
    } else {
      // Complex conjugate pair of equal modulus: |lambda|^2 = det(S_kk) / det(T_kk), with T_kk triangular.
      inside = numext::abs(S.template block<2, 2>(k, k).determinant()) < numext::abs(T(k, k) * T(k + 1, k + 1));
    }
    if (inside) {
      // Bubble the block up past the unselected ones.
      for (Index pos = k; pos > selected;) {
        const Index pbs =
            (!NumTraits<Scalar>::IsComplex && pos - 2 >= selected && !numext::is_exactly_zero(S(pos - 1, pos - 2))) ? 2
                                                                                                                    : 1;
        if (!swap_generalized_schur_blocks(S, T, Q, Z, pos - pbs, pbs, bs)) return -1;
        pos -= pbs;
      }
      selected += bs;
    }
    k += bs;
  }
  return selected;
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_GENERALIZED_SCHUR_REORDERING_H
