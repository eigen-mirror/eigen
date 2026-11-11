// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_QZ_TEST_HELPERS_H
#define EIGEN_TEST_QZ_TEST_HELPERS_H

// Shared by the RealQZ, ComplexQZ and GeneralizedEigenSolver tests. Include after main.h.

#include <complex>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include <Eigen/Eigenvalues>

namespace Eigen {

// T is upper triangular and S quasi-triangular, with T diagonal on each 2x2 diagonal block of S.
template <typename MatrixType, typename QZType>
void verify_qz_structure(const MatrixType& a, const MatrixType&, const QZType& qz,
                         const typename MatrixType::RealScalar&, std::false_type /* is complex */) {
  using Scalar = typename MatrixType::Scalar;
  VERIFY(qz.matrixT().template triangularView<StrictlyLower>().toDenseMatrix().isZero(0));
  for (Index j = 0; j + 2 < a.cols(); ++j) VERIFY(qz.matrixS().col(j).tail(a.rows() - j - 2).isZero(0));
  for (Index i = 0; i + 1 < a.rows(); ++i) {
    if (qz.matrixS()(i + 1, i) == Scalar(0)) continue;
    VERIFY_IS_EQUAL(qz.matrixT()(i, i + 1), Scalar(0));
    if (i + 2 < a.rows()) VERIFY_IS_EQUAL(qz.matrixS()(i + 2, i + 1), Scalar(0));
  }
}

// S and T are upper triangular up to the tolerance.
template <typename MatrixType, typename QZType>
void verify_qz_structure(const MatrixType& a, const MatrixType& b, const QZType& qz,
                         const typename MatrixType::RealScalar& tolerance, std::true_type /* is complex */) {
  const MatrixType lowerS = qz.matrixS().template triangularView<StrictlyLower>();
  const MatrixType lowerT = qz.matrixT().template triangularView<StrictlyLower>();
  VERIFY(lowerS.stableNorm() <= tolerance * a.stableNorm());
  VERIFY(lowerT.stableNorm() <= tolerance * b.stableNorm());
}

// Checks that qz is a successful decomposition of the pencil (a, b): ||a - Q S Z^*|| and ||b - Q T Z^*|| within the
// tolerance, Q and Z unitary, and the triangular form of S and T that RealQZ or ComplexQZ promises. stableNorm, since
// the squared entries of the badly scaled pencils underflow or overflow.
template <typename MatrixType, typename QZType>
void verify_qz_decomposition(const MatrixType& a, const MatrixType& b, const QZType& qz,
                             const typename MatrixType::RealScalar& tolerance) {
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY((a - qz.matrixQ() * qz.matrixS() * qz.matrixZ()).stableNorm() <= tolerance * a.stableNorm());
  VERIFY((b - qz.matrixQ() * qz.matrixT() * qz.matrixZ()).stableNorm() <= tolerance * b.stableNorm());
  VERIFY(qz.matrixQ().isUnitary(tolerance));
  VERIFY(qz.matrixZ().isUnitary(tolerance));
  verify_qz_structure(a, b, qz, tolerance,
                      internal::bool_constant<NumTraits<typename MatrixType::Scalar>::IsComplex>());
}

// The decomposition of (a, b) with the limit on the iterations between deflations. The first argument only selects
// the QZ class.
template <typename MatrixType>
RealQZ<MatrixType> qz_with_iteration_limit(const RealQZ<MatrixType>*, const MatrixType& a, const MatrixType& b,
                                           Index limit) {
  RealQZ<MatrixType> qz(a.rows());
  qz.setMaxIterations(limit);
  qz.compute(a, b);
  return qz;
}
template <typename MatrixType>
ComplexQZ<MatrixType> qz_with_iteration_limit(const ComplexQZ<MatrixType>*, const MatrixType& a, const MatrixType& b,
                                              Index limit) {
  return ComplexQZ<MatrixType>(a, b, true, static_cast<unsigned int>(limit));
}

// Entries of magnitude in [1/3, 1], so that no scaled entry becomes subnormal.
template <typename RealScalar>
RealScalar qz_away_from_zero(const RealScalar& x) {
  return (x < 0 ? x - RealScalar(0.5) : x + RealScalar(0.5)) / RealScalar(1.5);
}
template <typename RealScalar>
std::complex<RealScalar> qz_away_from_zero(const std::complex<RealScalar>& x) {
  return std::complex<RealScalar>(qz_away_from_zero(x.real()), qz_away_from_zero(x.imag()));
}

// A and B are scaled by powers of two near the ends of the exponent range, which made the iteration overflow or
// underflow (#3212). The scaling must leave Q and Z, and S and T up to the same powers of two, unchanged.
template <typename MatrixType, typename QZType>
void qz_extreme_scaling() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const Index dim = 6;
  const auto awayFromZero = [](const Scalar& x) { return qz_away_from_zero(x); };
  const MatrixType a0 = MatrixType::Random(dim, dim).unaryExpr(awayFromZero);
  const MatrixType b0 = MatrixType::Random(dim, dim).unaryExpr(awayFromZero);
  const int big = std::numeric_limits<RealScalar>::max_exponent - 16;
  const RealScalar tolerance = RealScalar(128 * dim) * NumTraits<RealScalar>::epsilon();
  for (int ea : {0, big, -big}) {
    for (int eb : {0, big, -big}) {
      const MatrixType A = a0 * numext::ldexp(RealScalar(1), ea), B = b0 * numext::ldexp(RealScalar(1), eb);
      QZType qz(A, B);
      VERIFY_IS_EQUAL(qz.info(), Success);
      const MatrixType S = qz.matrixS() * numext::ldexp(RealScalar(1), -ea);
      const MatrixType T = qz.matrixT() * numext::ldexp(RealScalar(1), -eb);
      VERIFY(S.allFinite() && T.allFinite());
      VERIFY((qz.matrixQ() * S * qz.matrixZ() - a0).norm() <= tolerance * a0.norm());
      VERIFY((qz.matrixQ() * T * qz.matrixZ() - b0).norm() <= tolerance * b0.norm());
    }
  }
}

// For A or B = c 1 with c = 0.9 highest, Q^* A Z (or Q^* B Z) is rank one with Frobenius norm c n, so its nonzeros form
// a p x q block with p + q <= n + 1 and one of them has modulus at least 2 c n / (n + 1) > highest: QZ must report
// NumericalIssue, not Success with inf in S or T. For complex scalars that bound applies to the larger of the real and
// imaginary parts only for n >= 4, so sizes lists n. For A = c Random, whose coefficient moduli overflow, QZ must
// neither report NoConvergence nor Success with a non-finite S or T.
template <typename MatrixType, typename QZType>
void qz_unrepresentable(std::initializer_list<Index> sizes) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const Scalar c(RealScalar(0.9) * NumTraits<RealScalar>::highest());
  for (Index n : sizes) {
    const MatrixType ones = MatrixType::Constant(n, n, c), random = MatrixType::Random(n, n);
    QZType qzA(ones, random), qzB(random, ones);
    VERIFY_IS_EQUAL(qzA.info(), NumericalIssue);
    VERIFY_IS_EQUAL(qzB.info(), NumericalIssue);
    const MatrixType big = random * c;
    QZType qzBig(big, random);
    VERIFY(qzBig.info() != NoConvergence);
    VERIFY(qzBig.info() != Success || (qzBig.matrixS().allFinite() && qzBig.matrixT().allFinite()));
  }
}

// A pencil that never deflates must still return: every pass through the iteration either deflates or counts towards
// the limit, chases of a zero on the diagonal of T included. With B = 0 every pass takes that branch.
template <typename MatrixType, typename QZType>
void qz_nan_pencil_terminates() {
  using RealScalar = typename MatrixType::RealScalar;
  for (Index n : {3, 5}) {
    MatrixType a = MatrixType::Random(n, n);
    a(n - 1, 0) = NumTraits<RealScalar>::quiet_NaN();
    const MatrixType b = MatrixType::Zero(n, n);
    QZType qz(a, b);
    VERIFY(qz.info() != Success);
  }
}

// An exact zero on the subdiagonal of S did not deflate when |S(k-1, k-1)| + |S(k, k)| and ||S|| were both 0. With
// A = 0 and n >= 3 the iteration made no progress: it did not return for B = 0, and reported NoConvergence for a
// nonzero B. Both pencils are already in generalized Schur form.
template <typename MatrixType, typename QZType>
void qz_zero_pencil() {
  using RealScalar = typename MatrixType::RealScalar;
  for (Index dim : {1, 2, 3, 5}) {
    const MatrixType zero = MatrixType::Zero(dim, dim), random = MatrixType::Random(dim, dim);
    const RealScalar tolerance = RealScalar(128 * dim) * NumTraits<RealScalar>::epsilon();
    for (const MatrixType* b : {&zero, &random}) {
      QZType qz(zero, *b);
      verify_qz_decomposition(zero, *b, qz, tolerance);
    }
  }
}

// setMaxIterations() bounds the steps between deflations. A 1x1 or 2x2 pencil needs no step, so a limit of 0 reduces
// it. A 3x3 pencil with T(0, 0) = 0 needs one chase of that zero, which deflates the infinite eigenvalue, and no
// QR-like step: a limit of 1 reduces it. The limit used to be checked before that deflation was found.
template <typename MatrixType, typename QZType>
void qz_iteration_limit() {
  using RealScalar = typename MatrixType::RealScalar;
  for (Index n : {1, 2}) {
    const MatrixType a = MatrixType::Random(n, n), b = MatrixType::Random(n, n);
    const QZType qz = qz_with_iteration_limit(static_cast<QZType*>(nullptr), a, b, 0);
    verify_qz_decomposition(a, b, qz, RealScalar(128 * n) * NumTraits<RealScalar>::epsilon());
  }
  // Already Hessenberg-triangular, so the reduction keeps T(0, 0) exactly 0.
  MatrixType a = MatrixType::Random(3, 3), b = MatrixType::Random(3, 3).template triangularView<Upper>();
  a(2, 0) = 0;
  b(0, 0) = 0;
  const QZType qz = qz_with_iteration_limit(static_cast<QZType*>(nullptr), a, b, 1);
  verify_qz_decomposition(a, b, qz, RealScalar(128 * 3) * NumTraits<RealScalar>::epsilon());
  VERIFY(qz.iterations() == 0);
}

#ifdef EIGEN_RUNTIME_NO_MALLOC
// Block Householder operations still allocate at larger sizes (issue #3163): compute() of a QZ object of the right
// size does not allocate, with or without Q and Z, and reproduces S and T. A 24 x 24 pencil is above the threshold
// of the block Householder operations.
template <typename MatrixType, typename QZType>
void qz_no_malloc() {
  const Index dim = 24;
  const MatrixType A = MatrixType::Random(dim, dim), B = MatrixType::Random(dim, dim);
  QZType qz(dim);
  internal::set_is_malloc_allowed(false);
  qz.compute(A, B, false);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(qz.info(), Success);
  const MatrixType S = qz.matrixS(), T = qz.matrixT();

  for (int i = 0; i < 2; ++i) {
    internal::set_is_malloc_allowed(false);
    qz.compute(A, B, true);
    internal::set_is_malloc_allowed(true);
    VERIFY_IS_EQUAL(qz.info(), Success);
    VERIFY_IS_APPROX(qz.matrixS(), S);
    VERIFY_IS_APPROX(qz.matrixT(), T);
    VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixS() * qz.matrixZ(), A);
    VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixT() * qz.matrixZ(), B);
    VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixQ().adjoint(), MatrixType::Identity(dim, dim));
    VERIFY_IS_APPROX(qz.matrixZ() * qz.matrixZ().adjoint(), MatrixType::Identity(dim, dim));
  }
}
#endif  // EIGEN_RUNTIME_NO_MALLOC

// blockdiag(R, ..., R) of order n for the rotation R = omega [0 -1; 1 0], whose eigenvalues are +-i omega.
template <typename MatrixType>
MatrixType qz_rotation_blocks(Index n, const typename MatrixType::Scalar& omega) {
  MatrixType r = MatrixType::Zero(n, n);
  for (Index k = 0; k + 1 < n; k += 2) {
    r(k, k + 1) = -omega;
    r(k + 1, k) = omega;
  }
  return r;
}

// eta = |(beta A - alpha B) v| / ((|beta| |A| + |alpha| |B|) |v|), the backward error of the eigenpair (alpha / beta,
// v) of the pencil (A, B): QZ and the back substitution for the eigenvectors are backward stable, so eta = O(n eps).
// alpha and beta may be scaled by any common factor, so callers can keep their products from overflowing.
template <typename MatrixType>
typename MatrixType::RealScalar generalized_eigenpair_backward_error(
    const MatrixType& a, const MatrixType& b, const std::complex<typename MatrixType::RealScalar>& alpha,
    const typename MatrixType::RealScalar& beta,
    const Matrix<std::complex<typename MatrixType::RealScalar>, Dynamic, 1>& v) {
  using Complex = std::complex<typename MatrixType::RealScalar>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;
  const ComplexVector residual = (a.template cast<Complex>() * v) * beta - (b.template cast<Complex>() * v) * alpha;
  return residual.stableNorm() /
         ((numext::abs(beta) * a.stableNorm() + numext::abs(alpha) * b.stableNorm()) * v.stableNorm());
}

}  // namespace Eigen

#endif  // EIGEN_TEST_QZ_TEST_HELPERS_H
