// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Alexey Korepanov <kaikaikai@yandex.ru>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>
#include "qz_test_helpers.h"

using RowMajorXf = Matrix<float, Dynamic, Dynamic, RowMajor>;
using RowMajorXd = Matrix<double, Dynamic, Dynamic, RowMajor>;

template <typename MatrixType>
void real_qz(const MatrixType& m) {
  /* this test covers the following files:
     RealQZ.h
  */
  using std::abs;

  Index dim = m.cols();

  MatrixType A = MatrixType::Random(dim, dim), B = MatrixType::Random(dim, dim);

  // Regression test for bug 985: Randomly set rows or columns to zero
  Index k = internal::random<Index>(0, dim - 1);
  switch (internal::random<int>(0, 10)) {
    case 0:
      A.row(k).setZero();
      break;
    case 1:
      A.col(k).setZero();
      break;
    case 2:
      B.row(k).setZero();
      break;
    case 3:
      B.col(k).setZero();
      break;
    default:
      break;
  }

  RealQZ<MatrixType> qz(dim);
  qz.compute(A, B);

  VERIFY_IS_EQUAL(qz.info(), Success);
  // check for zeros
  bool all_zeros = true;
  for (Index i = 0; i < A.cols(); i++)
    for (Index j = 0; j < i; j++) {
      if (!numext::is_exactly_zero(abs(qz.matrixT()(i, j)))) {
        std::cerr << "Error: T(" << i << "," << j << ") = " << qz.matrixT()(i, j) << std::endl;
        all_zeros = false;
      }
      if (j < i - 1 && !numext::is_exactly_zero(abs(qz.matrixS()(i, j)))) {
        std::cerr << "Error: S(" << i << "," << j << ") = " << qz.matrixS()(i, j) << std::endl;
        all_zeros = false;
      }
      if (j == i - 1 && j > 0 && !numext::is_exactly_zero(abs(qz.matrixS()(i, j))) &&
          !numext::is_exactly_zero(abs(qz.matrixS()(i - 1, j - 1)))) {
        std::cerr << "Error: S(" << i << "," << j << ") = " << qz.matrixS()(i, j) << " && S(" << i - 1 << "," << j - 1
                  << ") = " << qz.matrixS()(i - 1, j - 1) << std::endl;
        all_zeros = false;
      }
    }
  VERIFY_IS_EQUAL(all_zeros, true);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixS() * qz.matrixZ(), A);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixT() * qz.matrixZ(), B);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixQ().adjoint(), MatrixType::Identity(dim, dim));
  VERIFY_IS_APPROX(qz.matrixZ() * qz.matrixZ().adjoint(), MatrixType::Identity(dim, dim));
}

// On a 3x3 all-zero pencil an exact zero on the subdiagonal of S did not deflate, which also failed
// GeneralizedEigenSolver's eigenvalues. The decomposition of the same pencils is checked by qz_zero_pencil.
template <typename MatrixType>
void real_qz_zero_pencil_eigenvalues() {
  for (Index dim : {1, 2, 3, 5}) {
    GeneralizedEigenSolver<MatrixType> eig(MatrixType::Zero(dim, dim), MatrixType::Random(dim, dim), false);
    VERIFY_IS_EQUAL(eig.info(), Success);
    VERIFY((eig.alphas().array() == typename MatrixType::Scalar(0)).all());
  }
}

// RealQZ split a 2x2 block with real eigenvalues through S T^-1: with a diagonal entry delta of T in the block, the
// backward error grew like eps / delta, to 1e-3 relative for double at delta = 1e-12. The 3x3 pencils first move
// T(0, 0) = 0 to the bottom. A nearly scalar block a = lambda0 b + O(g) has a close pair of eigenvalues, which a
// discriminant formed without a shift resolves only to O(eps / g): the split must keep both its accuracy and its
// classification as real or complex. The backward error must stay O(n eps).
template <typename MatrixType>
void real_qz_ill_conditioned_block() {
  using Scalar = typename MatrixType::Scalar;
  using Complex = std::complex<Scalar>;
  const Scalar eps = NumTraits<Scalar>::epsilon();
  for (const Scalar delta : {numext::sqrt(eps), Scalar(1024) * eps}) {
    for (int kind = 0; kind < 3; ++kind) {
      const Index n = kind < 2 ? 2 : 3;
      for (int repeat = 0; repeat < 20; ++repeat) {
        MatrixType a = MatrixType::Random(n, n), b = MatrixType::Random(n, n).template triangularView<Upper>();
        if (kind == 0) b(1, 1) = delta;
        if (kind == 1) b(0, 0) = delta;
        if (kind == 2) {
          a(2, 0) = Scalar(0);
          b(0, 0) = Scalar(0);
          b(2, 2) = delta;
        }
        // Accumulated rounding in the orthogonal transformations is O(n eps).
        verify_qz_decomposition(a, b, RealQZ<MatrixType>(a, b), Scalar(128 * n) * eps);
      }
    }
  }

  // b = I: a = [1 t; s t 1] is already Hessenberg-triangular, with the eigenvalues 1 +- t (s = 1, split) or 1 +- i t
  // (s = -1). The pencil is normal, so the eigenvalue error is bounded by the backward error: a few eps.
  for (const Scalar t : {Scalar(1e-2), numext::sqrt(eps), Scalar(16) * eps}) {
    for (const int s : {1, -1}) {
      MatrixType a(2, 2);
      a << Scalar(1), t, Scalar(s) * t, Scalar(1);
      const MatrixType b = MatrixType::Identity(2, 2);
      const RealQZ<MatrixType> qz(a, b);
      verify_qz_decomposition(a, b, qz, Scalar(128 * 2) * eps);
      VERIFY_IS_EQUAL(numext::is_exactly_zero(qz.matrixS()(1, 0)), s > 0);
      const Complex shift = s > 0 ? Complex(t, 0) : Complex(0, t);
      const Complex plus = Scalar(1) + shift, minus = Scalar(1) - shift;
      const GeneralizedEigenSolver<MatrixType> solver(a, b, false);
      const auto lambda = solver.eigenvalues().eval();
      const Scalar error = numext::mini(numext::maxi(numext::abs(lambda(0) - plus), numext::abs(lambda(1) - minus)),
                                        numext::maxi(numext::abs(lambda(0) - minus), numext::abs(lambda(1) - plus)));
      VERIFY(error <= Scalar(16) * eps);
    }
  }

  // a = lambda0 b + g E for a random E and a triangular b whose diagonal entries have random signs and moduli in
  // delta [1, 2]: the largest entry of b is on its diagonal for delta = 1, often off it for delta = 1/4, and off it for
  // delta = sqrt(eps).
  for (const Scalar delta : {Scalar(1), Scalar(0.25), numext::sqrt(eps)}) {
    for (const Scalar g : {numext::sqrt(eps), Scalar(16) * eps}) {
      for (int repeat = 0; repeat < 20; ++repeat) {
        MatrixType b = MatrixType::Random(2, 2).template triangularView<Upper>();
        for (Index k = 0; k < 2; ++k)
          b(k, k) = delta * (internal::random<bool>() ? Scalar(1) : Scalar(-1)) *
                    internal::random<Scalar>(Scalar(1), Scalar(2));
        const Scalar lambda0 = internal::random<Scalar>(Scalar(0.5), Scalar(1.5));
        const MatrixType a = lambda0 * b + g * MatrixType::Random(2, 2);
        verify_qz_decomposition(a, b, RealQZ<MatrixType>(a, b), Scalar(128 * 2) * eps);
      }
    }
  }

  // a = [1 0; 1/2 1], b = I: a double eigenvalue 1, which the split leaves exact.
  MatrixType a(2, 2);
  a << Scalar(1), Scalar(0), Scalar(0.5), Scalar(1);
  const RealQZ<MatrixType> qz(a, MatrixType::Identity(2, 2));
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY_IS_EQUAL(qz.matrixS()(1, 0), Scalar(0));
  VERIFY_IS_EQUAL(qz.matrixS()(0, 0), qz.matrixT()(0, 0));
  VERIFY_IS_EQUAL(qz.matrixS()(1, 1), qz.matrixT()(1, 1));
}

// The split of a block with real eigenvalues puts the eigenvalue near (S T^-1)(0, 0) on top: here the eigenvalues are
// about top = a(0, 0) / b(0, 0) and 1, for every sign of the diagonal of b and of b(0, 1), with the largest entry of b
// on or off its diagonal.
template <typename MatrixType>
void real_qz_split_order() {
  using Scalar = typename MatrixType::Scalar;
  for (int k = 0; k < 16; ++k) {
    const Scalar s0 = (k & 1) ? Scalar(-1) : Scalar(1), s1 = (k & 2) ? Scalar(-1) : Scalar(1);
    const Scalar b01 = ((k & 4) ? Scalar(-1) : Scalar(1)) * ((k & 8) ? Scalar(1) : Scalar(0.25));
    for (const Scalar top : {Scalar(2), Scalar(-2)}) {
      MatrixType a(2, 2), b(2, 2);
      b << s0 * Scalar(0.5), b01, Scalar(0), s1 * Scalar(0.5);
      a << top * b(0, 0), Scalar(0.3), Scalar(1e-3), b(1, 1);
      const RealQZ<MatrixType> qz(a, b);
      VERIFY_IS_EQUAL(qz.info(), Success);
      VERIFY_IS_EQUAL(qz.matrixS()(1, 0), Scalar(0));
      const Scalar lambda = qz.matrixS()(0, 0) / qz.matrixT()(0, 0);
      VERIFY(numext::abs(lambda - top) < numext::abs(lambda - Scalar(1)));
    }
  }
}

// Under random orthogonal equivalence, repeated +-i pairs and real Jordan blocks did not converge in double: the
// subdiagonal entry between two 2x2 blocks of a +-i pair became negligible but did not deflate against diagonal entries
// that are rounding noise, and a defective eigenvalue, which converges linearly, was restarted by a random shift every
// 8 steps after the 24th. The backward error must be O(n eps), and the pencils (P R Z, P Z) equivalent to a normal
// (R, I) must have their eigenvalues +- omega i to the accuracy which that implies.
template <typename MatrixType>
void real_qz_repeated_eigenvalues() {
  using Scalar = typename MatrixType::Scalar;
  // omega = 0 (R not normal) skips the eigenvalue check.
  const auto verify = [](const MatrixType& a, const MatrixType& b, const Scalar& omega) {
    const Index n = a.rows();
    // QZ applies orthogonal transformations in O(n) stages for the reduction and O(n) per step, each perturbing by
    // c eps (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., sections 19.3 and 19.6): the backward
    // error is c n (1 + steps) eps. The constant is measured: at most 15 n eps in 2000 draws of each scalar type.
    const Scalar tolerance = Scalar(128 * n) * NumTraits<Scalar>::epsilon();
    const RealQZ<MatrixType> qz(a, b);
    verify_qz_decomposition(a, b, qz, tolerance);
    if (omega > Scalar(0)) {
      // Once its 2x2 blocks converge, the pencil deflates between them at once: at most 8 steps in 2000 draws of each
      // scalar type, against up to 136 while a negligible S(k, k-1) waited for a random shift.
      VERIFY(qz.iterations() < 16);
      const GeneralizedEigenSolver<MatrixType> eig(a, b, false);
      VERIFY_IS_EQUAL(eig.info(), Success);
      const auto lambda = eig.eigenvalues().eval();
      // A B^-1 = P R P^T is normal, ||A||_F = omega sqrt(n) and ||B||_F = sqrt(n): by Bauer-Fike, a backward error tau
      // moves each eigenvalue by at most 2 sqrt(n) tau omega (1 + O(tau)).
      const Scalar eigenvalueTolerance = Scalar(2) * numext::sqrt(Scalar(n)) * tolerance * omega;
      for (Index i = 0; i < n; ++i) {
        VERIFY(numext::abs(numext::real(lambda(i))) <= eigenvalueTolerance);
        VERIFY(numext::abs(numext::abs(numext::imag(lambda(i))) - omega) <= eigenvalueTolerance);
      }
    }
  };
  // The step and error bounds above were measured with these orthogonal factors; keep their distribution.
  const auto orthogonal = [](Index n) { return MatrixType(MatrixType::Random(n, n).householderQr().householderQ()); };
  for (int repeat = 0; repeat < 20; ++repeat) {
    // blockdiag(R, R), R = [0 -1; 1 0], and the same with the blocks coupled: +-i, defective in the second case.
    MatrixType r = qz_rotation_blocks<MatrixType>(4, Scalar(1));
    const MatrixType p = orthogonal(4), z = orthogonal(4);
    verify(MatrixType(p * r * z), MatrixType(p * z), Scalar(1));
    r(0, 3) = Scalar(0.5);
    verify(MatrixType(p * r * z), MatrixType(p * z), Scalar(0));
    const MatrixType r6 = qz_rotation_blocks<MatrixType>(6, Scalar(2));
    const MatrixType p6 = orthogonal(6), z6 = orthogonal(6);
    verify(MatrixType(p6 * r6 * z6), MatrixType(p6 * z6), Scalar(2));
    const Index n = 3 + repeat % 4;
    MatrixType jordan = Scalar(0.5) * MatrixType::Identity(n, n);
    jordan.diagonal(1).setOnes();
    const MatrixType pn = orthogonal(n), zn = orthogonal(n);
    verify(MatrixType(pn * jordan * zn), MatrixType(pn * zn), Scalar(0));
  }
}

// A deflation scale wide enough for the +-i pairs above must not deflate a graded pencil early: with g = 2^-e,
// g^2 < eps, the smallest eigenvalue keeps a relative error of O(n eps). Both pencils have T = I:
//   a = D A0 D, D = diag(1, g, g^2): lambda_min = g^4 det(A0) / det(A0(0:1, 0:1)) (1 + O(g^2)),
//   a = [0 -g g^2; g g^2 g^3; 0 g^3 g^4], whose top 2x2 block has complex eigenvalues: lambda_min = 2 g^4 (1 + O(g^4)).
// Neither is large enough to reach the added term; the third pencil, below, is.
template <typename MatrixType>
void real_qz_graded_accuracy() {
  using Scalar = typename MatrixType::Scalar;
  const Scalar eps = NumTraits<Scalar>::epsilon();
  const Scalar g = numext::ldexp(Scalar(1), -((NumTraits<Scalar>::digits() + 3) / 2 + 2));
  const Scalar g2 = g * g, g3 = g2 * g, g4 = g2 * g2;
  const auto smallest = [](const MatrixType& a) {
    const GeneralizedEigenSolver<MatrixType> eig(a, MatrixType::Identity(a.rows(), a.cols()), false);
    VERIFY_IS_EQUAL(eig.info(), Success);
    return eig.eigenvalues().cwiseAbs().minCoeff();
  };
  MatrixType a0(3, 3);
  a0 << Scalar(0.7), Scalar(-0.3), Scalar(0.5), Scalar(0.4), Scalar(0.9), Scalar(-0.6), Scalar(-0.8), Scalar(0.2),
      Scalar(0.3);
  const MatrixType d = Matrix<Scalar, 3, 1>(Scalar(1), g, g2).asDiagonal();
  const Scalar reference = g4 * a0.determinant() / a0.template topLeftCorner<2, 2>().determinant();
  // No theorem makes QZ relatively accurate on a graded pencil, and normwise stability gives only cond n eps ||a||, so
  // 16 n eps is an empirical bound: at most 2 eps is measured, against errors of 3.1 and 0.5 after an early deflation.
  VERIFY(numext::abs(smallest(MatrixType(d * a0 * d)) - numext::abs(reference)) <=
         Scalar(16 * 3) * eps * numext::abs(reference));
  MatrixType complexPair(3, 3);
  complexPair << Scalar(0), -g, g2, g, g2, g3, Scalar(0), g3, g4;
  VERIFY(numext::abs(smallest(complexPair) - Scalar(2) * g4) <= Scalar(16 * 3) * eps * Scalar(2) * g4);
  // Two blocks with real eigenvalues, a = [1 b 0 0; b 3b^2 1 0; 0 y 3b^2 b; 0 0 b 2], y = eps b / 2: the scale between
  // them must stay 6 b^2, as a term 2 sqrt(b b) would deflate y. The eigenvalues are the roots of f = fU fL - y,
  // fU = 3b^2 - lambda - b^2 / (1 - lambda), fL = 3b^2 - lambda - b^2 / (2 - lambda). Deflating y moves the two small
  // ones, near 2b^2 and 2.5b^2, by y / (b^2 / 2) = eps / b; their Newton correction |f / (fU + fL)| must stay below
  // eps ||a||.
  const Scalar b = numext::ldexp(Scalar(1), -((NumTraits<Scalar>::digits() + 2) / 3 + 1));
  const Scalar b2 = b * b, y = eps * b / Scalar(2);
  MatrixType realBlocks(4, 4);
  realBlocks << Scalar(1), b, Scalar(0), Scalar(0), b, Scalar(3) * b2, Scalar(1), Scalar(0), Scalar(0), y,
      Scalar(3) * b2, b, Scalar(0), Scalar(0), b, Scalar(2);
  const GeneralizedEigenSolver<MatrixType> eig(realBlocks, MatrixType::Identity(4, 4), false);
  VERIFY_IS_EQUAL(eig.info(), Success);
  Index smallCount = 0;
  for (Index i = 0; i < 4; ++i) {
    const Scalar lambda = numext::real(eig.eigenvalues()(i));
    if (numext::abs(lambda) > Scalar(0.5)) continue;
    ++smallCount;
    const Scalar fU = Scalar(3) * b2 - lambda - b2 / (Scalar(1) - lambda);
    const Scalar fL = Scalar(3) * b2 - lambda - b2 / (Scalar(2) - lambda);
    // Empirical as well: at most 2e-4 eps is measured, against eps / b >= 512 eps once y deflates.
    VERIFY(numext::abs(fU * fL - y) <= eps * realBlocks.norm() * numext::abs(fU + fL));
  }
  VERIFY_IS_EQUAL(smallCount, Index(2));
}

// RealQZ drew its exceptional shifts from the global std::rand() state (#3170): two compute() calls on one pencil could
// differ, and compute() advanced the caller's sequence. The coupled +-i pencil took those shifts before they were
// gated. Graded pencils (D A D, D B D), D = diag(g^i), still take them where the iteration stagnates, which for n = 5
// happens around g^8 = eps^1.5: on x86-64, 1 of the 32 below in float, 5 in double and 5 in 80-bit long double (none
// in binary128 or double-double). Their entries come from a generator independent of std::rand().
template <typename MatrixType>
void real_qz_leaves_rand_alone() {
  using Scalar = typename MatrixType::Scalar;
  // Restored on exit, so that the subtests after this one still follow EIGEN_SEED.
  const unsigned reseed = static_cast<unsigned>(std::rand());
  const auto verifyUntouched = [](const MatrixType& a, const MatrixType& b) {
    std::srand(1);
    const RealQZ<MatrixType> qz(a, b);
    const int next = std::rand();
    std::srand(1);
    VERIFY_IS_EQUAL(next, std::rand());
    VERIFY_IS_EQUAL(qz.info(), Success);
  };
  MatrixType r = qz_rotation_blocks<MatrixType>(4, Scalar(1));
  r(0, 3) = Scalar(0.5);
  const MatrixType p = generateRandomUnitaryMatrix<MatrixType>(4);
  verifyUntouched(MatrixType(p * r * p.transpose()), MatrixType::Identity(4, 4));
  const Scalar g = numext::pow(NumTraits<Scalar>::epsilon(), Scalar(3) / Scalar(16));
  for (numext::uint64_t seed = 1; seed <= 32; ++seed) {
    internal::inverse_iteration_rng rng(seed);
    const Index n = 5;
    MatrixType a(n, n), b(n, n);
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < n; ++i) {
        const Scalar grade = numext::pow(g, Scalar(i + j));
        a(i, j) = Scalar(rng.next<double>()) * grade;
        b(i, j) = Scalar(rng.next<double>()) * grade;
      }
    verifyUntouched(a, b);
  }
  std::srand(reseed);
}

// Where both diagonal entries next to S(k, k-1) are exactly zero, the deflation test compared S(k, k-1) with ||S||. The
// subdiagonal entry of a small block [0 -g; g 0] was then set to zero itself, and the pair +-ig came back as 0, 0:
//   a = [R X; 0 g R], R = [0 -1; 1 0], g = eps / 16: +-i and +-ig,
//   a = skew-symmetric tridiagonal (1, h, h^2), h^2 < eps: +-i sigma1 and +-i sigma2, with
//       sigma1^2 = (t + sqrt(t^2 - 4 h^4)) / 2, t = 1 + h^2 + h^4, and sigma2 = h^2 / sigma1.
template <typename MatrixType>
void real_qz_zero_diagonal() {
  using Scalar = typename MatrixType::Scalar;
  const Scalar eps = NumTraits<Scalar>::epsilon();
  const auto verify = [&eps](const MatrixType& a, const Scalar& sigmaLarge, const Scalar& sigmaSmall) {
    const GeneralizedEigenSolver<MatrixType> eig(a, MatrixType::Identity(4, 4), false);
    VERIFY_IS_EQUAL(eig.info(), Success);
    // Empirical, as in real_qz_graded_accuracy: the measured error is 0, against 1 for a pair that is lost.
    const Scalar tolerance = Scalar(16 * 4) * eps;
    Index smallCount = 0;
    for (Index i = 0; i < 4; ++i) {
      const Scalar re = numext::real(eig.eigenvalues()(i)), im = numext::abs(numext::imag(eig.eigenvalues()(i)));
      const bool isSmall = im < numext::sqrt(sigmaLarge) * numext::sqrt(sigmaSmall);
      const Scalar sigma = isSmall ? sigmaSmall : sigmaLarge;
      if (isSmall) ++smallCount;
      VERIFY(numext::abs(re) <= tolerance * sigma);
      VERIFY(numext::abs(im - sigma) <= tolerance * sigma);
    }
    VERIFY_IS_EQUAL(smallCount, Index(2));
  };
  const Scalar g = eps / Scalar(16);
  MatrixType blocks(4, 4);
  blocks << Scalar(0), Scalar(-1), Scalar(0.5), Scalar(-0.25), Scalar(1), Scalar(0), Scalar(0.75), Scalar(0.5),
      Scalar(0), Scalar(0), Scalar(0), -g, Scalar(0), Scalar(0), g, Scalar(0);
  verify(blocks, Scalar(1), g);
  const Scalar h = numext::ldexp(Scalar(1), -((NumTraits<Scalar>::digits() + 3) / 2 + 2)), h2 = h * h;
  MatrixType skew = MatrixType::Zero(4, 4);
  skew(1, 0) = Scalar(1);
  skew(2, 1) = h;
  skew(3, 2) = h2;
  skew.template triangularView<StrictlyUpper>() = -skew.transpose();
  const Scalar t = Scalar(1) + h2 + h2 * h2;
  const Scalar sigma1 = numext::sqrt((t + numext::sqrt(t * t - Scalar(4) * h2 * h2)) / Scalar(2));
  verify(skew, sigma1, h2 / sigma1);
}

// The deflation test keeps a tiny S(f+1, f) on top of a block where its neighboring diagonal entries are zero or as
// small. The shift vector of the next step, which divides by that entry, then overflowed, and S and T filled with NaN:
//   a = [d -1 0 1; u d -1 0; 0 1 0 -1; 0 0 1 0], b = diag(1, 1, t, t), u = 4 realmin / eps, t = eps^(3/4), d = 0 or u.
template <typename MatrixType>
void real_qz_tiny_subdiagonal() {
  using Scalar = typename MatrixType::Scalar;
  const Scalar eps = NumTraits<Scalar>::epsilon();
  const Scalar u = Scalar(4) * (std::numeric_limits<Scalar>::min)() / eps, t = numext::pow(eps, Scalar(0.75));
  for (const Scalar d : {Scalar(0), u}) {
    MatrixType a(4, 4), b = MatrixType::Zero(4, 4);
    a << d, Scalar(-1), Scalar(0), Scalar(1), u, d, Scalar(-1), Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(-1),
        Scalar(0), Scalar(0), Scalar(1), Scalar(0);
    b.diagonal() << Scalar(1), Scalar(1), t, t;
    const RealQZ<MatrixType> qz(a, b);
    // O(n eps), as in real_qz_repeated_eigenvalues.
    verify_qz_decomposition(a, b, qz, Scalar(128 * 4) * eps);
  }
}

EIGEN_DECLARE_TEST(real_qz) {
  CALL_SUBTEST_2((qz_no_malloc<MatrixXd, RealQZ<MatrixXd>>()));
  CALL_SUBTEST_1((qz_extreme_scaling<MatrixXf, RealQZ<MatrixXf>>()));
  CALL_SUBTEST_2((qz_extreme_scaling<MatrixXd, RealQZ<MatrixXd>>()));
  CALL_SUBTEST_1((qz_unrepresentable<MatrixXf, RealQZ<MatrixXf>>({2, 10})));
  CALL_SUBTEST_2((qz_unrepresentable<MatrixXd, RealQZ<MatrixXd>>({2, 10})));
  CALL_SUBTEST_1((real_qz_ill_conditioned_block<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_ill_conditioned_block<MatrixXd>()));
  CALL_SUBTEST_1((real_qz_split_order<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_split_order<MatrixXd>()));
  CALL_SUBTEST_1((qz_zero_pencil<MatrixXf, RealQZ<MatrixXf>>()));
  CALL_SUBTEST_2((qz_zero_pencil<MatrixXd, RealQZ<MatrixXd>>()));
  CALL_SUBTEST_1((real_qz_zero_pencil_eigenvalues<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_zero_pencil_eigenvalues<MatrixXd>()));
  CALL_SUBTEST_1((qz_nan_pencil_terminates<MatrixXf, RealQZ<MatrixXf>>()));
  CALL_SUBTEST_2((qz_nan_pencil_terminates<MatrixXd, RealQZ<MatrixXd>>()));
  CALL_SUBTEST_1((qz_iteration_limit<MatrixXf, RealQZ<MatrixXf>>()));
  CALL_SUBTEST_2((qz_iteration_limit<MatrixXd, RealQZ<MatrixXd>>()));
  CALL_SUBTEST_5((qz_no_malloc<RowMajorXf, RealQZ<RowMajorXf>>()));
  CALL_SUBTEST_6((qz_no_malloc<RowMajorXd, RealQZ<RowMajorXd>>()));
  CALL_SUBTEST_1((real_qz_repeated_eigenvalues<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_repeated_eigenvalues<MatrixXd>()));
  CALL_SUBTEST_7((real_qz_repeated_eigenvalues<Matrix<long double, Dynamic, Dynamic>>()));
  CALL_SUBTEST_1((real_qz_graded_accuracy<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_graded_accuracy<MatrixXd>()));
  CALL_SUBTEST_7((real_qz_graded_accuracy<Matrix<long double, Dynamic, Dynamic>>()));
  CALL_SUBTEST_1((real_qz_leaves_rand_alone<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_leaves_rand_alone<MatrixXd>()));
  CALL_SUBTEST_7((real_qz_leaves_rand_alone<Matrix<long double, Dynamic, Dynamic>>()));
  CALL_SUBTEST_1((real_qz_zero_diagonal<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_zero_diagonal<MatrixXd>()));
  CALL_SUBTEST_7((real_qz_zero_diagonal<Matrix<long double, Dynamic, Dynamic>>()));
  CALL_SUBTEST_1((real_qz_tiny_subdiagonal<MatrixXf>()));
  CALL_SUBTEST_2((real_qz_tiny_subdiagonal<MatrixXd>()));
  CALL_SUBTEST_7((real_qz_tiny_subdiagonal<Matrix<long double, Dynamic, Dynamic>>()));

  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(real_qz(Matrix4f()));
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    CALL_SUBTEST_2(real_qz(MatrixXd(s, s)));

    // some trivial but implementation-wise tricky cases
    CALL_SUBTEST_2(real_qz(MatrixXd(1, 1)));
    CALL_SUBTEST_2(real_qz(MatrixXd(2, 2)));
    CALL_SUBTEST_3(real_qz(Matrix<double, 1, 1>()));
    CALL_SUBTEST_4(real_qz(Matrix2d()));
    CALL_SUBTEST_5((real_qz(RowMajorXf(17, 17))));
    CALL_SUBTEST_6((real_qz(RowMajorXd(17, 17))));
  }

  TEST_SET_BUT_UNUSED_VARIABLE(s);
}
