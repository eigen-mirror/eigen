// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"

#include <Eigen/Eigenvalues>

/* this test covers the following files:
   ComplexQZ.h
*/

template <typename MatrixType>
void generate_random_matrix_pair(const Index dim, MatrixType& A, MatrixType& B) {
  A.setRandom(dim, dim);
  B.setRandom(dim, dim);
  // Zero out each row of B to with a probability of 10%.
  for (int i = 0; i < dim; i++) {
    if (internal::random<int>(0, 10) == 0) B.row(i).setZero();
  }
}

template <typename MatrixType>
void complex_qz(const MatrixType& A, const MatrixType& B) {
  using std::abs;
  const Index dim = A.rows();
  ComplexQZ<MatrixType> qz(A, B);
  VERIFY_IS_EQUAL(qz.info(), Success);
  auto T = qz.matrixT(), S = qz.matrixS();
  bool is_all_zero_T = true, is_all_zero_S = true;
  using RealScalar = typename MatrixType::RealScalar;
  RealScalar tol = dim * 10 * NumTraits<RealScalar>::epsilon();
  for (Index j = 0; j < dim; j++) {
    for (Index i = j + 1; i < dim; i++) {
      if (std::abs(T(i, j)) > tol) {
        std::cerr << std::abs(T(i, j)) << std::endl;
        is_all_zero_T = false;
      }
      if (std::abs(S(i, j)) > tol) {
        std::cerr << std::abs(S(i, j)) << std::endl;
        is_all_zero_S = false;
      }
    }
  }
  VERIFY_IS_EQUAL(is_all_zero_T, true);
  VERIFY_IS_EQUAL(is_all_zero_S, true);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixS() * qz.matrixZ(), A);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixT() * qz.matrixZ(), B);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixQ().adjoint(), MatrixType::Identity(dim, dim));
  VERIFY_IS_APPROX(qz.matrixZ() * qz.matrixZ().adjoint(), MatrixType::Identity(dim, dim));
}

template <typename MatrixType, typename QZType>
void verify_complex_qz_convergence(const MatrixType& a, const MatrixType& b, const QZType& qz) {
  using RealScalar = typename MatrixType::RealScalar;
  // Accumulated rounding in the unitary transformations is O(n * epsilon). stableNorm, since the squared entries of
  // complex_qz_scaled's pencils underflow.
  const RealScalar tolerance = RealScalar(128 * a.rows()) * NumTraits<RealScalar>::epsilon();
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY((a - qz.matrixQ() * qz.matrixS() * qz.matrixZ()).stableNorm() <= tolerance * a.stableNorm());
  VERIFY((b - qz.matrixQ() * qz.matrixT() * qz.matrixZ()).stableNorm() <= tolerance * b.stableNorm());
  VERIFY(qz.matrixQ().isUnitary(tolerance));
  VERIFY(qz.matrixZ().isUnitary(tolerance));
  const MatrixType lowerS = qz.matrixS().template triangularView<StrictlyLower>();
  const MatrixType lowerT = qz.matrixT().template triangularView<StrictlyLower>();
  VERIFY(lowerS.stableNorm() <= tolerance * a.stableNorm());
  VERIFY(lowerT.stableNorm() <= tolerance * b.stableNorm());
}

template <typename MatrixType>
void complex_qz_exceptional_shift() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  for (Index dim : {3, 4, 5, 8}) {
    for (Scalar phase : {Scalar(1), Scalar(0, 1)}) {
      // With corner -1 and phase 1, the exceptional shift 2 is equidistant from the eigenvalues exp(+-i*pi/dim).
      for (RealScalar corner : {RealScalar(1), RealScalar(-1)}) {
        // The trailing 2x2 block has two zero shifts: ordinary double shifts cycle without deflation.
        MatrixType a = MatrixType::Zero(dim, dim), b = MatrixType::Identity(dim, dim);
        a.diagonal(-1).setConstant(phase);
        a(0, dim - 1) = corner * phase;
        for (Index j = 0; j < dim; ++j) {
          b(j, j) = Scalar(1 << (j % 3));
          a.col(j) *= b(j, j);
        }
        const RealScalar tolerance = RealScalar(128 * dim) * NumTraits<RealScalar>::epsilon();

        ComplexQZ<MatrixType> qz(a, b);
        verify_complex_qz_convergence(a, b, qz);
        // a*b^-1 = phase*P with P^dim = corner*I, so lambda_k = phase*w_k with w_k^dim = corner. These are
        // 2*sin(pi/dim) apart with unit condition number (y^* b x = 1 for unit left y and |x| <= 1), so each moves by
        // at most the backward error, including the dropped strictly lower parts of S and T.
        const Matrix<Scalar, Dynamic, 1> lambda = qz.matrixS().diagonal().cwiseQuotient(qz.matrixT().diagonal());
        const RealScalar eigenvalueTolerance = RealScalar(2) * tolerance * (a.norm() + b.norm());
        for (Index k = 0; k < dim; ++k) {
          const RealScalar angle = (RealScalar(2 * k) + (corner < 0 ? RealScalar(1) : RealScalar(0))) *
                                   RealScalar(EIGEN_PI) / RealScalar(dim);
          const Scalar expected = phase * std::polar(RealScalar(1), angle);
          VERIFY((lambda.array() - expected).abs().minCoeff() <= eigenvalueTolerance);
        }

        ComplexQZ<MatrixType> limited(a, b, true, 1);
        VERIFY_IS_EQUAL(limited.info(), NoConvergence);
        VERIFY_IS_EQUAL(limited.iterations(), 1);

        MatrixType inplaceA = a, inplaceB = b;
        ComplexQZ<Ref<MatrixType>> inplace(inplaceA, inplaceB);
        verify_complex_qz_convergence(a, b, inplace);

        const MatrixType s = qz.matrixS(), t = qz.matrixT();
        qz.compute(a, b, false);
        VERIFY_IS_EQUAL(qz.info(), Success);
        VERIFY((qz.matrixS() - s).norm() <= tolerance * a.norm());
        VERIFY((qz.matrixT() - t).norm() <= tolerance * b.norm());
        qz.compute(a, b);
        verify_complex_qz_convergence(a, b, qz);
      }
    }
  }
}

// The 2x2 deflation compared the diagonal of T with an absolute epsilon, so a pencil with ||B|| << 1 failed.
template <typename MatrixType>
void complex_qz_small_b() {
  using RealScalar = typename MatrixType::RealScalar;
  for (Index dim : {2, 3, 5}) {
    const MatrixType a = MatrixType::Random(dim, dim),
                     b = MatrixType::Random(dim, dim) * (NumTraits<RealScalar>::epsilon() / 1024);
    ComplexQZ<MatrixType> qz(a, b);
    verify_complex_qz_convergence(a, b, qz);
  }
}

// The DARE example "darex #14" of SciPy's solve_discrete_are tests, from Benner, Laub and Mehrmann, "A Collection of
// Benchmark Examples for the Numerical Solution of Algebraic Riccati Equations II: Discrete-Time Case", TU
// Chemnitz-Zwickau SPC 95_23 (1995): the balanced symplectic pencil with the column of R deflated. Eigenvalues 0 (x3,
// one Jordan chain), infinity (x3) and 1 -+ sqrt(5) 1e-8. With AVX2 and FMA, an iterate had S(1, 0) = 1.8e-17 next to
// diagonal entries 2.4e-15 and 3e-3; sweeps starting at row 0 then stalled for 400 iterations.
template <typename MatrixType>
void complex_qz_darex14() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const Index m = 4, p = 2 * m + 1;
  // H = [A 0 B; -Q I 0; 0 0 R], J = [I 0 0; 0 A^* 0; 0 -B^* 0] with A = (1 - 1e-8) e0 e0^T + shift down,
  // B = 1e-8 e0, Q = e3 e3^T and R = 1/4.
  MatrixType H = MatrixType::Zero(p, p), J = MatrixType::Zero(p, p);
  H(0, 0) = J(m, m) = Scalar(RealScalar(1) - RealScalar(1e-8));
  for (Index i = 1; i < m; ++i) H(i, i - 1) = J(m + i - 1, m + i) = Scalar(1);
  H(0, 2 * m) = Scalar(RealScalar(1e-8));
  H(m + 3, 3) = Scalar(-1);
  H.block(m, m, m, m).setIdentity();
  H(2 * m, 2 * m) = Scalar(RealScalar(0.25));
  J.topLeftCorner(m, m).setIdentity();
  J(2 * m, m) = Scalar(RealScalar(-1e-8));
  // Symplectic balancing (Benner, SIAM J. Sci. Comput. 22(5), 2001) by diag(D, D^-1, 1/2) with
  // D = diag(2^20, 2^14, 2^8, 2^2).
  Matrix<RealScalar, Dynamic, 1> scale(p);
  const int exponents[m] = {20, 14, 8, 2};
  for (Index i = 0; i < m; ++i) {
    scale(i) = std::ldexp(RealScalar(1), exponents[i]);
    scale(m + i) = std::ldexp(RealScalar(1), -exponents[i]);
  }
  scale(2 * m) = RealScalar(0.5);
  H = scale.template cast<Scalar>().asDiagonal() * H * scale.cwiseInverse().template cast<Scalar>().asDiagonal();
  J = scale.template cast<Scalar>().asDiagonal() * J * scale.cwiseInverse().template cast<Scalar>().asDiagonal();
  const HouseholderQR<MatrixType> qr(H.rightCols(1));
  const MatrixType a = (qr.householderQ().adjoint() * H.leftCols(2 * m)).bottomRows(2 * m);
  const MatrixType b = (qr.householderQ().adjoint() * J.leftCols(2 * m)).bottomRows(2 * m);

  ComplexQZ<MatrixType> qz(a, b);
  verify_complex_qz_convergence(a, b, qz);
}

// a = diag(A1, s A2), b = I with A2 = U diag(lambda) U^* normal and |lambda_k| in [1/2, 1]. Relative deflation
// decomposes the small block as if it were alone, so by Bauer-Fike its eigenvalues move by at most its backward error,
// ~128 n eps s. A normwise deflation threshold eps ||a|| would allow errors ~eps ||A1|| >> s.
template <typename MatrixType>
void complex_qz_reducible(typename MatrixType::RealScalar s) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const Index n1 = 8, n2 = 8, n = n1 + n2;
  Vector lambda(n2);
  for (Index k = 0; k < n2; ++k)
    lambda(k) = std::polar(internal::random<RealScalar>(RealScalar(0.5), RealScalar(1)),
                           internal::random<RealScalar>(-RealScalar(EIGEN_PI), RealScalar(EIGEN_PI)));
  const MatrixType U = HouseholderQR<MatrixType>(MatrixType::Random(n2, n2)).householderQ();
  MatrixType a = MatrixType::Zero(n, n);
  a.topLeftCorner(n1, n1).setRandom();
  a.bottomRightCorner(n2, n2) = s * U * lambda.asDiagonal() * U.adjoint();
  const MatrixType b = MatrixType::Identity(n, n);

  ComplexQZ<MatrixType> qz(a, b);
  verify_complex_qz_convergence(a, b, qz);
  const Vector computed = qz.matrixS().diagonal().cwiseQuotient(qz.matrixT().diagonal());
  const RealScalar tolerance = RealScalar(128 * n) * NumTraits<RealScalar>::epsilon() * s;
  for (Index k = 0; k < n2; ++k) VERIFY((computed.array() - s * lambda(k)).abs().minCoeff() <= tolerance);
}

// Graded pencils leave 2x2 blocks that the closed-form reduction in reduceDiagonal2x2block does not deflate on its own.
template <typename MatrixType>
void complex_qz_graded() {
  using RealScalar = typename MatrixType::RealScalar;
  for (Index dim : {4, 8}) {
    for (int trial = 0; trial < 50; ++trial) {
      MatrixType a = MatrixType::Random(dim, dim), b = MatrixType::Random(dim, dim);
      for (Index i = 0; i < dim; ++i) {
        const RealScalar grade = std::pow(RealScalar(10), -RealScalar(8 * i) / RealScalar(dim));
        a.row(i) *= grade;
        b.col(i) *= grade;
      }
      ComplexQZ<MatrixType> qz(a, b);
      verify_complex_qz_convergence(a, b, qz);
    }
  }
}

// a = s N for the down-shift N, b = I. At s = 1 the 2x2 blocks have the eigenvalue 0, for which a left rotation formed
// from lambda T is the identity and leaves T(i+1, i) in place. At s = sqrt(min) eps the products of the sweep-start
// test underflow to 0, and an unscaled test accepted the first row it tried (0 <= 0) and dropped fill of size s.
template <typename MatrixType>
void complex_qz_scaled() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar tiny = std::sqrt((std::numeric_limits<RealScalar>::min)()) * NumTraits<RealScalar>::epsilon();
  for (RealScalar s : {RealScalar(1), tiny}) {
    for (Index dim : {4, 5, 8}) {
      MatrixType a = MatrixType::Zero(dim, dim);
      a.diagonal(-1).setConstant(Scalar(s));
      const MatrixType b = MatrixType::Identity(dim, dim);
      ComplexQZ<MatrixType> qz(a, b);
      verify_complex_qz_convergence(a, b, qz);
    }
  }
}

// a = V J W^*, b = V W^* with J the 2x2 Jordan block of the eigenvalue 0. On a defective block, refinement steps in
// reduceDiagonal2x2block can lower |S(1, 0)| / (|S(0, 0)| + |S(1, 1)|) by growing the diagonal while |S(1, 0)| grows
// past eps ||S||.
template <typename MatrixType>
void complex_qz_jordan() {
  using Scalar = typename MatrixType::Scalar;
  MatrixType J = MatrixType::Zero(2, 2);
  J(0, 1) = Scalar(1);
  for (int trial = 0; trial < 100; ++trial) {
    for (bool real : {true, false}) {
      MatrixType V = MatrixType::Random(2, 2), W = MatrixType::Random(2, 2);
      if (real) {
        V = V.real().template cast<Scalar>();
        W = W.real().template cast<Scalar>();
      }
      const MatrixType a = V * J * W.adjoint(), b = V * W.adjoint();
      ComplexQZ<MatrixType> qz(a, b);
      verify_complex_qz_convergence(a, b, qz);
    }
  }
}

// push_down_zero_ST checked the entries it leaves at T(l, l) and S(l, l - 1) against an absolute epsilon instead of
// eps ||T|| and eps ||S||, so whether this graded pencil reported NumericalIssue depended on the scale of b (#3212).
template <typename MatrixType>
void complex_qz_push_down_scale() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const int re[5][5] = {{0, -2, -1, -1, 2}, {1, -2, -2, 0, -2}, {1, 0, -1, 0, 0}, {-1, -1, 2, 0, -2}, {1, 2, 1, 1, 1}};
  const int im[5][5] = {{0, -2, 2, 0, 1}, {2, 1, -1, 2, -2}, {-1, -2, 0, 2, 1}, {2, -2, 2, -1, -1}, {2, 2, -2, 1, 2}};
  const int bre[5][5] = {
      {0, -1, -1, -1, -1}, {-1, -2, 1, 2, -2}, {0, 0, -2, 2, -1}, {2, -1, -2, -2, 2}, {1, -2, 1, 2, -1}};
  const int bim[5][5] = {{0, -1, 1, -2, 2}, {1, -2, -2, 1, -2}, {2, 0, 0, 1, 0}, {-2, 2, 1, 0, -2}, {1, 2, -1, -1, 2}};
  // Rows graded by 10^-2 (float) or 10^-4 (double), so that T gets diagonal entries below eps ||T||.
  const RealScalar step = NumTraits<RealScalar>::digits10() > 10 ? RealScalar(1e-4) : RealScalar(1e-2);
  MatrixType a(5, 5), b(5, 5);
  RealScalar grade = 1;
  for (Index i = 0; i < 5; ++i, grade *= step) {
    for (Index j = 0; j < 5; ++j) {
      a(i, j) = Scalar(RealScalar(re[i][j]), RealScalar(im[i][j])) * grade;
      b(i, j) = Scalar(RealScalar(bre[i][j]), RealScalar(bim[i][j])) * grade;
    }
  }
  for (int e : {-10, 0, 10}) {
    const MatrixType bs = b * numext::ldexp(RealScalar(1), e);
    ComplexQZ<MatrixType> qz(a, bs);
    verify_complex_qz_convergence(a, bs, qz);
  }
}

// A and B are scaled by powers of two near the ends of the exponent range, which made the iteration overflow or
// underflow (#3212). The scaling must leave Q and Z, and S and T up to the same powers of two, unchanged.
template <typename MatrixType>
void complex_qz_extreme_scaling() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const Index dim = 6;
  // Real and imaginary parts of magnitude in [1/3, 1], so that no scaled entry becomes subnormal.
  const auto away = [](const RealScalar& x) {
    return (x < 0 ? x - RealScalar(0.5) : x + RealScalar(0.5)) / RealScalar(1.5);
  };
  const auto awayFromZero = [&](const Scalar& x) { return Scalar(away(numext::real(x)), away(numext::imag(x))); };
  const MatrixType a0 = MatrixType::Random(dim, dim).unaryExpr(awayFromZero);
  const MatrixType b0 = MatrixType::Random(dim, dim).unaryExpr(awayFromZero);
  const int big = std::numeric_limits<RealScalar>::max_exponent - 16;
  const RealScalar tolerance = RealScalar(128 * dim) * NumTraits<RealScalar>::epsilon();
  for (int ea : {0, big, -big}) {
    for (int eb : {0, big, -big}) {
      const MatrixType A = a0 * numext::ldexp(RealScalar(1), ea), B = b0 * numext::ldexp(RealScalar(1), eb);
      ComplexQZ<MatrixType> qz(A, B);
      VERIFY_IS_EQUAL(qz.info(), Success);
      const MatrixType S = qz.matrixS() * numext::ldexp(RealScalar(1), -ea);
      const MatrixType T = qz.matrixT() * numext::ldexp(RealScalar(1), -eb);
      VERIFY(S.allFinite() && T.allFinite());
      VERIFY((qz.matrixQ() * S * qz.matrixZ() - a0).norm() <= tolerance * a0.norm());
      VERIFY((qz.matrixQ() * T * qz.matrixZ() - b0).norm() <= tolerance * b0.norm());
    }
  }
}

// Finite pencils whose abs-sum norm sum |t_ij| overflows, although ||B||_F does not. With that norm infinite, every
// relative threshold eps ||T|| accepted any entry, so T(l, l) of size ||B|| was zeroed with Success. The input scaling
// keeps the norms finite.
template <typename MatrixType>
void complex_qz_norm_overflow() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const MatrixType a3 = MatrixType::Random(3, 3);
  const MatrixType b3 = MatrixType::Identity(3, 3) * Scalar(RealScalar(0.39) * NumTraits<RealScalar>::highest());
  ComplexQZ<MatrixType> qz3(a3, b3);
  verify_complex_qz_convergence(a3, b3, qz3);
  const MatrixType a40 = MatrixType::Random(40, 40);
  const MatrixType b40 =
      MatrixType::Random(40, 40) * numext::ldexp(RealScalar(1), std::numeric_limits<RealScalar>::max_exponent - 6);
  ComplexQZ<MatrixType> qz40(a40, b40);
  verify_complex_qz_convergence(a40, b40, qz40);
}

// One step per eigenvalue cannot reduce this graded complex<float> pencil, so info() must report NoConvergence. It
// reported NumericalIssue instead (#3212), raised by the absolute-epsilon check that push_down_zero_ST then made. A 1x1
// pencil needs no step, so maxIters = 0 still reduces it.
void complex_qz_no_convergence_status() {
  using cf = std::complex<float>;
  MatrixXcf a(5, 5), b(5, 5);
  a << cf(-1, -1), cf(-2, -2), cf(2, -2), cf(0, 1), cf(-2, -1),  //
      cf(-2, 2), cf(2, 2), cf(2, 2), cf(-1, 2), cf(-1, -1),      //
      cf(-1, 0), cf(-2, -2), cf(-1, -1), cf(0, -2), cf(1, 1),    //
      cf(0, 2), cf(-2, -2), cf(1, 0), cf(1, 1), cf(-1, -1),      //
      cf(-2, -2), cf(-1, 2), cf(1, 1), cf(2, -1), cf(-1, 1);
  b << cf(1, -2), cf(-2, 2), cf(0, 0), cf(1, -2), cf(2, -1),   //
      cf(-1, -1), cf(2, -2), cf(-2, 1), cf(-2, 2), cf(2, -1),  //
      cf(-2, 1), cf(0, 1), cf(1, 0), cf(1, -2), cf(1, -2),     //
      cf(0, 1), cf(2, -1), cf(2, -1), cf(-2, 1), cf(2, 0),     //
      cf(1, 1), cf(1, -2), cf(2, 0), cf(2, 0), cf(0, 0);
  const float grade[] = {1.f, 1e-2f, 1e-4f, 1e-6f, 1e-8f};
  for (Index i = 0; i < 5; ++i) {
    a.row(i) *= grade[i];
    b.row(i) *= grade[i];
  }
  ComplexQZ<MatrixXcf> limited(a, b, true, 1);
  VERIFY_IS_EQUAL(limited.info(), NoConvergence);
  VERIFY((limited.matrixS().diagonal(-1).array() != cf(0)).any());

  const MatrixXcf a1 = MatrixXcf::Constant(1, 1, cf(2)), b1 = MatrixXcf::Constant(1, 1, cf(1));
  ComplexQZ<MatrixXcf> scalar(a1, b1, true, 0);
  VERIFY_IS_EQUAL(scalar.info(), Success);
}

// An exact zero on the subdiagonal of S did not deflate when |S(k-1, k-1)| + |S(k, k)| and ||S|| were both 0. With
// A = 0 and n >= 3 the iteration made no progress: it returned NoConvergence with NaN in S and T for a nonzero B, and
// did not return for B = 0. Both pencils are already in generalized Schur form.
template <typename MatrixType>
void complex_qz_zero_pencil() {
  for (Index dim : {1, 2, 3, 5}) {
    const MatrixType zero = MatrixType::Zero(dim, dim), random = MatrixType::Random(dim, dim);
    for (const MatrixType* b : {&zero, &random}) {
      ComplexQZ<MatrixType> qz(zero, *b);
      verify_complex_qz_convergence(zero, *b, qz);
    }
  }
}

// ComplexQZ returned Success with inf in S or T where Q^* A Z or Q^* B Z is not representable. For A or B = c 1 with
// c = 0.9 highest, that triangular factor has rank one and Frobenius norm c n, so its nonzeros form a p x q block with
// p + q <= n + 1, one of them at least 2 c n / (n + 1) > highest: info() must report NumericalIssue. For A = c Random,
// the moduli of the coefficients overflowed and left the pencil unscaled, so the iteration did not converge.
template <typename MatrixType>
void complex_qz_unrepresentable() {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const Scalar c(RealScalar(0.9) * NumTraits<RealScalar>::highest());
  for (Index n : {2, 10}) {
    const MatrixType ones = MatrixType::Constant(n, n, c), random = MatrixType::Random(n, n);
    ComplexQZ<MatrixType> qzA(ones, random), qzB(random, ones);
    VERIFY_IS_EQUAL(qzA.info(), NumericalIssue);
    VERIFY_IS_EQUAL(qzB.info(), NumericalIssue);
    const MatrixType big = random * c;
    ComplexQZ<MatrixType> qzBig(big, random);
    VERIFY(qzBig.info() != NoConvergence);
    VERIFY(qzBig.info() != Success || (qzBig.matrixS().allFinite() && qzBig.matrixT().allFinite()));
  }
}

// Block Householder operations still allocate at larger sizes (issue #3163).
template <typename MatrixType>
void complex_qz_no_malloc() {
  const Index dim = 24;
  MatrixType A = MatrixType::Random(dim, dim), B = MatrixType::Random(dim, dim);
  ComplexQZ<MatrixType> qz(dim);
  internal::set_is_malloc_allowed(false);
  qz.compute(A, B, false);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(qz.info(), Success);
  const MatrixType S = qz.matrixS(), T = qz.matrixT();

  ComplexQZ<MatrixType> resized(1, false);
  resized.compute(A, B, false);
  internal::set_is_malloc_allowed(false);
  resized.compute(A, B, false);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(resized.info(), Success);
  VERIFY_IS_APPROX(resized.matrixS(), S);
  VERIFY_IS_APPROX(resized.matrixT(), T);

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

EIGEN_DECLARE_TEST(complex_qz) {
  CALL_SUBTEST_3((complex_qz_no_malloc<MatrixXcf>()));
  CALL_SUBTEST_4((complex_qz_no_malloc<MatrixXcd>()));
  CALL_SUBTEST_5((complex_qz_no_malloc<Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_6((complex_qz_no_malloc<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_7((complex_qz_exceptional_shift<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_exceptional_shift<MatrixXcd>()));
  CALL_SUBTEST_7((complex_qz_small_b<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_small_b<MatrixXcd>()));
  CALL_SUBTEST_8((complex_qz_darex14<MatrixXcd>()));
  CALL_SUBTEST_10((complex_qz_darex14<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_7((complex_qz_reducible<MatrixXcf>(1e-6f)));
  CALL_SUBTEST_8((complex_qz_reducible<MatrixXcd>(1e-10)));
  CALL_SUBTEST_9((complex_qz_reducible<Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>>(1e-6f)));
  CALL_SUBTEST_10((complex_qz_reducible<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>(1e-10)));
  CALL_SUBTEST_7((complex_qz_graded<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_graded<MatrixXcd>()));
  CALL_SUBTEST_9((complex_qz_graded<Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_10((complex_qz_graded<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_7((complex_qz_scaled<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_scaled<MatrixXcd>()));
  CALL_SUBTEST_9((complex_qz_scaled<Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_10((complex_qz_scaled<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_7((complex_qz_jordan<MatrixXcf>()));
  CALL_SUBTEST_7(complex_qz_no_convergence_status());
  CALL_SUBTEST_7((complex_qz_zero_pencil<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_zero_pencil<MatrixXcd>()));
  CALL_SUBTEST_7((complex_qz_unrepresentable<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_unrepresentable<MatrixXcd>()));
  CALL_SUBTEST_8((complex_qz_jordan<MatrixXcd>()));
  CALL_SUBTEST_7((complex_qz_push_down_scale<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_push_down_scale<MatrixXcd>()));
  CALL_SUBTEST_7((complex_qz_extreme_scaling<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_extreme_scaling<MatrixXcd>()));
  CALL_SUBTEST_7((complex_qz_norm_overflow<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_norm_overflow<MatrixXcd>()));
  CALL_SUBTEST_9((complex_qz_jordan<Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_10((complex_qz_jordan<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_9((complex_qz_exceptional_shift<Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_10((complex_qz_exceptional_shift<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>()));

  for (int i = 0; i < g_repeat; i++) {
    // Check for very small, fixed-sized double- and float complex matrices
    Eigen::Matrix2cd A_2x2, B_2x2;
    A_2x2.setRandom();
    B_2x2.setRandom();
    B_2x2.row(1).setZero();
    Eigen::Matrix3cf A_3x3, B_3x3;
    A_3x3.setRandom();
    B_3x3.setRandom();
    B_3x3.col(i % 3).setRandom();
    CALL_SUBTEST_1(complex_qz(A_2x2, B_2x2));
    CALL_SUBTEST_2(complex_qz(A_3x3, B_3x3));

    // Test for float complex matrices
    const Index dim = internal::random<Index>(15, 80);
    Eigen::MatrixXcf A_float, B_float;
    generate_random_matrix_pair(dim, A_float, B_float);
    CALL_SUBTEST_3(complex_qz(A_float, B_float));
    CALL_SUBTEST_5((complex_qz(Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>(A_float),
                               Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>(B_float))));

    // Test for double complex matrices
    Eigen::MatrixXcd A_double, B_double;
    generate_random_matrix_pair(dim, A_double, B_double);
    CALL_SUBTEST_4(complex_qz(A_double, B_double));
    CALL_SUBTEST_6((complex_qz(Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>(A_double),
                               Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>(B_double))));
  }
}
