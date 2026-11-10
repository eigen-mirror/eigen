// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

template <typename MatrixType>
void generalized_eigensolver_real(const MatrixType& m) {
  /* this test covers the following files:
     GeneralizedEigenSolver.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  using Scalar = typename MatrixType::Scalar;
  using ComplexScalar = std::complex<Scalar>;
  using VectorType = Matrix<Scalar, MatrixType::RowsAtCompileTime, 1>;

  MatrixType a = MatrixType::Random(rows, cols);
  MatrixType b = MatrixType::Random(rows, cols);
  MatrixType a1 = MatrixType::Random(rows, cols);
  MatrixType b1 = MatrixType::Random(rows, cols);
  MatrixType spdA = a.adjoint() * a + a1.adjoint() * a1;
  MatrixType spdB = b.adjoint() * b + b1.adjoint() * b1;

  // lets compare to GeneralizedSelfAdjointEigenSolver
  {
    GeneralizedSelfAdjointEigenSolver<MatrixType> symmEig(spdA, spdB);
    GeneralizedEigenSolver<MatrixType> eig(spdA, spdB);

    VERIFY_IS_EQUAL(eig.eigenvalues().imag().cwiseAbs().maxCoeff(), 0);

    VectorType realEigenvalues = eig.eigenvalues().real();
    std::sort(realEigenvalues.data(), realEigenvalues.data() + realEigenvalues.size());
    VERIFY_IS_APPROX(realEigenvalues, symmEig.eigenvalues());

    // check eigenvectors
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType D = eig.eigenvalues().asDiagonal();
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType V = eig.eigenvectors();
    VERIFY_IS_APPROX(spdA * V, spdB * V * D);
  }

  // non symmetric case:
  {
    GeneralizedEigenSolver<MatrixType> eig(rows);
    eig.compute(a, b);
    for (Index k = 0; k < cols; ++k) {
      Matrix<ComplexScalar, Dynamic, Dynamic> tmp =
          (eig.betas()(k) * a).template cast<ComplexScalar>() - eig.alphas()(k) * b;
      if (tmp.size() > 1 && tmp.norm() > (std::numeric_limits<Scalar>::min)()) tmp /= tmp.norm();
      VERIFY_IS_MUCH_SMALLER_THAN(std::abs(tmp.determinant()), Scalar(1));
    }
    // check eigenvectors
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType D = eig.eigenvalues().asDiagonal();
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType V = eig.eigenvectors();
    VERIFY_IS_APPROX(a * V, b * V * D);
  }

  // regression test for bug 1098
  {
    GeneralizedSelfAdjointEigenSolver<MatrixType> eig1(a.adjoint() * a, b.adjoint() * b);
    eig1.compute(a.adjoint() * a, b.adjoint() * b);
    GeneralizedEigenSolver<MatrixType> eig2(a.adjoint() * a, b.adjoint() * b);
    eig2.compute(a.adjoint() * a, b.adjoint() * b);
  }

  // check without eigenvectors
  {
    GeneralizedEigenSolver<MatrixType> eig1(spdA, spdB, true);
    GeneralizedEigenSolver<MatrixType> eig2(spdA, spdB, false);
    VERIFY_IS_APPROX(eig1.eigenvalues(), eig2.eigenvalues());
  }
}

template <typename MatrixType>
void generalized_eigensolver_assert() {
  GeneralizedEigenSolver<MatrixType> eig;
  // all raise assert if uninitialized
  VERIFY_RAISES_ASSERT(eig.info());
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  VERIFY_RAISES_ASSERT(eig.eigenvalues());
  VERIFY_RAISES_ASSERT(eig.alphas());
  VERIFY_RAISES_ASSERT(eig.betas());

  // none raise assert after compute called
  eig.compute(MatrixType::Random(20, 20), MatrixType::Random(20, 20));
  VERIFY(eig.info() == Success);
  eig.eigenvectors();
  eig.eigenvalues();
  eig.alphas();
  eig.betas();

  // eigenvectors() raises assert, if eigenvectors were not requested
  eig.compute(MatrixType::Random(20, 20), MatrixType::Random(20, 20), false);
  VERIFY(eig.info() == Success);
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  eig.eigenvalues();
  eig.alphas();
  eig.betas();

  // all except info raise assert if realQZ did not converge
  eig.setMaxIterations(0);  // force real QZ to fail.
  eig.compute(MatrixType::Random(20, 20), MatrixType::Random(20, 20));
  VERIFY(eig.info() == NoConvergence);
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  VERIFY_RAISES_ASSERT(eig.eigenvalues());
  VERIFY_RAISES_ASSERT(eig.alphas());
  VERIFY_RAISES_ASSERT(eig.betas());
}

// A complex pair's alpha and beta came from products such as p * p and T(i, i) * T(i + 1, i + 1), which overflow or
// underflow once |S| |T| or |T|^2 leaves the exponent range, although alpha / beta is representable.
template <typename MatrixType>
void generalized_eigensolver_scaled_complex_pairs() {
  using Scalar = typename MatrixType::Scalar;
  using ComplexScalar = std::complex<Scalar>;
  using ComplexVector = Matrix<ComplexScalar, Dynamic, 1>;
  // QZ and the back substitution are backward stable: eta_k = O(n eps) for every eigenpair.
  const auto verifyComplexEigenpairs = [](const MatrixType& a, const MatrixType& b) {
    GeneralizedEigenSolver<MatrixType> eig(a, b);
    VERIFY_IS_EQUAL(eig.info(), Success);
    VERIFY((eig.eigenvalues().imag().array() != Scalar(0)).all());
    const Scalar tolerance = Scalar(4 * a.rows()) * NumTraits<Scalar>::epsilon();
    const Scalar normA = a.stableNorm(), normB = b.stableNorm();
    for (Index k = 0; k < a.rows(); ++k) {
      // eta_k = |(beta_k A - alpha_k B) v_k| / ((|beta_k| |A| + |alpha_k| |B|) |v_k|).
      const ComplexScalar alpha = eig.alphas()(k);
      const Scalar beta = eig.betas()(k);
      const ComplexVector vk = eig.eigenvectors().col(k);
      const ComplexVector residual =
          (a.template cast<ComplexScalar>() * vk) * beta - (b.template cast<ComplexScalar>() * vk) * alpha;
      const Scalar eta =
          residual.stableNorm() / ((numext::abs(beta) * normA + numext::abs(alpha) * normB) * vk.stableNorm());
      VERIFY(eta <= tolerance);
    }
  };

  const Index dim = 6;
  // A0 = U D V and B0 = U V with D = diag of blocks [c 1; -1 c], so every eigenvalue c +- i is complex.
  MatrixType d = MatrixType::Zero(dim, dim);
  for (Index k = 0; k < dim; k += 2) {
    d(k, k) = d(k + 1, k + 1) = Scalar(k + 1);
    d(k, k + 1) = Scalar(1);
    d(k + 1, k) = Scalar(-1);
  }
  const MatrixType u = MatrixType::Random(dim, dim), v = MatrixType::Random(dim, dim);
  const MatrixType a0 = u * d * v, b0 = u * v;
  // |S| |T| above sqrt(highest) or below sqrt(lowest), and |T|^2 above highest. The squared eigenvalues stay normal,
  // which RealQZ's own test for complex pairs needs.
  const int emax = std::numeric_limits<Scalar>::max_exponent;
  const int exponents[][2] = {{emax / 4 + 4, emax / 4 + 4}, {-emax / 4 - 4, -emax / 4 - 4}, {emax / 16, emax / 2 + 2}};
  for (const auto& e : exponents)
    verifyComplexEigenpairs(a0 * numext::ldexp(Scalar(1), e[0]), b0 * numext::ldexp(Scalar(1), e[1]));

  // A nearly real pair 2^k +- i 2^((k - 10 + m) / 2), B = I. RealQZ scales S by 2^-k, so m01 m10 = 2^(m - k - 10);
  // it keeps the pair while that is nonzero, and the smallest normal keeps it under FTZ as well.
  const int k = 5 * emax / 16, m = k + 10 + std::numeric_limits<Scalar>::min_exponent - 1;
  MatrixType a2(2, 2);
  a2 << numext::ldexp(Scalar(1), k), numext::ldexp(Scalar(1), m), -numext::ldexp(Scalar(1), k - 10),
      numext::ldexp(Scalar(1), k);
  verifyComplexEigenpairs(a2, MatrixType::Identity(2, 2));

  // |S_i| near highest: p = (m00 - m11) / 2 overflowed, and M = S_i diag(b', a') exceeded |S_i| for max(|a'|, |b'|)
  // > 1. |T_i| near highest: a b overflowed, and for t = 3.9 the power of two at or above |T_i|, 2^emax, is not
  // representable. A = 2^eS [x y; -y -x] and B = 2^eT t I have the eigenvalues +- i 2^(eS - eT) sqrt((y - x) (y + x))
  // / t. The back substitution forms beta S - alpha T, which overflows for |S| |T| > highest, so only the eigenvalues
  // are checked.
  const Scalar params[][3] = {{Scalar(0.55), Scalar(0.65), Scalar(1)},
                              {Scalar(0.4), Scalar(0.5), Scalar(1.9)},
                              {Scalar(0.48), Scalar(0.58), Scalar(3.9)}};
  const int pairExponents[][2] = {{emax, 0}, {emax / 2, emax - 2}};
  for (const auto& e : pairExponents) {
    const int eS = e[0], eT = e[1];
    for (const auto& c : params) {
      const Scalar x = c[0], y = c[1], t = c[2];
      MatrixType a3(2, 2);
      a3 << numext::ldexp(x, eS), numext::ldexp(y, eS), -numext::ldexp(y, eS), -numext::ldexp(x, eS);
      GeneralizedEigenSolver<MatrixType> eig(a3, numext::ldexp(t, eT) * MatrixType::Identity(2, 2), false);
      VERIFY_IS_EQUAL(eig.info(), Success);
      // y - x is exact, and the eigenvalues' condition number is about (x^2 + y^2) / (y^2 - x^2) <= 6.1.
      const Scalar exact = numext::ldexp(numext::sqrt((y - x) * (y + x)) / t, eS - eT);
      const Scalar tolerance = Scalar(16) * NumTraits<Scalar>::epsilon() * exact;
      for (Index j = 0; j < 2; ++j) {
        const ComplexScalar lambda = eig.eigenvalues()(j);
        VERIFY(numext::abs(lambda.real()) <= tolerance);
        VERIFY(numext::abs(numext::abs(lambda.imag()) - exact) <= tolerance);
      }
    }
  }
}

// Use an unbounded dynamic matrix below the block Householder threshold.
template <typename MatrixType>
void generalized_eigensolver_no_malloc() {
  const Index dim = 24;
  MatrixType A = MatrixType::Random(dim, dim), B = MatrixType::Random(dim, dim);
  GeneralizedEigenSolver<MatrixType> eig(dim);
  internal::set_is_malloc_allowed(false);
  eig.compute(A, B, false);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(eig.info(), Success);
  const auto alphas = eig.alphas().eval();
  const auto betas = eig.betas().eval();

  for (int i = 0; i < 2; ++i) {
    internal::set_is_malloc_allowed(false);
    eig.compute(A, B, true);
    internal::set_is_malloc_allowed(true);
    VERIFY_IS_EQUAL(eig.info(), Success);
    VERIFY_IS_APPROX(eig.alphas(), alphas);
    VERIFY_IS_APPROX(eig.betas(), betas);
    VERIFY_IS_APPROX(A * eig.eigenvectors() * eig.betas().asDiagonal(),
                     B * eig.eigenvectors() * eig.alphas().asDiagonal());
  }
}

EIGEN_DECLARE_TEST(eigensolver_generalized_real) {
  CALL_SUBTEST_1(generalized_eigensolver_scaled_complex_pairs<MatrixXf>());
  CALL_SUBTEST_1(generalized_eigensolver_scaled_complex_pairs<MatrixXd>());
  CALL_SUBTEST_2(generalized_eigensolver_no_malloc<MatrixXd>());
  CALL_SUBTEST_2((generalized_eigensolver_no_malloc<Matrix<double, Dynamic, Dynamic, RowMajor>>()));
  for (int i = 0; i < g_repeat; i++) {
    int s = 0;
    CALL_SUBTEST_1(generalized_eigensolver_real(Matrix4f()));
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    CALL_SUBTEST_2(generalized_eigensolver_real(MatrixXd(s, s)));

    // some trivial but implementation-wise special cases
    CALL_SUBTEST_2(generalized_eigensolver_real(MatrixXd(1, 1)));
    CALL_SUBTEST_2(generalized_eigensolver_real(MatrixXd(2, 2)));
    CALL_SUBTEST_3(generalized_eigensolver_real(Matrix<double, 1, 1>()));
    CALL_SUBTEST_4(generalized_eigensolver_real(Matrix2d()));
    CALL_SUBTEST_5(generalized_eigensolver_assert<MatrixXd>());
    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }
}
