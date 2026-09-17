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
  // Accumulated rounding in the unitary transformations is O(n * epsilon).
  const RealScalar tolerance = RealScalar(128 * a.rows()) * NumTraits<RealScalar>::epsilon();
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY((a - qz.matrixQ() * qz.matrixS() * qz.matrixZ()).norm() <= tolerance * a.norm());
  VERIFY((b - qz.matrixQ() * qz.matrixT() * qz.matrixZ()).norm() <= tolerance * b.norm());
  VERIFY(qz.matrixQ().isUnitary(tolerance));
  VERIFY(qz.matrixZ().isUnitary(tolerance));
  const MatrixType lowerS = qz.matrixS().template triangularView<StrictlyLower>();
  const MatrixType lowerT = qz.matrixT().template triangularView<StrictlyLower>();
  VERIFY(lowerS.norm() <= tolerance * a.norm());
  VERIFY(lowerT.norm() <= tolerance * b.norm());
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

EIGEN_DECLARE_TEST(complex_qz) {
  CALL_SUBTEST_7((complex_qz_exceptional_shift<MatrixXcf>()));
  CALL_SUBTEST_8((complex_qz_exceptional_shift<MatrixXcd>()));
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
