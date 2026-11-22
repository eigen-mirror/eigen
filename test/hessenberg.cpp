// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Eigenvalues>

template <typename Scalar, int Size>
void hessenberg(int size = Size) {
  using MatrixType = Matrix<Scalar, Size, Size>;

  // Test basic functionality: A = U H U* and H is Hessenberg
  for (int counter = 0; counter < g_repeat; ++counter) {
    MatrixType m = MatrixType::Random(size, size);
    HessenbergDecomposition<MatrixType> hess(m);
    MatrixType Q = hess.matrixQ();
    MatrixType H = hess.matrixH();
    VERIFY_IS_APPROX(m, Q * H * Q.adjoint());
    for (int row = 2; row < size; ++row) {
      for (int col = 0; col < row - 1; ++col) {
        VERIFY(H(row, col) == (typename MatrixType::Scalar)0);
      }
    }
  }

  // Test whether compute() and constructor returns same result
  MatrixType A = MatrixType::Random(size, size);
  HessenbergDecomposition<MatrixType> cs1;
  cs1.compute(A);
  HessenbergDecomposition<MatrixType> cs2(A);
  VERIFY_IS_EQUAL(cs1.matrixH().eval(), cs2.matrixH().eval());
  MatrixType cs1Q = cs1.matrixQ();
  MatrixType cs2Q = cs2.matrixQ();
  VERIFY_IS_EQUAL(cs1Q, cs2Q);

  // Test assertions for when used uninitialized
  HessenbergDecomposition<MatrixType> hessUninitialized;
  VERIFY_RAISES_ASSERT(hessUninitialized.matrixH());
  VERIFY_RAISES_ASSERT(hessUninitialized.matrixQ());
  VERIFY_RAISES_ASSERT(hessUninitialized.householderCoefficients());
  VERIFY_RAISES_ASSERT(hessUninitialized.packedMatrix());

  // TODO: Add tests for packedMatrix() and householderCoefficients()
}

// A = Q H Q^* must hold to working accuracy whatever the scale of A: the reflectors are formed from ratios, so
// neither a column whose entries are all far below sqrt(min()) nor one near overflow may be treated as negligible.
template <typename Scalar>
void hessenberg_scale_sweep() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = 6;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const int minExp = std::numeric_limits<RealScalar>::min_exponent - 1 + std::numeric_limits<RealScalar>::digits / 2;
  const int maxExp = std::numeric_limits<RealScalar>::max_exponent - 6;
  const int step = std::numeric_limits<RealScalar>::digits / 4;
  for (int e = minExp; e <= maxExp; e += step) {
    const RealScalar scale = std::ldexp(RealScalar(1), e);
    MatrixType unit = MatrixType::Random(n, n);
    // A tail far below sqrt(min()) relative to the rest of the matrix is negligible, at any scale of the matrix.
    if (e % 2 == 0) unit.col(1).tail(n - 3) *= Scalar(std::sqrt((std::numeric_limits<RealScalar>::min)()) / 1000);
    MatrixType A = unit * Scalar(scale);
    HessenbergDecomposition<MatrixType> hess(A);
    MatrixType Q = hess.matrixQ();
    MatrixType Hscaled = MatrixType(hess.matrixH()) / Scalar(scale);
    // |Q H Q^* - A| / |A|, entrywise relative to the largest entry; backward error ~ n * eps.
    const RealScalar err = (Q * Hscaled * Q.adjoint() - unit).cwiseAbs().maxCoeff() / unit.cwiseAbs().maxCoeff();
    VERIFY(err <= RealScalar(10 * n) * eps);
    const RealScalar orth = (Q * Q.adjoint() - MatrixType::Identity(n, n)).cwiseAbs().maxCoeff();
    VERIFY(orth <= RealScalar(10 * n) * eps);
  }
}

// A = diag(s B1, B2) with s*sqrt(min()) above the entries of B2: a column of the trailing block is negligible against
// max|A| but not against the entries the reflector touches. The trailing block of H must keep the eigenvalues of B2,
// which |Q H Q^* - A| / max|A| cannot show: dropping B2's tails changes B2 by O(1) and A by O(1/s).
template <typename Scalar>
void hessenberg_blockdiagonal() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using ComplexMatrix = Matrix<std::complex<RealScalar>, Dynamic, Dynamic>;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  for (int e :
       {std::numeric_limits<RealScalar>::max_exponent * 5 / 8, std::numeric_limits<RealScalar>::max_exponent * 7 / 8}) {
    MatrixType B1 = MatrixType::Random(3, 3), B2 = MatrixType::Random(3, 3);
    MatrixType A = MatrixType::Zero(6, 6);
    A.topLeftCorner(3, 3) = B1 * Scalar(std::ldexp(RealScalar(1), e));
    A.bottomRightCorner(3, 3) = B2;
    HessenbergDecomposition<MatrixType> hess(A);
    const MatrixType H = hess.matrixH();
    const MatrixType Q = hess.matrixQ();
    VERIFY((Q * H * Q.adjoint() - A).cwiseAbs().maxCoeff() <= RealScalar(60) * eps * A.cwiseAbs().maxCoeff());
    const ComplexMatrix trailing = H.bottomRightCorner(3, 3).template cast<std::complex<RealScalar>>();
    const auto computed = ComplexEigenSolver<ComplexMatrix>(trailing, false).eigenvalues();
    const auto expected =
        ComplexEigenSolver<ComplexMatrix>(B2.template cast<std::complex<RealScalar>>(), false).eigenvalues();
    for (Index i = 0; i < 3; ++i) {
      RealScalar nearest = (std::numeric_limits<RealScalar>::max)();
      for (Index j = 0; j < 3; ++j) nearest = numext::mini(nearest, RealScalar(numext::abs(computed(i) - expected(j))));
      VERIFY(nearest <= RealScalar(100) * eps * B2.cwiseAbs().maxCoeff());
    }
  }
}

EIGEN_DECLARE_TEST(hessenberg) {
  CALL_SUBTEST_1((hessenberg<std::complex<double>, 1>()));
  CALL_SUBTEST_2((hessenberg<std::complex<double>, 2>()));
  CALL_SUBTEST_3((hessenberg<std::complex<float>, 4>()));
  CALL_SUBTEST_4((hessenberg<float, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  CALL_SUBTEST_5((hessenberg<std::complex<double>, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));

  // Test problem size constructors
  CALL_SUBTEST_6(HessenbergDecomposition<MatrixXf>(10));

  CALL_SUBTEST_7(hessenberg_scale_sweep<float>());
  CALL_SUBTEST_7(hessenberg_scale_sweep<double>());
  CALL_SUBTEST_8(hessenberg_scale_sweep<std::complex<float>>());
  CALL_SUBTEST_8(hessenberg_scale_sweep<std::complex<double>>());
  CALL_SUBTEST_7(hessenberg_blockdiagonal<float>());
  CALL_SUBTEST_7(hessenberg_blockdiagonal<double>());
  CALL_SUBTEST_8(hessenberg_blockdiagonal<std::complex<float>>());
  CALL_SUBTEST_8(hessenberg_blockdiagonal<std::complex<double>>());
}
