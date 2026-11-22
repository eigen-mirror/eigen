// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>

template <typename EigType, typename MatType>
void check_eigensolver_for_given_mat(const EigType& eig, const MatType& a) {
  using RealScalar = typename NumTraits<typename MatType::Scalar>::Real;
  using RealVectorType = Matrix<RealScalar, MatType::RowsAtCompileTime, 1>;
  using Complex = std::complex<RealScalar>;
  Index n = a.rows();
  VERIFY_IS_EQUAL(eig.info(), Success);
  VERIFY_IS_APPROX(a * eig.pseudoEigenvectors(), eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix());
  VERIFY_IS_APPROX(a.template cast<Complex>() * eig.eigenvectors(),
                   eig.eigenvectors() * eig.eigenvalues().asDiagonal());
  VERIFY_IS_APPROX(eig.eigenvectors().colwise().norm(), RealVectorType::Ones(n).transpose());
  VERIFY_IS_APPROX(a.eigenvalues(), eig.eigenvalues());
}

template <typename MatrixType>
void eigensolver(const MatrixType& m) {
  /* this test covers the following files:
     EigenSolver.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<RealScalar>;

  MatrixType a = MatrixType::Random(rows, cols);
  MatrixType a1 = MatrixType::Random(rows, cols);
  MatrixType symmA = a.adjoint() * a + a1.adjoint() * a1;

  EigenSolver<MatrixType> ei0(symmA);
  VERIFY_IS_EQUAL(ei0.info(), Success);
  VERIFY_IS_APPROX(symmA * ei0.pseudoEigenvectors(), ei0.pseudoEigenvectors() * ei0.pseudoEigenvalueMatrix());
  VERIFY_IS_APPROX((symmA.template cast<Complex>()) * (ei0.pseudoEigenvectors().template cast<Complex>()),
                   (ei0.pseudoEigenvectors().template cast<Complex>()) * (ei0.eigenvalues().asDiagonal()));

  EigenSolver<MatrixType> ei1(a);
  CALL_SUBTEST(check_eigensolver_for_given_mat(ei1, a));

  // Both solves run in one object through compute(): without static alignment (s390x), packet-loop peeling depends
  // on the operands' addresses, and with a fused pmadd so does the rounding.
  EigenSolver<MatrixType> ei2;
  ei2.compute(a);
  const typename EigenSolver<MatrixType>::EigenvectorsType eivecs = ei2.eigenvectors();
  const typename EigenSolver<MatrixType>::EigenvalueType eivals = ei2.eigenvalues();
  ei2.setMaxIterations(RealSchur<MatrixType>::m_maxIterationsPerRow * rows).compute(a);
  VERIFY_IS_EQUAL(ei2.info(), Success);
  VERIFY_IS_EQUAL(ei2.eigenvectors(), eivecs);
  VERIFY_IS_EQUAL(ei2.eigenvalues(), eivals);
  if (rows > 2) {
    ei2.setMaxIterations(1).compute(a);
    VERIFY_IS_EQUAL(ei2.info(), NoConvergence);
    VERIFY_IS_EQUAL(ei2.getMaxIterations(), 1);
  }

  EigenSolver<MatrixType> eiNoEivecs(a, false);
  VERIFY_IS_EQUAL(eiNoEivecs.info(), Success);
  VERIFY_IS_APPROX(ei1.eigenvalues(), eiNoEivecs.eigenvalues());
  VERIFY_IS_APPROX(ei1.pseudoEigenvalueMatrix(), eiNoEivecs.pseudoEigenvalueMatrix());

  MatrixType id = MatrixType::Identity(rows, cols);
  VERIFY_IS_APPROX(id.operatorNorm(), RealScalar(1));

  if (rows > 2 && rows < 20) {
    // Test matrix with NaN
    a(0, 0) = std::numeric_limits<typename MatrixType::RealScalar>::quiet_NaN();
    EigenSolver<MatrixType> eiNaN(a);
    VERIFY_IS_NOT_EQUAL(eiNaN.info(), Success);
  }

  // regression test for bug 1098
  {
    EigenSolver<MatrixType> eig(a.adjoint() * a);
    eig.compute(a.adjoint() * a);
  }

  // regression test for bug 478
  {
    a.setZero();
    EigenSolver<MatrixType> ei3(a);
    VERIFY_IS_EQUAL(ei3.info(), Success);
    VERIFY_IS_MUCH_SMALLER_THAN(ei3.eigenvalues().norm(), RealScalar(1));
    RealScalar tol = 2 * a.cols() * NumTraits<RealScalar>::epsilon();
    VERIFY((ei3.eigenvectors().adjoint() * ei3.eigenvectors()).eval().isIdentity(tol));
  }
}

template <typename MatrixType>
void eigensolver_verify_assert(const MatrixType& m) {
  EigenSolver<MatrixType> eig;
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  VERIFY_RAISES_ASSERT(eig.pseudoEigenvectors());
  VERIFY_RAISES_ASSERT(eig.pseudoEigenvalueMatrix());
  VERIFY_RAISES_ASSERT(eig.eigenvalues());

  MatrixType a = MatrixType::Random(m.rows(), m.cols());
  eig.compute(a, false);
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  VERIFY_RAISES_ASSERT(eig.pseudoEigenvectors());
}

template <typename CoeffType>
Matrix<typename CoeffType::Scalar, Dynamic, Dynamic> make_companion(const CoeffType& coeffs) {
  Index n = coeffs.size() - 1;
  Matrix<typename CoeffType::Scalar, Dynamic, Dynamic> res(n, n);
  res.setZero();
  res.row(0) = -coeffs.tail(n) / coeffs(0);
  res.diagonal(-1).setOnes();
  return res;
}

template <int>
void eigensolver_generic_extra() {
  {
    // regression test for bug 793
    MatrixXd a(3, 3);
    a << 0, 0, 1, 1, 1, 1, 1, 1e+200, 1;
    Eigen::EigenSolver<MatrixXd> eig(a);
    double scale = 1e-200;  // scale to avoid overflow during the comparisons
    // The eigenvalues are 0 and +-1e100, which depend on entries 1e-200 times the norm of the matrix.
    VERIFY_IS_APPROX(eig.eigenvalues().cwiseAbs().maxCoeff() * scale * 1e100, 1.);
    VERIFY_IS_APPROX(a * eig.pseudoEigenvectors() * scale,
                     eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix() * scale);
    VERIFY_IS_APPROX(a * eig.eigenvectors() * scale, eig.eigenvectors() * eig.eigenvalues().asDiagonal() * scale);
  }
  {
    // check a case where all eigenvalues are null.
    MatrixXd a(2, 2);
    a << 1, 1, -1, -1;
    Eigen::EigenSolver<MatrixXd> eig(a);
    VERIFY_IS_APPROX(eig.pseudoEigenvectors().squaredNorm(), 2.);
    VERIFY_IS_APPROX((a * eig.pseudoEigenvectors()).norm() + 1., 1.);
    VERIFY_IS_APPROX((eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix()).norm() + 1., 1.);
    VERIFY_IS_APPROX((a * eig.eigenvectors()).norm() + 1., 1.);
    VERIFY_IS_APPROX((eig.eigenvectors() * eig.eigenvalues().asDiagonal()).norm() + 1., 1.);
  }

  // regression test for bug 933
  {
    {
      VectorXd coeffs(5);
      coeffs << 1, -3, -175, -225, 2250;
      MatrixXd C = make_companion(coeffs);
      EigenSolver<MatrixXd> eig(C);
      CALL_SUBTEST(check_eigensolver_for_given_mat(eig, C));
    }
    {
      // this test is tricky because it requires high accuracy in smallest eigenvalues
      VectorXd coeffs(5);
      coeffs << 6.154671e-15, -1.003870e-10, -9.819570e-01, 3.995715e+03, 2.211511e+08;
      MatrixXd C = make_companion(coeffs);
      EigenSolver<MatrixXd> eig(C);
      CALL_SUBTEST(check_eigensolver_for_given_mat(eig, C));
      Index n = C.rows();
      for (Index i = 0; i < n; ++i) {
        using Complex = std::complex<double>;
        MatrixXcd ac = C.cast<Complex>();
        ac.diagonal().array() -= eig.eigenvalues()(i);
        VectorXd sv = ac.jacobiSvd().singularValues();
        // comparing to sv(0) is not enough here to catch the "bug",
        // the hard-coded 1.0 is important!
        VERIFY_IS_MUCH_SMALLER_THAN(sv(n - 1), 1.0);
      }
    }
  }
  // regression test for bug 1557
  {
    // this test is interesting because it contains zeros on the diagonal.
    MatrixXd A_bug1557(3, 3);
    A_bug1557 << 0, 0, 0, 1, 0, 0.5887907064808635127, 0, 1, 0;
    EigenSolver<MatrixXd> eig(A_bug1557);
    CALL_SUBTEST(check_eigensolver_for_given_mat(eig, A_bug1557));
  }

  // regression test for bug 1174
  {
    Index n = 12;
    MatrixXf A_bug1174(n, n);
    A_bug1174 << 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0, 786432, 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0,
        786432, 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0, 786432, 262144, 0, 0, 262144, 786432, 0, 0, 0, 0, 0, 0,
        786432, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0,
        262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144,
        262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144,
        262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144,
        262144, 262144, 262144, 262144, 0, 0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0,
        0, 262144, 262144, 0, 0, 262144, 262144, 262144, 262144, 262144, 262144, 0;
    EigenSolver<MatrixXf> eig(A_bug1174);
    CALL_SUBTEST(check_eigensolver_for_given_mat(eig, A_bug1174));
  }
}

// max |A V - V D| / (max|A| max|V|) for the eigenpairs of the matrix A0 scaled by `scale`, computed in the scale-free
// coordinates A0, D / scale, so that the check itself cannot overflow.
template <typename RealScalar>
RealScalar eigensolver_scaled_residual(const Matrix<RealScalar, Dynamic, Dynamic>& A0, RealScalar scale,
                                       const EigenSolver<Matrix<RealScalar, Dynamic, Dynamic>>& eig) {
  using Complex = std::complex<RealScalar>;
  using ComplexMatrix = Matrix<Complex, Dynamic, Dynamic>;
  const ComplexMatrix& V = eig.eigenvectors();
  const ComplexMatrix scaledEigenvalues = (eig.eigenvalues() / scale).matrix();
  const ComplexMatrix residual = A0.template cast<Complex>() * V - V * scaledEigenvalues.col(0).asDiagonal();
  return residual.cwiseAbs().maxCoeff() / (A0.cwiseAbs().maxCoeff() * V.cwiseAbs().maxCoeff());
}

// A real matrix with complex pairs must give finite eigenvectors with a small residual at every scale: the
// back-substitution squares entries of T, which overflows or underflows far outside the normal range of the squares.
template <typename RealScalar>
void eigensolver_scale_sweep() {
  using MatrixType = Matrix<RealScalar, Dynamic, Dynamic>;
  const Index n = 6;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const int minExp = std::numeric_limits<RealScalar>::min_exponent - 1 + std::numeric_limits<RealScalar>::digits / 2;
  const int maxExp = std::numeric_limits<RealScalar>::max_exponent - 6;
  const int step = std::numeric_limits<RealScalar>::digits / 4;
  MatrixType A0 = RealScalar(0.3) * MatrixType::Random(n, n);
  A0.template topLeftCorner<2, 2>() << RealScalar(0), RealScalar(1.5), RealScalar(-1.5), RealScalar(0);
  A0.template block<2, 2>(2, 2) << RealScalar(1), RealScalar(2), RealScalar(-2), RealScalar(1);
  for (int e = minExp; e <= maxExp; e += step) {
    const RealScalar scale = std::ldexp(RealScalar(1), e);
    const MatrixType A = A0 * scale;
    EigenSolver<MatrixType> eig(A);
    VERIFY_IS_EQUAL(eig.info(), Success);
    VERIFY(eig.eigenvectors().allFinite());
    VERIFY((eigensolver_scaled_residual<RealScalar>(A0, scale, eig)) <= RealScalar(100 * n) * eps);
  }
}

// Every eigenvector column has unit norm: a finite, small-residual check alone passes a zero column.
template <typename RealScalar>
void eigensolver_verify_unit_columns(const EigenSolver<Matrix<RealScalar, Dynamic, Dynamic>>& eig) {
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const Index n = eig.eigenvectors().cols();
  for (Index j = 0; j < n; ++j)
    VERIFY(numext::abs(eig.eigenvectors().col(j).norm() - RealScalar(1)) <= RealScalar(10 * n) * eps);
}

// T = [d, g; 0, 2d] has the eigenvectors (1, 0) and (g / d, 1) / norm: the second one has a component g / d that
// overflows when d is tiny, though the normalized vector is representable.
template <typename RealScalar>
void eigensolver_triangular_overflow() {
  using MatrixType = Matrix<RealScalar, Dynamic, Dynamic>;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar d = RealScalar(16) * (std::numeric_limits<RealScalar>::min)();
  const RealScalar g = RealScalar(1024);
  MatrixType A0(2, 2);
  A0 << d, g, RealScalar(0), RealScalar(2) * d;
  EigenSolver<MatrixType> eig(A0);
  VERIFY_IS_EQUAL(eig.info(), Success);
  VERIFY(eig.eigenvectors().allFinite());
  VERIFY((eigensolver_scaled_residual<RealScalar>(A0, RealScalar(1), eig)) <= RealScalar(100) * eps);
  eigensolver_verify_unit_columns<RealScalar>(eig);
}

// A = [a, -1.5 s, 1.5 s; 0, s, 1.5 s; 0, 0, a / 2] with a << s: the back-substitution scales T by about 1/s, which
// makes the divisor a - a / 2 of the third eigenvector subnormal, and the factor |divisor| / |r| that rescales the
// column underflows to 0. The eigenvector (-1, 0, 0) must not be zeroed. sExp and aExp are the binary exponents of
// s and a.
template <typename RealScalar>
void eigensolver_subnormal_divisor(int sExp, int aExp) {
  using MatrixType = Matrix<RealScalar, Dynamic, Dynamic>;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar s = std::ldexp(RealScalar(1), sExp);
  const RealScalar a = std::ldexp(RealScalar(1), aExp);
  MatrixType A(3, 3);
  A << a, RealScalar(-1.5) * s, RealScalar(1.5) * s, RealScalar(0), s, RealScalar(1.5) * s, RealScalar(0),
      RealScalar(0), a / RealScalar(2);
  EigenSolver<MatrixType> eig(A);
  VERIFY_IS_EQUAL(eig.info(), Success);
  VERIFY(eig.eigenvectors().allFinite());
  eigensolver_verify_unit_columns<RealScalar>(eig);
  // A / s is exact: a power-of-two scaling of entries that stay representable.
  MatrixType A0 = A / s;
  VERIFY((eigensolver_scaled_residual<RealScalar>(A0, s, eig)) <= RealScalar(100 * 3) * eps);
}

// A pair of complex eigenvalues whose imaginary part is below 2*eps*|real part| is still a pair: the Schur form has
// the 2x2 block, and the pseudo-eigenvectors hold (Re v, Im v) in two columns. [1, g; -1, 1] has eigenvalues
// 1 +- i sqrt(g) = 1 +- i eps / 2.
template <typename RealScalar>
void eigensolver_tiny_imaginary_pair() {
  using MatrixType = Matrix<RealScalar, Dynamic, Dynamic>;
  using Complex = std::complex<RealScalar>;
  using ComplexMatrix = Matrix<Complex, Dynamic, Dynamic>;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  MatrixType A(2, 2);
  A << RealScalar(1), eps * eps / RealScalar(4), RealScalar(-1), RealScalar(1);
  EigenSolver<MatrixType> eig(A);
  VERIFY_IS_EQUAL(eig.info(), Success);
  VERIFY(numext::imag(eig.eigenvalues()(0)) != RealScalar(0));
  VERIFY(numext::abs(numext::imag(eig.eigenvalues()(0))) < RealScalar(2) * eps);
  const ComplexMatrix V = eig.eigenvectors();
  const RealScalar residual =
      (A.template cast<Complex>() * V - V * eig.eigenvalues().asDiagonal()).cwiseAbs().maxCoeff() /
      V.cwiseAbs().maxCoeff();
  VERIFY(residual <= RealScalar(100) * eps);
  const MatrixType& U = eig.pseudoEigenvectors();
  VERIFY((A * U - U * eig.pseudoEigenvalueMatrix()).cwiseAbs().maxCoeff() <= RealScalar(100) * eps);
}

EIGEN_DECLARE_TEST(eigensolver_generic) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(eigensolver(Matrix4f()));
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    CALL_SUBTEST_2(eigensolver(MatrixXd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    // some trivial but implementation-wise tricky cases
    CALL_SUBTEST_2(eigensolver(MatrixXd(1, 1)));
    CALL_SUBTEST_2(eigensolver(MatrixXd(2, 2)));
    CALL_SUBTEST_3(eigensolver(Matrix<double, 1, 1>()));
    CALL_SUBTEST_4(eigensolver(Matrix2d()));
    CALL_SUBTEST_6((eigensolver(Matrix<float, Dynamic, Dynamic, RowMajor>(17, 17))));
    CALL_SUBTEST_7((eigensolver(Matrix<double, Dynamic, Dynamic, RowMajor>(17, 17))));
  }

  CALL_SUBTEST_1(eigensolver_verify_assert(Matrix4f()));
  s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
  CALL_SUBTEST_2(eigensolver_verify_assert(MatrixXd(s, s)));
  CALL_SUBTEST_3(eigensolver_verify_assert(Matrix<double, 1, 1>()));
  CALL_SUBTEST_4(eigensolver_verify_assert(Matrix2d()));

  // Test problem size constructors
  CALL_SUBTEST_5(EigenSolver<MatrixXf> tmp(s));

  // regression test for bug 410
  CALL_SUBTEST_2({
    MatrixXd A(1, 1);
    A(0, 0) = std::sqrt(-1.);  // is Not-a-Number
    Eigen::EigenSolver<MatrixXd> solver(A);
    VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
  });

  // RealSchur took the maximum of no coefficients for an empty matrix.
  CALL_SUBTEST_2({
    MatrixXd A(0, 0);
    Eigen::EigenSolver<MatrixXd> solver(A);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_EQUAL(solver.eigenvectors().size(), 0);
    VERIFY_IS_EQUAL(A.eigenvalues().size(), 0);
  });

  CALL_SUBTEST_2(eigensolver_generic_extra<0>());

  CALL_SUBTEST_5(eigensolver_scale_sweep<float>());
  CALL_SUBTEST_5(eigensolver_scale_sweep<double>());
  CALL_SUBTEST_5(eigensolver_triangular_overflow<float>());
  CALL_SUBTEST_5(eigensolver_triangular_overflow<double>());
  CALL_SUBTEST_5(eigensolver_subnormal_divisor<float>(75, -74));
  CALL_SUBTEST_5(eigensolver_subnormal_divisor<double>(600, -473));
  CALL_SUBTEST_5(eigensolver_tiny_imaginary_pair<float>());
  CALL_SUBTEST_5(eigensolver_tiny_imaginary_pair<double>());

  TEST_SET_BUT_UNUSED_VARIABLE(s);
}
