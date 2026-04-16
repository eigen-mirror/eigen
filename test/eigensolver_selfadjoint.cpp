// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include "svd_fill.h"
#include "tridiag_test_matrices.h"
#include <limits>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/MatrixFunctions>

template <typename MatrixType>
void selfadjointeigensolver_essential_check(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  RealScalar eival_eps =
      numext::mini<RealScalar>(test_precision<RealScalar>(), NumTraits<Scalar>::dummy_precision() * 20000);

  SelfAdjointEigenSolver<MatrixType> eiSymm(m);
  VERIFY_IS_EQUAL(eiSymm.info(), Success);

  Index n = m.cols();
  RealScalar scaling = m.cwiseAbs().maxCoeff();

  if (scaling < (std::numeric_limits<RealScalar>::min)()) {
    VERIFY(eiSymm.eigenvalues().cwiseAbs().maxCoeff() <= (std::numeric_limits<RealScalar>::min)());
  } else {
    VERIFY_IS_APPROX((m.template selfadjointView<Lower>() * eiSymm.eigenvectors()) / scaling,
                     (eiSymm.eigenvectors() * eiSymm.eigenvalues().asDiagonal()) / scaling);
  }
  VERIFY_IS_APPROX(m.template selfadjointView<Lower>().eigenvalues(), eiSymm.eigenvalues());

  // Eigenvectors must be unitary. Use a tolerance proportional to n*epsilon,
  // which is the expected rounding error for Householder-based orthogonal transformations.
  RealScalar unitary_tol = RealScalar(4) * RealScalar(numext::maxi(Index(1), n)) * NumTraits<RealScalar>::epsilon();
  // But don't go below the test_precision floor (matters for float).
  unitary_tol = numext::maxi(unitary_tol, test_precision<RealScalar>());
  VERIFY(eiSymm.eigenvectors().isUnitary(unitary_tol));

  // Verify eigenvalues are sorted in non-decreasing order.
  for (Index i = 1; i < n; ++i) {
    VERIFY(eiSymm.eigenvalues()(i) >= eiSymm.eigenvalues()(i - 1));
  }

  if (m.cols() <= 4) {
    SelfAdjointEigenSolver<MatrixType> eiDirect;
    eiDirect.computeDirect(m);
    VERIFY_IS_EQUAL(eiDirect.info(), Success);
    if (!eiSymm.eigenvalues().isApprox(eiDirect.eigenvalues(), eival_eps)) {
      std::cerr << "reference eigenvalues: " << eiSymm.eigenvalues().transpose() << "\n"
                << "obtained eigenvalues:  " << eiDirect.eigenvalues().transpose() << "\n"
                << "diff:                  " << (eiSymm.eigenvalues() - eiDirect.eigenvalues()).transpose() << "\n"
                << "error (eps):           "
                << (eiSymm.eigenvalues() - eiDirect.eigenvalues()).norm() / eiSymm.eigenvalues().norm() << "  ("
                << eival_eps << ")\n";
    }
    if (scaling < (std::numeric_limits<RealScalar>::min)()) {
      VERIFY(eiDirect.eigenvalues().cwiseAbs().maxCoeff() <= (std::numeric_limits<RealScalar>::min)());
    } else {
      VERIFY_IS_APPROX(eiSymm.eigenvalues() / scaling, eiDirect.eigenvalues() / scaling);
      VERIFY_IS_APPROX((m.template selfadjointView<Lower>() * eiDirect.eigenvectors()) / scaling,
                       (eiDirect.eigenvectors() * eiDirect.eigenvalues().asDiagonal()) / scaling);
      VERIFY_IS_APPROX(m.template selfadjointView<Lower>().eigenvalues() / scaling, eiDirect.eigenvalues() / scaling);
    }

    // Direct solver eigenvectors must also be unitary.
    VERIFY(eiDirect.eigenvectors().isUnitary(unitary_tol));

    // Direct solver eigenvalues must also be sorted.
    for (Index i = 1; i < n; ++i) {
      VERIFY(eiDirect.eigenvalues()(i) >= eiDirect.eigenvalues()(i - 1));
    }
  }
}

template <typename MatrixType>
void selfadjointeigensolver(const MatrixType& m) {
  /* this test covers the following files:
     EigenSolver.h, SelfAdjointEigenSolver.h (and indirectly: Tridiagonalization.h)
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  RealScalar largerEps = 10 * test_precision<RealScalar>();

  MatrixType a = MatrixType::Random(rows, cols);
  MatrixType a1 = MatrixType::Random(rows, cols);
  MatrixType symmA = a.adjoint() * a + a1.adjoint() * a1;
  MatrixType symmC = symmA;

  svd_fill_random(symmA, SelfAdjoint);

  symmA.template triangularView<StrictlyUpper>().setZero();
  symmC.template triangularView<StrictlyUpper>().setZero();

  MatrixType b = MatrixType::Random(rows, cols);
  MatrixType b1 = MatrixType::Random(rows, cols);
  MatrixType symmB = b.adjoint() * b + b1.adjoint() * b1;
  symmB.template triangularView<StrictlyUpper>().setZero();

  CALL_SUBTEST(selfadjointeigensolver_essential_check(symmA));

  SelfAdjointEigenSolver<MatrixType> eiSymm(symmA);
  // generalized eigen pb
  GeneralizedSelfAdjointEigenSolver<MatrixType> eiSymmGen(symmC, symmB);

  SelfAdjointEigenSolver<MatrixType> eiSymmNoEivecs(symmA, false);
  VERIFY_IS_EQUAL(eiSymmNoEivecs.info(), Success);
  VERIFY_IS_APPROX(eiSymm.eigenvalues(), eiSymmNoEivecs.eigenvalues());

  // generalized eigen problem Ax = lBx
  eiSymmGen.compute(symmC, symmB, Ax_lBx);
  VERIFY_IS_EQUAL(eiSymmGen.info(), Success);
  VERIFY((symmC.template selfadjointView<Lower>() * eiSymmGen.eigenvectors())
             .isApprox(symmB.template selfadjointView<Lower>() *
                           (eiSymmGen.eigenvectors() * eiSymmGen.eigenvalues().asDiagonal()),
                       largerEps));

  // generalized eigen problem BAx = lx
  eiSymmGen.compute(symmC, symmB, BAx_lx);
  VERIFY_IS_EQUAL(eiSymmGen.info(), Success);
  VERIFY(
      (symmB.template selfadjointView<Lower>() * (symmC.template selfadjointView<Lower>() * eiSymmGen.eigenvectors()))
          .isApprox((eiSymmGen.eigenvectors() * eiSymmGen.eigenvalues().asDiagonal()), largerEps));

  // generalized eigen problem ABx = lx
  eiSymmGen.compute(symmC, symmB, ABx_lx);
  VERIFY_IS_EQUAL(eiSymmGen.info(), Success);
  VERIFY(
      (symmC.template selfadjointView<Lower>() * (symmB.template selfadjointView<Lower>() * eiSymmGen.eigenvectors()))
          .isApprox((eiSymmGen.eigenvectors() * eiSymmGen.eigenvalues().asDiagonal()), largerEps));

  eiSymm.compute(symmC);
  MatrixType sqrtSymmA = eiSymm.operatorSqrt();
  VERIFY_IS_APPROX(MatrixType(symmC.template selfadjointView<Lower>()), sqrtSymmA * sqrtSymmA);
  VERIFY_IS_APPROX(sqrtSymmA, symmC.template selfadjointView<Lower>() * eiSymm.operatorInverseSqrt());

  MatrixType id = MatrixType::Identity(rows, cols);
  VERIFY_IS_APPROX(id.template selfadjointView<Lower>().operatorNorm(), RealScalar(1));

  SelfAdjointEigenSolver<MatrixType> eiSymmUninitialized;
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.info());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.eigenvalues());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.eigenvectors());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorSqrt());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorInverseSqrt());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorExp());

  eiSymmUninitialized.compute(symmA, false);
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.eigenvectors());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorSqrt());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorInverseSqrt());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorExp());

  // test Tridiagonalization's methods
  Tridiagonalization<MatrixType> tridiag(symmC);
  VERIFY_IS_APPROX(tridiag.diagonal(), tridiag.matrixT().diagonal());
  VERIFY_IS_APPROX(tridiag.subDiagonal(), tridiag.matrixT().template diagonal<-1>());
  Matrix<RealScalar, Dynamic, Dynamic> T = tridiag.matrixT();
  if (rows > 2) {
    // Verify that the tridiagonal matrix is actually tridiagonal (zero outside the three central diagonals).
    for (Index i = 0; i < rows; ++i) {
      for (Index j = 0; j < cols; ++j) {
        if (numext::abs(i - j) > 1) {
          VERIFY(numext::is_exactly_zero(T(i, j)));
        }
      }
    }
  }
  VERIFY_IS_APPROX(tridiag.diagonal(), T.diagonal());
  VERIFY_IS_APPROX(tridiag.subDiagonal(), T.template diagonal<1>());
  VERIFY_IS_APPROX(MatrixType(symmC.template selfadjointView<Lower>()),
                   tridiag.matrixQ() * tridiag.matrixT().eval() * MatrixType(tridiag.matrixQ()).adjoint());
  VERIFY_IS_APPROX(MatrixType(symmC.template selfadjointView<Lower>()),
                   tridiag.matrixQ() * tridiag.matrixT() * tridiag.matrixQ().adjoint());

  // Test computation of eigenvalues from tridiagonal matrix
  if (rows > 1) {
    SelfAdjointEigenSolver<MatrixType> eiSymmTridiag;
    eiSymmTridiag.computeFromTridiagonal(tridiag.matrixT().diagonal(), tridiag.matrixT().diagonal(-1),
                                         ComputeEigenvectors);
    VERIFY_IS_APPROX(eiSymm.eigenvalues(), eiSymmTridiag.eigenvalues());
    VERIFY_IS_APPROX(tridiag.matrixT(), eiSymmTridiag.eigenvectors().real() * eiSymmTridiag.eigenvalues().asDiagonal() *
                                            eiSymmTridiag.eigenvectors().real().transpose());
  }

  // Test matrix exponential from eigendecomposition.
  // First scale to avoid overflow.
  symmB = symmB / symmB.norm();
  eiSymm.compute(symmB);
  MatrixType expSymmB = eiSymm.operatorExp();
  symmB = symmB.template selfadjointView<Lower>();
  VERIFY_IS_APPROX(expSymmB, symmB.exp());

  if (rows > 1 && rows < 20) {
    // Test matrix with NaN
    symmC(0, 0) = std::numeric_limits<typename MatrixType::RealScalar>::quiet_NaN();
    SelfAdjointEigenSolver<MatrixType> eiSymmNaN(symmC);
    VERIFY_IS_EQUAL(eiSymmNaN.info(), NoConvergence);
  }

  // regression test for bug 1098
  {
    SelfAdjointEigenSolver<MatrixType> eig(a.adjoint() * a);
    eig.compute(a.adjoint() * a);
  }

  // regression test for bug 478
  {
    a.setZero();
    SelfAdjointEigenSolver<MatrixType> ei3(a);
    VERIFY_IS_EQUAL(ei3.info(), Success);
    VERIFY_IS_MUCH_SMALLER_THAN(ei3.eigenvalues().norm(), RealScalar(1));
    RealScalar tol = 2 * a.cols() * NumTraits<RealScalar>::epsilon();
    VERIFY((ei3.eigenvectors().adjoint() * ei3.eigenvectors()).eval().isIdentity(tol));
  }
}

// Test matrices with exact eigenvalue multiplicities.
template <typename MatrixType>
void selfadjointeigensolver_repeated_eigenvalues(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  Index n = m.rows();
  if (n < 2) return;

  // Create a random unitary matrix via QR.
  MatrixType q = MatrixType::Random(n, n);
  HouseholderQR<MatrixType> qr(q);
  q = qr.householderQ();

  // All eigenvalues equal (scalar multiple of identity).
  {
    RealScalar lambda = internal::random<RealScalar>(-10, 10);
    MatrixType A = lambda * MatrixType::Identity(n, n);
    selfadjointeigensolver_essential_check(A);
  }

  // Eigenvalue of multiplicity n-1 (one distinct, rest equal).
  {
    Matrix<RealScalar, Dynamic, 1> d = Matrix<RealScalar, Dynamic, 1>::Constant(n, RealScalar(3));
    d(0) = RealScalar(-2);
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();
    selfadjointeigensolver_essential_check(A);
  }

  // Two clusters: first half one value, second half another.
  if (n >= 4) {
    Matrix<RealScalar, Dynamic, 1> d(n);
    for (Index i = 0; i < n / 2; ++i) d(i) = RealScalar(1);
    for (Index i = n / 2; i < n; ++i) d(i) = RealScalar(5);
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();
    selfadjointeigensolver_essential_check(A);
  }

  // Nearly repeated eigenvalues: separated by O(epsilon).
  {
    Matrix<RealScalar, Dynamic, 1> d(n);
    for (Index i = 0; i < n; ++i) {
      d(i) = RealScalar(1) + RealScalar(i) * NumTraits<RealScalar>::epsilon() * RealScalar(10);
    }
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();
    selfadjointeigensolver_essential_check(A);
  }
}

// Test matrices with extreme condition numbers and eigenvalue ranges.
template <typename MatrixType>
void selfadjointeigensolver_extreme_eigenvalues(const MatrixType& m) {
  using std::pow;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  Index n = m.rows();
  if (n < 2) return;

  // Create a random unitary matrix.
  MatrixType q = MatrixType::Random(n, n);
  HouseholderQR<MatrixType> qr(q);
  q = qr.householderQ();

  // Eigenvalues spanning many orders of magnitude (high condition number).
  {
    RealScalar maxExp = RealScalar(std::numeric_limits<RealScalar>::max_exponent10) / RealScalar(4);
    Matrix<RealScalar, Dynamic, 1> d(n);
    for (Index i = 0; i < n; ++i) {
      RealScalar exponent = -maxExp + RealScalar(2) * maxExp * RealScalar(i) / RealScalar(n - 1);
      d(i) = pow(RealScalar(10), exponent);
    }
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();

    SelfAdjointEigenSolver<MatrixType> eig(A);
    VERIFY_IS_EQUAL(eig.info(), Success);
    // For ill-conditioned matrices we can only check the relative residual.
    // ||A*V - V*D|| / ||A|| should be O(n * epsilon).
    RealScalar Anorm = A.template selfadjointView<Lower>().operatorNorm();
    if (Anorm > (std::numeric_limits<RealScalar>::min)()) {
      MatrixType residual = A.template selfadjointView<Lower>() * eig.eigenvectors() -
                            eig.eigenvectors() * eig.eigenvalues().asDiagonal();
      RealScalar rel_err = residual.norm() / Anorm;
      RealScalar tol = RealScalar(4) * RealScalar(n) * NumTraits<RealScalar>::epsilon();
      VERIFY(rel_err <= tol);
    }
    // Eigenvalues must still be sorted.
    for (Index i = 1; i < n; ++i) {
      VERIFY(eig.eigenvalues()(i) >= eig.eigenvalues()(i - 1));
    }
  }

  // Very tiny eigenvalues (near underflow).
  {
    RealScalar tiny = (std::numeric_limits<RealScalar>::min)() * RealScalar(100);
    Matrix<RealScalar, Dynamic, 1> d(n);
    for (Index i = 0; i < n; ++i) {
      d(i) = tiny * (RealScalar(1) + RealScalar(i));
    }
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();
    selfadjointeigensolver_essential_check(A);
  }

  // Very large eigenvalues (near overflow).
  {
    RealScalar huge = (std::numeric_limits<RealScalar>::max)() / (RealScalar(n) * RealScalar(100));
    Matrix<RealScalar, Dynamic, 1> d(n);
    for (Index i = 0; i < n; ++i) {
      d(i) = huge * (RealScalar(1) + RealScalar(i) * RealScalar(0.01));
    }
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();
    selfadjointeigensolver_essential_check(A);
  }

  // Mix of positive and negative eigenvalues.
  {
    Matrix<RealScalar, Dynamic, 1> d(n);
    for (Index i = 0; i < n; ++i) {
      d(i) = (i % 2 == 0) ? RealScalar(i + 1) : RealScalar(-(i + 1));
    }
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();
    selfadjointeigensolver_essential_check(A);
  }

  // One zero eigenvalue among non-zero ones (rank-deficient).
  {
    Matrix<RealScalar, Dynamic, 1> d = Matrix<RealScalar, Dynamic, 1>::LinSpaced(n, RealScalar(0), RealScalar(n - 1));
    MatrixType A = (q * d.template cast<Scalar>().asDiagonal() * q.adjoint()).eval();
    A.template triangularView<StrictlyUpper>().setZero();
    selfadjointeigensolver_essential_check(A);
  }
}

// Test computeFromTridiagonal with scaled inputs (regression for missing scaling).
template <typename MatrixType>
void selfadjointeigensolver_tridiagonal_scaled(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  Index n = m.rows();
  if (n < 2) return;

  // Create a tridiagonal matrix with large entries.
  typedef Matrix<RealScalar, Dynamic, 1> RealVectorType;
  RealVectorType diag(n), subdiag(n - 1);

  // Case 1: Large values.
  RealScalar scale = (std::numeric_limits<RealScalar>::max)() / (RealScalar(n) * RealScalar(100));
  for (Index i = 0; i < n; ++i) diag(i) = scale * RealScalar(i + 1);
  for (Index i = 0; i < n - 1; ++i) subdiag(i) = scale * RealScalar(0.5);

  SelfAdjointEigenSolver<MatrixType> eig1;
  eig1.computeFromTridiagonal(diag, subdiag, ComputeEigenvectors);
  VERIFY_IS_EQUAL(eig1.info(), Success);

  // Reconstruct tridiagonal and check residual.
  Matrix<RealScalar, Dynamic, Dynamic> T = Matrix<RealScalar, Dynamic, Dynamic>::Zero(n, n);
  T.diagonal() = diag;
  T.template diagonal<1>() = subdiag;
  T.template diagonal<-1>() = subdiag;
  VERIFY_IS_APPROX(
      T, eig1.eigenvectors().real() * eig1.eigenvalues().asDiagonal() * eig1.eigenvectors().real().transpose());

  // Case 2: Tiny values.
  scale = (std::numeric_limits<RealScalar>::min)() * RealScalar(100);
  for (Index i = 0; i < n; ++i) diag(i) = scale * RealScalar(i + 1);
  for (Index i = 0; i < n - 1; ++i) subdiag(i) = scale * RealScalar(0.5);

  SelfAdjointEigenSolver<MatrixType> eig2;
  eig2.computeFromTridiagonal(diag, subdiag, ComputeEigenvectors);
  VERIFY_IS_EQUAL(eig2.info(), Success);

  // Eigenvalues-only mode should produce the same eigenvalues.
  SelfAdjointEigenSolver<MatrixType> eig2v;
  eig2v.computeFromTridiagonal(diag, subdiag, EigenvaluesOnly);
  VERIFY_IS_EQUAL(eig2v.info(), Success);
  VERIFY_IS_APPROX(eig2.eigenvalues(), eig2v.eigenvalues());
}

// Test computeFromTridiagonal with structured hard-case matrices from the literature.
template <typename RealScalar>
void selfadjointeigensolver_structured_tridiagonal() {
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixType;

  test::for_all_symmetric_tridiag_test_matrices<RealScalar>([](const auto& diag, const auto& offdiag) {
    Index n = diag.size();

    // Build the full symmetric tridiagonal matrix for residual checking.
    MatrixType T = MatrixType::Zero(n, n);
    T.diagonal() = diag;
    if (n > 1) {
      T.template diagonal<1>() = offdiag;
      T.template diagonal<-1>() = offdiag;
    }
    RealScalar Tnorm = T.cwiseAbs().maxCoeff();

    // Test with eigenvectors.
    SelfAdjointEigenSolver<MatrixType> eig;
    eig.computeFromTridiagonal(diag, offdiag, ComputeEigenvectors);
    VERIFY_IS_EQUAL(eig.info(), Success);

    // Eigenvalues must be sorted.
    for (Index i = 1; i < n; ++i) {
      VERIFY(eig.eigenvalues()(i) >= eig.eigenvalues()(i - 1));
    }

    // Eigenvectors must be orthonormal.
    RealScalar unitary_tol =
        numext::maxi(RealScalar(4) * RealScalar(n) * NumTraits<RealScalar>::epsilon(), test_precision<RealScalar>());
    VERIFY(eig.eigenvectors().isUnitary(unitary_tol));

    // Residual check: ||T*V - V*D||_F / ||T||_F should be O(n*eps).
    // Scale T to avoid overflow in the matrix product when entries span extreme ranges.
    RealScalar Tnorm_F = T.norm();
    if (Tnorm_F > (std::numeric_limits<RealScalar>::min)()) {
      MatrixType Tscaled = T / Tnorm;
      MatrixType residual =
          Tscaled * eig.eigenvectors() - eig.eigenvectors() * (eig.eigenvalues() / Tnorm).asDiagonal();
      RealScalar rel_err = residual.norm() / Tscaled.norm();
      VERIFY(rel_err <= RealScalar(8) * RealScalar(n) * NumTraits<RealScalar>::epsilon());
    }

    // Eigenvalues-only mode must produce the same eigenvalues.
    SelfAdjointEigenSolver<MatrixType> eig_vals;
    eig_vals.computeFromTridiagonal(diag, offdiag, EigenvaluesOnly);
    VERIFY_IS_EQUAL(eig_vals.info(), Success);
    if (Tnorm > (std::numeric_limits<RealScalar>::min)()) {
      VERIFY_IS_APPROX(eig.eigenvalues() / Tnorm, eig_vals.eigenvalues() / Tnorm);
    }
  });
}

// Test with diagonal matrices (tridiagonalization is trivial).
template <typename MatrixType>
void selfadjointeigensolver_diagonal(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  Index n = m.rows();

  // Random diagonal matrix.
  MatrixType diag = MatrixType::Zero(n, n);
  for (Index i = 0; i < n; ++i) {
    diag(i, i) = internal::random<RealScalar>(-100, 100);
  }
  selfadjointeigensolver_essential_check(diag);

  // The eigenvalues should be the diagonal entries, sorted.
  SelfAdjointEigenSolver<MatrixType> eig(diag);
  VERIFY_IS_EQUAL(eig.info(), Success);

  Matrix<RealScalar, Dynamic, 1> expected_evals(n);
  for (Index i = 0; i < n; ++i) expected_evals(i) = numext::real(diag(i, i));
  std::sort(expected_evals.data(), expected_evals.data() + n);
  VERIFY_IS_APPROX(eig.eigenvalues(), expected_evals);
}

// Test operatorInverseSqrt more thoroughly.
template <typename MatrixType>
void selfadjointeigensolver_inverse_sqrt(const MatrixType& m) {
  Index n = m.rows();
  if (n < 1) return;

  // Create a positive-definite matrix.
  MatrixType a = MatrixType::Random(n, n);
  MatrixType spd = a.adjoint() * a + MatrixType::Identity(n, n);
  spd.template triangularView<StrictlyUpper>().setZero();

  SelfAdjointEigenSolver<MatrixType> eig(spd);
  VERIFY_IS_EQUAL(eig.info(), Success);

  MatrixType sqrtA = eig.operatorSqrt();
  MatrixType invSqrtA = eig.operatorInverseSqrt();

  // sqrtA * invSqrtA should be identity.
  VERIFY_IS_APPROX(sqrtA * invSqrtA, MatrixType::Identity(n, n));

  // invSqrtA * A * invSqrtA should be identity.
  VERIFY_IS_APPROX(invSqrtA * spd.template selfadjointView<Lower>() * invSqrtA, MatrixType::Identity(n, n));

  // invSqrtA should be symmetric/selfadjoint.
  VERIFY_IS_APPROX(invSqrtA, invSqrtA.adjoint());
}

// Test that RowMajor matrices work correctly with computeDirect.
template <int>
void selfadjointeigensolver_rowmajor() {
  typedef Matrix<double, 3, 3, RowMajor> RowMajorMatrix3d;
  typedef Matrix<double, 2, 2, RowMajor> RowMajorMatrix2d;
  typedef Matrix<float, 3, 3, RowMajor> RowMajorMatrix3f;
  typedef Matrix<float, 2, 2, RowMajor> RowMajorMatrix2f;

  // 3x3 RowMajor double
  {
    RowMajorMatrix3d a = RowMajorMatrix3d::Random();
    RowMajorMatrix3d symmA = a.transpose() * a;
    SelfAdjointEigenSolver<RowMajorMatrix3d> eig;
    eig.computeDirect(symmA);
    VERIFY_IS_EQUAL(eig.info(), Success);
    // Compare with iterative solver.
    SelfAdjointEigenSolver<RowMajorMatrix3d> eigRef(symmA);
    VERIFY_IS_APPROX(eigRef.eigenvalues(), eig.eigenvalues());
  }

  // 2x2 RowMajor double
  {
    RowMajorMatrix2d a = RowMajorMatrix2d::Random();
    RowMajorMatrix2d symmA = a.transpose() * a;
    SelfAdjointEigenSolver<RowMajorMatrix2d> eig;
    eig.computeDirect(symmA);
    VERIFY_IS_EQUAL(eig.info(), Success);
    SelfAdjointEigenSolver<RowMajorMatrix2d> eigRef(symmA);
    VERIFY_IS_APPROX(eigRef.eigenvalues(), eig.eigenvalues());
  }

  // 3x3 RowMajor float
  {
    RowMajorMatrix3f a = RowMajorMatrix3f::Random();
    RowMajorMatrix3f symmA = a.transpose() * a;
    SelfAdjointEigenSolver<RowMajorMatrix3f> eig;
    eig.computeDirect(symmA);
    VERIFY_IS_EQUAL(eig.info(), Success);
    SelfAdjointEigenSolver<RowMajorMatrix3f> eigRef(symmA);
    VERIFY_IS_APPROX(eigRef.eigenvalues(), eig.eigenvalues());
  }

  // 2x2 RowMajor float
  {
    RowMajorMatrix2f a = RowMajorMatrix2f::Random();
    RowMajorMatrix2f symmA = a.transpose() * a;
    SelfAdjointEigenSolver<RowMajorMatrix2f> eig;
    eig.computeDirect(symmA);
    VERIFY_IS_EQUAL(eig.info(), Success);
    SelfAdjointEigenSolver<RowMajorMatrix2f> eigRef(symmA);
    VERIFY_IS_APPROX(eigRef.eigenvalues(), eig.eigenvalues());
  }

  // Dynamic RowMajor with iterative solver
  {
    typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMajorMatrixXd;
    int s = internal::random<int>(2, 20);
    RowMajorMatrixXd a = RowMajorMatrixXd::Random(s, s);
    RowMajorMatrixXd symmA = a.transpose() * a;
    SelfAdjointEigenSolver<RowMajorMatrixXd> eig(symmA);
    VERIFY_IS_EQUAL(eig.info(), Success);
    double scaling = symmA.cwiseAbs().maxCoeff();
    if (scaling > (std::numeric_limits<double>::min)()) {
      VERIFY_IS_APPROX((symmA.template selfadjointView<Lower>() * eig.eigenvectors()) / scaling,
                       (eig.eigenvectors() * eig.eigenvalues().asDiagonal()) / scaling);
    }
  }
}

// Test matrix with Inf entries returns NoConvergence (similar to NaN test).
template <int>
void selfadjointeigensolver_inf() {
  Matrix3d m;
  m.setRandom();
  m = m * m.transpose();
  m(1, 1) = std::numeric_limits<double>::infinity();
  SelfAdjointEigenSolver<Matrix3d> eig(m);
  VERIFY_IS_EQUAL(eig.info(), NoConvergence);
}

template <int>
void bug_854() {
  Matrix3d m;
  m << 850.961, 51.966, 0, 51.966, 254.841, 0, 0, 0, 0;
  selfadjointeigensolver_essential_check(m);
}

template <int>
void bug_1014() {
  Matrix3d m;
  m << 0.11111111111111114658, 0, 0, 0, 0.11111111111111109107, 0, 0, 0, 0.11111111111111107719;
  selfadjointeigensolver_essential_check(m);
}

template <int>
void bug_1225() {
  Matrix3d m1, m2;
  m1.setRandom();
  m1 = m1 * m1.transpose();
  m2 = m1.triangularView<Upper>();
  SelfAdjointEigenSolver<Matrix3d> eig1(m1);
  SelfAdjointEigenSolver<Matrix3d> eig2(m2.selfadjointView<Upper>());
  VERIFY_IS_APPROX(eig1.eigenvalues(), eig2.eigenvalues());
}

template <int>
void bug_1204() {
  SparseMatrix<double> A(2, 2);
  A.setIdentity();
  SelfAdjointEigenSolver<Eigen::SparseMatrix<double> > eig(A);
}

template <int>
void selfadjointeigensolver_tridiagonal_zerosized() {
  SelfAdjointEigenSolver<MatrixXd> eig;
  VectorXd diag(0), subdiag(0);

  eig.computeFromTridiagonal(diag, subdiag, EigenvaluesOnly);
  VERIFY_IS_EQUAL(eig.info(), Success);
  VERIFY_IS_EQUAL(eig.eigenvalues().size(), 0);
  VERIFY_RAISES_ASSERT(eig.eigenvectors());

  eig.computeFromTridiagonal(diag, subdiag, ComputeEigenvectors);
  VERIFY_IS_EQUAL(eig.info(), Success);
  VERIFY_IS_EQUAL(eig.eigenvalues().size(), 0);
  VERIFY_IS_EQUAL(eig.eigenvectors().rows(), 0);
  VERIFY_IS_EQUAL(eig.eigenvectors().cols(), 0);
}

// Specific 3x3 test cases that stress the direct solver.
template <int>
void direct_3x3_stress() {
  // Near-planar point cloud covariance: two large eigenvalues, one near-zero.
  {
    Matrix3d m;
    m << 100, 50, 0.001, 50, 100, 0.002, 0.001, 0.002, 1e-10;
    selfadjointeigensolver_essential_check(m);
  }

  // All equal diagonal entries (triple eigenvalue).
  {
    Matrix3d m = Matrix3d::Identity() * 7.0;
    selfadjointeigensolver_essential_check(m);
  }

  // Two exactly equal eigenvalues (from explicit construction).
  {
    Matrix3d q;
    q << 1, 0, 0, 0, 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0, -1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    Vector3d d(1.0, 5.0, 5.0);
    Matrix3d m = q * d.asDiagonal() * q.transpose();
    selfadjointeigensolver_essential_check(m);
  }

  // Large off-diagonal relative to diagonal.
  {
    Matrix3d m;
    m << 1, 1000, 1000, 1000, 1, 1000, 1000, 1000, 1;
    selfadjointeigensolver_essential_check(m);
  }

  // Nearly singular: one eigenvalue much smaller than others.
  {
    Matrix3d m;
    m << 1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1;
    m *= 1e15;
    Matrix3d perturbation = Matrix3d::Zero();
    perturbation(0, 0) = 1e-15;
    m += perturbation;
    selfadjointeigensolver_essential_check(m);
  }
}

// Specific 2x2 test cases that stress the direct solver.
template <int>
void direct_2x2_stress() {
  // Equal eigenvalues.
  {
    Matrix2d m = Matrix2d::Identity() * 42.0;
    selfadjointeigensolver_essential_check(m);
  }

  // Very small off-diagonal.
  {
    Matrix2d m;
    m << 1.0, 1e-15, 1e-15, 1.0;
    selfadjointeigensolver_essential_check(m);
  }

  // Huge ratio between diagonal entries.
  {
    Matrix2d m;
    m << 1e100, 0, 0, 1e-100;
    selfadjointeigensolver_essential_check(m);
  }

  // Anti-diagonal dominant.
  {
    Matrix2d m;
    m << 0, 1e10, 1e10, 0;
    selfadjointeigensolver_essential_check(m);
  }

  // Negative entries.
  {
    Matrix2d m;
    m << -5.0, 3.0, 3.0, -5.0;
    selfadjointeigensolver_essential_check(m);
  }
}

EIGEN_DECLARE_TEST(eigensolver_selfadjoint) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    // trivial test for 1x1 matrices:
    CALL_SUBTEST_1(selfadjointeigensolver(Matrix<float, 1, 1>()));
    CALL_SUBTEST_10(selfadjointeigensolver(Matrix<double, 1, 1>()));
    CALL_SUBTEST_11(selfadjointeigensolver(Matrix<std::complex<double>, 1, 1>()));

    // very important to test 3x3 and 2x2 matrices since we provide special paths for them
    CALL_SUBTEST_12(selfadjointeigensolver(Matrix2f()));
    CALL_SUBTEST_15(selfadjointeigensolver(Matrix2d()));
    CALL_SUBTEST_16(selfadjointeigensolver(Matrix2cd()));
    CALL_SUBTEST_13(selfadjointeigensolver(Matrix3f()));
    CALL_SUBTEST_17(selfadjointeigensolver(Matrix3d()));
    CALL_SUBTEST_18(selfadjointeigensolver(Matrix3cd()));
    CALL_SUBTEST_2(selfadjointeigensolver(Matrix4d()));
    CALL_SUBTEST_14(selfadjointeigensolver(Matrix4cd()));

    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    CALL_SUBTEST_3(selfadjointeigensolver(MatrixXf(s, s)));
    CALL_SUBTEST_4(selfadjointeigensolver(MatrixXd(s, s)));
    CALL_SUBTEST_5(selfadjointeigensolver(MatrixXcd(s, s)));
    CALL_SUBTEST_9(selfadjointeigensolver(Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    // some trivial but implementation-wise tricky cases
    CALL_SUBTEST_4(selfadjointeigensolver(MatrixXd(1, 1)));
    CALL_SUBTEST_4(selfadjointeigensolver(MatrixXd(2, 2)));
    CALL_SUBTEST_5(selfadjointeigensolver(MatrixXcd(1, 1)));
    CALL_SUBTEST_5(selfadjointeigensolver(MatrixXcd(2, 2)));
    CALL_SUBTEST_6(selfadjointeigensolver(Matrix<double, 1, 1>()));
    CALL_SUBTEST_7(selfadjointeigensolver(Matrix<double, 2, 2>()));

    // repeated eigenvalues
    CALL_SUBTEST_17(selfadjointeigensolver_repeated_eigenvalues(Matrix3d()));
    CALL_SUBTEST_15(selfadjointeigensolver_repeated_eigenvalues(Matrix2d()));
    CALL_SUBTEST_2(selfadjointeigensolver_repeated_eigenvalues(Matrix4d()));
    CALL_SUBTEST_4(selfadjointeigensolver_repeated_eigenvalues(MatrixXd(s, s)));
    CALL_SUBTEST_13(selfadjointeigensolver_repeated_eigenvalues(Matrix3f()));
    CALL_SUBTEST_12(selfadjointeigensolver_repeated_eigenvalues(Matrix2f()));
    CALL_SUBTEST_18(selfadjointeigensolver_repeated_eigenvalues(Matrix3cd()));

    // extreme eigenvalues (near overflow/underflow, high condition number)
    CALL_SUBTEST_17(selfadjointeigensolver_extreme_eigenvalues(Matrix3d()));
    CALL_SUBTEST_2(selfadjointeigensolver_extreme_eigenvalues(Matrix4d()));
    CALL_SUBTEST_4(selfadjointeigensolver_extreme_eigenvalues(MatrixXd(s, s)));
    CALL_SUBTEST_13(selfadjointeigensolver_extreme_eigenvalues(Matrix3f()));
    CALL_SUBTEST_3(selfadjointeigensolver_extreme_eigenvalues(MatrixXf(s, s)));

    // computeFromTridiagonal with scaled inputs
    CALL_SUBTEST_4(selfadjointeigensolver_tridiagonal_scaled(MatrixXd(s, s)));
    CALL_SUBTEST_3(selfadjointeigensolver_tridiagonal_scaled(MatrixXf(s, s)));

    // structured tridiagonal hard cases from the literature
    CALL_SUBTEST_4(selfadjointeigensolver_structured_tridiagonal<double>());
    CALL_SUBTEST_3(selfadjointeigensolver_structured_tridiagonal<float>());

    // diagonal matrices
    CALL_SUBTEST_17(selfadjointeigensolver_diagonal(Matrix3d()));
    CALL_SUBTEST_4(selfadjointeigensolver_diagonal(MatrixXd(s, s)));

    // operatorInverseSqrt
    CALL_SUBTEST_17(selfadjointeigensolver_inverse_sqrt(Matrix3d()));
    CALL_SUBTEST_2(selfadjointeigensolver_inverse_sqrt(Matrix4d()));
    CALL_SUBTEST_4(selfadjointeigensolver_inverse_sqrt(MatrixXd(s, s)));
    CALL_SUBTEST_13(selfadjointeigensolver_inverse_sqrt(Matrix3f()));

    // RowMajor
    CALL_SUBTEST_19(selfadjointeigensolver_rowmajor<0>());

    // Larger matrices to exercise the blocked tridiagonalization path (n >= 96).
    CALL_SUBTEST_4(selfadjointeigensolver(MatrixXd(256, 256)));
    CALL_SUBTEST_5(selfadjointeigensolver(MatrixXcd(256, 256)));
    CALL_SUBTEST_3(selfadjointeigensolver(MatrixXf(256, 256)));
    CALL_SUBTEST_9(selfadjointeigensolver(Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>(256, 256)));
  }

  CALL_SUBTEST_17(bug_854<0>());
  CALL_SUBTEST_17(bug_1014<0>());
  CALL_SUBTEST_17(bug_1204<0>());
  CALL_SUBTEST_17(bug_1225<0>());
  CALL_SUBTEST_8(selfadjointeigensolver_tridiagonal_zerosized<0>());

  // Stress tests for direct 3x3 and 2x2 solvers.
  CALL_SUBTEST_17(direct_3x3_stress<0>());
  CALL_SUBTEST_15(direct_2x2_stress<0>());

  // Test Inf input handling.
  CALL_SUBTEST_17(selfadjointeigensolver_inf<0>());

  // Test problem size constructors
  s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
  CALL_SUBTEST_8(SelfAdjointEigenSolver<MatrixXf> tmp1(s));
  CALL_SUBTEST_8(Tridiagonalization<MatrixXf> tmp2(s));

  TEST_SET_BUT_UNUSED_VARIABLE(s);
}
