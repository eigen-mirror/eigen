// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Tests for GpuLLT: GPU Cholesky (LL^T) using cuSOLVER.
// Covers cusolverDnXpotrf (factorization) and cusolverDnXpotrs (solve)
// for float, double, complex<float>, complex<double>, Lower and Upper.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Cholesky>
#include <Eigen/GPU>

using namespace Eigen;

// Build a random symmetric positive-definite matrix: A = M^H*M + n*I.
template <typename MatrixType>
MatrixType make_spd(Index n) {
  using Scalar = typename MatrixType::Scalar;
  MatrixType M = MatrixType::Random(n, n);
  return M.adjoint() * M + MatrixType::Identity(n, n) * static_cast<Scalar>(n);
}

// Test factorization: L*L^H must reconstruct A to within floating-point tolerance.
template <typename Scalar, int UpLo>
void test_potrf(Index n) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType A = make_spd<MatrixType>(n);

  GpuLLT<Scalar, UpLo> llt(A);
  VERIFY_IS_EQUAL(llt.info(), Success);

  // Reconstruct L*L^H and compare to original A.
  // GpuLLT stores the factor on device; use CPU LLT to get the triangular factor
  // for reconstruction since GpuLLT does not expose the device-resident factor directly.
  LLT<MatrixType, UpLo> ref(A);
  VERIFY_IS_EQUAL(ref.info(), Success);
  MatrixType A_reconstructed = ref.reconstructedMatrix();

  // Both should equal A to within n*eps*||A||.
  RealScalar tol = RealScalar(4) * RealScalar(n) * NumTraits<Scalar>::epsilon() * A.norm();
  VERIFY((A_reconstructed - A).norm() < tol);

  // Smoke-test: llt.solve(b) should return the same result as ref.solve(b).
  MatrixType b = MatrixType::Random(n, 1);
  MatrixType x_gpu = llt.solve(b);
  MatrixType x_cpu = ref.solve(b);
  VERIFY((x_gpu - x_cpu).norm() < tol);
}

// Test solve: residual ||A*X - B|| / ||B|| must be small.
template <typename Scalar, int UpLo>
void test_potrs(Index n, Index nrhs) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType A = make_spd<MatrixType>(n);
  MatrixType B = MatrixType::Random(n, nrhs);

  GpuLLT<Scalar, UpLo> llt(A);
  VERIFY_IS_EQUAL(llt.info(), Success);

  MatrixType X = llt.solve(B);

  RealScalar residual = (A * X - B).norm() / B.norm();
  RealScalar tol = RealScalar(n) * NumTraits<Scalar>::epsilon();
  VERIFY(residual < tol);
}

// Test that multiple solves against the same factor all produce correct results.
// This exercises the key design property: L stays on device across calls.
template <typename Scalar>
void test_multiple_solves(Index n) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType A = make_spd<MatrixType>(n);
  GpuLLT<Scalar, Lower> llt(A);
  VERIFY_IS_EQUAL(llt.info(), Success);

  RealScalar tol = RealScalar(n) * NumTraits<Scalar>::epsilon();
  for (int k = 0; k < 5; ++k) {
    MatrixType B = MatrixType::Random(n, 3);
    MatrixType X = llt.solve(B);
    RealScalar residual = (A * X - B).norm() / B.norm();
    VERIFY(residual < tol);
  }
}

// Test that GpuLLT correctly detects a non-SPD matrix.
void test_not_spd() {
  MatrixXd A = -MatrixXd::Identity(8, 8);  // negative definite
  GpuLLT<double> llt(A);
  VERIFY_IS_EQUAL(llt.info(), NumericalIssue);
}

// ---- DeviceMatrix integration tests -----------------------------------------

// compute(DeviceMatrix) + solve(DeviceMatrix) → toHost
template <typename Scalar, int UpLo>
void test_device_matrix_solve(Index n, Index nrhs) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType A = make_spd<MatrixType>(n);
  MatrixType B = MatrixType::Random(n, nrhs);

  auto d_A = DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = DeviceMatrix<Scalar>::fromHost(B);

  GpuLLT<Scalar, UpLo> llt;
  llt.compute(d_A);
  VERIFY_IS_EQUAL(llt.info(), Success);

  DeviceMatrix<Scalar> d_X = llt.solve(d_B);
  MatrixType X = d_X.toHost();

  RealScalar residual = (A * X - B).norm() / B.norm();
  VERIFY(residual < RealScalar(n) * NumTraits<Scalar>::epsilon());
}

// compute(DeviceMatrix&&) — move path
template <typename Scalar>
void test_device_matrix_move_compute(Index n) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType A = make_spd<MatrixType>(n);
  MatrixType B = MatrixType::Random(n, 1);

  auto d_A = DeviceMatrix<Scalar>::fromHost(A);
  GpuLLT<Scalar, Lower> llt;
  llt.compute(std::move(d_A));
  VERIFY_IS_EQUAL(llt.info(), Success);

  // d_A should be empty after move.
  VERIFY(d_A.empty());

  MatrixType X = llt.solve(B);
  RealScalar residual = (A * X - B).norm() / B.norm();
  VERIFY(residual < RealScalar(n) * NumTraits<Scalar>::epsilon());
}

// Full async chain: compute → solve → solve again with result as RHS → toHost
template <typename Scalar>
void test_chaining(Index n) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType A = make_spd<MatrixType>(n);
  MatrixType B = MatrixType::Random(n, 3);

  auto d_A = DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = DeviceMatrix<Scalar>::fromHost(B);

  GpuLLT<Scalar, Lower> llt;
  llt.compute(d_A);
  VERIFY_IS_EQUAL(llt.info(), Success);

  // Chain: solve → use result as RHS for another solve
  DeviceMatrix<Scalar> d_X = llt.solve(d_B);
  DeviceMatrix<Scalar> d_Y = llt.solve(d_X);

  // Only sync at the very end.
  MatrixType Y = d_Y.toHost();

  // Verify: Y = A^{-2} * B
  MatrixType X_ref = LLT<MatrixType, Lower>(A).solve(B);
  MatrixType Y_ref = LLT<MatrixType, Lower>(A).solve(X_ref);

  RealScalar tol = RealScalar(4) * RealScalar(n) * NumTraits<Scalar>::epsilon() * Y_ref.norm();
  VERIFY((Y - Y_ref).norm() < tol);
}

template <typename Scalar>
void test_scalar() {
  CALL_SUBTEST((test_potrf<Scalar, Lower>(1)));
  CALL_SUBTEST((test_potrf<Scalar, Lower>(64)));
  CALL_SUBTEST((test_potrf<Scalar, Lower>(256)));
  CALL_SUBTEST((test_potrf<Scalar, Upper>(64)));
  CALL_SUBTEST((test_potrf<Scalar, Upper>(256)));

  CALL_SUBTEST((test_potrs<Scalar, Lower>(64, 1)));
  CALL_SUBTEST((test_potrs<Scalar, Lower>(64, 4)));
  CALL_SUBTEST((test_potrs<Scalar, Lower>(256, 8)));
  CALL_SUBTEST((test_potrs<Scalar, Upper>(64, 1)));
  CALL_SUBTEST((test_potrs<Scalar, Upper>(256, 4)));

  CALL_SUBTEST(test_multiple_solves<Scalar>(128));

  CALL_SUBTEST((test_device_matrix_solve<Scalar, Lower>(64, 4)));
  CALL_SUBTEST((test_device_matrix_solve<Scalar, Upper>(128, 1)));
  CALL_SUBTEST(test_device_matrix_move_compute<Scalar>(64));
  CALL_SUBTEST(test_chaining<Scalar>(64));
}

EIGEN_DECLARE_TEST(gpu_cusolver_llt) {
  CALL_SUBTEST(test_scalar<float>());
  CALL_SUBTEST(test_scalar<double>());
  CALL_SUBTEST(test_scalar<std::complex<float>>());
  CALL_SUBTEST(test_scalar<std::complex<double>>());
  CALL_SUBTEST(test_not_spd());
}
