// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Tests for GpuSelfAdjointEigenSolver: GPU symmetric/Hermitian eigenvalue
// decomposition via cuSOLVER.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Eigenvalues>
#include <Eigen/GPU>

using namespace Eigen;

// ---- Reconstruction: V * diag(W) * V^H ≈ A ---------------------------------

template <typename Scalar>
void test_eigen_reconstruction(Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  // Build a symmetric/Hermitian matrix.
  Mat R = Mat::Random(n, n);
  Mat A = R + R.adjoint();

  GpuSelfAdjointEigenSolver<Scalar> es(A);
  VERIFY_IS_EQUAL(es.info(), Success);

  auto W = es.eigenvalues();
  Mat V = es.eigenvectors();

  VERIFY_IS_EQUAL(W.size(), n);
  VERIFY_IS_EQUAL(V.rows(), n);
  VERIFY_IS_EQUAL(V.cols(), n);

  // Reconstruct: A_hat = V * diag(W) * V^H.
  Mat A_hat = V * W.asDiagonal() * V.adjoint();
  RealScalar tol = RealScalar(5) * std::sqrt(static_cast<RealScalar>(n)) * NumTraits<Scalar>::epsilon() * A.norm();
  VERIFY((A_hat - A).norm() < tol);

  // Orthogonality: V^H * V ≈ I.
  Mat VhV = V.adjoint() * V;
  Mat eye = Mat::Identity(n, n);
  VERIFY((VhV - eye).norm() < tol);
}

// ---- Eigenvalues match CPU SelfAdjointEigenSolver ---------------------------

template <typename Scalar>
void test_eigen_values(Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat R = Mat::Random(n, n);
  Mat A = R + R.adjoint();

  GpuSelfAdjointEigenSolver<Scalar> gpu_es(A);
  VERIFY_IS_EQUAL(gpu_es.info(), Success);
  auto W_gpu = gpu_es.eigenvalues();

  SelfAdjointEigenSolver<Mat> cpu_es(A);
  auto W_cpu = cpu_es.eigenvalues();

  RealScalar tol = RealScalar(5) * std::sqrt(static_cast<RealScalar>(n)) * NumTraits<Scalar>::epsilon() *
                   W_cpu.cwiseAbs().maxCoeff();
  VERIFY((W_gpu - W_cpu).norm() < tol);
}

// ---- Eigenvalues-only mode --------------------------------------------------

template <typename Scalar>
void test_eigen_values_only(Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat R = Mat::Random(n, n);
  Mat A = R + R.adjoint();

  GpuSelfAdjointEigenSolver<Scalar> gpu_es(A, GpuSelfAdjointEigenSolver<Scalar>::EigenvaluesOnly);
  VERIFY_IS_EQUAL(gpu_es.info(), Success);
  auto W_gpu = gpu_es.eigenvalues();

  SelfAdjointEigenSolver<Mat> cpu_es(A, EigenvaluesOnly);
  auto W_cpu = cpu_es.eigenvalues();

  RealScalar tol = RealScalar(5) * std::sqrt(static_cast<RealScalar>(n)) * NumTraits<Scalar>::epsilon() *
                   W_cpu.cwiseAbs().maxCoeff();
  VERIFY((W_gpu - W_cpu).norm() < tol);
}

// ---- DeviceMatrix input path ------------------------------------------------

template <typename Scalar>
void test_eigen_device_matrix(Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat R = Mat::Random(n, n);
  Mat A = R + R.adjoint();

  auto d_A = DeviceMatrix<Scalar>::fromHost(A);
  GpuSelfAdjointEigenSolver<Scalar> es;
  es.compute(d_A);
  VERIFY_IS_EQUAL(es.info(), Success);

  auto W_gpu = es.eigenvalues();
  Mat V = es.eigenvectors();

  // Verify reconstruction.
  Mat A_hat = V * W_gpu.asDiagonal() * V.adjoint();
  RealScalar tol = RealScalar(5) * std::sqrt(static_cast<RealScalar>(n)) * NumTraits<Scalar>::epsilon() * A.norm();
  VERIFY((A_hat - A).norm() < tol);
}

// ---- Recompute (reuse solver object) ----------------------------------------

template <typename Scalar>
void test_eigen_recompute(Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  GpuSelfAdjointEigenSolver<Scalar> es;

  for (int trial = 0; trial < 3; ++trial) {
    Mat R = Mat::Random(n, n);
    Mat A = R + R.adjoint();
    es.compute(A);
    VERIFY_IS_EQUAL(es.info(), Success);

    auto W = es.eigenvalues();
    Mat V = es.eigenvectors();
    Mat A_hat = V * W.asDiagonal() * V.adjoint();
    RealScalar tol = RealScalar(5) * std::sqrt(static_cast<RealScalar>(n)) * NumTraits<Scalar>::epsilon() * A.norm();
    VERIFY((A_hat - A).norm() < tol);
  }
}

// ---- Empty matrix -----------------------------------------------------------

void test_eigen_empty() {
  GpuSelfAdjointEigenSolver<double> es(MatrixXd(0, 0));
  VERIFY_IS_EQUAL(es.info(), Success);
  VERIFY_IS_EQUAL(es.rows(), 0);
  VERIFY_IS_EQUAL(es.cols(), 0);
}

// ---- Per-scalar driver ------------------------------------------------------

template <typename Scalar>
void test_scalar() {
  // Reconstruction + orthogonality.
  CALL_SUBTEST(test_eigen_reconstruction<Scalar>(64));
  CALL_SUBTEST(test_eigen_reconstruction<Scalar>(128));

  // Eigenvalues match CPU.
  CALL_SUBTEST(test_eigen_values<Scalar>(64));
  CALL_SUBTEST(test_eigen_values<Scalar>(128));

  // Values-only mode.
  CALL_SUBTEST(test_eigen_values_only<Scalar>(64));

  // DeviceMatrix input.
  CALL_SUBTEST(test_eigen_device_matrix<Scalar>(64));

  // Recompute.
  CALL_SUBTEST(test_eigen_recompute<Scalar>(32));
}

EIGEN_DECLARE_TEST(gpu_cusolver_eigen) {
  CALL_SUBTEST(test_scalar<float>());
  CALL_SUBTEST(test_scalar<double>());
  CALL_SUBTEST(test_scalar<std::complex<float>>());
  CALL_SUBTEST(test_scalar<std::complex<double>>());
  CALL_SUBTEST(test_eigen_empty());
}
