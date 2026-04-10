// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Tests for GpuQR: GPU QR decomposition via cuSOLVER.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/QR>
#include <Eigen/GPU>

using namespace Eigen;

// ---- Solve square system: A * X = B -----------------------------------------

template <typename Scalar>
void test_qr_solve_square(Index n, Index nrhs) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(n, n);
  Mat B = Mat::Random(n, nrhs);

  GpuQR<Scalar> qr(A);
  VERIFY_IS_EQUAL(qr.info(), Success);

  Mat X = qr.solve(B);
  RealScalar residual = (A * X - B).norm() / (A.norm() * X.norm());
  VERIFY(residual < RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon());
}

// ---- Solve overdetermined system: m > n (least-squares) ---------------------

template <typename Scalar>
void test_qr_solve_overdetermined(Index m, Index n, Index nrhs) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  eigen_assert(m >= n);
  Mat A = Mat::Random(m, n);
  Mat B = Mat::Random(m, nrhs);

  GpuQR<Scalar> qr(A);
  VERIFY_IS_EQUAL(qr.info(), Success);

  Mat X = qr.solve(B);
  VERIFY_IS_EQUAL(X.rows(), n);
  VERIFY_IS_EQUAL(X.cols(), nrhs);

  // Compare with CPU QR.
  Mat X_cpu = HouseholderQR<Mat>(A).solve(B);
  RealScalar tol = RealScalar(100) * RealScalar(m) * NumTraits<Scalar>::epsilon();
  VERIFY((X - X_cpu).norm() / X_cpu.norm() < tol);
}

// ---- Solve with DeviceMatrix input ------------------------------------------

template <typename Scalar>
void test_qr_solve_device(Index n, Index nrhs) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(n, n);
  Mat B = Mat::Random(n, nrhs);

  auto d_A = DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = DeviceMatrix<Scalar>::fromHost(B);

  GpuQR<Scalar> qr;
  qr.compute(d_A);
  VERIFY_IS_EQUAL(qr.info(), Success);

  DeviceMatrix<Scalar> d_X = qr.solve(d_B);
  Mat X = d_X.toHost();

  RealScalar residual = (A * X - B).norm() / (A.norm() * X.norm());
  VERIFY(residual < RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon());
}

// ---- Solve overdetermined via device path -----------------------------------

template <typename Scalar>
void test_qr_solve_overdetermined_device(Index m, Index n, Index nrhs) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  eigen_assert(m >= n);
  Mat A = Mat::Random(m, n);
  Mat B = Mat::Random(m, nrhs);

  auto d_A = DeviceMatrix<Scalar>::fromHost(A);
  auto d_B = DeviceMatrix<Scalar>::fromHost(B);

  GpuQR<Scalar> qr;
  qr.compute(d_A);
  VERIFY_IS_EQUAL(qr.info(), Success);

  DeviceMatrix<Scalar> d_X = qr.solve(d_B);
  VERIFY_IS_EQUAL(d_X.rows(), n);
  VERIFY_IS_EQUAL(d_X.cols(), nrhs);

  Mat X = d_X.toHost();
  Mat X_cpu = HouseholderQR<Mat>(A).solve(B);
  RealScalar tol = RealScalar(100) * RealScalar(m) * NumTraits<Scalar>::epsilon();
  VERIFY((X - X_cpu).norm() / X_cpu.norm() < tol);
}

// ---- Multiple solves reuse the factorization --------------------------------

template <typename Scalar>
void test_qr_multiple_solves(Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(n, n);
  GpuQR<Scalar> qr(A);
  VERIFY_IS_EQUAL(qr.info(), Success);

  RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  for (int k = 0; k < 5; ++k) {
    Mat B = Mat::Random(n, 3);
    Mat X = qr.solve(B);
    RealScalar residual = (A * X - B).norm() / (A.norm() * X.norm());
    VERIFY(residual < tol);
  }
}

// ---- Agreement with CPU HouseholderQR ---------------------------------------

template <typename Scalar>
void test_qr_vs_cpu(Index n, Index nrhs) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(n, n);
  Mat B = Mat::Random(n, nrhs);

  GpuQR<Scalar> gpu_qr(A);
  VERIFY_IS_EQUAL(gpu_qr.info(), Success);

  Mat X_gpu = gpu_qr.solve(B);
  Mat X_cpu = HouseholderQR<Mat>(A).solve(B);

  RealScalar tol = RealScalar(100) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  VERIFY((X_gpu - X_cpu).norm() / X_cpu.norm() < tol);
}

// ---- Per-scalar driver ------------------------------------------------------

template <typename Scalar>
void test_scalar() {
  CALL_SUBTEST(test_qr_solve_square<Scalar>(1, 1));
  CALL_SUBTEST(test_qr_solve_square<Scalar>(64, 1));
  CALL_SUBTEST(test_qr_solve_square<Scalar>(64, 4));
  CALL_SUBTEST(test_qr_solve_square<Scalar>(256, 8));

  CALL_SUBTEST(test_qr_solve_overdetermined<Scalar>(128, 64, 4));
  CALL_SUBTEST(test_qr_solve_overdetermined<Scalar>(256, 128, 1));

  CALL_SUBTEST(test_qr_solve_device<Scalar>(64, 4));
  CALL_SUBTEST(test_qr_solve_overdetermined_device<Scalar>(128, 64, 4));
  CALL_SUBTEST(test_qr_multiple_solves<Scalar>(64));
  CALL_SUBTEST(test_qr_vs_cpu<Scalar>(64, 4));
  CALL_SUBTEST(test_qr_vs_cpu<Scalar>(256, 8));
}

void test_qr_empty() {
  GpuQR<double> qr(MatrixXd(0, 0));
  VERIFY_IS_EQUAL(qr.info(), Success);
  VERIFY_IS_EQUAL(qr.rows(), 0);
  VERIFY_IS_EQUAL(qr.cols(), 0);
}

EIGEN_DECLARE_TEST(gpu_cusolver_qr) {
  CALL_SUBTEST(test_scalar<float>());
  CALL_SUBTEST(test_scalar<double>());
  CALL_SUBTEST(test_scalar<std::complex<float>>());
  CALL_SUBTEST(test_scalar<std::complex<double>>());
  CALL_SUBTEST(test_qr_empty());
}
