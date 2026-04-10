// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Tests for GpuSVD: GPU SVD via cuSOLVER.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/SVD>
#include <Eigen/GPU>

using namespace Eigen;

// ---- SVD reconstruction: U * diag(S) * VT ≈ A ------------------------------

template <typename Scalar, unsigned int Options>
void test_svd_reconstruction(Index m, Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(m, n);
  gpu::SVD<Scalar> svd(A, Options);
  VERIFY_IS_EQUAL(svd.info(), Success);

  auto S = svd.singularValues();
  Mat U = svd.matrixU();
  Mat VT = svd.matrixVT();

  const Index k = (std::min)(m, n);

  // Reconstruct: A_hat = U[:,:k] * diag(S) * VT[:k,:].
  Mat A_hat = U.leftCols(k) * S.asDiagonal() * VT.topRows(k);
  RealScalar tol = RealScalar(5) * std::sqrt(static_cast<RealScalar>(k)) * NumTraits<Scalar>::epsilon() * A.norm();
  VERIFY((A_hat - A).norm() < tol);

  // Orthogonality: U^H * U ≈ I.
  Mat UtU = U.adjoint() * U;
  Mat I_u = Mat::Identity(U.cols(), U.cols());
  VERIFY((UtU - I_u).norm() < tol);

  // Orthogonality: VT * VT^H ≈ I.
  Mat VtVh = VT * VT.adjoint();
  Mat I_v = Mat::Identity(VT.rows(), VT.rows());
  VERIFY((VtVh - I_v).norm() < tol);
}

// ---- Singular values match CPU BDCSVD ---------------------------------------

template <typename Scalar>
void test_svd_singular_values(Index m, Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(m, n);
  gpu::SVD<Scalar> svd(A, 0);  // values only
  VERIFY_IS_EQUAL(svd.info(), Success);

  auto S_gpu = svd.singularValues();
  auto S_cpu = BDCSVD<Mat>(A, 0).singularValues();

  RealScalar tol =
      RealScalar(5) * std::sqrt(static_cast<RealScalar>((std::min)(m, n))) * NumTraits<Scalar>::epsilon() * S_cpu(0);
  VERIFY((S_gpu - S_cpu).norm() < tol);
}

// ---- Solve: pseudoinverse ---------------------------------------------------

template <typename Scalar>
void test_svd_solve(Index m, Index n, Index nrhs) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(m, n);
  Mat B = Mat::Random(m, nrhs);

  gpu::SVD<Scalar> svd(A, ComputeThinU | ComputeThinV);
  VERIFY_IS_EQUAL(svd.info(), Success);

  Mat X = svd.solve(B);
  VERIFY_IS_EQUAL(X.rows(), n);
  VERIFY_IS_EQUAL(X.cols(), nrhs);

  // Compare with CPU BDCSVD solve.
  Mat X_cpu = BDCSVD<Mat>(A, ComputeThinU | ComputeThinV).solve(B);
  RealScalar tol = RealScalar(100) * RealScalar((std::max)(m, n)) * NumTraits<Scalar>::epsilon();
  VERIFY((X - X_cpu).norm() / X_cpu.norm() < tol);
}

// ---- Solve: truncated -------------------------------------------------------

template <typename Scalar>
void test_svd_solve_truncated(Index m, Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(m, n);
  Mat B = Mat::Random(m, 1);
  const Index k = (std::min)(m, n);
  const Index trunc = k / 2;
  eigen_assert(trunc > 0);

  gpu::SVD<Scalar> svd(A, ComputeThinU | ComputeThinV);
  Mat X_trunc = svd.solve(B, trunc);

  // Build CPU reference: truncated pseudoinverse.
  auto cpu_svd = BDCSVD<Mat>(A, ComputeThinU | ComputeThinV);
  auto S = cpu_svd.singularValues();
  Mat U = cpu_svd.matrixU();
  Mat V = cpu_svd.matrixV();

  // D_ii = 1/S_i for i < trunc, 0 otherwise.
  Matrix<RealScalar, Dynamic, 1> D = Matrix<RealScalar, Dynamic, 1>::Zero(k);
  for (Index i = 0; i < trunc; ++i) D(i) = RealScalar(1) / S(i);
  Mat X_ref = V * D.asDiagonal() * U.adjoint() * B;

  RealScalar tol = RealScalar(100) * RealScalar(k) * NumTraits<Scalar>::epsilon();
  VERIFY((X_trunc - X_ref).norm() / X_ref.norm() < tol);
}

// ---- Solve: Tikhonov regularized --------------------------------------------

template <typename Scalar>
void test_svd_solve_regularized(Index m, Index n) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Mat A = Mat::Random(m, n);
  Mat B = Mat::Random(m, 1);
  RealScalar lambda = RealScalar(0.1);
  const Index k = (std::min)(m, n);

  gpu::SVD<Scalar> svd(A, ComputeThinU | ComputeThinV);
  Mat X_reg = svd.solve(B, lambda);

  // CPU reference: D_ii = S_i / (S_i^2 + lambda^2).
  auto cpu_svd = BDCSVD<Mat>(A, ComputeThinU | ComputeThinV);
  auto S = cpu_svd.singularValues();
  Mat U = cpu_svd.matrixU();
  Mat V = cpu_svd.matrixV();

  Matrix<RealScalar, Dynamic, 1> D(k);
  for (Index i = 0; i < k; ++i) D(i) = S(i) / (S(i) * S(i) + lambda * lambda);
  Mat X_ref = V * D.asDiagonal() * U.adjoint() * B;

  RealScalar tol = RealScalar(100) * RealScalar(k) * NumTraits<Scalar>::epsilon();
  VERIFY((X_reg - X_ref).norm() / X_ref.norm() < tol);
}

// ---- Empty matrix -----------------------------------------------------------

void test_svd_empty() {
  gpu::SVD<double> svd(MatrixXd(0, 0), 0);
  VERIFY_IS_EQUAL(svd.info(), Success);
  VERIFY_IS_EQUAL(svd.rows(), 0);
  VERIFY_IS_EQUAL(svd.cols(), 0);
}

// ---- Per-scalar driver ------------------------------------------------------

template <typename Scalar>
void test_scalar() {
  // Reconstruction + orthogonality (thin and full, identical test logic).
  CALL_SUBTEST((test_svd_reconstruction<Scalar, ComputeThinU | ComputeThinV>(64, 64)));
  CALL_SUBTEST((test_svd_reconstruction<Scalar, ComputeThinU | ComputeThinV>(128, 64)));
  CALL_SUBTEST((test_svd_reconstruction<Scalar, ComputeThinU | ComputeThinV>(64, 128)));  // wide (m < n)
  CALL_SUBTEST((test_svd_reconstruction<Scalar, ComputeFullU | ComputeFullV>(64, 64)));
  CALL_SUBTEST((test_svd_reconstruction<Scalar, ComputeFullU | ComputeFullV>(128, 64)));

  // Singular values.
  CALL_SUBTEST(test_svd_singular_values<Scalar>(64, 64));
  CALL_SUBTEST(test_svd_singular_values<Scalar>(128, 64));

  // Solve.
  CALL_SUBTEST(test_svd_solve<Scalar>(64, 64, 4));
  CALL_SUBTEST(test_svd_solve<Scalar>(128, 64, 4));
  CALL_SUBTEST(test_svd_solve<Scalar>(64, 128, 4));  // wide (m < n)

  // Truncated and regularized solve.
  CALL_SUBTEST(test_svd_solve_truncated<Scalar>(64, 64));
  CALL_SUBTEST(test_svd_solve_regularized<Scalar>(64, 64));
}

EIGEN_DECLARE_TEST(gpu_cusolver_svd) {
  CALL_SUBTEST(test_scalar<float>());
  CALL_SUBTEST(test_scalar<double>());
  CALL_SUBTEST(test_scalar<std::complex<float>>());
  CALL_SUBTEST(test_scalar<std::complex<double>>());
  CALL_SUBTEST(test_svd_empty());
}
