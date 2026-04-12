// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Tests for gpu::SparseContext: GPU SpMV/SpMM via cuSPARSE.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Sparse>
#include <Eigen/GPU>

using namespace Eigen;

// ---- Helper: build a random sparse matrix -----------------------------------

template <typename Scalar>
SparseMatrix<Scalar, ColMajor, int> make_sparse(Index rows, Index cols, double density = 0.1) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  SpMat R(rows, cols);
  R.reserve(VectorXi::Constant(cols, static_cast<int>(rows * density) + 1));
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      if ((std::rand() / double(RAND_MAX)) < density) {
        R.insert(i, j) = Scalar(RealScalar(std::rand() / double(RAND_MAX) - 0.5));
      }
    }
  }
  R.makeCompressed();
  return R;
}

// ---- SpMV: y = A * x -------------------------------------------------------

template <typename Scalar>
void test_spmv(Index rows, Index cols) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  SpMat A = make_sparse<Scalar>(rows, cols);
  Vec x = Vec::Random(cols);

  gpu::SparseContext<Scalar> ctx;
  Vec y_gpu = ctx.multiply(A, x);
  Vec y_cpu = A * x;

  RealScalar tol = RealScalar(10) * RealScalar((std::max)(rows, cols)) * NumTraits<Scalar>::epsilon();
  VERIFY_IS_EQUAL(y_gpu.size(), rows);
  VERIFY((y_gpu - y_cpu).norm() / (y_cpu.norm() + RealScalar(1)) < tol);
}

// ---- SpMV with alpha/beta: y = alpha*A*x + beta*y ---------------------------

template <typename Scalar>
void test_spmv_alpha_beta(Index n) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  SpMat A = make_sparse<Scalar>(n, n);
  Vec x = Vec::Random(n);
  Vec y_init = Vec::Random(n);

  Scalar alpha(2);
  Scalar beta(3);

  Vec y_cpu = alpha * (A * x) + beta * y_init;

  gpu::SparseContext<Scalar> ctx;
  Vec y_gpu = y_init;
  ctx.multiply(A, x, y_gpu, alpha, beta);

  RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();
  VERIFY((y_gpu - y_cpu).norm() / (y_cpu.norm() + RealScalar(1)) < tol);
}

// ---- Transpose: y = A^T * x ------------------------------------------------

template <typename Scalar>
void test_spmv_transpose(Index rows, Index cols) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  SpMat A = make_sparse<Scalar>(rows, cols);
  Vec x = Vec::Random(rows);

  gpu::SparseContext<Scalar> ctx;
  Vec y_gpu = ctx.multiplyT(A, x);
  Vec y_cpu = A.transpose() * x;

  RealScalar tol = RealScalar(10) * RealScalar((std::max)(rows, cols)) * NumTraits<Scalar>::epsilon();
  VERIFY_IS_EQUAL(y_gpu.size(), cols);
  VERIFY((y_gpu - y_cpu).norm() / (y_cpu.norm() + RealScalar(1)) < tol);
}

// ---- SpMM: Y = A * X (multiple RHS) ----------------------------------------

template <typename Scalar>
void test_spmm(Index rows, Index cols, Index nrhs) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  SpMat A = make_sparse<Scalar>(rows, cols);
  Mat X = Mat::Random(cols, nrhs);

  gpu::SparseContext<Scalar> ctx;
  Mat Y_gpu = ctx.multiplyMat(A, X);
  Mat Y_cpu = A * X;

  RealScalar tol = RealScalar(10) * RealScalar((std::max)(rows, cols)) * NumTraits<Scalar>::epsilon();
  VERIFY_IS_EQUAL(Y_gpu.rows(), rows);
  VERIFY_IS_EQUAL(Y_gpu.cols(), nrhs);
  VERIFY((Y_gpu - Y_cpu).norm() / (Y_cpu.norm() + RealScalar(1)) < tol);
}

// ---- Identity matrix: I * x = x --------------------------------------------

template <typename Scalar>
void test_identity(Index n) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  // Build sparse identity.
  SpMat eye(n, n);
  eye.setIdentity();
  eye.makeCompressed();

  Vec x = Vec::Random(n);

  gpu::SparseContext<Scalar> ctx;
  Vec y = ctx.multiply(eye, x);

  RealScalar tol = NumTraits<Scalar>::epsilon();
  VERIFY((y - x).norm() < tol);
}

// ---- Context reuse ----------------------------------------------------------

template <typename Scalar>
void test_reuse(Index n) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  gpu::SparseContext<Scalar> ctx;
  RealScalar tol = RealScalar(10) * RealScalar(n) * NumTraits<Scalar>::epsilon();

  for (int trial = 0; trial < 3; ++trial) {
    SpMat A = make_sparse<Scalar>(n, n);
    Vec x = Vec::Random(n);
    Vec y_gpu = ctx.multiply(A, x);
    Vec y_cpu = A * x;
    VERIFY((y_gpu - y_cpu).norm() / (y_cpu.norm() + RealScalar(1)) < tol);
  }
}

// ---- Empty ------------------------------------------------------------------

template <typename Scalar>
void test_empty() {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;

  SpMat A(0, 0);
  A.makeCompressed();
  Vec x(0);

  gpu::SparseContext<Scalar> ctx;
  Vec y = ctx.multiply(A, x);
  VERIFY_IS_EQUAL(y.size(), 0);
}

// ---- Per-scalar driver ------------------------------------------------------

template <typename Scalar>
void test_scalar() {
  CALL_SUBTEST(test_spmv<Scalar>(64, 64));
  CALL_SUBTEST(test_spmv<Scalar>(128, 64));  // non-square
  CALL_SUBTEST(test_spmv<Scalar>(64, 128));  // wide
  CALL_SUBTEST(test_spmv_alpha_beta<Scalar>(64));
  CALL_SUBTEST(test_spmv_transpose<Scalar>(128, 64));
  CALL_SUBTEST(test_spmm<Scalar>(64, 64, 4));
  CALL_SUBTEST(test_identity<Scalar>(64));
  CALL_SUBTEST(test_reuse<Scalar>(64));
  CALL_SUBTEST(test_empty<Scalar>());
}

EIGEN_DECLARE_TEST(gpu_cusparse_spmv) {
  CALL_SUBTEST(test_scalar<float>());
  CALL_SUBTEST(test_scalar<double>());
  CALL_SUBTEST(test_scalar<std::complex<float>>());
  CALL_SUBTEST(test_scalar<std::complex<double>>());
}
