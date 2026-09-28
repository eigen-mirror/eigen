// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// End-to-end tests of conjugate gradient on the GPU types: Eigen's ConjugateGradient
// class on DeviceSparseView and DeviceMatrix through solveWithGuessInPlace(), and the
// module's hand-written loop with DeviceScalar reductions. Both are checked against
// the CPU ConjugateGradient.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <contrib/Eigen/GPU>
#include "gpu_test_helpers.h"

using namespace Eigen;

// ---- Helpers ----------------------------------------------------------------

// A = R^T R + n I: lambda_min >= n and lambda_max <= ||A||_F, so kappa(A) <= ||A||_F / n.
template <typename Scalar>
SparseMatrix<Scalar, ColMajor, int> make_spd(Index n, double density = 0.1) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  SpMat R(n, n);
  R.reserve(VectorXi::Constant(n, static_cast<int>(n * density) + 1));
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) {
      if (i == j || (std::rand() / double(RAND_MAX)) < density) {
        R.insert(i, j) = Scalar(std::rand() / double(RAND_MAX) - 0.5);
      }
    }
  }
  R.makeCompressed();
  SpMat A = R.adjoint() * R;
  for (Index i = 0; i < n; ++i) A.coeffRef(i, i) += Scalar(RealScalar(n));
  A.makeCompressed();
  return A;
}

template <typename Scalar>
double condition_bound(const SparseMatrix<Scalar, ColMajor, int>& A) {
  return double(A.norm()) / double(A.rows());
}

// The recursively updated residual keeps shrinking below eps, so a tolerance under eps
// still reports convergence while certifying nothing about x.
template <typename Scalar>
typename NumTraits<Scalar>::Real cg_tolerance() {
  return typename NumTraits<Scalar>::Real(100) * NumTraits<Scalar>::epsilon();
}

template <typename Scalar>
Matrix<Scalar, Dynamic, 1> cpu_reference(const SparseMatrix<Scalar, ColMajor, int>& A,
                                         const Matrix<Scalar, Dynamic, 1>& b, typename NumTraits<Scalar>::Real tol) {
  ConjugateGradient<SparseMatrix<Scalar, ColMajor, int>, Lower | Upper, IdentityPreconditioner> cpu_cg;
  cpu_cg.setMaxIterations(1000);
  cpu_cg.setTolerance(tol);
  cpu_cg.compute(A);
  Matrix<Scalar, Dynamic, 1> x = cpu_cg.solve(b);
  VERIFY_IS_EQUAL(cpu_cg.info(), Success);
  return x;
}

// At exit the recursive residual is below tol ||b||; the true residual differs by the
// rounding of k <~ 10 iterations of SpMV and vector updates, ~ k sqrt(n) eps ||b|| for random
// signs. Both solutions satisfy that bound, so
//   ||x_gpu - x_cpu|| <= ||A^-1|| (||r_gpu|| + ||r_cpu||) <= 2 kappa relres_bound ||x_cpu||.
// The residuals are evaluated in double so that a float check carries no rounding of its own.
template <typename Scalar>
void check_cg_solution(const SparseMatrix<Scalar, ColMajor, int>& A, const Matrix<Scalar, Dynamic, 1>& b,
                       const Matrix<Scalar, Dynamic, 1>& x_gpu, const Matrix<Scalar, Dynamic, 1>& x_cpu,
                       typename NumTraits<Scalar>::Real tol) {
  const double eps = double(NumTraits<Scalar>::epsilon());
  const double relres_bound = double(tol) + 10 * std::sqrt(double(A.rows())) * eps;
  const VectorXd r = A.template cast<double>() * x_gpu.template cast<double>() - b.template cast<double>();
  VERIFY(r.norm() <= relres_bound * b.template cast<double>().norm());
  const VectorXd dx = (x_gpu - x_cpu).template cast<double>();
  VERIFY(dx.norm() <= 2 * condition_bound(A) * relres_bound * x_cpu.template cast<double>().norm());
}

// ---- Eigen's ConjugateGradient class on device types ------------------------
// DeviceSparseView is a matrix-free matrix type, so the public solver class
// instantiates with it; solveWithGuessInPlace() is the entry point that takes
// non-Eigen vectors.
template <typename Scalar>
void test_gpu_cg_class(Index n) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  SpMat A = make_spd<Scalar>(n);
  Vec b = Vec::Random(n);
  const RealScalar tol = cg_tolerance<Scalar>();
  Vec x_cpu = cpu_reference(A, b, tol);

  gpu::Context ctx;
  gpu::Context::setThreadLocal(&ctx);
  gpu::SparseContext<Scalar> spmv_ctx(ctx);
  auto mat = spmv_ctx.deviceView(A);
  auto d_b = gpu::DeviceMatrix<Scalar>::fromHost(b, ctx.stream());
  gpu::DeviceMatrix<Scalar> d_x(n, 1);
  d_x.setZero(ctx);

  ConjugateGradient<gpu::DeviceSparseView<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.setMaxIterations(1000);
  cg.setTolerance(tol);
  cg.compute(mat);
  cg.solveWithGuessInPlace(d_b, d_x);
  VERIFY_IS_EQUAL(cg.info(), Success);
  VERIFY(cg.iterations() > 0 && cg.iterations() < 1000);
  VERIFY(cg.error() <= tol);

  // The deep copy and the scalar division the algorithm relies on (`p = precond.solve(r)`,
  // `residual /= residualScale`). 3 is not a power of two, so the quotient rounds; each element
  // must be within 2 eps of the host division.
  gpu::DeviceMatrix<Scalar> d_c = d_b;
  d_c /= Scalar(3);
  gpu::Context::setThreadLocal(nullptr);
  const Vec c = d_c.toHost(ctx.stream());
  const Vec c_ref = b / Scalar(3);
  VERIFY(((c - c_ref).cwiseAbs().array() <= RealScalar(2) * NumTraits<Scalar>::epsilon() * c_ref.cwiseAbs().array())
             .all());

  Vec x_gpu = d_x.toHost(ctx.stream());
  check_cg_solution(A, b, x_gpu, x_cpu, tol);
}

// ---- Extreme right-hand sides -----------------------------------------------
// GPU analog of test_conjugate_gradient_extreme_rhs, at scales where ||b||^2 underflows to 0 or
// overflows: a nrm2 without scaling then gives rhsNorm = 0 and x = 0 reported as converged, or
// rhsNorm = inf and NaN in x. At max/4 the residual norm is within a factor 3 of max, so
// `residual /= residualScale` must divide and `x += (residualScale * alpha) * p` stay finite.
template <typename Scalar>
void test_gpu_cg_extreme_rhs() {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Limits = std::numeric_limits<RealScalar>;

  SpMat A(2, 2);
  A.insert(0, 0) = Scalar(1);
  A.insert(1, 1) = Scalar(1);
  A.makeCompressed();
  Vec direction(2);
  direction << Scalar(1), Scalar(-1);

  gpu::Context ctx;
  gpu::Context::setThreadLocal(&ctx);
  gpu::SparseContext<Scalar> spmv_ctx(ctx);
  auto mat = spmv_ctx.deviceView(A);
  ConjugateGradient<gpu::DeviceSparseView<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.setTolerance(cg_tolerance<Scalar>());
  cg.compute(mat);

  const RealScalar scales[] = {numext::sqrt(Limits::denorm_min()) * RealScalar(1e-10),
                               numext::sqrt((Limits::max)()) * RealScalar(1e10), (Limits::max)() / RealScalar(4)};
  for (RealScalar scale : scales) {
    const Vec rhs = scale * direction;
    auto d_b = gpu::DeviceMatrix<Scalar>::fromHost(rhs, ctx.stream());
    for (RealScalar guess : {RealScalar(0), RealScalar(0.5)}) {
      const Vec x0 = guess * rhs;
      auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x0, ctx.stream());
      cg.solveWithGuessInPlace(d_b, d_x);
      VERIFY_IS_EQUAL(cg.info(), Success);
      VERIFY(cg.iterations() <= 1);
      const Vec x = d_x.toHost(ctx.stream());
      VERIFY(x.allFinite());
      VERIFY_IS_APPROX(x / scale, direction);
    }
  }
  gpu::Context::setThreadLocal(nullptr);
}

// ---- The module's hand-written CG loop --------------------------------------
// The README's loop: every scalar intermediate stays on device as a DeviceScalar and the
// convergence check is the one host sync per iteration. `jacobi` selects the diagonal
// preconditioner, applied as a cwiseProduct with 1 / diag(A).
template <typename Scalar>
void test_gpu_cg_loop(Index n, bool jacobi) {
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;

  SpMat A = make_spd<Scalar>(n);
  Vec b = Vec::Random(n);
  const RealScalar tol = cg_tolerance<Scalar>();
  Vec x_cpu = cpu_reference(A, b, tol);

  Vec invdiag(n);
  for (Index j = 0; j < n; ++j) invdiag(j) = Scalar(1) / A.coeff(j, j);

  gpu::Context ctx;
  gpu::Context::setThreadLocal(&ctx);
  gpu::SparseContext<Scalar> spmv_ctx(ctx);
  auto mat = spmv_ctx.deviceView(A);
  auto d_invdiag = gpu::DeviceMatrix<Scalar>::fromHost(invdiag, ctx.stream());

  auto d_b = gpu::DeviceMatrix<Scalar>::fromHost(b, ctx.stream());
  gpu::DeviceMatrix<Scalar> d_x(n, 1);
  d_x.setZero(ctx);

  // r = b (since x = 0)
  gpu::DeviceMatrix<Scalar> residual(n, 1);
  residual.copyFrom(ctx, d_b);

  RealScalar rhsNorm2 = d_b.squaredNorm(ctx);
  RealScalar threshold = tol * tol * rhsNorm2;
  RealScalar residualNorm2 = residual.squaredNorm(ctx);

  // p = precond.solve(r)
  gpu::DeviceMatrix<Scalar> p(n, 1);
  if (jacobi) {
    p.cwiseProduct(ctx, d_invdiag, residual);
  } else {
    p.copyFrom(ctx, residual);
  }
  gpu::DeviceMatrix<Scalar> z(n, 1), tmp(n, 1);

  auto absNew = residual.dot(ctx, p);
  Index maxIters = 1000;
  Index i = 0;
  while (i < maxIters) {
    tmp.noalias() = mat * p;

    auto alpha = absNew / p.dot(ctx, tmp);
    d_x += alpha * p;
    residual -= alpha * tmp;

    residualNorm2 = residual.squaredNorm(ctx);
    if (residualNorm2 < threshold) break;

    // z = precond.solve(r)
    if (jacobi) {
      z.cwiseProduct(ctx, d_invdiag, residual);
    } else {
      z.copyFrom(ctx, residual);
    }

    auto absOld = std::move(absNew);
    absNew = residual.dot(ctx, z);
    auto beta = absNew / absOld;

    p *= beta;
    p += z;
    i++;
  }

  gpu::Context::setThreadLocal(nullptr);

  Vec x_gpu = d_x.toHost(ctx.stream());
  check_cg_solution(A, b, x_gpu, x_cpu, tol);
}

EIGEN_DECLARE_TEST(gpu_cg) {
  gpu_test::require_cusparse_context();

  // Split by scalar so each part compiles in parallel.
  CALL_SUBTEST_1(test_gpu_cg_class<double>(64));
  CALL_SUBTEST_1(test_gpu_cg_class<double>(256));
  CALL_SUBTEST_1(test_gpu_cg_extreme_rhs<double>());
  CALL_SUBTEST_1(test_gpu_cg_loop<double>(64, false));
  CALL_SUBTEST_1(test_gpu_cg_loop<double>(256, false));
  CALL_SUBTEST_1(test_gpu_cg_loop<double>(256, true));
  CALL_SUBTEST_2(test_gpu_cg_class<float>(64));
  CALL_SUBTEST_2(test_gpu_cg_class<float>(256));
  CALL_SUBTEST_2(test_gpu_cg_extreme_rhs<float>());
  CALL_SUBTEST_2(test_gpu_cg_loop<float>(64, false));
  CALL_SUBTEST_2(test_gpu_cg_loop<float>(256, false));
  CALL_SUBTEST_2(test_gpu_cg_loop<float>(256, true));
}
