// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 desire Nuentsa <desire.nuentsa_wakam@inria.fr
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T>
void test_dgmres_T() {
  DGMRES<SparseMatrix<T>, DiagonalPreconditioner<T> > dgmres_colmajor_diag;
  DGMRES<SparseMatrix<T>, IdentityPreconditioner> dgmres_colmajor_I;
  DGMRES<SparseMatrix<T>, IncompleteLUT<T> > dgmres_colmajor_ilut;
  // GMRES<SparseMatrix<T>, SSORPreconditioner<T> >     dgmres_colmajor_ssor;

  CALL_SUBTEST(check_sparse_square_solving(dgmres_colmajor_diag));
  //   CALL_SUBTEST( check_sparse_square_solving(dgmres_colmajor_I)     );
  CALL_SUBTEST(check_sparse_square_solving(dgmres_colmajor_ilut));
  // CALL_SUBTEST( check_sparse_square_solving(dgmres_colmajor_ssor)     );
}

// Regression: Arnoldi breakdown used to divide by zero (producing NaN in the
// Krylov basis) and solve a singular triangular system, silently returning
// Inf with info() == Success. Exercise both the pathological (rank-deficient
// pivot) and benign (exact Krylov subspace) breakdown paths.
template <typename T>
void test_dgmres_breakdown_T() {
  typedef SparseMatrix<T> Mat;
  typedef Matrix<T, 2, 1> Vec;

  // Nilpotent A with singular Hessenberg pivot on the first step.
  Mat A(2, 2);
  A.insert(0, 1) = T(1);
  A.makeCompressed();
  Vec b;
  b << T(1), T(0);

  DGMRES<Mat, IdentityPreconditioner> solver;
  solver.compute(A);
  Vec x = solver.solve(b);
  VERIFY(x.allFinite());
  VERIFY(solver.info() != Success);

  // Diagonal A with b in an eigenspace: Arnoldi converges after one step.
  Mat D(2, 2);
  D.insert(0, 0) = T(2);
  D.insert(1, 1) = T(2);
  D.makeCompressed();
  Vec d;
  d << T(2), T(2);

  DGMRES<Mat, DiagonalPreconditioner<T> > solver2;
  solver2.compute(D);
  Vec y = solver2.solve(d);
  VERIFY_IS_EQUAL(solver2.info(), Success);
  VERIFY_IS_APPROX(y, (Vec() << T(1), T(1)).finished());
}

// Regression: dgmres() used m_iterations only as the iteration cap and never
// wrote the performed count back, so iterations() returned maxIterations()
// after every solve, however quickly it converged.
template <typename T>
void test_dgmres_iterations_T() {
  using Mat = SparseMatrix<T>;
  using DenseMat = Matrix<T, Dynamic, Dynamic>;
  using Vec = Matrix<T, Dynamic, 1>;
  using RealScalar = typename NumTraits<T>::Real;

  // Well-conditioned tridiagonal system. Its size stays below the default
  // restart length of 30, so a converged solve ends inside the first cycle.
  const Index n = 20;
  Mat A(n, n);
  A.reserve(3 * n);
  for (Index i = 0; i < n; ++i) {
    if (i > 0) A.insert(i, i - 1) = T(-1);
    A.insert(i, i) = T(4);
    if (i + 1 < n) A.insert(i, i + 1) = T(-1);
  }
  A.makeCompressed();

  const Index max_iters = 500;
  // The system is well conditioned, so the true residual tracks the tolerance
  // the solver converged to, which defaults to NumTraits<Scalar>::epsilon().
  const RealScalar res_bound = RealScalar(64) * NumTraits<RealScalar>::epsilon();

  Vec b = Vec::Constant(n, T(1));
  DGMRES<Mat, DiagonalPreconditioner<T> > solver;
  solver.setMaxIterations(max_iters);
  solver.compute(A);
  Vec x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY(solver.iterations() > 0);
  VERIFY(solver.iterations() < solver.maxIterations());
  VERIFY((A * x - b).norm() <= res_bound * b.norm());

  // Zero right hand side: the early return reports no iteration at all.
  Vec zero = Vec::Zero(n);
  Vec x0 = solver.solve(zero);
  VERIFY(x0.isZero());
  VERIFY_IS_EQUAL(solver.iterations(), Index(0));

  // Several right hand sides: the cap is restored for every column, so a cheap
  // first column must not throttle a costlier second one.
  DenseMat B = DenseMat::Zero(n, 2);
  B.col(0).setConstant(T(1));
  B(0, 1) = T(1);
  DGMRES<Mat, DiagonalPreconditioner<T> > multi;
  multi.setMaxIterations(max_iters);
  multi.compute(A);
  DenseMat X = multi.solve(B);
  VERIFY_IS_EQUAL(multi.info(), Success);
  VERIFY(multi.iterations() > 0);
  VERIFY(multi.iterations() < multi.maxIterations());
  VERIFY((A * X - B).norm() <= res_bound * B.norm());

  // Non-converging direction: GMRES stagnates on a cyclic shift matrix, so the
  // reported count saturates at the cap and never exceeds it.
  const Index m = 16;
  Mat S(m, m);
  S.reserve(m);
  for (Index i = 0; i < m; ++i) S.insert((i + 1) % m, i) = T(1);
  S.makeCompressed();
  Vec e = Vec::Zero(m);
  e(0) = T(1);
  for (Index k = 1; k <= 3; ++k) {
    DGMRES<Mat, IdentityPreconditioner> stalled;
    stalled.setMaxIterations(k);
    stalled.compute(S);
    Vec xs = stalled.solve(e);
    VERIFY(xs.allFinite());
    VERIFY_IS_EQUAL(stalled.info(), NoConvergence);
    VERIFY(stalled.iterations() <= stalled.maxIterations());
    VERIFY_IS_EQUAL(stalled.iterations(), k);
    // A zero right hand side after a failed solve reports success, not the previous solve's status.
    xs = stalled.solve(Vec::Zero(m));
    VERIFY(xs.isZero());
    VERIFY_IS_EQUAL(stalled.info(), Success);
  }
}

// The Arnoldi coefficient is h(i,j) = v_i^H A v_j; computing its conjugate broke the orthogonalization for complex
// scalars, so restarted DGMRES needed about twice the iterations of the equivalent GMRES(30).
void test_dgmres_complex_arnoldi() {
  using T = std::complex<double>;
  const Index n = 60;
  SparseMatrix<T> A(n, n);
  for (Index i = 0; i < n; ++i) {
    A.insert(i, i) = T(4, 1.0 + 0.05 * double(i));
    if (i > 0) A.insert(i, i - 1) = T(-1, 0.7);
    if (i + 1 < n) A.insert(i, i + 1) = T(0.5, -1.3);
  }
  DGMRES<SparseMatrix<T>, IdentityPreconditioner> solver(A);
  solver.setTolerance(1e-12);
  VectorXcd x = solver.solve(VectorXcd::Ones(n));
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY(solver.iterations() <= 45);
}

// Exposes the deflation data, U and T = U^H A M^{-1} U, to the test below.
template <typename MatrixType, typename Preconditioner>
struct DGMRESDeflation : DGMRES<MatrixType, Preconditioner> {
  using Base = DGMRES<MatrixType, Preconditioner>;
  using Base::m_r;
  using Base::m_T;
  using Base::m_U;
};

// Deflation took Schur vectors that do not span an invariant subspace, built T from M^{-1} A instead of the Arnoldi
// operator A M^{-1}, used transpose() for complex scalars, kept the deflation subspace of the previous right-hand
// side, and reset lambda_N after setting it, so the first deflated preconditioner I - U U^H was singular. On this
// row-scaled convection-diffusion stencil setEigenv(k) stagnated for k >= 2.
template <typename T, typename Preconditioner>
void test_dgmres_deflation(const T& imagPart) {
  using RealScalar = typename NumTraits<T>::Real;
  using Vector = Matrix<T, Dynamic, 1>;
  using DenseMatrix = Matrix<T, Dynamic, Dynamic>;
  const Index g = 20, n = g * g;
  SparseMatrix<T> A(n, n);
  for (Index i = 0; i < g; ++i) {
    for (Index j = 0; j < g; ++j) {
      const Index k = i * g + j;
      const T s = T(1 + 0.5 * double(k % 3));
      A.insert(k, k) = s * T(4);
      if (j + 1 < g) A.insert(k, k + 1) = s * (T(-1.3) + T(0.3) * imagPart);
      if (j > 0) A.insert(k, k - 1) = s * (T(-0.7) - T(0.3) * imagPart);
      if (i + 1 < g) A.insert(k, k + g) = s * T(-1.1);
      if (i > 0) A.insert(k, k - g) = s * T(-0.9);
    }
  }
  A.makeCompressed();
  Matrix<T, Dynamic, 2> B(n, 2);
  B.col(0).setOnes();
  B.col(1) = Vector::LinSpaced(n, T(-1), T(1));
  Index iterations0 = 0, iterationsB0 = 0;
  for (Index neig : {0, 1, 2, 4}) {
    DGMRESDeflation<SparseMatrix<T>, Preconditioner> solver;
    solver.set_restart(20);
    solver.setEigenv(neig);
    solver.setTolerance(RealScalar(1e-10));
    solver.setMaxIterations(1000);
    solver.compute(A);
    Vector x = solver.solve(B.col(0));
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY((A * x - B.col(0)).norm() <= RealScalar(1e-8) * B.col(0).norm());
    if (neig == 0) {
      iterations0 = solver.iterations();
    } else {
      VERIFY(solver.iterations() <= iterations0);
      const Index r = solver.m_r;
      VERIFY(r >= neig);
      const DenseMatrix U = solver.m_U.leftCols(r);
      VERIFY_IS_APPROX(U.adjoint() * U, DenseMatrix::Identity(r, r));
      DenseMatrix BU(n, r);
      for (Index j = 0; j < r; ++j) BU.col(j) = A * solver.preconditioner().solve(U.col(j));
      VERIFY_IS_APPROX(solver.m_T.topLeftCorner(r, r), U.adjoint() * BU);
    }
    // Each right-hand side builds its own deflation subspace.
    Matrix<T, Dynamic, 2> X = solver.solve(B);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY((A * X - B).norm() <= RealScalar(1e-8) * B.norm());
    if (neig == 0)
      iterationsB0 = solver.iterations();
    else
      VERIFY(solver.iterations() < iterationsB0);
  }
}

// A real conjugate pair of Ritz values is deflated whole. The smallest eigenvalues here are about 0.002 +- 0.04i, so
// setEigenv(1) with room for one update deflates a two-dimensional subspace.
void test_dgmres_deflation_conjugate_pair() {
  const Index g = 20, n = g * g;
  SparseMatrix<double> A(n, n);
  for (Index i = 0; i < g; ++i) {
    for (Index j = 0; j < g; ++j) {
      const Index k = i * g + j;
      const double s = k < 2 ? 0.1 : 1.0;
      A.insert(k, k) = 4 * s;
      if (j + 1 < g) A.insert(k, k + 1) = -1.3 * s;
      if (j > 0) A.insert(k, k - 1) = -0.7 * s;
      if (i + 1 < g) A.insert(k, k + g) = -1.1 * s;
      if (i > 0) A.insert(k, k - g) = -0.9 * s;
    }
  }
  A.coeffRef(0, 0) = A.coeffRef(1, 1) = A.coeffRef(0, 1) = 0.05;
  A.coeffRef(1, 0) = -0.05;
  const VectorXd b = VectorXd::Ones(n);
  DGMRES<SparseMatrix<double>, IdentityPreconditioner> solver;
  solver.set_restart(20);
  solver.setEigenv(1);
  solver.setMaxEigenv(2);
  solver.setTolerance(1e-10);
  solver.setMaxIterations(2000);
  solver.compute(A);
  VectorXd x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(solver.deflSize(), Index(2));
  VERIFY((A * x - b).norm() <= 1e-8 * b.norm());
}

// The deflation subspace never exceeds the dimension: with n = 3 and restart 2 it reached four vectors, which made T
// singular.
void test_dgmres_deflation_small() {
  const Index n = 3;
  SparseMatrix<double> A(n, n);
  for (Index i = 0; i < n; ++i) {
    A.insert(i, i) = 2.0 + 0.3 * double(i);
    if (i > 0) A.insert(i, i - 1) = -1.2;
    if (i + 1 < n) A.insert(i, i + 1) = -0.8;
  }
  const VectorXd b = VectorXd::LinSpaced(n, 1, 2);
  DGMRES<SparseMatrix<double>, IdentityPreconditioner> solver;
  solver.set_restart(2);
  solver.setEigenv(1);
  solver.setTolerance(1e-12);
  solver.setMaxIterations(200);
  solver.compute(A);
  VectorXd x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY(solver.deflSize() <= n);
  VERIFY_IS_APPROX(A * x, b);
}

EIGEN_DECLARE_TEST(dgmres) {
  CALL_SUBTEST_1(test_dgmres_T<double>());
  CALL_SUBTEST_2(test_dgmres_T<std::complex<double> >());
  CALL_SUBTEST_3(test_dgmres_breakdown_T<double>());
  CALL_SUBTEST_4(test_dgmres_breakdown_T<std::complex<double> >());
  CALL_SUBTEST_5(test_dgmres_iterations_T<double>());
  CALL_SUBTEST_6(test_dgmres_iterations_T<std::complex<double> >());
  CALL_SUBTEST_6(test_dgmres_complex_arnoldi());
  CALL_SUBTEST_7((test_dgmres_deflation<double, IdentityPreconditioner>(0.0)));
  CALL_SUBTEST_7((test_dgmres_deflation<double, DiagonalPreconditioner<double> >(0.0)));
  CALL_SUBTEST_7(test_dgmres_deflation_conjugate_pair());
  CALL_SUBTEST_7(test_dgmres_deflation_small());
  CALL_SUBTEST_8((test_dgmres_deflation<std::complex<double>, IdentityPreconditioner>(std::complex<double>(0, 1))));
  CALL_SUBTEST_8((test_dgmres_deflation<std::complex<double>, DiagonalPreconditioner<std::complex<double> > >(
      std::complex<double>(0, 1))));
}
