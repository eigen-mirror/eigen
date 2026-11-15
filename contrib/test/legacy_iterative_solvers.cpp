// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

// The legacy header must keep making the modules that Eigen 5.0 included from it available: Sparse (including
// SparseCholesky, SparseLU and SparseQR), Jacobi and Householder, besides the iterative solvers.
#define EIGEN_NO_DEPRECATED_WARNING
#include <unsupported/Eigen/IterativeSolvers>

void check_legacy_iterative_solvers() {
  using SpMat = SparseMatrix<double>;
  const int n = 8;
  SpMat A(n, n);
  for (int i = 0; i < n; ++i) {
    A.insert(i, i) = 4.0;
    if (i > 0) A.insert(i, i - 1) = -1.0;
    if (i + 1 < n) A.insert(i, i + 1) = -1.0;
  }
  A.makeCompressed();
  VectorXd b = VectorXd::LinSpaced(n, 1.0, 2.0);

  SimplicialLDLT<SpMat> ldlt(A);
  SparseLU<SpMat> lu(A);
  SparseQR<SpMat, COLAMDOrdering<int>> qr(A);
  GMRES<SpMat, IdentityPreconditioner> gmres(A);
  VectorXd x_ldlt = ldlt.solve(b);
  VectorXd x_lu = lu.solve(b);
  VectorXd x_qr = qr.solve(b);
  VectorXd x_gmres = gmres.solve(b);
  VERIFY_IS_APPROX(A * x_ldlt, b);
  VERIFY_IS_APPROX(A * x_lu, b);
  VERIFY_IS_APPROX(A * x_qr, b);
  VERIFY_IS_APPROX(A * x_gmres, b);

  JacobiRotation<double> rot;
  rot.makeGivens(3.0, 4.0);
  VERIFY_IS_APPROX(rot.c(), 0.6);
  VERIFY_IS_APPROX(rot.s(), -0.8);
  Vector3d essential;
  double tau, beta;
  Vector4d h = Vector4d::Random();
  h.makeHouseholder(essential, tau, beta);
  VERIFY_IS_APPROX(std::abs(beta), h.norm());
}

EIGEN_DECLARE_TEST(legacy_iterative_solvers) { CALL_SUBTEST(check_legacy_iterative_solvers()); }
