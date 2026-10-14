// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Steady convection-diffusion -eps (u_xx + u_yy) + b_x u_x + b_y u_y = 1 on the
// unit square, u = 0 on the boundary, with centered differences. The 1-D
// operators eps D + b G (D = tridiag(-1, 2, -1) / h^2, G = tridiag(-1, 0, 1) / 2h)
// are not symmetric, so BartelsStewart takes the Schur path; the result agrees
// with SparseLU on the assembled matrix.

#include <contrib/Eigen/StructuredMatrices>
#include <iostream>

using namespace Eigen;

SparseMatrix<double> convectionDiffusion(int n, double eps, double b) {
  const double h = 1.0 / (n + 1);
  SparseMatrix<double> A(n, n);
  A.reserve(VectorXi::Constant(n, 3));  // at most 3 nonzeros per column
  for (int i = 0; i < n; ++i) {
    A.insert(i, i) = 2 * eps / (h * h);
    if (i > 0) A.insert(i, i - 1) = -eps / (h * h) - b / (2 * h);
    if (i + 1 < n) A.insert(i, i + 1) = -eps / (h * h) + b / (2 * h);
  }
  A.makeCompressed();
  return A;
}

int main() {
  const int n = 100;
  const double eps = 0.05;
  const auto L = makeKroneckerSum(convectionDiffusion(n, eps, 0.5), convectionDiffusion(n, eps, 1.0));
  const BartelsStewart<std::decay_t<decltype(L)>> solver(L);
  const VectorXd f = VectorXd::Ones(n * n);
  const VectorXd u = solver.solve(f);

  SparseMatrix<double> A;
  A = L;  // the assembled n^2 x n^2 matrix, for comparison
  const VectorXd uLU = SparseLU<SparseMatrix<double>>(A).solve(f);
  std::cout << "fast diagonalization: " << (solver.isHermitian() ? "yes" : "no") << "\n";
  std::cout << "max u:                " << u.maxCoeff() << "\n";
  std::cout << "relative residual:    " << (A * u - f).norm() / f.norm() << "\n";
  std::cout << "|u - u_SparseLU|/|u|: " << (u - uLU).norm() / u.norm() << "\n";
}
