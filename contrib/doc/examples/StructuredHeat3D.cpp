// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Implicit Euler for the heat equation u_t = u_xx + u_yy + u_zz on the unit
// cube: each step solves (I + tau (D (+) D (+) D)) u_new = u_old, with
// D = tridiag(-1, 2, -1) / h^2. The identity goes into one factor,
// I + tau (D (+) D (+) D) = (I + tau D) (+) tau D (+) tau D, and the solver
// computes three n x n eigendecompositions once for all steps. The initial
// state is a discrete eigenmode, so the computed decay can be checked exactly.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>

using namespace Eigen;

SparseMatrix<double> secondDifference(int n) {
  const double h = 1.0 / (n + 1);
  SparseMatrix<double> D(n, n);
  D.reserve(VectorXi::Constant(n, 3));  // at most 3 nonzeros per column
  for (int i = 0; i < n; ++i) {
    D.insert(i, i) = 2 / (h * h);
    if (i > 0) D.insert(i, i - 1) = -1 / (h * h);
    if (i + 1 < n) D.insert(i, i + 1) = -1 / (h * h);
  }
  D.makeCompressed();
  return D;
}

int main() {
  const double pi = std::acos(-1.0);
  const int n = 24, steps = 100;
  const double h = 1.0 / (n + 1), tau = 1e-3;
  const SparseMatrix<double> D = secondDifference(n);
  SparseMatrix<double> I(n, n);
  I.setIdentity();
  const auto M =
      makeKroneckerSum(SparseMatrix<double>(I + tau * D), SparseMatrix<double>(tau * D), SparseMatrix<double>(tau * D));
  const BartelsStewart<std::decay_t<decltype(M)>> step(M);

  // u0 = sin(pi x) sin(pi y) sin(pi z), an eigenvector of D (+) D (+) D.
  VectorXd s(n);
  for (int i = 0; i < n; ++i) s[i] = std::sin(pi * (i + 1) * h);
  const MatrixXd modes = (s * s.transpose()).reshaped() * s.transpose();
  const VectorXd u0 = modes.reshaped();
  VectorXd u = u0;
  for (int k = 0; k < steps; ++k) u = step.solve(u);

  const double lambda = 3 * 4 / (h * h) * std::pow(std::sin(pi * h / 2), 2);
  const double amplitude = u.dot(u0) / u0.squaredNorm();
  std::cout << n * n * n << " unknowns, " << steps << " steps\n";
  std::cout << "amplitude:               " << amplitude << "\n";
  std::cout << "implicit Euler, exact:   " << std::pow(1 + tau * lambda, -steps) << "\n";
  std::cout << "heat equation, exact:    " << std::exp(-3 * pi * pi * tau * steps) << "\n";
}
