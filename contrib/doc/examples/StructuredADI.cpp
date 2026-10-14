// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Peaceman-Rachford ADI for u_t = u_xx + u_yy on the unit square. With x the
// fast grid index, the two directions are I (x) D and D (x) I, and each half
// step is an explicit product in one direction and a tridiagonal solve in the
// other:
//   (I + tau/2 I (x) D) u_half = (I - tau/2 D (x) I) u_old,
//   (I + tau/2 D (x) I) u_new  = (I - tau/2 I (x) D) u_half.
// The Identity() factors cost nothing: the products are sparse products with D
// and the solves one SparseLU of the tridiagonal I + tau/2 D.

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
  const int n = 200, steps = 50;
  const double h = 1.0 / (n + 1), tau = 1e-3;
  const SparseMatrix<double> D = secondDifference(n);
  SparseMatrix<double> I(n, n);
  I.setIdentity();
  const auto Id = MatrixXd::Identity(n, n);
  const auto xImplicit = makeKroneckerOperator(Id, SparseMatrix<double>(I + tau / 2 * D));
  const auto yImplicit = makeKroneckerOperator(SparseMatrix<double>(I + tau / 2 * D), Id);
  const auto xExplicit = makeKroneckerOperator(Id, SparseMatrix<double>(I - tau / 2 * D));
  const auto yExplicit = makeKroneckerOperator(SparseMatrix<double>(I - tau / 2 * D), Id);

  // u0 = sin(pi x) sin(2 pi y), an eigenvector of both directions.
  VectorXd sx(n), sy(n);
  for (int i = 0; i < n; ++i) {
    sx[i] = std::sin(pi * (i + 1) * h);
    sy[i] = std::sin(2 * pi * (i + 1) * h);
  }
  const VectorXd u0 = (sx * sy.transpose()).reshaped();  // entry j*n + i is (x_i, y_j)
  VectorXd u = u0;
  for (int k = 0; k < steps; ++k) {
    const VectorXd half = xImplicit.solve(yExplicit * u);
    u = yImplicit.solve(xExplicit * half);
  }
  const auto factor = [&](double lambda) { return (1 - tau / 2 * lambda) / (1 + tau / 2 * lambda); };
  const double lx = 4 / (h * h) * std::pow(std::sin(pi * h / 2), 2), ly = 4 / (h * h) * std::pow(std::sin(pi * h), 2);
  std::cout << n * n << " unknowns, " << steps << " ADI steps\n";
  std::cout << "amplitude:       " << u.dot(u0) / u0.squaredNorm() << "\n";
  std::cout << "ADI, exact:      " << std::pow(factor(lx) * factor(ly), steps) << "\n";
  std::cout << "PDE, exact:      " << std::exp(-5 * pi * pi * tau * steps) << "\n";
}
