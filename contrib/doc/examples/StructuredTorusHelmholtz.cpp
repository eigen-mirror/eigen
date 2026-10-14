// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The screened Poisson equation -(u_xx + u_yy) + kappa^2 u = f on the periodic
// unit square, five-point stencil on an n x n grid. Periodic boundaries make
// the matrix a block circulant with circulant blocks, solved by Bccb in
// O(N log N) through the 2-D FFT. The same matrix is the Kronecker sum
// (C + kappa^2 I) (+) C of the 1-D periodic second-difference circulant C,
// which BartelsStewart solves by fast diagonalization; the two agree.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>

using namespace Eigen;

int main() {
  const double pi = std::acos(-1.0), kappa = 2.0;
  const int n = 64;
  const double h = 1.0 / n;
  // Stencil generator: column k holds the first column of block k.
  MatrixXd G = MatrixXd::Zero(n, n);
  G(0, 0) = 4 / (h * h) + kappa * kappa;
  G(1, 0) = G(n - 1, 0) = G(0, 1) = G(0, n - 1) = -1 / (h * h);
  const auto A = makeBccb(G);

  // u = cos(2 pi x) sin(4 pi y) is exact up to the O(h^2) stencil error.
  VectorXd exact(n * n), f(n * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      exact[j * n + i] = std::cos(2 * pi * i * h) * std::sin(4 * pi * j * h);
      f[j * n + i] = (20 * pi * pi + kappa * kappa) * exact[j * n + i];
    }
  const VectorXd u = A.solve(f);
  std::cout << "Bccb:               max error " << (u - exact).cwiseAbs().maxCoeff() << "\n";

  VectorXd column = VectorXd::Zero(n);
  column[0] = 2 / (h * h);
  column[1] = column[n - 1] = -1 / (h * h);
  MatrixXd C;
  C = makeCirculant(column);  // dense, for the Kronecker sum
  const auto L = makeKroneckerSum(MatrixXd(C + kappa * kappa * MatrixXd::Identity(n, n)), C);
  const VectorXd v = L.solve(f);
  std::cout << "KroneckerSum:       max error " << (v - exact).cwiseAbs().maxCoeff() << "\n";
  std::cout << "max |u_Bccb - u_KroneckerSum|: " << (u - v).cwiseAbs().maxCoeff() << "\n";
}
