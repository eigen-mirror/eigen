// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The 2-D Poisson equation -(u_xx + u_yy) = f on the unit square, u = 0 on the
// boundary, with second-order finite differences on an n x n interior grid.
// Its matrix is the Kronecker sum D (+) D of the 1-D matrix
// D = tridiag(-1, 2, -1) / h^2, which BartelsStewart solves by fast
// diagonalization: two n x n eigendecompositions instead of a sparse
// factorization of the n^2 x n^2 matrix. The error falls as O(h^2).

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
  for (int n : {16, 32, 64, 128}) {
    const double h = 1.0 / (n + 1);
    // Grid values in Kronecker order: entry j*n + i belongs to (x_i, y_j).
    VectorXd exact(n * n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) exact[j * n + i] = std::sin(pi * (i + 1) * h) * std::sin(2 * pi * (j + 1) * h);
    const VectorXd f = 5 * pi * pi * exact;

    const SparseMatrix<double> D = secondDifference(n);
    const auto L = makeKroneckerSum(D, D);  // y on the left (slow index), x on the right
    const BartelsStewart<std::decay_t<decltype(L)>> poisson(L);
    const VectorXd u = poisson.solve(f);
    std::cout << "n = " << n << ": max error " << (u - exact).cwiseAbs().maxCoeff() << "\n";
  }
}
