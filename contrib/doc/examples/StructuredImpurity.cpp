// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// A Koster-Slater impurity: a tight-binding ring of N sites with hopping t and
// one site shifted by V < 0. In the Bloch basis the band is diagonal,
// e_k = -2t cos(2 pi k / N), and the impurity is the rank-one update
// V z z^T with z_k = 1/sqrt(N), so H = D + V z z^T is diagonal plus rank one.
// DPR1EigenSolver finds all N eigenvalues in O(N^2) from the secular equation;
// one of them splits off below the band, approaching the infinite-chain bound
// state -sqrt(V^2 + 4 t^2).

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Eigen;

int main() {
  const double pi = std::acos(-1.0), t = 1.0, V = -1.5;
  std::cout << "     N  lowest eigenvalue  band bottom\n";
  for (int N : {16, 64, 256, 1024}) {
    VectorXd d(N);
    for (int k = 0; k < N; ++k) d[k] = -2 * t * std::cos(2 * pi * k / N);
    const VectorXd z = VectorXd::Constant(N, 1 / std::sqrt(double(N)));
    const DPR1EigenSolver<double> es(d, V, z, /*options=*/0);
    std::cout << std::setw(6) << N << "  " << std::setw(16) << es.eigenvalues()[0] << "  " << d.minCoeff() << "\n";
  }
  std::cout << "infinite chain: " << -std::sqrt(V * V + 4 * t * t) << "\n";
}
