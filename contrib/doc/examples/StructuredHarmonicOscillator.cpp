// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The 2-D quantum harmonic oscillator H = -(d_xx + d_yy) / 2 + (x^2 + y^2) / 2
// separates as H = H_1 (+) H_1 with the 1-D Hamiltonian H_1, so its spectrum
// is the set of pairwise sums of the 1-D energies -- here from a
// finite-difference H_1 on [-8, 8], without ever forming the 2-D matrix. The
// exact energies are n_x + n_y + 1, with degeneracy n_x + n_y + 1.

#include <contrib/Eigen/StructuredMatrices>
#include <algorithm>
#include <iostream>

using namespace Eigen;

int main() {
  const int n = 400;
  const double a = 8, h = 2 * a / (n + 1);
  MatrixXd H1 = MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    const double x = -a + (i + 1) * h;
    H1(i, i) = 1 / (h * h) + x * x / 2;
    if (i + 1 < n) H1(i, i + 1) = H1(i + 1, i) = -1 / (2 * h * h);
  }
  const auto H = makeKroneckerSum(H1, H1);
  VectorXd energies = H.eigenvalues().real();
  std::sort(energies.data(), energies.data() + energies.size());
  std::cout << H.rows() << " x " << H.cols() << " Hamiltonian, lowest energies:\n"
            << energies.head(10).transpose() << "\n";
}
