// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Deconvolution as a rectangular Toeplitz least-squares problem. A signal s of
// length n recorded through a causal kernel h of length k is the full
// convolution y = T s, with T the (n + k - 1) x n Toeplitz matrix whose first
// column holds h. LSMR recovers s from y without forming T: every iteration
// applies T and T^T through the FFT, in O(p log p) for the embedding length
// p >= n + k - 1, the adjoint reusing the embedding symbol. For the kernel
// h_j = a^j, |a| < 1, the system is well conditioned, kappa ~ (1 + a)/(1 - a),
// so noise in y passes into s amplified by at most that factor. The dense QR
// of the materialized matrix gives the same solution at O(m n^2).

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>
#include <random>

using namespace Eigen;

int main() {
  const int n = 1000, k = 50, m = n + k - 1;
  const double a = 0.9;
  // The signal: two Gaussian pulses and a step.
  VectorXd s(n);
  for (int i = 0; i < n; ++i)
    s[i] = std::exp(-std::pow((i - 300) / 20.0, 2)) + 0.5 * std::exp(-std::pow((i - 450) / 10.0, 2)) +
           (i > 700 && i < 850 ? 0.8 : 0.0);
  VectorXd column = VectorXd::Zero(m), row = VectorXd::Zero(n);
  for (int j = 0; j < k; ++j) column[j] = std::pow(a, j);
  row[0] = column[0];
  const auto T = makeToeplitz(column, row);  // m x n
  const VectorXd y = T * s;

  LSMR<std::decay_t<decltype(T)>, IdentityPreconditioner> lsmr(T);
  lsmr.setTolerance(1e-12);
  const VectorXd recovered = lsmr.solve(y);
  std::cout << "LSMR, exact data:  " << lsmr.iterations() << " iterations, max error "
            << (recovered - s).cwiseAbs().maxCoeff() << "\n";

  // 1% noise from a generator whose sequence the C++ standard fixes.
  std::minstd_rand generator(1);
  VectorXd noisy = y;
  for (int i = 0; i < m; ++i) noisy[i] += 0.01 * (2.0 * generator() / generator.max() - 1);
  const VectorXd fromNoisy = lsmr.solve(noisy);
  std::cout << "LSMR, noisy data:  " << lsmr.iterations() << " iterations, max error "
            << (fromNoisy - s).cwiseAbs().maxCoeff() << "\n";

  const MatrixXd dense = T;
  std::cout << "dense QR of the " << m << " x " << n << " matrix, noisy data: max error "
            << (dense.householderQr().solve(noisy) - s).cwiseAbs().maxCoeff() << "\n";
}
