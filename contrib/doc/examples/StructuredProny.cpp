// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Prony's method fits a sum of r complex exponentials, f_t = sum_j c_j z_j^t,
// to equally spaced samples f_0, ..., f_{m-1}. Every such sequence obeys the
// linear recurrence whose characteristic polynomial has the z_j as roots,
//   f_{t+r} + a_{r-1} f_{t+r-1} + ... + a_0 f_t = 0,
// so the coefficients a solve a Hankel system in the samples, H(t, k) = f_{t+k},
// and the recurrence residual over all samples is a rectangular Hankel product.
// The roots are the nodes z_j, with damping -log|z_j| and frequency arg z_j; the
// amplitudes solve the dual Vandermonde system V^T c = f, V(j, t) = z_j^t, with
// complex nodes, by Bjorck-Pereyra. A real signal gives conjugate pairs.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>

using namespace Eigen;

int main() {
  const int r = 4, m = 64;
  // Two damped sinusoids A exp(-alpha t) cos(omega t + phi): four exponentials.
  const double alpha[2] = {0.02, 0.05}, omega[2] = {0.3, 1.1}, A[2] = {1.0, 0.5}, phi[2] = {0.0, 0.7};
  VectorXd f = VectorXd::Zero(m);
  for (int t = 0; t < m; ++t)
    for (int j = 0; j < 2; ++j) f[t] += A[j] * std::exp(-alpha[j] * t) * std::cos(omega[j] * t + phi[j]);

  // The r x r Hankel system on the first 2r samples, and the recurrence on all m.
  const VectorXd a = makeHankel(f.head(r), f.segment(r - 1, r)).solve(-f.segment(r, r));
  const auto H = makeHankel(f.head(m - r), f.segment(m - r - 1, r));
  std::cout << "recurrence residual over " << m << " samples: " << (H * a + f.tail(m - r)).norm() << "\n";

  // The roots of z^r + a_{r-1} z^{r-1} + ... + a_0 are the eigenvalues of its companion matrix.
  MatrixXd companion = MatrixXd::Zero(r, r);
  companion.bottomLeftCorner(r - 1, r - 1).setIdentity();
  companion.col(r - 1) = -a;
  const VectorXcd z = EigenSolver<MatrixXd>(companion, false).eigenvalues();
  const BjorckPereyra<std::complex<double>> amplitudes(makeVandermonde(z));
  const VectorXcd c = amplitudes.transpose().solve(f.head(r).cast<std::complex<double>>());

  std::cout << std::fixed << std::setprecision(4) << "  damping  frequency  amplitude  phase\n";
  for (int j = 0; j < r; ++j)
    if (z[j].imag() > 0)  // one root of each conjugate pair; c_j = (A / 2) exp(i phi)
      std::cout << std::setw(9) << -std::log(std::abs(z[j])) << std::setw(11) << std::arg(z[j]) << std::setw(11)
                << 2 * std::abs(c[j]) << std::setw(8) << std::arg(c[j]) << "\n";
}
