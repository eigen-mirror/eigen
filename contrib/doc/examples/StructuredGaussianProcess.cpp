// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The log-likelihood of observations y under a Gaussian process with k Fourier
// features on a fixed frequency grid and noise variance s^2,
//   log N(y | 0, S) = -(y^T S^{-1} y + log det S + n log(2 pi)) / 2,
//   S = s^2 I + U U^T,  U = (n x k feature matrix),
// costs O(n k^2) with DiagonalPlusLowRank: the Woodbury identity for the solve
// and the matrix determinant lemma for the determinant, instead of the O(n^3)
// Cholesky factorization of the dense covariance.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>

using namespace Eigen;

int main() {
  const double pi = std::acos(-1.0), noise = 1.0;
  const int n = 1000, k = 20;
  VectorXd t(n), y(n);
  for (int i = 0; i < n; ++i) {
    t[i] = 10.0 * i / n;
    y[i] = std::sin(t[i]) + 0.3 * std::cos(3 * t[i]);
  }
  // Features sqrt(2/k) cos(omega_j t + phase_j) on a fixed frequency grid.
  MatrixXd U(n, k);
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < n; ++i) U(i, j) = std::sqrt(2.0 / k) * std::cos(0.25 * j * t[i] + 0.5 * j);

  const auto S = makeDiagonalPlusLowRank(VectorXd::Constant(n, noise * noise), U, U);
  const double logLikelihood = -(y.dot(S.solve(y)) + std::log(S.determinant()) + n * std::log(2 * pi)) / 2;

  const MatrixXd dense = noise * noise * MatrixXd::Identity(n, n) + U * U.transpose();
  const LLT<MatrixXd> llt(dense);
  const double logDet = 2 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
  const double reference = -(y.dot(llt.solve(y)) + logDet + n * std::log(2 * pi)) / 2;
  std::cout << "log-likelihood, Woodbury: " << logLikelihood << "\n";
  std::cout << "log-likelihood, dense:    " << reference << "\n";
}
