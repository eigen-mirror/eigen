// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The measurement update of a Kalman filter in information form. A field x on
// n grid points has the prior N(0, tau^2 I); k sensors record weighted sums
// y = H x + e over short footprints, with noise e ~ N(0, s^2 I). The posterior
// precision
//   P^{-1} = tau^{-2} I + s^{-2} H^T H
// is diagonal plus rank k, and DiagonalPlusLowRank::inverse returns the
// posterior covariance P as another DiagonalPlusLowRank: the Woodbury identity,
// which is the Kalman update. The posterior mean P H^T y / s^2 is then one O(nk)
// product and the posterior variances are the diagonal of P, with no n x n
// matrix anywhere.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using namespace Eigen;

int main() {
  const int n = 400, k = 40, width = 4;
  const double tau = 1.0, s = 0.05;
  // Sensor j averages the points [100 + 2 j, 100 + 2 j + width); neighboring footprints overlap.
  MatrixXd H = MatrixXd::Zero(k, n);
  for (int j = 0; j < k; ++j) H.row(j).segment(100 + 2 * j, width).setConstant(1.0 / width);
  VectorXd x(n);
  for (int i = 0; i < n; ++i) x[i] = std::sin(0.05 * i) + 0.5 * std::cos(0.2 * i);
  std::minstd_rand generator(1);
  VectorXd y = H * x;
  for (int j = 0; j < k; ++j) y[j] += s * (2.0 * generator() / generator.max() - 1);

  const auto information = makeDiagonalPlusLowRank(VectorXd::Constant(n, 1 / (tau * tau)), H.transpose() / (s * s),
                                                   H.transpose());  // P^{-1}
  const auto P = information.inverse();
  const VectorXd mean = P * VectorXd(H.transpose() * y / (s * s));

  const MatrixXd dense = (MatrixXd::Identity(n, n) / (tau * tau) + H.transpose() * H / (s * s)).inverse();
  const VectorXd denseMean = dense * H.transpose() * y / (s * s);
  double varianceError = 0;
  for (int i = 0; i < n; ++i) varianceError = std::max(varianceError, std::abs(P.coeff(i, i) - dense(i, i)));

  std::cout << "point  footprints  truth    posterior mean  posterior std\n";
  for (int i : {50, 100, 101, 140, 179, 250}) {
    const int footprints = int((H.col(i).array() != 0).count());
    std::cout << std::setw(5) << i << std::setw(12) << footprints << std::setw(7) << std::setprecision(3) << x[i]
              << std::setw(16) << mean[i] << std::setw(15) << std::sqrt(P.coeff(i, i)) << "\n";
  }
  std::cout << std::setprecision(6) << "max |mean - dense|: " << (mean - denseMean).cwiseAbs().maxCoeff()
            << ", max |variance - dense|: " << varianceError << "\n";
}
