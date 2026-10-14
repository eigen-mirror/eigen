// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Least-squares polynomial approximation in the monomial basis. The normal
// equations of the continuous problem, min \int_0^1 (f - sum_j c_j x^j)^2 dx,
// have the Hilbert matrix H_ij = 1/(i + j + 1), the Cauchy matrix with nodes
// x_i = i + 1 and y_j = -j, and the moments m_i = \int_0^1 x^i f dx on the right.
// CauchyLU solves them in O(n^2), and Cauchy::determinant evaluates det H in
// closed form where dense LU, at kappa(H_n) ~ e^{3.5 n}, loses digits. The
// coefficients inherit the conditioning, the fitted polynomial much less: the
// coefficient error d = H^{-1} r of a residual r = H c - m perturbs the fit by
// (d^T H d)^{1/2} = (r^T H^{-1} r)^{1/2} in L^2, so the fit loses at most about
// sqrt(kappa) eps where the coefficients can lose kappa eps. The discrete
// least-squares fit -- a dense QR of the rectangular Vandermonde matrix of 200
// sample points -- is the reference; Vandermonde products (Horner's rule)
// evaluate both fits.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Eigen;

int main() {
  const double e = std::exp(1.0);
  const ArrayXd samples = ArrayXd::LinSpaced(200, 0, 1), fine = ArrayXd::LinSpaced(1001, 0, 1);
  std::cout << " n  det H: Cauchy      dense LU       exact          fit error: Hilbert  QR\n";
  for (int n : {4, 6, 8, 10}) {
    const auto H = makeCauchy(VectorXd::LinSpaced(n, 1, n), -VectorXd::LinSpaced(n, 0, n - 1));
    // m_i = e - i m_{i-1}, run backwards from m_40 ~ e / 41: the forward recurrence amplifies errors by i!.
    VectorXd moments(n);
    double mi = e / 41;
    for (int i = 40; i > 0; --i) {
      mi = (e - mi) / i;
      if (i <= n) moments[i - 1] = mi;
    }
    const VectorXd c = CauchyLU<double>(H).solve(moments);

    const MatrixXd V = makeVandermonde(samples.matrix(), n);  // 200 x n
    const VectorXd cq = V.colPivHouseholderQr().solve(samples.exp().matrix());

    // det H_n = c_n^4 / c_{2n} with c_n = prod_{i<n} i!
    double cn = 1, c2n = 1, factorial = 1;
    for (int i = 1; i < 2 * n; ++i) {
      factorial *= i;
      if (i < n) cn *= factorial;
      c2n *= factorial;
    }
    const auto E = makeVandermonde(fine.matrix(), n);
    std::cout << std::setw(2) << n << std::setprecision(8) << std::setw(16) << H.determinant() << std::setw(15)
              << MatrixXd(H).determinant() << std::setw(15) << std::pow(cn, 4) / c2n << std::setprecision(3)
              << std::setw(16) << (E * c - fine.exp().matrix()).cwiseAbs().maxCoeff() << std::setw(13)
              << (E * cq - fine.exp().matrix()).cwiseAbs().maxCoeff() << "\n";
  }
}
