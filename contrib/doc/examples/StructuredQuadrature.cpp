// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Interpolatory quadrature: the weights w of a rule sum_i w_i f(x_i) that is
// exact for polynomials of degree < n solve the transposed Vandermonde system
// V^T w = m with the moments m_k = \int_{-1}^{1} x^k dx. Bjorck-Pereyra solves
// it (the dual system) in O(n^2); on Chebyshev extreme points the weights are
// those of Clenshaw-Curtis quadrature. The primal system V a = f gives the
// monomial coefficients of the interpolating polynomial.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Eigen;

int main() {
  const double pi = std::acos(-1.0);
  const double exact = 2 * std::sinh(1.0);  // \int_{-1}^{1} e^x dx
  std::cout << "  n  sum(w)  error on e^x\n";
  for (int n : {5, 9, 13, 17}) {
    VectorXd x(n), moments(n);
    for (int i = 0; i < n; ++i) x[i] = -std::cos(pi * i / (n - 1));  // increasing nodes
    for (int k = 0; k < n; ++k) moments[k] = k % 2 == 0 ? 2.0 / (k + 1) : 0.0;
    const BjorckPereyra<double> bp(makeVandermonde(x));
    const VectorXd w = bp.transpose().solve(moments);
    std::cout << std::setw(3) << n << "  " << std::setw(6) << w.sum() << "  "
              << std::abs(w.dot(x.array().exp().matrix()) - exact) << "\n";
  }

  // Interpolation: the cubic through four samples of x^3 - 2x + 1.
  const Vector4d x(-1, 0, 0.5, 2);
  const Vector4d f = x.array().cube() - 2 * x.array() + 1;
  std::cout << "monomial coefficients of the interpolant: "
            << BjorckPereyra<double>(makeVandermonde(x)).solve(f).transpose() << "\n";
}
