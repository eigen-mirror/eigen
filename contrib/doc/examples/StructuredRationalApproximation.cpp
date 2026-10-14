// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The Cauchy integral formula f(z) = (1/2 pi i) \int_{|w|=rho} f(w) / (w - z) dw
// discretized: a function analytic in |z| < R is represented on the unit disk
// by n point charges, r(z) = sum_j c_j / (z - w_j), on the circle |w| = rho
// between 1 and R. Fitting r to f at n points z_i of the unit circle is a
// square Cauchy system with entries 1/(z_i - w_j), factored in O(n^2) by
// CauchyLU; the error decays like (R/rho)^-n + rho^-n, fastest for
// rho = sqrt(R). Here f(z) = exp(z) / (z - 1.5), so R = 1.5.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>

using namespace Eigen;
using Complex = std::complex<double>;

int main() {
  const double pi = std::acos(-1.0), R = 1.5, rho = std::sqrt(R);
  const auto f = [R](const Complex& z) { return std::exp(z) / (z - R); };
  std::cout << "   n  max |r(z) - f(z)| on |z| = 1\n";
  for (int n : {16, 32, 64, 128}) {
    VectorXcd z(n), w(n), fz(n);
    for (int j = 0; j < n; ++j) {
      z[j] = std::polar(1.0, 2 * pi * j / n);
      w[j] = std::polar(rho, 2 * pi * (j + 0.5) / n);
      fz[j] = f(z[j]);
    }
    const VectorXcd c = CauchyLU<Complex>(makeCauchy(z, w)).solve(fz);

    // By the maximum principle the error on the circle bounds it on the disk.
    double error = 0;
    for (int k = 0; k < 1000; ++k) {
      const Complex t = std::polar(1.0, 2 * pi * (k + 0.37) / 1000);
      error = std::max(error, std::abs((c.array() / (t - w.array())).sum() - f(t)));
    }
    std::cout << std::setw(4) << n << "  " << error << "\n";
  }
}
