// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The [m/m] Pade approximant p(x)/q(x) of exp(x), q(0) = 1, from its Taylor
// coefficients. Matching the series through degree 2m gives the denominator
// from a Hankel system in the coefficients c_k,
//   sum_{j=1}^{m} c_{m+i-j} q_j = -c_{m+i},  i = 1..m,
// written with H(i,j) = c_{i+j+1} (j reversed), and the numerator by a
// truncated convolution p_i = sum_{j<=i} c_{i-j} q_j.
//
// The raw coefficients 1/k! fall from 1 to 1/(2m)! across H, which leaves it
// badly scaled: kappa(H) ~ 7e15 at m = 8. Approximating exp(t y) instead, with
// coefficients c_k = t^k / k!, scales H to t D H D, D = diag(t^i) -- still
// Hankel -- and t = m equilibrates it: kappa(H) ~ 5e7 at m = 8. The
// approximant of exp(x) is then p(x/t) / q(x/t), since the substitution
// preserves both the degrees and the order of contact.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Eigen;

int main() {
  std::cout << " m  |r(1) - e|      |r(-2) - e^-2|\n";
  for (int m : {2, 4, 6, 8, 10}) {
    const double t = m;
    VectorXd c(2 * m + 1);
    c[0] = 1;
    for (int k = 1; k <= 2 * m; ++k) c[k] = c[k - 1] * t / k;
    // H(i, j) = c_{i+j+1} for i, j = 0..m-1: first column c_1..c_m, last row c_m..c_{2m-1}.
    const auto H = makeHankel(c.segment(1, m), c.segment(m, m));
    // The unknowns are q_m, ..., q_1 (reversed): row i reads c_{i+1} q_m + ... + c_{m+i} q_1.
    const VectorXd reversed = H.solve(-c.segment(m + 1, m));
    VectorXd q(m + 1), p(m + 1);
    q[0] = 1;
    q.tail(m) = reversed.reverse();
    for (int i = 0; i <= m; ++i) p[i] = c.head(i + 1).reverse().dot(q.head(i + 1));
    const auto r = [&](double x) {
      const double y = x / t;
      double num = 0, den = 0;
      for (int i = m; i >= 0; --i) {
        num = num * y + p[i];
        den = den * y + q[i];
      }
      return num / den;
    };
    std::cout << std::setw(2) << m << "  " << std::setw(15) << std::abs(r(1) - std::exp(1.0)) << "  "
              << std::abs(r(-2) - std::exp(-2.0)) << "\n";
  }
}
