// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Two Toeplitz problems.
//
// Love's integral equation for the field between two coaxial disks at unit
// distance (a classical test problem of electrostatics),
//   u(x) - (1/pi) \int_{-1}^{1} u(t) / (1 + (x - t)^2) dt = 1,   -1 <= x <= 1.
// The midpoint rule on n equal cells gives (I - K) u = 1 with a symmetric
// Toeplitz K, K_ij = h / (pi (1 + (x_i - x_j)^2)), which LookAheadLevinson
// solves directly in O(n^2).
//
// A large dense Toeplitz system, T_ij = (1 + |i - j|)^(-1.1), solved
// by conjugate gradients: the Toeplitz product costs O(n log n) through the
// FFT, and the Strang preconditioner -- the central diagonals of T wrapped into
// a circulant C, inverted by Circulant::solve in O(n log n) -- clusters the
// spectrum of C^{-1} T around 1.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Eigen;

// The iterative solvers' preconditioner interface around Circulant::solve.
class StrangPreconditioner {
 public:
  void setToeplitzColumn(const VectorXd& t) {
    const Index n = t.size();
    VectorXd c(n);
    for (Index k = 0; k < n; ++k) c[k] = t[k <= n / 2 ? k : n - k];
    m_circulant = Circulant<double>(c);
  }
  template <typename MatrixType>
  StrangPreconditioner& analyzePattern(const MatrixType&) {
    return *this;
  }
  template <typename MatrixType>
  StrangPreconditioner& factorize(const MatrixType&) {
    return *this;
  }
  template <typename MatrixType>
  StrangPreconditioner& compute(const MatrixType&) {
    return *this;
  }
  template <typename Rhs>
  VectorXd solve(const MatrixBase<Rhs>& b) const {
    return m_circulant.solve(b);
  }
  ComputationInfo info() const { return Success; }

 private:
  Circulant<double> m_circulant{VectorXd::Ones(1)};
};

int main() {
  const double pi = std::acos(-1.0);
  std::cout << "Love's equation:\n     n  u(0)\n";
  for (int n : {64, 128, 256, 512}) {
    const double h = 2.0 / n;
    VectorXd c(n);
    for (int k = 0; k < n; ++k) c[k] = -h / (pi * (1 + (k * h) * (k * h)));
    c[0] += 1;
    const VectorXd u = LookAheadLevinson<double>(makeToeplitz(c, c)).solve(VectorXd::Ones(n));
    std::cout << std::setw(6) << n << "  " << std::setprecision(8) << (u[n / 2 - 1] + u[n / 2]) / 2 << "\n";
  }

  const int n = 1 << 16;
  VectorXd t(n);
  for (int k = 0; k < n; ++k) t[k] = std::pow(1.0 + k, -1.1);
  const auto T = makeToeplitz(t, t);
  const VectorXd b = VectorXd::Ones(n);
  ConjugateGradient<std::decay_t<decltype(T)>, Lower | Upper, IdentityPreconditioner> plain(T);
  plain.setTolerance(1e-10);
  ConjugateGradient<std::decay_t<decltype(T)>, Lower | Upper, StrangPreconditioner> strang;
  strang.setTolerance(1e-10);
  strang.preconditioner().setToeplitzColumn(t);
  strang.compute(T);
  const VectorXd x0 = plain.solve(b), x1 = strang.solve(b);
  std::cout << "Toeplitz system, n = " << n << ", CG iterations to a 1e-10 residual:\n";
  std::cout << "  no preconditioner: " << plain.iterations() << "\n";
  std::cout << "  Strang circulant:  " << strang.iterations() << "\n";
}
