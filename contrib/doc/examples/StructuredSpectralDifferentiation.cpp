// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Fourier spectral differentiation on the periodic grid x_j = j h, h = 2 pi / N,
// N even. The differentiation matrix is the circulant with first column
//   D_0 = 0,  D_j = (-1)^j / (2 tan(j h / 2)),  j = 1, ..., N-1,
// and the second-derivative matrix the circulant with
//   D2_0 = -pi^2 / (3 h^2) - 1/6,  D2_j = -(-1)^j / (2 sin^2(j h / 2)).
// Their eigenvalues are the wavenumbers: D has i k for |k| < N/2 and 0 for the
// constant and for the sawtooth mode k = N/2, so it is singular of rank N - 2,
// and Circulant::solve returns the minimum-norm antiderivative, the one with
// zero mean and no sawtooth component. The derivatives, the antiderivative and
// the periodic boundary value problem u'' - u = f all converge geometrically
// for u = exp(sin x).

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Eigen;

// First column of the N x N spectral differentiation matrix of the given order.
VectorXd spectralColumn(int N, int order) {
  const double pi = std::acos(-1.0), h = 2 * pi / N;
  VectorXd column(N);
  column[0] = order == 1 ? 0 : -pi * pi / (3 * h * h) - 1.0 / 6;
  for (int j = 1; j < N; ++j) {
    const double sign = j % 2 == 0 ? 1 : -1;
    column[j] = order == 1 ? sign / (2 * std::tan(j * h / 2)) : -sign / (2 * std::pow(std::sin(j * h / 2), 2));
  }
  return column;
}

int main() {
  const double pi = std::acos(-1.0);
  const double mean = 1.2660658777520084;  // mean of exp(sin x) over a period, the Bessel function I_0(1)
  std::cout << " N  rank  error of u'  error of u''  antiderivative  u'' - u = f\n";
  for (int N : {8, 16, 24, 32}) {
    const auto D = makeCirculant(spectralColumn(N, 1));
    const auto D2 = makeCirculant(spectralColumn(N, 2));
    VectorXd shifted = spectralColumn(N, 2);
    shifted[0] -= 1;
    const auto D2mI = makeCirculant(shifted);  // D2 - I

    const ArrayXd x = ArrayXd::LinSpaced(N, 0, 2 * pi * (N - 1) / N), s = x.sin(), c = x.cos();
    const VectorXd u = s.exp(), du = c * s.exp(), d2u = (c * c - s) * s.exp();
    std::cout << std::setw(2) << N << std::setw(5) << D.rank() << "  " << std::setw(11)
              << (D * u - du).cwiseAbs().maxCoeff() << "  " << std::setw(12) << (D2 * u - d2u).cwiseAbs().maxCoeff()
              << "  " << std::setw(14) << (D.solve(du).array() + mean - s.exp()).abs().maxCoeff() << "  "
              << (D2mI.solve(VectorXd(d2u - u)) - u).cwiseAbs().maxCoeff() << "\n";
  }
  const auto D = makeCirculant(spectralColumn(8, 1));
  std::cout << "N = 8, eigenvalues / i: " << D.eigenvalues().imag().array().round().cast<int>().transpose() << "\n";
}
