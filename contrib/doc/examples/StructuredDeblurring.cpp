// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Image deblurring with periodic boundary conditions. Blurring an n x n image
// X with a point spread function G is the 2-D circular convolution G (*) X, so
// the blur matrix A is a Bccb. For a motion blur over L pixels, the box filter,
// the symbol vanishes on the L - 1 frequency lines k = j n / L: A is exactly
// rank deficient, Bccb::rank counts the retained modes, and the pseudo-inverse
// A.solve recovers nothing on those lines while amplifying the noise of the
// recorded image on the lines next to them. Tikhonov regularization,
//   x_lambda = (A^H A + lambda I)^{-1} A^H b,
// keeps the structure: A^H A + lambda I is the Bccb whose generator is the
// autocorrelation A^H (A e_0) of the PSF plus lambda at the origin, so each
// candidate lambda costs one O(N log N) solve.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>
#include <random>

using namespace Eigen;

int main() {
  const int n = 128, N = n * n;
  const int L = 8;  // the camera moves L pixels along the columns during the exposure
  MatrixXd G = MatrixXd::Zero(n, n);
  G.col(0).head(L).setConstant(1.0 / L);
  const auto A = makeBccb(G);

  // The image: a bright rectangle and a disc on a dark background.
  MatrixXd X = MatrixXd::Constant(n, n, 0.1);
  X.block(32, 40, 32, 48).setConstant(1.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      if ((i - 90) * (i - 90) + (j - 40) * (j - 40) < 400) X(i, j) = 0.6;
  const VectorXd x = X.reshaped();

  // Blur and add 1% noise from a generator whose sequence the C++ standard fixes.
  std::minstd_rand generator(1);
  VectorXd b = A * x;
  for (int i = 0; i < N; ++i) b[i] += 0.01 * (2.0 * generator() / generator.max() - 1);

  const VectorXd s = A.singularValues();  // decreasing
  std::cout << "rank " << A.rank() << " of " << N << ", smallest retained singular value " << s[A.rank() - 1] << "\n";
  std::cout << "relative error of the blurred image:  " << (b - x).norm() / x.norm() << "\n";
  std::cout << "relative error of the pseudo-inverse: " << (A.solve(b) - x).norm() / x.norm() << "\n";

  VectorXd e0 = VectorXd::Zero(N);
  e0[0] = 1;
  const VectorXd autocorrelation = A.adjoint() * VectorXd(A * e0);  // first column of A^H A
  const VectorXd Ahb = A.adjoint() * b;
  for (double lambda : {1e-4, 1e-3, 1e-2, 1e-1}) {
    VectorXd g = autocorrelation;
    g[0] += lambda;
    const VectorXd xl = makeBccb(g.reshaped(n, n)).solve(Ahb);
    std::cout << "Tikhonov, lambda = " << lambda << ": relative error " << (xl - x).norm() / x.norm() << "\n";
  }
}
