// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <contrib/Eigen/Riccati>
#include <iostream>

using namespace Eigen;

int main() {
  // Discrete-time LQR for a double integrator sampled with period h:
  // minimize sum_k x_k^T Q x_k + u_k^T R u_k subject to x_{k+1} = A x_k + B u_k.
  const double h = 0.1;
  Matrix2d A;
  A << 1, h, 0, 1;
  Vector2d B(h * h / 2, h);
  Matrix2d Q = Matrix2d::Identity();
  Matrix<double, 1, 1> R;
  R << 0.1;

  DiscreteAlgebraicRiccatiSolver<Matrix2d> dare(A, B, Q, R);
  if (dare.info() != Success) {
    std::cout << "The Riccati equation could not be solved.\n";
    return 1;
  }
  const Matrix2d& X = dare.solution();
  std::cout << "The stabilizing solution X is:\n" << X << "\n\n";

  // The optimal state feedback u_k = -K x_k.
  const RowVector2d K = (R + B.transpose() * X * B).ldlt().solve(B.transpose() * X * A);
  std::cout << "The optimal gain K is:\n" << K << "\n\n";
  std::cout << "The closed-loop eigenvalues of A - B K are:\n" << (A - B * K).eigenvalues() << "\n";
}
