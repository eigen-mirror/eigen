// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The controllability Gramian X of a damped mass-spring chain x' = A x + B w,
// forced at the last mass, solves the Lyapunov equation A X + X A^T + B B^T = 0.
// With vec stacking columns, vec(A X + X A^T) = (A (+) A) vec(X), so the
// equation is one Kronecker-sum solve. A is not symmetric and has complex
// eigenvalue pairs (an underdamped chain); X is symmetric positive definite
// when the chain is controllable.

#include <contrib/Eigen/StructuredMatrices>
#include <iostream>

using namespace Eigen;

int main() {
  const int masses = 20, n = 2 * masses;
  const double damping = 0.1;
  // State (positions, velocities): A = [0 I; -K -c I] with K = tridiag(-1, 2, -1).
  MatrixXd A = MatrixXd::Zero(n, n);
  A.topRightCorner(masses, masses).setIdentity();
  for (int i = 0; i < masses; ++i) {
    A(masses + i, i) = -2;
    if (i > 0) A(masses + i, i - 1) = 1;
    if (i + 1 < masses) A(masses + i, i + 1) = 1;
    A(masses + i, masses + i) = -damping;
  }
  VectorXd B = VectorXd::Zero(n);
  B[n - 1] = 1;
  const MatrixXd Q = B * B.transpose();

  const MatrixXd X = makeKroneckerSum(A, A).solve(-Q.reshaped()).reshaped(n, n);
  std::cout << "residual |AX + XA^T + BB^T| / |BB^T|: " << (A * X + X * A.transpose() + Q).norm() / Q.norm() << "\n";
  std::cout << "asymmetry |X - X^T| / |X|:            " << (X - X.transpose()).norm() / X.norm() << "\n";
  std::cout << "smallest eigenvalue of X:             "
            << SelfAdjointEigenSolver<MatrixXd>((X + X.transpose()) / 2).eigenvalues()[0] << "\n";
}
