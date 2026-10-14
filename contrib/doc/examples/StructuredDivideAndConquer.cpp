// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The merge step of Cuppen's divide-and-conquer eigensolver. A symmetric
// tridiagonal T splits at the coupling b = T(m, m-1) into
//   T = diag(T_1, T_2) + b v v^T,   v = e_{m-1} + e_m,
// where T_1 and T_2 are the two halves with b subtracted from their diagonal
// entries at the cut. With T_i = Q_i L_i Q_i^T and Q = diag(Q_1, Q_2),
//   T = Q (D + b z z^T) Q^T,   D = diag(L_1, L_2),   z = Q^T v,
// a diagonal-plus-rank-one eigenproblem that DPR1EigenSolver solves in O(n^2)
// from the secular equation, with eigenvectors orthogonal to working accuracy;
// those of T are Q times them. In the real algorithm the halves recurse; here
// SelfAdjointEigenSolver stands in for the recursion.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>

using namespace Eigen;

int main() {
  const int n = 200, m = n / 2;
  // T = tridiag(-1, 2 + cos(i), -1).
  MatrixXd T = MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) T(i, i) = 2 + std::cos(i);
  T.diagonal(1).setConstant(-1);
  T.diagonal(-1).setConstant(-1);

  const double b = T(m, m - 1);
  MatrixXd T1 = T.topLeftCorner(m, m), T2 = T.bottomRightCorner(n - m, n - m);
  T1(m - 1, m - 1) -= b;
  T2(0, 0) -= b;
  const SelfAdjointEigenSolver<MatrixXd> half1(T1), half2(T2);
  VectorXd d(n), z(n);
  d << half1.eigenvalues(), half2.eigenvalues();
  z << half1.eigenvectors().row(m - 1).transpose(), half2.eigenvectors().row(0).transpose();

  const DPR1EigenSolver<double> merged(d, b, z);
  const VectorXd& lambda = merged.eigenvalues();
  MatrixXd V(n, n);
  V.topRows(m) = half1.eigenvectors() * merged.eigenvectors().topRows(m);
  V.bottomRows(n - m) = half2.eigenvectors() * merged.eigenvectors().bottomRows(n - m);

  std::cout << "residual |T V - V L| / |T|:               " << (T * V - V * lambda.asDiagonal()).norm() / T.norm()
            << "\n";
  std::cout << "orthogonality |V^T V - I|:                " << (V.transpose() * V - MatrixXd::Identity(n, n)).norm()
            << "\n";
  std::cout << "max difference to SelfAdjointEigenSolver: "
            << (lambda - SelfAdjointEigenSolver<MatrixXd>(T, EigenvaluesOnly).eigenvalues()).cwiseAbs().maxCoeff()
            << "\n";
}
