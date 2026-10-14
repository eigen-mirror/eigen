// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Least-squares fitting of a surface sampled on a grid by a tensor-product
// polynomial sum_{p,q} c_pq x^p y^q. With the samples F(x_i, y_j) stored with x
// as the fast index, the design matrix is the Kronecker product V_y (x) V_x of
// the two rectangular Vandermonde matrices, and KroneckerOperator solves the
// least-squares problem through one complete orthogonal decomposition per
// factor, (A (x) B)^+ = A^+ (x) B^+: for m points per direction and degrees
// P, Q, the two decompositions cost O(m (P^2 + Q^2)) and applying them
// O(m^2 (P + Q)), against O(m^2 P^2 Q^2) for the assembled m^2 x (P+1)(Q+1)
// matrix. Its rank and singular values also come from the factors.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>

using namespace Eigen;

int main() {
  const int m = 60;  // grid points per direction
  const ArrayXd x = ArrayXd::LinSpaced(m, -1, 1), y = ArrayXd::LinSpaced(m, -1, 1);
  MatrixXd F(m, m);  // F(i, j) = exp(x_i y_j), with x the fast index of vec(F)
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i) F(i, j) = std::exp(x[i] * y[j]);

  std::cout << "degree  rank  kappa        max fit error  |c - c_dense|\n";
  for (int degree : {2, 4, 6, 8}) {
    const MatrixXd Vx = makeVandermonde(x.matrix(), degree + 1), Vy = makeVandermonde(y.matrix(), degree + 1);
    const auto design = makeKroneckerOperator(Vy, Vx);  // m^2 x (degree + 1)^2
    const VectorXd c = design.leastSquaresSolve(F.reshaped());
    const VectorXd s = design.singularValues();

    const MatrixXd dense = design;
    const VectorXd reference = dense.completeOrthogonalDecomposition().solve(F.reshaped());
    std::cout << "  " << degree << "     " << design.rank() << "    " << s.maxCoeff() / s.minCoeff() << "  "
              << (F.reshaped() - design * c).cwiseAbs().maxCoeff() << "    " << (c - reference).cwiseAbs().maxCoeff()
              << "\n";
  }
}
