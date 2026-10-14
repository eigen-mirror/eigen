// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/LU>
int main() {
  Eigen::Matrix<double, 2, 3> matrix = Eigen::Matrix<double, 2, 3>::Zero();
  const auto solver = matrix.fullPivLu();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Vector2d rhs = Eigen::Vector2d::Ones();
#else
  Eigen::Vector3d rhs = Eigen::Vector3d::Ones();
#endif
  const auto solution = solver.transpose().solve(rhs);
  return solution.rows() != 2;
}
