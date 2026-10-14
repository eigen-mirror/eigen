// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix<double, 3, 4> matrix = Eigen::Matrix<double, 3, 4>::Ones();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::RowVector4d vector = Eigen::RowVector4d::Ones();
#else
  Eigen::RowVector3d vector = Eigen::RowVector3d::Ones();
#endif
  const Eigen::MatrixXd result = vector.asDiagonal() * matrix;
  return result.sum() != 12;
}
