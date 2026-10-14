// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix<double, 3, 4> matrix = Eigen::Matrix<double, 3, 4>::Ones();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Vector3d vector = Eigen::Vector3d::Ones();
#else
  Eigen::Vector4d vector = Eigen::Vector4d::Ones();
#endif
  const Eigen::MatrixXd result = matrix * vector.asDiagonal();
  return result.sum() != 12;
}
