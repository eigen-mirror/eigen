// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>

int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Vector3d vector = Eigen::Vector3d::Ones();
#else
  Eigen::Vector4d vector = Eigen::Vector4d::Ones();
#endif
  const auto reshaped = vector.reshaped(Eigen::fix<2>, Eigen::fix<2>);
  const Eigen::Reshaped<decltype(vector), 2, 2> direct(vector);
  return reshaped.rows() != 2 || direct.rows() != 2;
}
