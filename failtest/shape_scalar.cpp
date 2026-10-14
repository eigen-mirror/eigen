// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix2d value = Eigen::Matrix2d::Zero();
#else
  Eigen::Matrix<double, 1, 1> value = Eigen::Matrix<double, 1, 1>::Zero();
#endif
  return value.value() != 0;
}
