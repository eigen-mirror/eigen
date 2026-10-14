// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix<double, 2, 3> lhs;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 3, 4> rhs;
#else
  Eigen::Matrix<double, 2, 4> rhs;
#endif
  const auto result = Eigen::hcat(lhs, rhs);
  return result.cols() != 7;
}
