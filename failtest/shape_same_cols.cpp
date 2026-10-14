// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix<double, 2, 3> lhs;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 4, 4> rhs;
#else
  Eigen::Matrix<double, 4, 3> rhs;
#endif
  const auto result = Eigen::vcat(lhs, rhs);
  return result.rows() != 6;
}
