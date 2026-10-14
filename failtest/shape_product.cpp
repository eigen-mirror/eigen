// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>

int main() {
  Eigen::Matrix<double, 2, 3> lhs;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 4, 2> rhs;
#else
  Eigen::Matrix<double, 3, 2> rhs;
#endif
  const Eigen::Product<decltype(lhs), decltype(rhs)> product(lhs, rhs);
  return product.rows() != 2;
}
