// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <contrib/Eigen/Tensor>
int main() {
  Eigen::TensorFixedSize<double, Eigen::Sizes<2, 3>> lhs;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::TensorFixedSize<double, Eigen::Sizes<3, 2>> rhs;
#else
  Eigen::TensorFixedSize<double, Eigen::Sizes<2, 3>> rhs;
#endif
  lhs.setConstant(2);
  rhs.setConstant(3);
  Eigen::TensorFixedSize<double, Eigen::Sizes<2, 3>> result;
  result = lhs + rhs;
  return result(0, 0) != 5;
}
