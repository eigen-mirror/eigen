// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_NO_STATIC_ASSERT
#include "../Eigen/Core"

int main() {
  Eigen::PermutationMatrix<2> permutation;
  permutation.setIdentity();
  Eigen::Matrix2d matrix = Eigen::Matrix2d::Identity();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::internal::BandMatrix<double, 2, 2, 1, 1> operand;
#else
  auto operand = matrix;
#endif
  Eigen::Matrix2d result = Eigen::Matrix2d::Constant(42);
  result.noalias() = Eigen::Product<decltype(permutation), decltype(operand)>(permutation, operand);
  return result.isIdentity() ? 0 : 1;
}
