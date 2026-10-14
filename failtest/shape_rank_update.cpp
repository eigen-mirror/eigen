// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix2d value = Eigen::Matrix2d::Zero();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 3, 2> rhs;
#else
  Eigen::Matrix2d rhs;
#endif
  rhs.setZero();
  value.selfadjointView<Eigen::Lower>().rankUpdate(rhs);
}
