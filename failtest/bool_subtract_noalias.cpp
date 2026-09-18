// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> a(8, 8), b(8, 8), c(8, 8);
  a.setOnes();
  b.setOnes();
  c.setOnes();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  c.noalias() -= a * b;
#else
  Eigen::MatrixXi d = c.cast<int>();
  d.noalias() -= a.cast<int>() * b.cast<int>();
#endif
}
