// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  Eigen::Matrix<bool, Eigen::Dynamic, 1> a(8), b(8);
  Eigen::Matrix<bool, 1, 1> c;
  a.setOnes();
  b.setOnes();
  c.setOnes();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  c.noalias() -= a.transpose() * b;
#else
  Eigen::Matrix<int, 1, 1> d = c.cast<int>();
  d.noalias() -= a.cast<int>().transpose() * b.cast<int>();
#endif
}
