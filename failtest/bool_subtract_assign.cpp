// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  Eigen::Matrix<bool, 2, 2> a, b;
  a.setOnes();
  b.setOnes();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  a -= b;
#else
  Eigen::Matrix2i c = a.cast<int>();
  c -= b.cast<int>();
#endif
}
