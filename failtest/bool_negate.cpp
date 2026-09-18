// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> a(8, 8);
  a.setOnes();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  -a;
#else
  -a.cast<int>();
#endif
}
