// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  Eigen::Array<bool, 2, 1> a, b;
  a.setOnes();
  b.setOnes();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  a - b;
#else
  a.cast<int>() - b.cast<int>();
#endif
}
