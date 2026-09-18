// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  Eigen::Array<bool, Eigen::Dynamic, 1> a(8);
  a.setOnes();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  a - true;
#else
  a.cast<int>() - 1;
#endif
}
