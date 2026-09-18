// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> a(8, 8), b(8, 8);
  a.setOnes();
  b.setOnes();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> c = a * b - a * b;
#else
  Eigen::MatrixXi c = a.cast<int>() * b.cast<int>() - a.cast<int>() * b.cast<int>();
#endif
}
