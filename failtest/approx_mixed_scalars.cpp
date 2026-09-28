// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

int main() {
  const Eigen::Matrix<Eigen::half, 3, 1> a = Eigen::Matrix<Eigen::half, 3, 1>::Ones();
  const Eigen::Vector3f b = Eigen::Vector3f::Ones();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  return int(a.isApprox(b));
#else
  return int(a.cast<float>().isApprox(b));
#endif
}
