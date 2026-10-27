// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>

struct Scalar {};

int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  return Eigen::NumTraits<Scalar>::digits();
#else
  return Eigen::NumTraits<Scalar>::digits10() + Eigen::NumTraits<Scalar>::max_digits10();
#endif
}
