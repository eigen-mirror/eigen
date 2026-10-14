// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>

int main() {
  Eigen::VectorXd destination;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix2d source;
#else
  Eigen::RowVector2d source;
#endif
  destination.resizeLike(source);
  return destination.size() != source.size();
}
