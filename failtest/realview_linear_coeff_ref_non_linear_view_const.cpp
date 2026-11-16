// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

// The const single-index coeffRef() of a RealView is only for views with linear access, like that of a Map.
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define VIEW_XPR m.block(0, 0, 2, 2)
#else
#define VIEW_XPR m
#endif

using namespace Eigen;

double foo(MatrixXcd &m) {
  const auto view = VIEW_XPR.realView();
  return view.coeffRef(4);
}

int main() {}
