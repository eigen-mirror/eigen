// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

// The single-index coeffRef() of a RealView is only for views with linear access, like that of a Map. A runtime-sized
// block of a matrix has none: the linear index of the view would be read as an offset from the block's first
// coefficient, outside the block.
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define VIEW_XPR m.block(0, 0, 2, 2)
#else
#define VIEW_XPR m
#endif

using namespace Eigen;

double foo(MatrixXcd &m) { return VIEW_XPR.realView().coeffRef(4); }

int main() {}
