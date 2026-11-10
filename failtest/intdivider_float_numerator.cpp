// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

// A floating-point numerator would be truncated to the divider's Scalar: 7.5 / IntDivider<int>(2) would be 3.
int main() {
  Eigen::IntDivider<int> by(2);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  double numerator = 7.5;
#else
  int numerator = 7;
#endif
  return numerator / by == 3 ? 0 : 1;
}
