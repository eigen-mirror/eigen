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

// A numerator wider than both the divider's Scalar and int would be truncated: (1LL << 40) / IntDivider<int>(1) would
// be 0.
int main() {
  Eigen::IntDivider<int> by(1);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  long long numerator = 7;
#else
  int numerator = 7;
#endif
  return numerator / by == 7 ? 0 : 1;
}
