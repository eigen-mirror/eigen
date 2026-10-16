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

int main() {
  const Eigen::Vector2f a(1.0f, 0.0f);
  const Eigen::Vector2d b(0.0, 1e300);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  return !a.isOrthogonal(b);
#else
  return !a.cast<double>().isOrthogonal(b);
#endif
}
