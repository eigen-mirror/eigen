// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The checks have to fire with assertions compiled out, as in a release build.
#define EIGEN_NO_DEBUG 1
#include "runtime_checks.cpp"  // NOLINT(bugprone-suspicious-include): the same suite under EIGEN_NO_DEBUG.
