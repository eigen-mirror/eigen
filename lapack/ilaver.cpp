// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "lapack.h"

extern "C" {
// Returns the LAPACK version that this implementation conforms to.
void ilaver_(int* major, int* minor, int* patch) {
  *major = 3;
  *minor = 4;
  *patch = 1;
}
}
