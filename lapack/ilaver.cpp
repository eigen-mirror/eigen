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
// Returns the reference LAPACK release that this implementation follows; lapack/CMakeLists.txt defines it.
void ilaver_(EIGEN_BLAS_INT* major, EIGEN_BLAS_INT* minor, EIGEN_BLAS_INT* patch) {
  *major = EIGEN_LAPACK_VERSION_MAJOR;
  *minor = EIGEN_LAPACK_VERSION_MINOR;
  *patch = EIGEN_LAPACK_VERSION_PATCH;
}
}
