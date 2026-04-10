// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// cuSPARSE support utilities: error checking macro.

#ifndef EIGEN_GPU_CUSPARSE_SUPPORT_H
#define EIGEN_GPU_CUSPARSE_SUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSupport.h"
#include <cusparse.h>

namespace Eigen {
namespace internal {

#define EIGEN_CUSPARSE_CHECK(x)                                                 \
  do {                                                                          \
    cusparseStatus_t _s = (x);                                                  \
    eigen_assert(_s == CUSPARSE_STATUS_SUCCESS && "cuSPARSE call failed: " #x); \
    EIGEN_UNUSED_VARIABLE(_s);                                                  \
  } while (0)

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_GPU_CUSPARSE_SUPPORT_H
