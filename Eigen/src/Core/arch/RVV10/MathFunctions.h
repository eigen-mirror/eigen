// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATH_FUNCTIONS_RVV10_H
#define EIGEN_MATH_FUNCTIONS_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(PacketMul1Xf)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(PacketMul2Xf)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(PacketMul4Xf)

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(PacketMul1Xd)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(PacketMul2Xd)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(PacketMul4Xd)

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_RVV10_H
