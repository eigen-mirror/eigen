// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MATH_FUNCTIONS_ZVECTOR_H
#define EIGEN_MATH_FUNCTIONS_ZVECTOR_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

EIGEN_DOUBLE_PACKET_FUNCTION(atanh, Packet2d)
EIGEN_DOUBLE_PACKET_FUNCTION(log, Packet2d)
EIGEN_DOUBLE_PACKET_FUNCTION(log2, Packet2d)
EIGEN_DOUBLE_PACKET_FUNCTION(tanh, Packet2d)

EIGEN_FLOAT_PACKET_FUNCTION(atanh, Packet4f)
EIGEN_FLOAT_PACKET_FUNCTION(log, Packet4f)
EIGEN_FLOAT_PACKET_FUNCTION(log2, Packet4f)

EIGEN_GENERIC_PACKET_FUNCTION(atan, Packet2d)
EIGEN_GENERIC_PACKET_FUNCTION(atan, Packet4f)
EIGEN_GENERIC_PACKET_FUNCTION(exp2, Packet2d)
EIGEN_GENERIC_PACKET_FUNCTION(exp2, Packet4f)

EIGEN_DOUBLE_PACKET_FUNCTION(exp, Packet2d)
EIGEN_FLOAT_PACKET_FUNCTION(exp, Packet4f)

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet2d psqrt<Packet2d>(const Packet2d& x) {
  return vec_sqrt(x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet4f psqrt<Packet4f>(const Packet4f& x) {
  return vec_sqrt(x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet2d prsqrt<Packet2d>(const Packet2d& x) {
  return pset1<Packet2d>(1.0) / psqrt<Packet2d>(x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet4f prsqrt<Packet4f>(const Packet4f& x) {
  return pset1<Packet4f>(1.0) / psqrt<Packet4f>(x);
}

// Hyperbolic Tangent function.
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet4f ptanh<Packet4f>(const Packet4f& x) {
  return ptanh_float(x);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_ZVECTOR_H
