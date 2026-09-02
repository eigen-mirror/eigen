// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TYPE_CASTING_GPU_H
#define EIGEN_TYPE_CASTING_GPU_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

#if defined(EIGEN_GPU_COMPILE_PHASE)

// Both ratios follow from the packet sizes (8 halves, 4 floats), which is exactly what
// vectorized_type_casting_traits computes; restating them by hand let them drift from the packets.
template <>
struct type_casting_traits<Eigen::half, float> : vectorized_type_casting_traits<Eigen::half, float> {};

template <>
struct type_casting_traits<float, Eigen::half> : vectorized_type_casting_traits<float, Eigen::half> {};

// Widening: one Packet4h2 covers two float4, so the evaluator calls this once per output packet with the same
// source and takes the first four lanes, then the second four. Narrowing: two float4 make one Packet4h2.
// CoreEvaluators.h loads a source segment of DstPacketSize elements for TgtCoeffRatio == 2 and TensorConversion.h
// does the same, so the one-argument form returning the low half is the protocol, not a truncation bug.
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcast<Packet4h2, float4>(const Packet4h2& a) {
  const float2 low = __half22float2(lane_half2(a, 0));
  const float2 high = __half22float2(lane_half2(a, 1));
  return make_float4(low.x, low.y, high.x, high.y);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pcast<float4, Packet4h2>(const float4& a, const float4& b) {
  return make_packet4h2(__floats2half2_rn(a.x, a.y), __floats2half2_rn(a.z, a.w), __floats2half2_rn(b.x, b.y),
                        __floats2half2_rn(b.z, b.w));
}

#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TYPE_CASTING_GPU_H
