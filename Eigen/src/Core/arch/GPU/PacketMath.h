// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_PACKET_MATH_GPU_H
#define EIGEN_PACKET_MATH_GPU_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Read-only data cached load (__ldg) and native FP16 arithmetic are available
// on all supported GPU architectures (sm_60+ for CUDA, GFX906+ for HIP).

// Make sure this is only available when targeting a GPU: we don't want to
// introduce conflicts between these packet_traits definitions and the ones
// we'll use on the host side (SSE, AVX, ...)
#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)

template <>
struct is_arithmetic<float4> : std::true_type {};
template <>
struct is_arithmetic<double2> : std::true_type {};

template <>
struct packet_traits<float> : default_packet_traits {
  using type = float4;
  using half = float4;
  static constexpr int Vectorizable = 1;
  static constexpr int AlignedOnScalar = 1;
  static constexpr int size = 4;

  static constexpr int HasDiv = 1;
  static constexpr int HasSin = 0;
  static constexpr int HasCos = 0;
  static constexpr int HasLog = 1;
  static constexpr int HasExp = 1;
  static constexpr int HasSqrt = 1;
  static constexpr int HasRsqrt = 1;
  static constexpr int HasLGamma = 1;
  static constexpr int HasDiGamma = 1;
  static constexpr int HasZeta = 1;
  static constexpr int HasPolygamma = 1;
  static constexpr int HasErf = 1;
  static constexpr int HasErfc = 1;
  static constexpr int HasNdtri = 1;
  static constexpr int HasBessel = 1;
  static constexpr int HasIGamma = 1;
  static constexpr int HasIGammaDerA = 1;
  static constexpr int HasGammaSampleDerAlpha = 1;
  static constexpr int HasIGammac = 1;
  static constexpr int HasBetaInc = 1;

  static constexpr int HasCmp = 1;
};

template <>
struct packet_traits<double> : default_packet_traits {
  using type = double2;
  using half = double2;
  static constexpr int Vectorizable = 1;
  static constexpr int AlignedOnScalar = 1;
  static constexpr int size = 2;

  static constexpr int HasDiv = 1;
  static constexpr int HasLog = 1;
  static constexpr int HasExp = 1;
  static constexpr int HasSqrt = 1;
  static constexpr int HasRsqrt = 1;
  static constexpr int HasLGamma = 1;
  static constexpr int HasDiGamma = 1;
  static constexpr int HasZeta = 1;
  static constexpr int HasPolygamma = 1;
  static constexpr int HasErf = 1;
  static constexpr int HasErfc = 1;
  static constexpr int HasNdtri = 1;
  static constexpr int HasBessel = 1;
  static constexpr int HasIGamma = 1;
  static constexpr int HasIGammaDerA = 1;
  static constexpr int HasGammaSampleDerAlpha = 1;
  static constexpr int HasIGammac = 1;
  static constexpr int HasBetaInc = 1;

  static constexpr int HasCmp = 1;
};

template <>
struct unpacket_traits<float4> {
  using type = float;
  static constexpr int size = 4;
  static constexpr int alignment = Aligned16;
  static constexpr bool vectorizable = true;
  static constexpr bool masked_load_available = false;
  static constexpr bool masked_store_available = false;
  using half = float4;
};
template <>
struct unpacket_traits<double2> {
  using type = double;
  static constexpr int size = 2;
  static constexpr int alignment = Aligned16;
  static constexpr bool vectorizable = true;
  static constexpr bool masked_load_available = false;
  static constexpr bool masked_store_available = false;
  using half = double2;
};

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pset1<float4>(const float& from) {
  return make_float4(from, from, from, from);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pset1<double2>(const double& from) {
  return make_double2(from, from);
}

// Bit-level helpers on the scalar lanes. numext::bit_cast rather than the __int_as_float family, which are device
// intrinsics: a .cu translation unit that defines EIGEN_USE_GPU instantiates these packet types in the host pass
// too, and an operation that exists in only one of the two passes makes packet_traits differ between them.
template <typename T>
using lane_bits_t = typename numext::get_integer_by_size<sizeof(T)>::unsigned_type;

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T bitwise_and(const T& a, const T& b) {
  using Bits = lane_bits_t<T>;
  return numext::bit_cast<T>(static_cast<Bits>(numext::bit_cast<Bits>(a) & numext::bit_cast<Bits>(b)));
}
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T bitwise_or(const T& a, const T& b) {
  using Bits = lane_bits_t<T>;
  return numext::bit_cast<T>(static_cast<Bits>(numext::bit_cast<Bits>(a) | numext::bit_cast<Bits>(b)));
}
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T bitwise_xor(const T& a, const T& b) {
  using Bits = lane_bits_t<T>;
  return numext::bit_cast<T>(static_cast<Bits>(numext::bit_cast<Bits>(a) ^ numext::bit_cast<Bits>(b)));
}
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T bitwise_andnot(const T& a, const T& b) {
  using Bits = lane_bits_t<T>;
  return numext::bit_cast<T>(static_cast<Bits>(numext::bit_cast<Bits>(a) & ~numext::bit_cast<Bits>(b)));
}

// A comparison returns an all-ones lane where it holds and an all-zero lane elsewhere, so that the result can be
// consumed bitwise by pselect, pand and pandnot.
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T mask_from(bool condition) {
  using Bits = lane_bits_t<T>;
  return numext::bit_cast<T>(condition ? ~Bits(0) : Bits(0));
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T eq_mask(const T& a, const T& b) {
  return mask_from<T>(a == b);
}
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T lt_mask(const T& a, const T& b) {
  return mask_from<T>(a < b);
}
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T le_mask(const T& a, const T& b) {
  return mask_from<T>(a <= b);
}
// !(a >= b), so a NaN operand makes the lane true.
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T lt_or_nan_mask(const T& a, const T& b) {
  return mask_from<T>(!(a >= b));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pand<float4>(const float4& a, const float4& b) {
  return make_float4(bitwise_and(a.x, b.x), bitwise_and(a.y, b.y), bitwise_and(a.z, b.z), bitwise_and(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pand<double2>(const double2& a, const double2& b) {
  return make_double2(bitwise_and(a.x, b.x), bitwise_and(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 por<float4>(const float4& a, const float4& b) {
  return make_float4(bitwise_or(a.x, b.x), bitwise_or(a.y, b.y), bitwise_or(a.z, b.z), bitwise_or(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 por<double2>(const double2& a, const double2& b) {
  return make_double2(bitwise_or(a.x, b.x), bitwise_or(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pxor<float4>(const float4& a, const float4& b) {
  return make_float4(bitwise_xor(a.x, b.x), bitwise_xor(a.y, b.y), bitwise_xor(a.z, b.z), bitwise_xor(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pxor<double2>(const double2& a, const double2& b) {
  return make_double2(bitwise_xor(a.x, b.x), bitwise_xor(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pandnot<float4>(const float4& a, const float4& b) {
  return make_float4(bitwise_andnot(a.x, b.x), bitwise_andnot(a.y, b.y), bitwise_andnot(a.z, b.z),
                     bitwise_andnot(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pandnot<double2>(const double2& a, const double2& b) {
  return make_double2(bitwise_andnot(a.x, b.x), bitwise_andnot(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcmp_eq<float4>(const float4& a, const float4& b) {
  return make_float4(eq_mask(a.x, b.x), eq_mask(a.y, b.y), eq_mask(a.z, b.z), eq_mask(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcmp_lt<float4>(const float4& a, const float4& b) {
  return make_float4(lt_mask(a.x, b.x), lt_mask(a.y, b.y), lt_mask(a.z, b.z), lt_mask(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcmp_le<float4>(const float4& a, const float4& b) {
  return make_float4(le_mask(a.x, b.x), le_mask(a.y, b.y), le_mask(a.z, b.z), le_mask(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcmp_lt_or_nan<float4>(const float4& a, const float4& b) {
  return make_float4(lt_or_nan_mask(a.x, b.x), lt_or_nan_mask(a.y, b.y), lt_or_nan_mask(a.z, b.z),
                     lt_or_nan_mask(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pcmp_eq<double2>(const double2& a, const double2& b) {
  return make_double2(eq_mask(a.x, b.x), eq_mask(a.y, b.y));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pcmp_lt<double2>(const double2& a, const double2& b) {
  return make_double2(lt_mask(a.x, b.x), lt_mask(a.y, b.y));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pcmp_le<double2>(const double2& a, const double2& b) {
  return make_double2(le_mask(a.x, b.x), le_mask(a.y, b.y));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pcmp_lt_or_nan<double2>(const double2& a, const double2& b) {
  return make_double2(lt_or_nan_mask(a.x, b.x), lt_or_nan_mask(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predux_any(const float4& a) {
  using Bits = lane_bits_t<float>;
  return (numext::bit_cast<Bits>(a.x) | numext::bit_cast<Bits>(a.y) | numext::bit_cast<Bits>(a.z) |
          numext::bit_cast<Bits>(a.w)) != 0;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predux_any(const double2& a) {
  using Bits = lane_bits_t<double>;
  return (numext::bit_cast<Bits>(a.x) | numext::bit_cast<Bits>(a.y)) != 0;
}

// numext::sign: 0 for either zero, NaN for NaN, +/-1 otherwise. default_packet_traits advertises HasSign, so
// without these the flag was a promise the backend did not keep (the generic form does not compile for float4).
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 psign<float4>(const float4& a) {
  return make_float4(numext::sign(a.x), numext::sign(a.y), numext::sign(a.z), numext::sign(a.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 psign<double2>(const double2& a) {
  return make_double2(numext::sign(a.x), numext::sign(a.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 plset<float4>(const float& a) {
  return make_float4(a, a + 1, a + 2, a + 3);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 plset<double2>(const double& a) {
  return make_double2(a, a + 1);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 padd<float4>(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 padd<double2>(const double2& a, const double2& b) {
  return make_double2(a.x + b.x, a.y + b.y);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 psub<float4>(const float4& a, const float4& b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 psub<double2>(const double2& a, const double2& b) {
  return make_double2(a.x - b.x, a.y - b.y);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pnegate(const float4& a) {
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pnegate(const double2& a) {
  return make_double2(-a.x, -a.y);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmul<float4>(const float4& a, const float4& b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmul<double2>(const double2& a, const double2& b) {
  return make_double2(a.x * b.x, a.y * b.y);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pdiv<float4>(const float4& a, const float4& b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pdiv<double2>(const double2& a, const double2& b) {
  return make_double2(a.x / b.x, a.y / b.y);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmin<float4>(const float4& a, const float4& b) {
  return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmin<double2>(const double2& a, const double2& b) {
  return make_double2(fmin(a.x, b.x), fmin(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmax<float4>(const float4& a, const float4& b) {
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmax<double2>(const double2& a, const double2& b) {
  return make_double2(fmax(a.x, b.x), fmax(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pload<float4>(const float* from) {
  return *reinterpret_cast<const float4*>(from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pload<double2>(const double* from) {
  return *reinterpret_cast<const double2*>(from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 ploadu<float4>(const float* from) {
  return make_float4(from[0], from[1], from[2], from[3]);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 ploadu<double2>(const double* from) {
  return make_double2(from[0], from[1]);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 ploaddup<float4>(const float* from) {
  return make_float4(from[0], from[0], from[1], from[1]);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 ploaddup<double2>(const double* from) {
  return make_double2(from[0], from[0]);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<float>(float* to, const float4& from) {
  *reinterpret_cast<float4*>(to) = from;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<double>(double* to, const double2& from) {
  *reinterpret_cast<double2*>(to) = from;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const float4& from) {
  to[0] = from.x;
  to[1] = from.y;
  to[2] = from.z;
  to[3] = from.w;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const double2& from) {
  to[0] = from.x;
  to[1] = from.y;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float4 ploadt_ro<float4, Aligned>(const float* from) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  return __ldg(reinterpret_cast<const float4*>(from));
#else
  return make_float4(from[0], from[1], from[2], from[3]);
#endif
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double2 ploadt_ro<double2, Aligned>(const double* from) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  return __ldg(reinterpret_cast<const double2*>(from));
#else
  return make_double2(from[0], from[1]);
#endif
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float4 ploadt_ro<float4, Unaligned>(const float* from) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  return make_float4(__ldg(from + 0), __ldg(from + 1), __ldg(from + 2), __ldg(from + 3));
#else
  return make_float4(from[0], from[1], from[2], from[3]);
#endif
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double2 ploadt_ro<double2, Unaligned>(const double* from) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  return make_double2(__ldg(from + 0), __ldg(from + 1));
#else
  return make_double2(from[0], from[1]);
#endif
}

template <>
EIGEN_DEVICE_FUNC inline float4 pgather<float, float4>(const float* from, Index stride) {
  return make_float4(from[0 * stride], from[1 * stride], from[2 * stride], from[3 * stride]);
}

template <>
EIGEN_DEVICE_FUNC inline double2 pgather<double, double2>(const double* from, Index stride) {
  return make_double2(from[0 * stride], from[1 * stride]);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, float4>(float* to, const float4& from, Index stride) {
  to[stride * 0] = from.x;
  to[stride * 1] = from.y;
  to[stride * 2] = from.z;
  to[stride * 3] = from.w;
}
template <>
EIGEN_DEVICE_FUNC inline void pscatter<double, double2>(double* to, const double2& from, Index stride) {
  to[stride * 0] = from.x;
  to[stride * 1] = from.y;
}

template <>
EIGEN_DEVICE_FUNC inline float4 preverse(const float4& a) {
  return make_float4(a.w, a.z, a.y, a.x);
}
template <>
EIGEN_DEVICE_FUNC inline double2 preverse(const double2& a) {
  return make_double2(a.y, a.x);
}

template <>
EIGEN_DEVICE_FUNC inline float pfirst<float4>(const float4& a) {
  return a.x;
}
template <>
EIGEN_DEVICE_FUNC inline double pfirst<double2>(const double2& a) {
  return a.x;
}

template <>
EIGEN_DEVICE_FUNC inline float predux<float4>(const float4& a) {
  return a.x + a.y + a.z + a.w;
}
template <>
EIGEN_DEVICE_FUNC inline double predux<double2>(const double2& a) {
  return a.x + a.y;
}

template <>
EIGEN_DEVICE_FUNC inline float predux_max<float4>(const float4& a) {
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double predux_max<double2>(const double2& a) {
  return fmax(a.x, a.y);
}

template <>
EIGEN_DEVICE_FUNC inline float predux_min<float4>(const float4& a) {
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double predux_min<double2>(const double2& a) {
  return fmin(a.x, a.y);
}

template <>
EIGEN_DEVICE_FUNC inline float predux_mul<float4>(const float4& a) {
  return a.x * a.y * a.z * a.w;
}
template <>
EIGEN_DEVICE_FUNC inline double predux_mul<double2>(const double2& a) {
  return a.x * a.y;
}

template <>
EIGEN_DEVICE_FUNC inline float4 pabs<float4>(const float4& a) {
  return make_float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double2 pabs<double2>(const double2& a) {
  return make_double2(fabs(a.x), fabs(a.y));
}

template <>
EIGEN_DEVICE_FUNC inline float4 pfloor<float4>(const float4& a) {
  return make_float4(floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double2 pfloor<double2>(const double2& a) {
  return make_double2(floor(a.x), floor(a.y));
}

template <>
EIGEN_DEVICE_FUNC inline float4 pceil<float4>(const float4& a) {
  return make_float4(ceilf(a.x), ceilf(a.y), ceilf(a.z), ceilf(a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double2 pceil<double2>(const double2& a) {
  return make_double2(ceil(a.x), ceil(a.y));
}

template <>
EIGEN_DEVICE_FUNC inline float4 print<float4>(const float4& a) {
  return make_float4(rintf(a.x), rintf(a.y), rintf(a.z), rintf(a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double2 print<double2>(const double2& a) {
  return make_double2(rint(a.x), rint(a.y));
}

template <>
EIGEN_DEVICE_FUNC inline float4 ptrunc<float4>(const float4& a) {
  return make_float4(truncf(a.x), truncf(a.y), truncf(a.z), truncf(a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double2 ptrunc<double2>(const double2& a) {
  return make_double2(trunc(a.x), trunc(a.y));
}

template <>
EIGEN_DEVICE_FUNC inline float4 pround<float4>(const float4& a) {
  return make_float4(roundf(a.x), roundf(a.y), roundf(a.z), roundf(a.w));
}
template <>
EIGEN_DEVICE_FUNC inline double2 pround<double2>(const double2& a) {
  return make_double2(round(a.x), round(a.y));
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<float4, 4>& kernel) {
  float tmp = kernel.packet[0].y;
  kernel.packet[0].y = kernel.packet[1].x;
  kernel.packet[1].x = tmp;

  tmp = kernel.packet[0].z;
  kernel.packet[0].z = kernel.packet[2].x;
  kernel.packet[2].x = tmp;

  tmp = kernel.packet[0].w;
  kernel.packet[0].w = kernel.packet[3].x;
  kernel.packet[3].x = tmp;

  tmp = kernel.packet[1].z;
  kernel.packet[1].z = kernel.packet[2].y;
  kernel.packet[2].y = tmp;

  tmp = kernel.packet[1].w;
  kernel.packet[1].w = kernel.packet[3].y;
  kernel.packet[3].y = tmp;

  tmp = kernel.packet[2].w;
  kernel.packet[2].w = kernel.packet[3].z;
  kernel.packet[3].z = tmp;
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<double2, 2>& kernel) {
  double tmp = kernel.packet[0].y;
  kernel.packet[0].y = kernel.packet[1].x;
  kernel.packet[1].x = tmp;
}

#endif  // defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)

// Half-packet functions are only available in GPU device compilation — they use
// intrinsics (__half2, etc.) that have no host-side benefit.
#if defined(EIGEN_GPU_COMPILE_PHASE)

// Two packets over Eigen::half: the native two-lane half2, and Packet4h2, eight halves in four half2 lanes.
// Packet4h2 stays an alias of ulonglong2 because the Tensor GPU kernels and TensorFlow name it that way.
using Packet4h2 = ulonglong2;
template <>
struct unpacket_traits<Packet4h2> {
  using type = Eigen::half;
  static constexpr int size = 8;
  static constexpr int alignment = Aligned16;
  static constexpr bool vectorizable = true;
  static constexpr bool masked_load_available = false;
  static constexpr bool masked_store_available = false;
  using half = Packet4h2;
};
template <>
struct is_arithmetic<Packet4h2> : std::true_type {};

template <>
struct unpacket_traits<half2> {
  using type = Eigen::half;
  static constexpr int size = 2;
  // half2 needs 4-byte alignment; Aligned8 is the smallest value the enum offers that satisfies it.
  static constexpr int alignment = Aligned8;
  static constexpr bool vectorizable = true;
  static constexpr bool masked_load_available = false;
  static constexpr bool masked_store_available = false;
  using half = half2;
};
template <>
struct is_arithmetic<half2> : std::true_type {};

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  using type = Packet4h2;
  using half = Packet4h2;
  static constexpr int Vectorizable = 1;
  static constexpr int AlignedOnScalar = 1;
  static constexpr int size = 8;
  static constexpr int HasAdd = 1;
  static constexpr int HasSub = 1;
  static constexpr int HasMul = 1;
  static constexpr int HasDiv = 1;
  static constexpr int HasSqrt = 1;
  static constexpr int HasRsqrt = 1;
  static constexpr int HasExp = 1;
  static constexpr int HasExpm1 = 1;
  static constexpr int HasLog = 1;
  static constexpr int HasLog1p = 1;
  // default_packet_traits turns these on, but there is no pfloor/pceil/print/ptrunc/pround and no psign for the
  // half packets; advertising them makes the generic fallback the evaluator picks fail to compile.
  static constexpr int HasRound = 0;
  static constexpr int HasSign = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// half2, the native two-lane packet.

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pset1<half2>(const Eigen::half& from) {
  return __half2half2(from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pload<half2>(const Eigen::half* from) {
  return *reinterpret_cast<const half2*>(from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ploadu<half2>(const Eigen::half* from) {
  return __halves2half2(from[0], from[1]);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ploaddup<half2>(const Eigen::half* from) {
  return __halves2half2(from[0], from[0]);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const half2& from) {
  *reinterpret_cast<half2*>(to) = from;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const half2& from) {
  to[0] = __low2half(from);
  to[1] = __high2half(from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Aligned>(const Eigen::half* from) {
  // Input is guaranteed to be properly aligned.
  return __ldg(reinterpret_cast<const half2*>(from));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Unaligned>(const Eigen::half* from) {
  return __halves2half2(__ldg(from + 0), __ldg(from + 1));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pgather<Eigen::half, half2>(const Eigen::half* from, Index stride) {
  return __halves2half2(from[0 * stride], from[1 * stride]);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pscatter<Eigen::half, half2>(Eigen::half* to, const half2& from,
                                                                        Index stride) {
  to[stride * 0] = __low2half(from);
  to[stride * 1] = __high2half(from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 preverse(const half2& a) {
  return __halves2half2(__high2half(a), __low2half(a));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half pfirst<half2>(const half2& a) {
  return __low2half(a);
}

// __low2half returns the CUDA type; Eigen::half is what carries the accessible raw field.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE numext::uint16_t low_bits(const half2& a) {
  const Eigen::half low = __low2half(a);
  return low.x;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE numext::uint16_t high_bits(const half2& a) {
  const Eigen::half high = __high2half(a);
  return high.x;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 half2_from_bits(numext::uint16_t low, numext::uint16_t high) {
  return __halves2half2(half_impl::raw_uint16_to_half(low), half_impl::raw_uint16_to_half(high));
}
// An all-ones or all-zero lane: the mask form pselect and the bitwise operations consume.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half half_mask_from(bool condition) {
  return half_impl::raw_uint16_to_half(condition ? 0xffffu : 0x0000u);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pabs<half2>(const half2& a) {
  return half2_from_bits(low_bits(a) & 0x7FFF, high_bits(a) & 0x7FFF);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ptrue<half2>(const half2& /*a*/) {
  return pset1<half2>(half_mask_from(true));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pzero<half2>(const half2& /*a*/) {
  return pset1<half2>(half_mask_from(false));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<half2, 2>& kernel) {
  const Eigen::half a1 = __low2half(kernel.packet[0]);
  const Eigen::half a2 = __high2half(kernel.packet[0]);
  const Eigen::half b1 = __low2half(kernel.packet[1]);
  const Eigen::half b2 = __high2half(kernel.packet[1]);
  kernel.packet[0] = __halves2half2(a1, b1);
  kernel.packet[1] = __halves2half2(a2, b2);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plset<half2>(const Eigen::half& a) {
  return __halves2half2(a, __hadd(a, __float2half(1.0f)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pselect<half2>(const half2& mask, const half2& a, const half2& b) {
  const Eigen::half mask_low = __low2half(mask);
  const Eigen::half mask_high = __high2half(mask);
  const Eigen::half low = mask_low == Eigen::half(0) ? __low2half(b) : __low2half(a);
  const Eigen::half high = mask_high == Eigen::half(0) ? __high2half(b) : __high2half(a);
  return __halves2half2(low, high);
}

// Conversion to float is exact for both half operands.
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pcmp_eq<half2>(const half2& a, const half2& b) {
  return __halves2half2(half_mask_from(__low2float(a) == __low2float(b)),
                        half_mask_from(__high2float(a) == __high2float(b)));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pcmp_lt<half2>(const half2& a, const half2& b) {
  return __halves2half2(half_mask_from(__low2float(a) < __low2float(b)),
                        half_mask_from(__high2float(a) < __high2float(b)));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pcmp_le<half2>(const half2& a, const half2& b) {
  return __halves2half2(half_mask_from(__low2float(a) <= __low2float(b)),
                        half_mask_from(__high2float(a) <= __high2float(b)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pand<half2>(const half2& a, const half2& b) {
  return half2_from_bits(low_bits(a) & low_bits(b), high_bits(a) & high_bits(b));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 por<half2>(const half2& a, const half2& b) {
  return half2_from_bits(low_bits(a) | low_bits(b), high_bits(a) | high_bits(b));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pxor<half2>(const half2& a, const half2& b) {
  return half2_from_bits(low_bits(a) ^ low_bits(b), high_bits(a) ^ high_bits(b));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pandnot<half2>(const half2& a, const half2& b) {
  return half2_from_bits(low_bits(a) & ~low_bits(b), high_bits(a) & ~high_bits(b));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 padd<half2>(const half2& a, const half2& b) {
  return __hadd2(a, b);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 psub<half2>(const half2& a, const half2& b) {
  return __hsub2(a, b);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pnegate(const half2& a) {
  return __hneg2(a);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmul<half2>(const half2& a, const half2& b) {
  return __hmul2(a, b);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmadd<half2>(const half2& a, const half2& b, const half2& c) {
  return __hfma2(a, b, c);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pdiv<half2>(const half2& a, const half2& b) {
  return __h2div(a, b);
}

// Compared through float, and every comparison against a NaN is false, so the result is b. This differs from
// the fminf/fmaxf of the float packets, which return the non-NaN operand: here a NaN in b propagates and a
// NaN in a does not.
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmin<half2>(const half2& a, const half2& b) {
  const __half low = __low2float(a) < __low2float(b) ? __low2half(a) : __low2half(b);
  const __half high = __high2float(a) < __high2float(b) ? __high2half(a) : __high2half(b);
  return __halves2half2(low, high);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmax<half2>(const half2& a, const half2& b) {
  const __half low = __low2float(a) > __low2float(b) ? __low2half(a) : __low2half(b);
  const __half high = __high2float(a) > __high2float(b) ? __high2half(a) : __high2half(b);
  return __halves2half2(low, high);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux<half2>(const half2& a) {
  return __hadd(__low2half(a), __high2half(a));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predux_any(const half2& a) {
  return (low_bits(a) | high_bits(a)) != 0;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_max<half2>(const half2& a) {
  const __half first = __low2half(a);
  const __half second = __high2half(a);
  return __hgt(first, second) ? first : second;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_min<half2>(const half2& a) {
  const __half first = __low2half(a);
  const __half second = __high2half(a);
  return __hlt(first, second) ? first : second;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_mul<half2>(const half2& a) {
  return __hmul(__low2half(a), __high2half(a));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plog<half2>(const half2& a) {
  return h2log(a);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pexp<half2>(const half2& a) {
  return h2exp(a);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 psqrt<half2>(const half2& a) {
  return h2sqrt(a);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 prsqrt<half2>(const half2& a) {
  return h2rsqrt(a);
}

// No native h2log1p/h2expm1; both go through float.
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plog1p<half2>(const half2& a) {
  return __floats2half2_rn(log1pf(__low2float(a)), log1pf(__high2float(a)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pexpm1<half2>(const half2& a) {
  return __floats2half2_rn(expm1f(__low2float(a)), expm1f(__high2float(a)));
}

// ---------------------------------------------------------------------------------------------------------------
// Packet4h2 is four half2 lanes in the two words of a ulonglong2, the alias the Tensor GPU kernels expect. Do not
// replace the cast: reaching the lanes conformingly (shift and reassemble, or __half2_raw) stops the compiler
// keeping them in registers, costing 168 SASS instructions against 152 for eight mixed half operations and 152
// against 64 for an 8x8 ptranspose (sm_89, nvcc 13.3).
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 lane_half2(const Packet4h2& p, int i) {
  return reinterpret_cast<const half2*>(&p)[i];
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 make_packet4h2(const half2& l0, const half2& l1, const half2& l2,
                                                               const half2& l3) {
  Packet4h2 r;
  half2* lanes = reinterpret_cast<half2*>(&r);
  lanes[0] = l0;
  lanes[1] = l1;
  lanes[2] = l2;
  lanes[3] = l3;
  return r;
}

// Every lane-wise operation is its half2 form applied to the four lanes; the macros keep that unroll in one place.
#define EIGEN_GPU_PACKET4H2_UNARY(NAME)                                                           \
  template <>                                                                                     \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 NAME<Packet4h2>(const Packet4h2& a) {           \
    return make_packet4h2(NAME(lane_half2(a, 0)), NAME(lane_half2(a, 1)), NAME(lane_half2(a, 2)), \
                          NAME(lane_half2(a, 3)));                                                \
  }

#define EIGEN_GPU_PACKET4H2_BINARY(NAME)                                                                       \
  template <>                                                                                                  \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 NAME<Packet4h2>(const Packet4h2& a, const Packet4h2& b) {    \
    return make_packet4h2(NAME(lane_half2(a, 0), lane_half2(b, 0)), NAME(lane_half2(a, 1), lane_half2(b, 1)),  \
                          NAME(lane_half2(a, 2), lane_half2(b, 2)), NAME(lane_half2(a, 3), lane_half2(b, 3))); \
  }

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pset1<Packet4h2>(const Eigen::half& from) {
  const half2 lane = pset1<half2>(from);
  return make_packet4h2(lane, lane, lane, lane);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pload<Packet4h2>(const Eigen::half* from) {
  return *reinterpret_cast<const Packet4h2*>(from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 ploadu<Packet4h2>(const Eigen::half* from) {
  return make_packet4h2(ploadu<half2>(from + 0), ploadu<half2>(from + 2), ploadu<half2>(from + 4),
                        ploadu<half2>(from + 6));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 ploaddup<Packet4h2>(const Eigen::half* from) {
  return make_packet4h2(ploaddup<half2>(from + 0), ploaddup<half2>(from + 1), ploaddup<half2>(from + 2),
                        ploaddup<half2>(from + 3));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const Packet4h2& from) {
  *reinterpret_cast<Packet4h2*>(to) = from;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const Packet4h2& from) {
  pstoreu<Eigen::half>(to + 0, lane_half2(from, 0));
  pstoreu<Eigen::half>(to + 2, lane_half2(from, 1));
  pstoreu<Eigen::half>(to + 4, lane_half2(from, 2));
  pstoreu<Eigen::half>(to + 6, lane_half2(from, 3));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet4h2 ploadt_ro<Packet4h2, Aligned>(const Eigen::half* from) {
  return __ldg(reinterpret_cast<const Packet4h2*>(from));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet4h2 ploadt_ro<Packet4h2, Unaligned>(const Eigen::half* from) {
  return make_packet4h2(ploadt_ro<half2, Unaligned>(from + 0), ploadt_ro<half2, Unaligned>(from + 2),
                        ploadt_ro<half2, Unaligned>(from + 4), ploadt_ro<half2, Unaligned>(from + 6));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pgather<Eigen::half, Packet4h2>(const Eigen::half* from, Index stride) {
  return make_packet4h2(
      __halves2half2(from[0 * stride], from[1 * stride]), __halves2half2(from[2 * stride], from[3 * stride]),
      __halves2half2(from[4 * stride], from[5 * stride]), __halves2half2(from[6 * stride], from[7 * stride]));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pscatter<Eigen::half, Packet4h2>(Eigen::half* to, const Packet4h2& from,
                                                                            Index stride) {
  pscatter<Eigen::half, half2>(to + stride * 0, lane_half2(from, 0), stride);
  pscatter<Eigen::half, half2>(to + stride * 2, lane_half2(from, 1), stride);
  pscatter<Eigen::half, half2>(to + stride * 4, lane_half2(from, 2), stride);
  pscatter<Eigen::half, half2>(to + stride * 6, lane_half2(from, 3), stride);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 preverse(const Packet4h2& a) {
  return make_packet4h2(preverse(lane_half2(a, 3)), preverse(lane_half2(a, 2)), preverse(lane_half2(a, 1)),
                        preverse(lane_half2(a, 0)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half pfirst<Packet4h2>(const Packet4h2& a) {
  return pfirst<half2>(lane_half2(a, 0));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 ptrue<Packet4h2>(const Packet4h2& /*a*/) {
  return pset1<Packet4h2>(half_impl::raw_uint16_to_half(0xffffu));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pzero<Packet4h2>(const Packet4h2& /*a*/) {
  return pset1<Packet4h2>(half_impl::raw_uint16_to_half(0x0000u));
}

// An 8x8 transpose of halves: packet r of the result is column r of the input.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet4h2, 8>& kernel) {
  Eigen::half elements[8][8];
  EIGEN_UNROLL_LOOP
  for (int row = 0; row < 8; ++row) {
    EIGEN_UNROLL_LOOP
    for (int lane = 0; lane < 4; ++lane) {
      const half2 pair = lane_half2(kernel.packet[row], lane);
      elements[row][2 * lane] = __low2half(pair);
      elements[row][2 * lane + 1] = __high2half(pair);
    }
  }
  EIGEN_UNROLL_LOOP
  for (int row = 0; row < 8; ++row) {
    kernel.packet[row] = make_packet4h2(
        __halves2half2(elements[0][row], elements[1][row]), __halves2half2(elements[2][row], elements[3][row]),
        __halves2half2(elements[4][row], elements[5][row]), __halves2half2(elements[6][row], elements[7][row]));
  }
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 plset<Packet4h2>(const Eigen::half& a) {
  // Add each offset once: (a + 2*k) + 1 can round twice. Preserve a's sign in lane zero.
  const half2 base = pset1<half2>(a);
  return make_packet4h2(plset<half2>(a), __hadd2(base, __floats2half2_rn(2.0f, 3.0f)),
                        __hadd2(base, __floats2half2_rn(4.0f, 5.0f)), __hadd2(base, __floats2half2_rn(6.0f, 7.0f)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pselect<Packet4h2>(const Packet4h2& mask, const Packet4h2& a,
                                                                   const Packet4h2& b) {
  return make_packet4h2(pselect<half2>(lane_half2(mask, 0), lane_half2(a, 0), lane_half2(b, 0)),
                        pselect<half2>(lane_half2(mask, 1), lane_half2(a, 1), lane_half2(b, 1)),
                        pselect<half2>(lane_half2(mask, 2), lane_half2(a, 2), lane_half2(b, 2)),
                        pselect<half2>(lane_half2(mask, 3), lane_half2(a, 3), lane_half2(b, 3)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pmadd<Packet4h2>(const Packet4h2& a, const Packet4h2& b,
                                                                 const Packet4h2& c) {
  return make_packet4h2(pmadd<half2>(lane_half2(a, 0), lane_half2(b, 0), lane_half2(c, 0)),
                        pmadd<half2>(lane_half2(a, 1), lane_half2(b, 1), lane_half2(c, 1)),
                        pmadd<half2>(lane_half2(a, 2), lane_half2(b, 2), lane_half2(c, 2)),
                        pmadd<half2>(lane_half2(a, 3), lane_half2(b, 3), lane_half2(c, 3)));
}

EIGEN_GPU_PACKET4H2_UNARY(pabs)
EIGEN_GPU_PACKET4H2_UNARY(pnegate)
EIGEN_GPU_PACKET4H2_UNARY(plog)
EIGEN_GPU_PACKET4H2_UNARY(pexp)
EIGEN_GPU_PACKET4H2_UNARY(psqrt)
EIGEN_GPU_PACKET4H2_UNARY(prsqrt)
EIGEN_GPU_PACKET4H2_UNARY(plog1p)
EIGEN_GPU_PACKET4H2_UNARY(pexpm1)

EIGEN_GPU_PACKET4H2_BINARY(padd)
EIGEN_GPU_PACKET4H2_BINARY(psub)
EIGEN_GPU_PACKET4H2_BINARY(pmul)
EIGEN_GPU_PACKET4H2_BINARY(pdiv)
EIGEN_GPU_PACKET4H2_BINARY(pmin)
EIGEN_GPU_PACKET4H2_BINARY(pmax)
EIGEN_GPU_PACKET4H2_BINARY(pand)
EIGEN_GPU_PACKET4H2_BINARY(por)
EIGEN_GPU_PACKET4H2_BINARY(pxor)
EIGEN_GPU_PACKET4H2_BINARY(pandnot)
EIGEN_GPU_PACKET4H2_BINARY(pcmp_eq)
EIGEN_GPU_PACKET4H2_BINARY(pcmp_lt)
EIGEN_GPU_PACKET4H2_BINARY(pcmp_le)

#undef EIGEN_GPU_PACKET4H2_UNARY
#undef EIGEN_GPU_PACKET4H2_BINARY

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux<Packet4h2>(const Packet4h2& a) {
  const half2 sum =
      padd<half2>(padd<half2>(lane_half2(a, 0), lane_half2(a, 1)), padd<half2>(lane_half2(a, 2), lane_half2(a, 3)));
  return predux<half2>(sum);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predux_any(const Packet4h2& a) {
  return predux_any(lane_half2(a, 0)) | predux_any(lane_half2(a, 1)) | predux_any(lane_half2(a, 2)) |
         predux_any(lane_half2(a, 3));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_max<Packet4h2>(const Packet4h2& a) {
  const half2 m =
      pmax<half2>(pmax<half2>(lane_half2(a, 0), lane_half2(a, 1)), pmax<half2>(lane_half2(a, 2), lane_half2(a, 3)));
  return predux_max<half2>(m);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_min<Packet4h2>(const Packet4h2& a) {
  const half2 m =
      pmin<half2>(pmin<half2>(lane_half2(a, 0), lane_half2(a, 1)), pmin<half2>(lane_half2(a, 2), lane_half2(a, 3)));
  return predux_min<half2>(m);
}

// Likely to overflow or underflow: eight halves multiplied together.
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_mul<Packet4h2>(const Packet4h2& a) {
  const half2 product =
      pmul<half2>(pmul<half2>(lane_half2(a, 0), lane_half2(a, 1)), pmul<half2>(lane_half2(a, 2), lane_half2(a, 3)));
  return predux_mul<half2>(product);
}

#endif  // defined(EIGEN_GPU_COMPILE_PHASE)

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_PACKET_MATH_GPU_H
