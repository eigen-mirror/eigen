// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_PACKET_MATH_BF16_NEON_H
#define EIGEN_PACKET_MATH_BF16_NEON_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// TODO: Add support for native armv8.6-a bfloat16_t

// TODO: Guard if we have native bfloat16 support
typedef eigen_packet_wrapper<uint16x4_t, 19> Packet4bf;
using Packet8bf = eigen_packet_wrapper<uint16x8_t, 20>;

template <>
struct is_arithmetic<Packet4bf> : std::true_type {};
template <>
struct is_arithmetic<Packet8bf> : std::true_type {};

// Packet8bf is the full packet: every bfloat16 op widens to float, and the float work of two
// Packet4f halves overlaps where one Packet4bf's chain would leave the pipes idle.
template <>
struct packet_traits<bfloat16> : default_packet_traits {
  using type = Packet8bf;
  using half = Packet4bf;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 8,

    HasCmp = 1,
    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbsDiff = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 1,
    HasDiv = 1,
    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH,
    HasBessel = 0,  // Issues with accuracy.
    HasNdtri = 0
  };
};

template <>
struct unpacket_traits<Packet4bf> : neon_unpacket_default<Packet4bf, bfloat16> {};
template <>
struct unpacket_traits<Packet8bf> : neon_unpacket_default<Packet8bf, bfloat16> {
  using half = Packet4bf;
};

namespace detail {
template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet4bf>(Packet4bf& p1, Packet4bf& p2) {
  const uint16x4x2_t tmp = vzip_u16(p1, p2);
  p1 = tmp.val[0];
  p2 = tmp.val[1];
}
template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet8bf>(Packet8bf& p1, Packet8bf& p2) {
  const uint16x8x2_t tmp = vzipq_u16(p1, p2);
  p1 = tmp.val[0];
  p2 = tmp.val[1];
}
}  // namespace detail

// Round to nearest even on the bfloat16 boundary. With lsb the lowest bit kept, the scalar
// formula is (p + 0x7fff + lsb) >> 16, and RADDHN computes (p + d + 0x8000) >> 16, so d = lsb - 1,
// that is -1 where lsb is clear. Written as an add and RSHRN, the add stays a separate op in
// loops with GCC and clang on both targets. The caller guarantees that every NaN lane has a zero low
// half, which the FPU's NaN propagation does for bfloat16 operands: the rounding then leaves the
// NaN's high half alone. A NaN with a nonzero low half can instead round to +-inf or carry into a
// zero.
EIGEN_STRONG_INLINE Packet4bf F32ToBf16KeepNaN(const Packet4f& p) {
  Packet4ui bits = vreinterpretq_u32_f32(p);
  Packet4ui round_down = vceqq_u32(vandq_u32(bits, vdupq_n_u32(0x10000)), vdupq_n_u32(0));
  return vraddhn_u32(bits, round_down);
}

// F32ToBf16KeepNaN for any float: NaN lanes become the bfloat16 NaN 0x7fc0, as on AVX. The scalar
// conversion keeps the NaN's sign; doing that here costs an op on the critical path of every round
// trip, 25% of a cast loop and 6% of exp on a Cortex-A72, for a bit that carries no value.
// |p| <= inf is false only for NaN. The NaN lanes are selected after narrowing, so the rounding
// reads the raw bits. Selecting first costs clang an extra AND and MOV.
EIGEN_STRONG_INLINE Packet4bf F32ToBf16(const Packet4f& p) {
  uint16x4_t not_nan = vmovn_u32(vcageq_f32(pinf<Packet4f>(), p));
  return vbsl_u16(not_nan, F32ToBf16KeepNaN(p), vdup_n_u16(0x7fc0));
}

// Discard the low 16 bits of each float. Only valid when every lane already holds an exact
// bfloat16 value, in which case this agrees with F32ToBf16 but skips its rounding and NaN
// canonicalization. pmin/pmax qualify: they return one of their operands bit for bit.
EIGEN_STRONG_INLINE Packet4bf F32ToBf16Truncate(const Packet4f& p) { return vshrn_n_u32(vreinterpretq_u32_f32(p), 16); }

EIGEN_STRONG_INLINE Packet4f Bf16ToF32(const Packet4bf& p) {
  return Packet4f(vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(p), 16)));
}

EIGEN_STRONG_INLINE Packet4bf F32MaskToBf16Mask(const Packet4f& p) { return vmovn_u32(vreinterpretq_u32_f32(p)); }

// Packet8bf conversions work on two Packet4f halves, lanes 0-3 and 4-7, with the same rounding as
// the Packet4bf ones.
EIGEN_STRONG_INLINE Packet4f Bf16ToF32Low(const Packet8bf& p) {
  return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(p), 16));
}

EIGEN_STRONG_INLINE Packet4f Bf16ToF32High(const Packet8bf& p) {
#if EIGEN_ARCH_ARM64
  return vreinterpretq_f32_u32(vshll_high_n_u16(p, 16));
#else
  return vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(p), 16));
#endif
}

// (v + 0x8000) >> 16 on each lane of lo and hi.
EIGEN_STRONG_INLINE Packet8bf RoundingNarrowBf16(const Packet4ui& lo, const Packet4ui& hi) {
#if EIGEN_ARCH_ARM64
  return vrshrn_high_n_u32(vrshrn_n_u32(lo, 16), hi, 16);
#else
  return vcombine_u16(vrshrn_n_u32(lo, 16), vrshrn_n_u32(hi, 16));
#endif
}

EIGEN_STRONG_INLINE Packet8bf F32ToBf16KeepNaN(const Packet4f& lo, const Packet4f& hi) {
  Packet4ui lo_bits = vreinterpretq_u32_f32(lo);
  Packet4ui hi_bits = vreinterpretq_u32_f32(hi);
  Packet4ui lo_round_down = vceqq_u32(vandq_u32(lo_bits, vdupq_n_u32(0x10000)), vdupq_n_u32(0));
  Packet4ui hi_round_down = vceqq_u32(vandq_u32(hi_bits, vdupq_n_u32(0x10000)), vdupq_n_u32(0));
#if EIGEN_ARCH_ARM64
  return vraddhn_high_u32(vraddhn_u32(lo_bits, lo_round_down), hi_bits, hi_round_down);
#else
  return vcombine_u16(vraddhn_u32(lo_bits, lo_round_down), vraddhn_u32(hi_bits, hi_round_down));
#endif
}

// As the Packet4bf F32ToBf16, with one UZP1 narrowing both NaN masks.
EIGEN_STRONG_INLINE Packet8bf F32ToBf16(const Packet4f& lo, const Packet4f& hi) {
  Packet4f inf = pinf<Packet4f>();
  uint16x8_t not_nan =
      vuzpq_u16(vreinterpretq_u16_u32(vcageq_f32(inf, lo)), vreinterpretq_u16_u32(vcageq_f32(inf, hi))).val[0];
  return vbslq_u16(not_nan, F32ToBf16KeepNaN(lo, hi), vdupq_n_u16(0x7fc0));
}

// The high and low 16 bits of every float: one UZP2 or UZP1 on AArch64, one VUZP on ARMv7.
EIGEN_STRONG_INLINE Packet8bf F32ToBf16Truncate(const Packet4f& lo, const Packet4f& hi) {
  return vuzpq_u16(vreinterpretq_u16_f32(lo), vreinterpretq_u16_f32(hi)).val[1];
}

EIGEN_STRONG_INLINE Packet8bf F32MaskToBf16Mask(const Packet4f& lo, const Packet4f& hi) {
  return vuzpq_u16(vreinterpretq_u16_f32(lo), vreinterpretq_u16_f32(hi)).val[0];
}

template <>
EIGEN_STRONG_INLINE Packet4bf pset1<Packet4bf>(const bfloat16& from) {
  return Packet4bf(pset1<Packet4us>(from.value));
}

template <>
EIGEN_STRONG_INLINE bfloat16 pfirst<Packet4bf>(const Packet4bf& from) {
  return bfloat16_impl::raw_uint16_to_bfloat16(static_cast<uint16_t>(pfirst<Packet4us>(Packet4us(from))));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pload<Packet4bf>(const bfloat16* from) {
  return Packet4bf(
      pload<Packet4us>(reinterpret_cast<const uint16_t*>(assume_aligned<unpacket_traits<Packet4bf>::alignment>(from))));
}

template <>
EIGEN_STRONG_INLINE Packet4bf ploadu<Packet4bf>(const bfloat16* from) {
  return Packet4bf(ploadu<Packet4us>(reinterpret_cast<const uint16_t*>(from)));
}

template <>
EIGEN_STRONG_INLINE void pstore<bfloat16>(bfloat16* to, const Packet4bf& from) {
  EIGEN_DEBUG_ALIGNED_STORE vst1_u16(
      reinterpret_cast<uint16_t*>(assume_aligned<unpacket_traits<Packet4bf>::alignment>(to)), from);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<bfloat16>(bfloat16* to, const Packet4bf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE vst1_u16(reinterpret_cast<uint16_t*>(to), from);
}

template <>
EIGEN_STRONG_INLINE Packet4bf ploaddup<Packet4bf>(const bfloat16* from) {
  return Packet4bf(ploaddup<Packet4us>(reinterpret_cast<const uint16_t*>(from)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pabs(const Packet4bf& a) {
  return Packet4bf(vand_u16(a, vdup_n_u16(0x7fff)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pmin<PropagateNumbers, Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16Truncate(pmin<PropagateNumbers, Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}
template <>
EIGEN_STRONG_INLINE Packet4bf pmin<PropagateNaN, Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16Truncate(pmin<PropagateNaN, Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pmin<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16Truncate(pmin<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pmax<PropagateNumbers, Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16Truncate(pmax<PropagateNumbers, Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}
template <>
EIGEN_STRONG_INLINE Packet4bf pmax<PropagateNaN, Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16Truncate(pmax<PropagateNaN, Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pmax<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16Truncate(pmax<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf plset<Packet4bf>(const bfloat16& a) {
  return F32ToBf16(plset<Packet4f>(static_cast<float>(a)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf por(const Packet4bf& a, const Packet4bf& b) {
  return Packet4bf(por<Packet4us>(Packet4us(a), Packet4us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pxor(const Packet4bf& a, const Packet4bf& b) {
  return Packet4bf(pxor<Packet4us>(Packet4us(a), Packet4us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pand(const Packet4bf& a, const Packet4bf& b) {
  return Packet4bf(pand<Packet4us>(Packet4us(a), Packet4us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pandnot(const Packet4bf& a, const Packet4bf& b) {
  return Packet4bf(pandnot<Packet4us>(Packet4us(a), Packet4us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pnot(const Packet4bf& a) {
  return Packet4bf(pnot<Packet4us>(Packet4us(a)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf ptrue(const Packet4bf& /*a*/) {
  return Packet4bf(vdup_n_u16(0xffff));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4bf pselect(const Packet4bf& mask, const Packet4bf& a, const Packet4bf& b) {
  return Packet4bf(pselect<Packet4us>(Packet4us(mask), Packet4us(a), Packet4us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf print<Packet4bf>(const Packet4bf& a) {
  return F32ToBf16Truncate(print<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pfloor<Packet4bf>(const Packet4bf& a) {
  return F32ToBf16Truncate(pfloor<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pceil<Packet4bf>(const Packet4bf& a) {
  return F32ToBf16Truncate(pceil<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pround<Packet4bf>(const Packet4bf& a) {
  return F32ToBf16Truncate(pround<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf ptrunc<Packet4bf>(const Packet4bf& a) {
  return F32ToBf16Truncate(ptrunc<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf padd<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16KeepNaN(padd<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf psub<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16KeepNaN(psub<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pmul<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16KeepNaN(pmul<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pmadd<Packet4bf>(const Packet4bf& a, const Packet4bf& b, const Packet4bf& c) {
  return F32ToBf16KeepNaN(pmadd<Packet4f>(Bf16ToF32(a), Bf16ToF32(b), Bf16ToF32(c)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pmsub<Packet4bf>(const Packet4bf& a, const Packet4bf& b, const Packet4bf& c) {
  return F32ToBf16KeepNaN(pmsub<Packet4f>(Bf16ToF32(a), Bf16ToF32(b), Bf16ToF32(c)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pnmadd<Packet4bf>(const Packet4bf& a, const Packet4bf& b, const Packet4bf& c) {
  return F32ToBf16KeepNaN(pnmadd<Packet4f>(Bf16ToF32(a), Bf16ToF32(b), Bf16ToF32(c)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pnmsub<Packet4bf>(const Packet4bf& a, const Packet4bf& b, const Packet4bf& c) {
  return F32ToBf16KeepNaN(pnmsub<Packet4f>(Bf16ToF32(a), Bf16ToF32(b), Bf16ToF32(c)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pdiv<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16KeepNaN(pdiv<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pgather<bfloat16, Packet4bf>(const bfloat16* from, Index stride) {
  return Packet4bf(pgather<uint16_t, Packet4us>(reinterpret_cast<const uint16_t*>(from), stride));
}

template <>
EIGEN_STRONG_INLINE void pscatter<bfloat16, Packet4bf>(bfloat16* to, const Packet4bf& from, Index stride) {
  pscatter<uint16_t, Packet4us>(reinterpret_cast<uint16_t*>(to), Packet4us(from), stride);
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux<Packet4bf>(const Packet4bf& a) {
  return static_cast<bfloat16>(predux<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux_max<Packet4bf>(const Packet4bf& a) {
  return bfloat16_impl::exact_float_to_bfloat16(predux_max<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux_min<Packet4bf>(const Packet4bf& a) {
  return bfloat16_impl::exact_float_to_bfloat16(predux_min<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux_mul<Packet4bf>(const Packet4bf& a) {
  return static_cast<bfloat16>(predux_mul<Packet4f>(Bf16ToF32(a)));
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet4bf& a) {
  return vget_lane_u64(vreinterpret_u64_u16(Packet4us(a)), 0) != 0;
}

template <>
EIGEN_STRONG_INLINE Packet4bf preverse<Packet4bf>(const Packet4bf& a) {
  return Packet4bf(preverse<Packet4us>(Packet4us(a)));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet4bf, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

template <>
EIGEN_STRONG_INLINE Packet4bf pabsdiff<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32ToBf16KeepNaN(pabsdiff<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pcmp_eq<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32MaskToBf16Mask(pcmp_eq<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pcmp_lt<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32MaskToBf16Mask(pcmp_lt<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pcmp_lt_or_nan<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32MaskToBf16Mask(pcmp_lt_or_nan<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

// Classify on the raw bits, as the scalar isinf/isnan/isfinite do: |a| ==, >, < 0x7f80.
template <>
EIGEN_STRONG_INLINE Packet4bf pisinf<Packet4bf>(const Packet4bf& a) {
  constexpr uint16_t kInf = ((1 << 8) - 1) << 7;
  constexpr uint16_t kAbsMask = (1 << 15) - 1;
  return vceq_u16(vand_u16(a, vdup_n_u16(kAbsMask)), vdup_n_u16(kInf));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pisnan<Packet4bf>(const Packet4bf& a) {
  constexpr uint16_t kInf = ((1 << 8) - 1) << 7;
  constexpr uint16_t kAbsMask = (1 << 15) - 1;
  return vcgt_u16(vand_u16(a, vdup_n_u16(kAbsMask)), vdup_n_u16(kInf));
}

// Compare encoded lanes: widening bf16 subnormals to float loses them under DAZ/FZ.
template <>
EIGEN_STRONG_INLINE Packet4bf psign<Packet4bf>(const Packet4bf& a) {
  const uint16x4_t abs_mask = vdup_n_u16(0x7fff);
  const uint16x4_t is_nan = vcgt_u16(vand_u16(a, abs_mask), vdup_n_u16(0x7f80));
  const uint16x4_t keep = vorr_u16(is_nan, vdup_n_u16(0x8000));
  return vand_u16(vbsl_u16(keep, a, vdup_n_u16(0x3f80)), vtst_u16(a, abs_mask));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pisfinite<Packet4bf>(const Packet4bf& a) {
  constexpr uint16_t kInf = ((1 << 8) - 1) << 7;
  constexpr uint16_t kAbsMask = (1 << 15) - 1;
  return vclt_u16(vand_u16(a, vdup_n_u16(kAbsMask)), vdup_n_u16(kInf));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pcmp_le<Packet4bf>(const Packet4bf& a, const Packet4bf& b) {
  return F32MaskToBf16Mask(pcmp_le<Packet4f>(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pnegate<Packet4bf>(const Packet4bf& a) {
  return Packet4bf(pxor<Packet4us>(Packet4us(a), pset1<Packet4us>(static_cast<uint16_t>(0x8000))));
}

//---------- Packet8bf ----------

template <>
EIGEN_STRONG_INLINE Packet8bf pset1<Packet8bf>(const bfloat16& from) {
  return Packet8bf(pset1<Packet8us>(from.value));
}

template <>
EIGEN_STRONG_INLINE bfloat16 pfirst<Packet8bf>(const Packet8bf& from) {
  return bfloat16_impl::raw_uint16_to_bfloat16(static_cast<uint16_t>(pfirst<Packet8us>(Packet8us(from))));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pload<Packet8bf>(const bfloat16* from) {
  return Packet8bf(
      pload<Packet8us>(reinterpret_cast<const uint16_t*>(assume_aligned<unpacket_traits<Packet8bf>::alignment>(from))));
}

template <>
EIGEN_STRONG_INLINE Packet8bf ploadu<Packet8bf>(const bfloat16* from) {
  return Packet8bf(ploadu<Packet8us>(reinterpret_cast<const uint16_t*>(from)));
}

template <>
EIGEN_STRONG_INLINE void pstore<bfloat16>(bfloat16* to, const Packet8bf& from) {
  EIGEN_DEBUG_ALIGNED_STORE vst1q_u16(
      reinterpret_cast<uint16_t*>(assume_aligned<unpacket_traits<Packet8bf>::alignment>(to)), from);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<bfloat16>(bfloat16* to, const Packet8bf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE vst1q_u16(reinterpret_cast<uint16_t*>(to), from);
}

template <>
EIGEN_STRONG_INLINE Packet8bf ploaddup<Packet8bf>(const bfloat16* from) {
  return Packet8bf(ploaddup<Packet8us>(reinterpret_cast<const uint16_t*>(from)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf ploadquad<Packet8bf>(const bfloat16* from) {
  return Packet8bf(ploadquad<Packet8us>(reinterpret_cast<const uint16_t*>(from)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pabs(const Packet8bf& a) {
  return Packet8bf(vandq_u16(a, vdupq_n_u16(0x7fff)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pmin<PropagateNumbers, Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16Truncate(pmin<PropagateNumbers, Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pmin<PropagateNumbers, Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}
template <>
EIGEN_STRONG_INLINE Packet8bf pmin<PropagateNaN, Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16Truncate(pmin<PropagateNaN, Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pmin<PropagateNaN, Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pmin<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16Truncate(pmin<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pmin<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pmax<PropagateNumbers, Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16Truncate(pmax<PropagateNumbers, Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pmax<PropagateNumbers, Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}
template <>
EIGEN_STRONG_INLINE Packet8bf pmax<PropagateNaN, Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16Truncate(pmax<PropagateNaN, Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pmax<PropagateNaN, Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pmax<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16Truncate(pmax<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pmax<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf plset<Packet8bf>(const bfloat16& a) {
  Packet4f lo = plset<Packet4f>(static_cast<float>(a));
  return F32ToBf16(lo, padd(lo, pset1<Packet4f>(4.0f)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf por(const Packet8bf& a, const Packet8bf& b) {
  return Packet8bf(por<Packet8us>(Packet8us(a), Packet8us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pxor(const Packet8bf& a, const Packet8bf& b) {
  return Packet8bf(pxor<Packet8us>(Packet8us(a), Packet8us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pand(const Packet8bf& a, const Packet8bf& b) {
  return Packet8bf(pand<Packet8us>(Packet8us(a), Packet8us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pandnot(const Packet8bf& a, const Packet8bf& b) {
  return Packet8bf(pandnot<Packet8us>(Packet8us(a), Packet8us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pnot(const Packet8bf& a) {
  return Packet8bf(pnot<Packet8us>(Packet8us(a)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf ptrue(const Packet8bf& /*a*/) {
  return Packet8bf(vdupq_n_u16(0xffff));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8bf pselect(const Packet8bf& mask, const Packet8bf& a, const Packet8bf& b) {
  return Packet8bf(pselect<Packet8us>(Packet8us(mask), Packet8us(a), Packet8us(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf print<Packet8bf>(const Packet8bf& a) {
  return F32ToBf16Truncate(print<Packet4f>(Bf16ToF32Low(a)), print<Packet4f>(Bf16ToF32High(a)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pfloor<Packet8bf>(const Packet8bf& a) {
  return F32ToBf16Truncate(pfloor<Packet4f>(Bf16ToF32Low(a)), pfloor<Packet4f>(Bf16ToF32High(a)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pceil<Packet8bf>(const Packet8bf& a) {
  return F32ToBf16Truncate(pceil<Packet4f>(Bf16ToF32Low(a)), pceil<Packet4f>(Bf16ToF32High(a)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pround<Packet8bf>(const Packet8bf& a) {
  return F32ToBf16Truncate(pround<Packet4f>(Bf16ToF32Low(a)), pround<Packet4f>(Bf16ToF32High(a)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf ptrunc<Packet8bf>(const Packet8bf& a) {
  return F32ToBf16Truncate(ptrunc<Packet4f>(Bf16ToF32Low(a)), ptrunc<Packet4f>(Bf16ToF32High(a)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf padd<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16KeepNaN(padd<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                          padd<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf psub<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16KeepNaN(psub<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                          psub<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pmul<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16KeepNaN(pmul<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                          pmul<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pmadd<Packet8bf>(const Packet8bf& a, const Packet8bf& b, const Packet8bf& c) {
  return F32ToBf16KeepNaN(pmadd<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b), Bf16ToF32Low(c)),
                          pmadd<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b), Bf16ToF32High(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pmsub<Packet8bf>(const Packet8bf& a, const Packet8bf& b, const Packet8bf& c) {
  return F32ToBf16KeepNaN(pmsub<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b), Bf16ToF32Low(c)),
                          pmsub<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b), Bf16ToF32High(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pnmadd<Packet8bf>(const Packet8bf& a, const Packet8bf& b, const Packet8bf& c) {
  return F32ToBf16KeepNaN(pnmadd<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b), Bf16ToF32Low(c)),
                          pnmadd<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b), Bf16ToF32High(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pnmsub<Packet8bf>(const Packet8bf& a, const Packet8bf& b, const Packet8bf& c) {
  return F32ToBf16KeepNaN(pnmsub<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b), Bf16ToF32Low(c)),
                          pnmsub<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b), Bf16ToF32High(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pdiv<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16KeepNaN(pdiv<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                          pdiv<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pgather<bfloat16, Packet8bf>(const bfloat16* from, Index stride) {
  return Packet8bf(pgather<uint16_t, Packet8us>(reinterpret_cast<const uint16_t*>(from), stride));
}

template <>
EIGEN_STRONG_INLINE void pscatter<bfloat16, Packet8bf>(bfloat16* to, const Packet8bf& from, Index stride) {
  pscatter<uint16_t, Packet8us>(reinterpret_cast<uint16_t*>(to), Packet8us(from), stride);
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux<Packet8bf>(const Packet8bf& a) {
  return static_cast<bfloat16>(predux<Packet4f>(padd<Packet4f>(Bf16ToF32Low(a), Bf16ToF32High(a))));
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux_max<Packet8bf>(const Packet8bf& a) {
  return bfloat16_impl::exact_float_to_bfloat16(
      predux_max<Packet4f>(pmax<Packet4f>(Bf16ToF32Low(a), Bf16ToF32High(a))));
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux_min<Packet8bf>(const Packet8bf& a) {
  return bfloat16_impl::exact_float_to_bfloat16(
      predux_min<Packet4f>(pmin<Packet4f>(Bf16ToF32Low(a), Bf16ToF32High(a))));
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux_mul<Packet8bf>(const Packet8bf& a) {
  return static_cast<bfloat16>(predux_mul<Packet4f>(pmul<Packet4f>(Bf16ToF32Low(a), Bf16ToF32High(a))));
}

template <>
EIGEN_STRONG_INLINE Packet4bf predux_half<Packet8bf>(const Packet8bf& a) {
  return F32ToBf16KeepNaN(padd<Packet4f>(Bf16ToF32Low(a), Bf16ToF32High(a)));
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet8bf& a) {
  return predux_any(Packet8us(a));
}

template <>
EIGEN_STRONG_INLINE Packet8bf preverse<Packet8bf>(const Packet8bf& a) {
  return Packet8bf(preverse<Packet8us>(Packet8us(a)));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8bf, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8bf, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

template <>
EIGEN_STRONG_INLINE Packet8bf pabsdiff<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32ToBf16KeepNaN(pabsdiff<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                          pabsdiff<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pcmp_eq<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32MaskToBf16Mask(pcmp_eq<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pcmp_eq<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pcmp_lt<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32MaskToBf16Mask(pcmp_lt<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pcmp_lt<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pcmp_lt_or_nan<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  // Negate a >= b once on the narrowed mask rather than on each float half.
  return vmvnq_u16(F32MaskToBf16Mask(pcmp_le<Packet4f>(Bf16ToF32Low(b), Bf16ToF32Low(a)),
                                     pcmp_le<Packet4f>(Bf16ToF32High(b), Bf16ToF32High(a))));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pcmp_le<Packet8bf>(const Packet8bf& a, const Packet8bf& b) {
  return F32MaskToBf16Mask(pcmp_le<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(b)),
                           pcmp_le<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pisinf<Packet8bf>(const Packet8bf& a) {
  constexpr uint16_t kInf = ((1 << 8) - 1) << 7;
  constexpr uint16_t kAbsMask = (1 << 15) - 1;
  return vceqq_u16(vandq_u16(a, vdupq_n_u16(kAbsMask)), vdupq_n_u16(kInf));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pisnan<Packet8bf>(const Packet8bf& a) {
  constexpr uint16_t kInf = ((1 << 8) - 1) << 7;
  constexpr uint16_t kAbsMask = (1 << 15) - 1;
  return vcgtq_u16(vandq_u16(a, vdupq_n_u16(kAbsMask)), vdupq_n_u16(kInf));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pisfinite<Packet8bf>(const Packet8bf& a) {
  constexpr uint16_t kInf = ((1 << 8) - 1) << 7;
  constexpr uint16_t kAbsMask = (1 << 15) - 1;
  return vcltq_u16(vandq_u16(a, vdupq_n_u16(kAbsMask)), vdupq_n_u16(kInf));
}

template <>
EIGEN_STRONG_INLINE Packet8bf psign<Packet8bf>(const Packet8bf& a) {
  uint16x8_t abs_mask = vdupq_n_u16(0x7fff);
  uint16x8_t is_nan = vcgtq_u16(vandq_u16(a, abs_mask), vdupq_n_u16(0x7f80));
  uint16x8_t keep = vorrq_u16(is_nan, vdupq_n_u16(0x8000));
  return vandq_u16(vbslq_u16(keep, a, vdupq_n_u16(0x3f80)), vtstq_u16(a, abs_mask));
}

template <>
EIGEN_STRONG_INLINE Packet8bf pnegate<Packet8bf>(const Packet8bf& a) {
  return Packet8bf(pxor<Packet8us>(Packet8us(a), pset1<Packet8us>(static_cast<uint16_t>(0x8000))));
}

// psqrt and prsqrt have their own kernels below.
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_BF16(Packet4f, Packet4bf)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_BF16_WITH(BF16_PACKET_FUNCTION_HALVES, Packet4f, Packet8bf)

template <>
EIGEN_STRONG_INLINE Packet4bf psqrt<Packet4bf>(const Packet4bf& a) {
  // y = x r with r = FRSQRTE(x), then one FRSQRTS step y <- y (3 - y r) / 2. FRSQRTE is specified
  // bit for bit, and rounding y to bfloat16 equals bfloat16(sqrtf(x)) for all 65536 inputs
  // (packetmath checks them all); the raw estimate alone misrounds 7366 of them. RSHRN adds half a
  // bfloat16 ulp and truncates. That is round half up, which differs from F32ToBf16 only on an
  // exact float midpoint, and no input produces one; NaNs stay NaN since their low half is zero.
  Packet4f x = Bf16ToF32(a);
  Packet4f r = vrsqrteq_f32(x);
  Packet4f y = pmul(x, r);
  y = pmul(y, vrsqrtsq_f32(y, r));
  Packet4bf ybf = vrshrn_n_u32(vreinterpretq_u32_f32(y), 16);
  // y is NaN where r is +-inf (x is +-0, or a subnormal the FPU flushes) or +0 (x is +inf), and
  // the answer there is the sign and exponent of a. Reading r rather than a keeps that right under
  // FPCR.FZ and on ARMv7. The high half of r plus 0x80 is 0x80, 0 or 0x8000 on exactly those
  // lanes; every other r, a positive normal or a NaN, has a bit set under 0x7f7f.
  uint16x4_t u = vadd_u16(vshrn_n_u32(vreinterpretq_u32_f32(r), 16), vdup_n_u16(0x80));
  uint16x4_t step_lane = vtst_u16(u, vdup_n_u16(0x7f7f));
  return vbsl_u16(step_lane, ybf, vand_u16(a, vdup_n_u16(0xff80)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf prsqrt<Packet4bf>(const Packet4bf& a) {
  // r = FRSQRTE(x) refined by one FRSQRTS step, r1 = r (3 - x r r) / 2, then the same RSHRN
  // narrowing as psqrt. r1 rounds to the correctly rounded bfloat16 rsqrt for all 65536 inputs
  // (packetmath checks them all). FRSQRTE has 8 fraction bits, so the raw estimate always lands on
  // a bfloat16 value or midpoint and misrounds 1270 of them.
  Packet4f x = Bf16ToF32(a);
  Packet4f r = vrsqrteq_f32(x);
#if EIGEN_ARCH_ARM64
  // FMULX gives 2 for 0 * inf where FMUL gives NaN, so no lane needs a select. Where the FPU reads
  // x as +-0 (a subnormal too under FPCR.FZ), r is +-inf, y = 2 and the step is -+inf, and |step|
  // restores r1 = +-inf. Where x is +inf, r is 0, y = 2 and r1 = 0 * 1.5. Negative and NaN x stay
  // NaN. On every other lane FMULX is FMUL and the step is positive, so r1 is the plain step.
  Packet4f y = vmulxq_f32(x, r);
  Packet4f r1 = pmul(r, vabsq_f32(vrsqrtsq_f32(y, r)));
  return vrshrn_n_u32(vreinterpretq_u32_f32(r1), 16);
#else
  // ARMv7 has no FMULX. r1 is NaN where x is +-0, +inf or a subnormal, which ARMv7 NEON flushes,
  // and there r is already the answer (+-inf, 0, +-inf) with a zero low half, so those lanes take
  // r truncated. They are the bit patterns with a - 0x80 >= 0x7f00 unsigned: the exponent-0 range
  // wraps around, and +inf, NaN and every negative value sit above.
  Packet4f r1 = pmul(r, vrsqrtsq_f32(pmul(x, r), r));
  Packet4bf refined = vrshrn_n_u32(vreinterpretq_u32_f32(r1), 16);
  Packet4bf raw = vshrn_n_u32(vreinterpretq_u32_f32(r), 16);
  uint16x4_t special = vcge_u16(vsub_u16(a, vdup_n_u16(0x80)), vdup_n_u16(0x7f00));
  return vbsl_u16(special, raw, refined);
#endif
}

template <>
EIGEN_STRONG_INLINE Packet4bf pfrexp(const Packet4bf& a, Packet4bf& exponent) {
  // Both results are exact: the mantissa keeps the input's significand, and the exponent is a
  // small integer.
  Packet4f fexponent;
  const Packet4bf out = F32ToBf16Truncate(pfrexp<Packet4f>(Bf16ToF32(a), fexponent));
  exponent = F32ToBf16Truncate(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet4bf pldexp(const Packet4bf& a, const Packet4bf& exponent) {
  return F32ToBf16(pldexp<Packet4f>(Bf16ToF32(a), Bf16ToF32(exponent)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf psqrt<Packet8bf>(const Packet8bf& a) {
  // psqrt<Packet4bf> on each float half, narrowed and selected once over all 8 lanes.
  Packet4f x_lo = Bf16ToF32Low(a);
  Packet4f x_hi = Bf16ToF32High(a);
  Packet4f r_lo = vrsqrteq_f32(x_lo);
  Packet4f r_hi = vrsqrteq_f32(x_hi);
  Packet4f y_lo = pmul(x_lo, r_lo);
  Packet4f y_hi = pmul(x_hi, r_hi);
  y_lo = pmul(y_lo, vrsqrtsq_f32(y_lo, r_lo));
  y_hi = pmul(y_hi, vrsqrtsq_f32(y_hi, r_hi));
  Packet8bf ybf = RoundingNarrowBf16(vreinterpretq_u32_f32(y_lo), vreinterpretq_u32_f32(y_hi));
  uint16x8_t u = vaddq_u16(F32ToBf16Truncate(r_lo, r_hi), vdupq_n_u16(0x80));
  uint16x8_t step_lane = vtstq_u16(u, vdupq_n_u16(0x7f7f));
  return vbslq_u16(step_lane, ybf, vandq_u16(a, vdupq_n_u16(0xff80)));
}

template <>
EIGEN_STRONG_INLINE Packet8bf prsqrt<Packet8bf>(const Packet8bf& a) {
  // prsqrt<Packet4bf> on each float half.
  Packet4f x_lo = Bf16ToF32Low(a);
  Packet4f x_hi = Bf16ToF32High(a);
  Packet4f r_lo = vrsqrteq_f32(x_lo);
  Packet4f r_hi = vrsqrteq_f32(x_hi);
#if EIGEN_ARCH_ARM64
  Packet4f r1_lo = pmul(r_lo, vabsq_f32(vrsqrtsq_f32(vmulxq_f32(x_lo, r_lo), r_lo)));
  Packet4f r1_hi = pmul(r_hi, vabsq_f32(vrsqrtsq_f32(vmulxq_f32(x_hi, r_hi), r_hi)));
  return RoundingNarrowBf16(vreinterpretq_u32_f32(r1_lo), vreinterpretq_u32_f32(r1_hi));
#else
  Packet4f r1_lo = pmul(r_lo, vrsqrtsq_f32(pmul(x_lo, r_lo), r_lo));
  Packet4f r1_hi = pmul(r_hi, vrsqrtsq_f32(pmul(x_hi, r_hi), r_hi));
  Packet8bf refined = RoundingNarrowBf16(vreinterpretq_u32_f32(r1_lo), vreinterpretq_u32_f32(r1_hi));
  Packet8bf raw = F32ToBf16Truncate(r_lo, r_hi);
  uint16x8_t special = vcgeq_u16(vsubq_u16(a, vdupq_n_u16(0x80)), vdupq_n_u16(0x7f00));
  return vbslq_u16(special, raw, refined);
#endif
}

template <>
EIGEN_STRONG_INLINE Packet8bf pfrexp(const Packet8bf& a, Packet8bf& exponent) {
  Packet4f fexponent_lo;
  Packet4f fexponent_hi;
  Packet4f lo = pfrexp<Packet4f>(Bf16ToF32Low(a), fexponent_lo);
  Packet4f hi = pfrexp<Packet4f>(Bf16ToF32High(a), fexponent_hi);
  exponent = F32ToBf16Truncate(fexponent_lo, fexponent_hi);
  return F32ToBf16Truncate(lo, hi);
}

template <>
EIGEN_STRONG_INLINE Packet8bf pldexp(const Packet8bf& a, const Packet8bf& exponent) {
  return F32ToBf16(pldexp<Packet4f>(Bf16ToF32Low(a), Bf16ToF32Low(exponent)),
                   pldexp<Packet4f>(Bf16ToF32High(a), Bf16ToF32High(exponent)));
}

template <>
struct type_casting_traits<bfloat16, float> : vectorized_type_casting_traits<bfloat16, float> {};
template <>
struct type_casting_traits<float, bfloat16> : vectorized_type_casting_traits<float, bfloat16> {};

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet4bf, Packet4f>(const Packet4bf& a) {
  return Bf16ToF32(a);
}

template <>
EIGEN_STRONG_INLINE Packet4bf pcast<Packet4f, Packet4bf>(const Packet4f& a) {
  return F32ToBf16(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet8bf, Packet4f>(const Packet8bf& a) {
  return Bf16ToF32Low(a);
}

template <>
EIGEN_STRONG_INLINE Packet8bf pcast<Packet4f, Packet8bf>(const Packet4f& a, const Packet4f& b) {
  return F32ToBf16(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet2f pcast<Packet4bf, Packet2f>(const Packet4bf& a) {
  return vget_low_f32(Bf16ToF32(a));
}

template <>
EIGEN_STRONG_INLINE Packet4bf pcast<Packet2f, Packet4bf>(const Packet2f& a, const Packet2f& b) {
  return F32ToBf16(vcombine_f32(a, b));
}
}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_PACKET_MATH_BF16_NEON_H
