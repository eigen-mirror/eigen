// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET2_MATH_RVV10_H
#define EIGEN_PACKET2_MATH_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

/********************************* PacketMul2Xi ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pset1<PacketMul2Xi>(const numext::int32_t& from) {
  return __riscv_vmv_v_x_i32m2(from, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi plset<PacketMul2Xi>(const numext::int32_t& a) {
  PacketMul2Xi idx = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vid_v_u32m2(unpacket_traits<PacketMul2Xi>::size));
  return __riscv_vadd_vx_i32m2(idx, a, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pzero<PacketMul2Xi>(const PacketMul2Xi& /*a*/) {
  return __riscv_vmv_v_x_i32m2(0, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi padd<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vadd_vv_i32m2(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi psub<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pnegate(const PacketMul2Xi& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pconj(const PacketMul2Xi& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pmul<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pdiv<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pmadd(const PacketMul2Xi& a, const PacketMul2Xi& b, const PacketMul2Xi& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pmsub(const PacketMul2Xi& a, const PacketMul2Xi& b, const PacketMul2Xi& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pnmadd(const PacketMul2Xi& a, const PacketMul2Xi& b, const PacketMul2Xi& c) {
  return __riscv_vnmsub_vv_i32m2(a, b, c, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pnmsub(const PacketMul2Xi& a, const PacketMul2Xi& b, const PacketMul2Xi& c) {
  return __riscv_vnmsub_vv_i32m2(a, b, pnegate(c), unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pmin<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pmax<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcmp_le<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  PacketMask16 mask = __riscv_vmsle_vv_i32m2_b16(a, b, unpacket_traits<PacketMul2Xi>::size);
  return __riscv_vmerge_vxm_i32m2(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcmp_lt<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  PacketMask16 mask = __riscv_vmslt_vv_i32m2_b16(a, b, unpacket_traits<PacketMul2Xi>::size);
  return __riscv_vmerge_vxm_i32m2(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcmp_eq<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  PacketMask16 mask = __riscv_vmseq_vv_i32m2_b16(a, b, unpacket_traits<PacketMul2Xi>::size);
  return __riscv_vmerge_vxm_i32m2(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi ptrue<PacketMul2Xi>(const PacketMul2Xi& /*a*/) {
  return __riscv_vmv_v_x_i32m2(0xffffffffu, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pand<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vand_vv_i32m2(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi por<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vor_vv_i32m2(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pxor<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vxor_vv_i32m2(a, b, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pandnot<PacketMul2Xi>(const PacketMul2Xi& a, const PacketMul2Xi& b) {
  return __riscv_vand_vv_i32m2(a, __riscv_vnot_v_i32m2(b, unpacket_traits<PacketMul2Xi>::size),
                               unpacket_traits<PacketMul2Xi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xi parithmetic_shift_right(PacketMul2Xi a) {
  return __riscv_vsra_vx_i32m2(a, N, unpacket_traits<PacketMul2Xi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xi plogical_shift_right(PacketMul2Xi a) {
  return __riscv_vreinterpret_i32m2(
      __riscv_vsrl_vx_u32m2(__riscv_vreinterpret_u32m2(a), N, unpacket_traits<PacketMul2Xi>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xi plogical_shift_left(PacketMul2Xi a) {
  return __riscv_vsll_vx_i32m2(a, N, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pload<PacketMul2Xi>(const numext::int32_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_i32m2(from, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi ploadu<PacketMul2Xi>(const numext::int32_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_i32m2(from, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi ploaddup<PacketMul2Xi>(const numext::int32_t* from) {
  PacketMul2Xu idx = __riscv_vid_v_u32m2(unpacket_traits<PacketMul2Xi>::size);
  idx = __riscv_vsll_vx_u32m2(__riscv_vand_vx_u32m2(idx, 0xfffffffeu, unpacket_traits<PacketMul2Xi>::size), 1,
                              unpacket_traits<PacketMul2Xi>::size);
  // idx = 0 0 sizeof(int32_t) sizeof(int32_t) 2*sizeof(int32_t) 2*sizeof(int32_t) ...
  return __riscv_vloxei32_v_i32m2(from, idx, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi ploadquad<PacketMul2Xi>(const numext::int32_t* from) {
  PacketMul2Xu idx = __riscv_vid_v_u32m2(unpacket_traits<PacketMul2Xi>::size);
  idx = __riscv_vand_vx_u32m2(idx, 0xfffffffcu, unpacket_traits<PacketMul2Xi>::size);
  return __riscv_vloxei32_v_i32m2(from, idx, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int32_t>(numext::int32_t* to, const PacketMul2Xi& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_i32m2(to, from, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int32_t>(numext::int32_t* to, const PacketMul2Xi& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_i32m2(to, from, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul2Xi pgather<numext::int32_t, PacketMul2Xi>(const numext::int32_t* from,
                                                                             Index stride) {
  return __riscv_vlse32_v_i32m2(from, stride * sizeof(numext::int32_t), unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int32_t, PacketMul2Xi>(numext::int32_t* to, const PacketMul2Xi& from,
                                                                      Index stride) {
  __riscv_vsse32(to, stride * sizeof(numext::int32_t), from, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t pfirst<PacketMul2Xi>(const PacketMul2Xi& a) {
  return __riscv_vmv_x_s_i32m2_i32(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi preverse(const PacketMul2Xi& a) {
  PacketMul2Xu idx =
      __riscv_vrsub_vx_u32m2(__riscv_vid_v_u32m2(unpacket_traits<PacketMul2Xi>::size),
                             unpacket_traits<PacketMul2Xi>::size - 1, unpacket_traits<PacketMul2Xi>::size);
  return __riscv_vrgather_vv_i32m2(a, idx, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pabs(const PacketMul2Xi& a) {
  PacketMul2Xi mask = __riscv_vsra_vx_i32m2(a, 31, unpacket_traits<PacketMul2Xi>::size);
  return __riscv_vsub_vv_i32m2(__riscv_vxor_vv_i32m2(a, mask, unpacket_traits<PacketMul2Xi>::size), mask,
                               unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux<PacketMul2Xi>(const PacketMul2Xi& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i32m2_i32m1(
      a, __riscv_vmv_v_x_i32m1(0, unpacket_traits<PacketMul2Xi>::size / 2), unpacket_traits<PacketMul2Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_mul<PacketMul2Xi>(const PacketMul2Xi& a) {
  return predux_mul<PacketMul1Xi>(__riscv_vmul_vv_i32m1(__riscv_vget_v_i32m2_i32m1(a, 0), __riscv_vget_v_i32m2_i32m1(a, 1),
                                                    unpacket_traits<PacketMul1Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_min<PacketMul2Xi>(const PacketMul2Xi& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i32m2_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::max)(), unpacket_traits<PacketMul2Xi>::size / 2),
      unpacket_traits<PacketMul2Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_max<PacketMul2Xi>(const PacketMul2Xi& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i32m2_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::min)(), unpacket_traits<PacketMul2Xi>::size / 2),
      unpacket_traits<PacketMul2Xi>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul2Xi, N>& kernel) {
  numext::int32_t buffer[unpacket_traits<PacketMul2Xi>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(numext::int32_t), kernel.packet[i], unpacket_traits<PacketMul2Xi>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_i32m2(&buffer[i * unpacket_traits<PacketMul2Xi>::size], unpacket_traits<PacketMul2Xi>::size);
  }
}

template <typename Packet = PacketMul4Xi>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul4Xi>::value && (unpacket_traits<PacketMul4Xi>::size % 8) == 0,
                        PacketMul2Xi>::type
predux_half_dowto4(const PacketMul4Xi& a) {
  return __riscv_vadd_vv_i32m2(__riscv_vget_v_i32m4_i32m2(a, 0), __riscv_vget_v_i32m4_i32m2(a, 1),
                               unpacket_traits<PacketMul2Xi>::size);
}

template <typename Packet = PacketMul2Xi>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul2Xi>::value && (unpacket_traits<PacketMul2Xi>::size % 8) == 0,
                        PacketMul1Xi>::type
predux_half_dowto4(const PacketMul2Xi& a) {
  return __riscv_vadd_vv_i32m1(__riscv_vget_v_i32m2_i32m1(a, 0), __riscv_vget_v_i32m2_i32m1(a, 1),
                               unpacket_traits<PacketMul1Xi>::size);
}

/********************************* PacketMul2Xf ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul2Xf ptrue<PacketMul2Xf>(const PacketMul2Xf& /*a*/) {
  return __riscv_vreinterpret_f32m2(__riscv_vmv_v_x_u32m2(0xffffffffu, unpacket_traits<PacketMul2Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pzero<PacketMul2Xf>(const PacketMul2Xf& /*a*/) {
  return __riscv_vfmv_v_f_f32m2(0.0f, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pabs(const PacketMul2Xf& a) {
  return __riscv_vfabs_v_f32m2(a, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pset1<PacketMul2Xf>(const float& from) {
  return __riscv_vfmv_v_f_f32m2(from, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pset1frombits<PacketMul2Xf>(numext::uint32_t from) {
  return __riscv_vreinterpret_f32m2(__riscv_vmv_v_x_u32m2(from, unpacket_traits<PacketMul2Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf plset<PacketMul2Xf>(const float& a) {
  PacketMul2Xf idx = __riscv_vfcvt_f_x_v_f32m2(
      __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vid_v_u32m2(unpacket_traits<PacketMul4Xi>::size)),
      unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vfadd_vf_f32m2(idx, a, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf padd<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vfadd_vv_f32m2(a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf psub<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vfsub_vv_f32m2(a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pnegate(const PacketMul2Xf& a) {
  return __riscv_vfneg_v_f32m2(a, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pconj(const PacketMul2Xf& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmul<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vfmul_vv_f32m2(a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pdiv<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vfdiv_vv_f32m2(a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmadd(const PacketMul2Xf& a, const PacketMul2Xf& b, const PacketMul2Xf& c) {
  return __riscv_vfmadd_vv_f32m2(a, b, c, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmsub(const PacketMul2Xf& a, const PacketMul2Xf& b, const PacketMul2Xf& c) {
  return __riscv_vfmsub_vv_f32m2(a, b, c, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pnmadd(const PacketMul2Xf& a, const PacketMul2Xf& b, const PacketMul2Xf& c) {
  return __riscv_vfnmsub_vv_f32m2(a, b, c, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pnmsub(const PacketMul2Xf& a, const PacketMul2Xf& b, const PacketMul2Xf& c) {
  return __riscv_vfnmadd_vv_f32m2(a, b, c, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmin<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  PacketMul2Xf nans =
      __riscv_vfmv_v_f_f32m2((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul2Xf>::size);
  PacketMask16 mask = __riscv_vmfeq_vv_f32m2_b16(a, a, unpacket_traits<PacketMul2Xf>::size);
  PacketMask16 mask2 = __riscv_vmfeq_vv_f32m2_b16(b, b, unpacket_traits<PacketMul2Xf>::size);
  mask = __riscv_vmand_mm_b16(mask, mask2, unpacket_traits<PacketMul2Xf>::size);

  return __riscv_vfmin_vv_f32m2_tumu(mask, nans, a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmin<PropagateNaN, PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return pmin<PacketMul2Xf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmin<PropagateNumbers, PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vfmin_vv_f32m2(a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmax<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  PacketMul2Xf nans =
      __riscv_vfmv_v_f_f32m2((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul2Xf>::size);
  PacketMask16 mask = __riscv_vmfeq_vv_f32m2_b16(a, a, unpacket_traits<PacketMul2Xf>::size);
  PacketMask16 mask2 = __riscv_vmfeq_vv_f32m2_b16(b, b, unpacket_traits<PacketMul2Xf>::size);
  mask = __riscv_vmand_mm_b16(mask, mask2, unpacket_traits<PacketMul2Xf>::size);

  return __riscv_vfmax_vv_f32m2_tumu(mask, nans, a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmax<PropagateNaN, PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return pmax<PacketMul2Xf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pmax<PropagateNumbers, PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vfmax_vv_f32m2(a, b, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcmp_le<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  PacketMask16 mask = __riscv_vmfle_vv_f32m2_b16(a, b, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vmerge_vvm_f32m2(pzero<PacketMul2Xf>(a), ptrue<PacketMul2Xf>(a), mask,
                                  unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcmp_lt<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  PacketMask16 mask = __riscv_vmflt_vv_f32m2_b16(a, b, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vmerge_vvm_f32m2(pzero<PacketMul2Xf>(a), ptrue<PacketMul2Xf>(a), mask,
                                  unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcmp_eq<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  PacketMask16 mask = __riscv_vmfeq_vv_f32m2_b16(a, b, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vmerge_vvm_f32m2(pzero<PacketMul2Xf>(a), ptrue<PacketMul2Xf>(a), mask,
                                  unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcmp_lt_or_nan<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  PacketMask16 mask = __riscv_vmfge_vv_f32m2_b16(a, b, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vfmerge_vfm_f32m2(ptrue<PacketMul2Xf>(a), 0.0f, mask, unpacket_traits<PacketMul2Xf>::size);
}

// Logical Operations are not supported for float, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketMul2Xf pand<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(a),
                                                                  __riscv_vreinterpret_v_f32m2_u32m2(b),
                                                                  unpacket_traits<PacketMul2Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf por<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(a),
                                                                 __riscv_vreinterpret_v_f32m2_u32m2(b),
                                                                 unpacket_traits<PacketMul2Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pxor<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vxor_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(a),
                                                                  __riscv_vreinterpret_v_f32m2_u32m2(b),
                                                                  unpacket_traits<PacketMul2Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pandnot<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& b) {
  return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(
      __riscv_vreinterpret_v_f32m2_u32m2(a),
      __riscv_vnot_v_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(b), unpacket_traits<PacketMul2Xf>::size),
      unpacket_traits<PacketMul2Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pload<PacketMul2Xf>(const float* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_f32m2(from, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf ploadu<PacketMul2Xf>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_f32m2(from, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf ploaddup<PacketMul2Xf>(const float* from) {
  PacketMul2Xu idx = __riscv_vid_v_u32m2(unpacket_traits<PacketMul2Xf>::size);
  idx = __riscv_vsll_vx_u32m2(__riscv_vand_vx_u32m2(idx, 0xfffffffeu, unpacket_traits<PacketMul2Xf>::size), 1,
                              unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vloxei32_v_f32m2(from, idx, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf ploadquad<PacketMul2Xf>(const float* from) {
  PacketMul2Xu idx = __riscv_vid_v_u32m2(unpacket_traits<PacketMul2Xf>::size);
  idx = __riscv_vand_vx_u32m2(idx, 0xfffffffcu, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vloxei32_v_f32m2(from, idx, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const PacketMul2Xf& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_f32m2(to, from, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const PacketMul2Xf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_f32m2(to, from, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul2Xf pgather<float, PacketMul2Xf>(const float* from, Index stride) {
  return __riscv_vlse32_v_f32m2(from, stride * sizeof(float), unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, PacketMul2Xf>(float* to, const PacketMul2Xf& from, Index stride) {
  __riscv_vsse32(to, stride * sizeof(float), from, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE float pfirst<PacketMul2Xf>(const PacketMul2Xf& a) {
  return __riscv_vfmv_f_s_f32m2_f32(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf psqrt(const PacketMul2Xf& a) {
  return __riscv_vfsqrt_v_f32m2(a, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf print<PacketMul2Xf>(const PacketMul2Xf& a) {
  const PacketMul2Xf limit = pset1<PacketMul2Xf>(static_cast<float>(1 << 23));
  const PacketMul2Xf abs_a = pabs(a);

  PacketMask16 mask = __riscv_vmfne_vv_f32m2_b16(a, a, unpacket_traits<PacketMul2Xf>::size);
  const PacketMul2Xf x = __riscv_vfadd_vv_f32m2_tumu(mask, a, a, a, unpacket_traits<PacketMul2Xf>::size);
  const PacketMul2Xf new_x = __riscv_vfcvt_f_x_v_f32m2(
      __riscv_vfcvt_x_f_v_i32m2(a, unpacket_traits<PacketMul2Xf>::size), unpacket_traits<PacketMul2Xf>::size);

  mask = __riscv_vmflt_vv_f32m2_b16(abs_a, limit, unpacket_traits<PacketMul2Xf>::size);
  PacketMul2Xf signed_x = __riscv_vfsgnj_vv_f32m2(new_x, x, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vmerge_vvm_f32m2(x, signed_x, mask, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pfloor<PacketMul2Xf>(const PacketMul2Xf& a) {
  PacketMul2Xf tmp = print<PacketMul2Xf>(a);
  // If greater, subtract one.
  PacketMask16 mask = __riscv_vmflt_vv_f32m2_b16(a, tmp, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vfsub_vf_f32m2_tumu(mask, tmp, tmp, 1.0f, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf preverse(const PacketMul2Xf& a) {
  PacketMul2Xu idx =
      __riscv_vrsub_vx_u32m2(__riscv_vid_v_u32m2(unpacket_traits<PacketMul2Xf>::size),
                             unpacket_traits<PacketMul2Xf>::size - 1, unpacket_traits<PacketMul2Xf>::size);
  return __riscv_vrgather_vv_f32m2(a, idx, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pfrexp<PacketMul2Xf>(const PacketMul2Xf& a, PacketMul2Xf& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE float predux<PacketMul2Xf>(const PacketMul2Xf& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f32m2_f32m1(
      a, __riscv_vfmv_v_f_f32m1(0.0, unpacket_traits<PacketMul2Xf>::size / 2), unpacket_traits<PacketMul2Xf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_mul<PacketMul2Xf>(const PacketMul2Xf& a) {
  return predux_mul<PacketMul1Xf>(__riscv_vfmul_vv_f32m1(__riscv_vget_v_f32m2_f32m1(a, 0), __riscv_vget_v_f32m2_f32m1(a, 1),
                                                     unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketMul2Xf>(const PacketMul2Xf& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f32m2_f32m1(
                        a,
                        __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(),
                                               unpacket_traits<PacketMul2Xf>::size / 2),
                        unpacket_traits<PacketMul2Xf>::size)),
                    (std::numeric_limits<float>::max)());
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketMul2Xf>(const PacketMul2Xf& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f32m2_f32m1(
                        a,
                        __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(),
                                               unpacket_traits<PacketMul2Xf>::size / 2),
                        unpacket_traits<PacketMul2Xf>::size)),
                    -(std::numeric_limits<float>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul2Xf, N>& kernel) {
  float buffer[unpacket_traits<PacketMul2Xf>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(float), kernel.packet[i], unpacket_traits<PacketMul2Xf>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_f32m2(&buffer[i * unpacket_traits<PacketMul2Xf>::size], unpacket_traits<PacketMul2Xf>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pldexp<PacketMul2Xf>(const PacketMul2Xf& a, const PacketMul2Xf& exponent) {
  return pldexp_generic(a, exponent);
}

template <typename Packet = PacketMul4Xf>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul4Xf>::value && (unpacket_traits<PacketMul4Xf>::size % 8) == 0,
                        PacketMul2Xf>::type
predux_half_dowto4(const PacketMul4Xf& a) {
  return __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(a, 0), __riscv_vget_v_f32m4_f32m2(a, 1),
                                unpacket_traits<PacketMul2Xf>::size);
}

template <typename Packet = PacketMul2Xf>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul2Xf>::value && (unpacket_traits<PacketMul2Xf>::size % 8) == 0,
                        PacketMul1Xf>::type
predux_half_dowto4(const PacketMul2Xf& a) {
  return __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(a, 0), __riscv_vget_v_f32m2_f32m1(a, 1),
                                unpacket_traits<PacketMul1Xf>::size);
}

/********************************* PacketMul2Xl ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pset1<PacketMul2Xl>(const numext::int64_t& from) {
  return __riscv_vmv_v_x_i64m2(from, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl plset<PacketMul2Xl>(const numext::int64_t& a) {
  PacketMul2Xl idx = __riscv_vreinterpret_v_u64m2_i64m2(__riscv_vid_v_u64m2(unpacket_traits<PacketMul2Xl>::size));
  return __riscv_vadd_vx_i64m2(idx, a, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pzero<PacketMul2Xl>(const PacketMul2Xl& /*a*/) {
  return __riscv_vmv_v_x_i64m2(0, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl padd<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vadd_vv_i64m2(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl psub<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pnegate(const PacketMul2Xl& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pconj(const PacketMul2Xl& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pmul<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pdiv<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pmadd(const PacketMul2Xl& a, const PacketMul2Xl& b, const PacketMul2Xl& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pmsub(const PacketMul2Xl& a, const PacketMul2Xl& b, const PacketMul2Xl& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pnmadd(const PacketMul2Xl& a, const PacketMul2Xl& b, const PacketMul2Xl& c) {
  return __riscv_vnmsub_vv_i64m2(a, b, c, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pnmsub(const PacketMul2Xl& a, const PacketMul2Xl& b, const PacketMul2Xl& c) {
  return __riscv_vnmsub_vv_i64m2(a, b, pnegate(c), unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pmin<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pmax<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcmp_le<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  PacketMask32 mask = __riscv_vmsle_vv_i64m2_b32(a, b, unpacket_traits<PacketMul2Xl>::size);
  return __riscv_vmerge_vxm_i64m2(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcmp_lt<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  PacketMask32 mask = __riscv_vmslt_vv_i64m2_b32(a, b, unpacket_traits<PacketMul2Xl>::size);
  return __riscv_vmerge_vxm_i64m2(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcmp_eq<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  PacketMask32 mask = __riscv_vmseq_vv_i64m2_b32(a, b, unpacket_traits<PacketMul2Xl>::size);
  return __riscv_vmerge_vxm_i64m2(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl ptrue<PacketMul2Xl>(const PacketMul2Xl& /*a*/) {
  return __riscv_vmv_v_x_i64m2(0xffffffffffffffffu, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pand<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vand_vv_i64m2(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl por<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vor_vv_i64m2(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pxor<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vxor_vv_i64m2(a, b, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pandnot<PacketMul2Xl>(const PacketMul2Xl& a, const PacketMul2Xl& b) {
  return __riscv_vand_vv_i64m2(a, __riscv_vnot_v_i64m2(b, unpacket_traits<PacketMul2Xl>::size),
                               unpacket_traits<PacketMul2Xl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xl parithmetic_shift_right(PacketMul2Xl a) {
  return __riscv_vsra_vx_i64m2(a, N, unpacket_traits<PacketMul2Xl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xl plogical_shift_right(PacketMul2Xl a) {
  return __riscv_vreinterpret_i64m2(
      __riscv_vsrl_vx_u64m2(__riscv_vreinterpret_u64m2(a), N, unpacket_traits<PacketMul2Xl>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xl plogical_shift_left(PacketMul2Xl a) {
  return __riscv_vsll_vx_i64m2(a, N, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pload<PacketMul2Xl>(const numext::int64_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_i64m2(from, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl ploadu<PacketMul2Xl>(const numext::int64_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_i64m2(from, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl ploaddup<PacketMul2Xl>(const numext::int64_t* from) {
  PacketMul2Xul idx = __riscv_vid_v_u64m2(unpacket_traits<PacketMul2Xl>::size);
  idx = __riscv_vsll_vx_u64m2(__riscv_vand_vx_u64m2(idx, 0xfffffffffffffffeu, unpacket_traits<PacketMul2Xl>::size), 2,
                              unpacket_traits<PacketMul2Xl>::size);
  // idx = 0 0 sizeof(int64_t) sizeof(int64_t) 2*sizeof(int64_t) 2*sizeof(int64_t) ...
  return __riscv_vloxei64_v_i64m2(from, idx, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl ploadquad<PacketMul2Xl>(const numext::int64_t* from) {
  PacketMul2Xul idx = __riscv_vid_v_u64m2(unpacket_traits<PacketMul2Xl>::size);
  idx = __riscv_vsll_vx_u64m2(__riscv_vand_vx_u64m2(idx, 0xfffffffffffffffcu, unpacket_traits<PacketMul2Xl>::size), 1,
                              unpacket_traits<PacketMul2Xl>::size);
  return __riscv_vloxei64_v_i64m2(from, idx, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int64_t>(numext::int64_t* to, const PacketMul2Xl& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_i64m2(to, from, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int64_t>(numext::int64_t* to, const PacketMul2Xl& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_i64m2(to, from, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul2Xl pgather<numext::int64_t, PacketMul2Xl>(const numext::int64_t* from,
                                                                             Index stride) {
  return __riscv_vlse64_v_i64m2(from, stride * sizeof(numext::int64_t), unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int64_t, PacketMul2Xl>(numext::int64_t* to, const PacketMul2Xl& from,
                                                                      Index stride) {
  __riscv_vsse64(to, stride * sizeof(numext::int64_t), from, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t pfirst<PacketMul2Xl>(const PacketMul2Xl& a) {
  return __riscv_vmv_x_s_i64m2_i64(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl preverse(const PacketMul2Xl& a) {
  PacketMul2Xul idx =
      __riscv_vrsub_vx_u64m2(__riscv_vid_v_u64m2(unpacket_traits<PacketMul2Xl>::size),
                             unpacket_traits<PacketMul2Xl>::size - 1, unpacket_traits<PacketMul2Xl>::size);
  return __riscv_vrgather_vv_i64m2(a, idx, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pabs(const PacketMul2Xl& a) {
  PacketMul2Xl mask = __riscv_vsra_vx_i64m2(a, 63, unpacket_traits<PacketMul2Xl>::size);
  return __riscv_vsub_vv_i64m2(__riscv_vxor_vv_i64m2(a, mask, unpacket_traits<PacketMul2Xl>::size), mask,
                               unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux<PacketMul2Xl>(const PacketMul2Xl& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i64m2_i64m1(
      a, __riscv_vmv_v_x_i64m1(0, unpacket_traits<PacketMul2Xl>::size / 2), unpacket_traits<PacketMul2Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_mul<PacketMul2Xl>(const PacketMul2Xl& a) {
  return predux_mul<PacketMul1Xl>(__riscv_vmul_vv_i64m1(__riscv_vget_v_i64m2_i64m1(a, 0), __riscv_vget_v_i64m2_i64m1(a, 1),
                                                    unpacket_traits<PacketMul1Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_min<PacketMul2Xl>(const PacketMul2Xl& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i64m2_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::max)(), unpacket_traits<PacketMul2Xl>::size / 2),
      unpacket_traits<PacketMul2Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_max<PacketMul2Xl>(const PacketMul2Xl& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i64m2_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::min)(), unpacket_traits<PacketMul2Xl>::size / 2),
      unpacket_traits<PacketMul2Xl>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul2Xl, N>& kernel) {
  numext::int64_t buffer[unpacket_traits<PacketMul2Xl>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(numext::int64_t), kernel.packet[i], unpacket_traits<PacketMul2Xl>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_i64m2(&buffer[i * unpacket_traits<PacketMul2Xl>::size], unpacket_traits<PacketMul2Xl>::size);
  }
}

template <typename Packet = PacketMul4Xl>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul4Xl>::value && (unpacket_traits<PacketMul4Xl>::size % 8) == 0,
                        PacketMul2Xl>::type
predux_half_dowto4(const PacketMul4Xl& a) {
  return __riscv_vadd_vv_i64m2(__riscv_vget_v_i64m4_i64m2(a, 0), __riscv_vget_v_i64m4_i64m2(a, 1),
                               unpacket_traits<PacketMul2Xl>::size);
}

template <typename Packet = PacketMul2Xl>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul2Xl>::value && (unpacket_traits<PacketMul2Xl>::size % 8) == 0,
                        PacketMul1Xl>::type
predux_half_dowto4(const PacketMul2Xl& a) {
  return __riscv_vadd_vv_i64m1(__riscv_vget_v_i64m2_i64m1(a, 0), __riscv_vget_v_i64m2_i64m1(a, 1),
                               unpacket_traits<PacketMul1Xl>::size);
}

/********************************* PacketMul2Xd ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul2Xd ptrue<PacketMul2Xd>(const PacketMul2Xd& /*a*/) {
  return __riscv_vreinterpret_f64m2(__riscv_vmv_v_x_u64m2(0xffffffffffffffffu, unpacket_traits<PacketMul2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pzero<PacketMul2Xd>(const PacketMul2Xd& /*a*/) {
  return __riscv_vfmv_v_f_f64m2(0.0, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pabs(const PacketMul2Xd& a) {
  return __riscv_vfabs_v_f64m2(a, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pset1<PacketMul2Xd>(const double& from) {
  return __riscv_vfmv_v_f_f64m2(from, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pset1frombits<PacketMul2Xd>(numext::uint64_t from) {
  return __riscv_vreinterpret_f64m2(__riscv_vmv_v_x_u64m2(from, unpacket_traits<PacketMul2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd plset<PacketMul2Xd>(const double& a) {
  PacketMul2Xd idx = __riscv_vfcvt_f_x_v_f64m2(
      __riscv_vreinterpret_v_u64m2_i64m2(__riscv_vid_v_u64m2(unpacket_traits<PacketMul4Xi>::size)),
      unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vfadd_vf_f64m2(idx, a, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd padd<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vfadd_vv_f64m2(a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd psub<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vfsub_vv_f64m2(a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pnegate(const PacketMul2Xd& a) {
  return __riscv_vfneg_v_f64m2(a, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pconj(const PacketMul2Xd& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmul<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vfmul_vv_f64m2(a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pdiv<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vfdiv_vv_f64m2(a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmadd(const PacketMul2Xd& a, const PacketMul2Xd& b, const PacketMul2Xd& c) {
  return __riscv_vfmadd_vv_f64m2(a, b, c, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmsub(const PacketMul2Xd& a, const PacketMul2Xd& b, const PacketMul2Xd& c) {
  return __riscv_vfmsub_vv_f64m2(a, b, c, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pnmadd(const PacketMul2Xd& a, const PacketMul2Xd& b, const PacketMul2Xd& c) {
  return __riscv_vfnmsub_vv_f64m2(a, b, c, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pnmsub(const PacketMul2Xd& a, const PacketMul2Xd& b, const PacketMul2Xd& c) {
  return __riscv_vfnmadd_vv_f64m2(a, b, c, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmin<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  PacketMul2Xd nans =
      __riscv_vfmv_v_f_f64m2((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul2Xd>::size);
  PacketMask32 mask = __riscv_vmfeq_vv_f64m2_b32(a, a, unpacket_traits<PacketMul2Xd>::size);
  PacketMask32 mask2 = __riscv_vmfeq_vv_f64m2_b32(b, b, unpacket_traits<PacketMul2Xd>::size);
  mask = __riscv_vmand_mm_b32(mask, mask2, unpacket_traits<PacketMul2Xd>::size);

  return __riscv_vfmin_vv_f64m2_tumu(mask, nans, a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmin<PropagateNaN, PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return pmin<PacketMul2Xd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmin<PropagateNumbers, PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vfmin_vv_f64m2(a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmax<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  PacketMul2Xd nans =
      __riscv_vfmv_v_f_f64m2((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul2Xd>::size);
  PacketMask32 mask = __riscv_vmfeq_vv_f64m2_b32(a, a, unpacket_traits<PacketMul2Xd>::size);
  PacketMask32 mask2 = __riscv_vmfeq_vv_f64m2_b32(b, b, unpacket_traits<PacketMul2Xd>::size);
  mask = __riscv_vmand_mm_b32(mask, mask2, unpacket_traits<PacketMul2Xd>::size);

  return __riscv_vfmax_vv_f64m2_tumu(mask, nans, a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmax<PropagateNaN, PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return pmax<PacketMul2Xd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pmax<PropagateNumbers, PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vfmax_vv_f64m2(a, b, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcmp_le<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  PacketMask32 mask = __riscv_vmfle_vv_f64m2_b32(a, b, unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vmerge_vvm_f64m2(pzero<PacketMul2Xd>(a), ptrue<PacketMul2Xd>(a), mask,
                                  unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcmp_lt<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  PacketMask32 mask = __riscv_vmflt_vv_f64m2_b32(a, b, unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vmerge_vvm_f64m2(pzero<PacketMul2Xd>(a), ptrue<PacketMul2Xd>(a), mask,
                                  unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcmp_eq<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  PacketMask32 mask = __riscv_vmfeq_vv_f64m2_b32(a, b, unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vmerge_vvm_f64m2(pzero<PacketMul2Xd>(a), ptrue<PacketMul2Xd>(a), mask,
                                  unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcmp_lt_or_nan<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  PacketMask32 mask = __riscv_vmfge_vv_f64m2_b32(a, b, unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vfmerge_vfm_f64m2(ptrue<PacketMul2Xd>(a), 0.0, mask, unpacket_traits<PacketMul2Xd>::size);
}

// Logical Operations are not supported for double, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketMul2Xd pand<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(a),
                                                                  __riscv_vreinterpret_v_f64m2_u64m2(b),
                                                                  unpacket_traits<PacketMul2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd por<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vv_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(a),
                                                                 __riscv_vreinterpret_v_f64m2_u64m2(b),
                                                                 unpacket_traits<PacketMul2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pxor<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vxor_vv_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(a),
                                                                  __riscv_vreinterpret_v_f64m2_u64m2(b),
                                                                  unpacket_traits<PacketMul2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pandnot<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& b) {
  return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(
      __riscv_vreinterpret_v_f64m2_u64m2(a),
      __riscv_vnot_v_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(b), unpacket_traits<PacketMul2Xd>::size),
      unpacket_traits<PacketMul2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pload<PacketMul2Xd>(const double* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_f64m2(from, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd ploadu<PacketMul2Xd>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_f64m2(from, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd ploaddup<PacketMul2Xd>(const double* from) {
  PacketMul2Xul idx = __riscv_vid_v_u64m2(unpacket_traits<PacketMul2Xd>::size);
  idx = __riscv_vsll_vx_u64m2(__riscv_vand_vx_u64m2(idx, 0xfffffffffffffffeu, unpacket_traits<PacketMul2Xd>::size), 2,
                              unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vloxei64_v_f64m2(from, idx, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd ploadquad<PacketMul2Xd>(const double* from) {
  PacketMul2Xul idx = __riscv_vid_v_u64m2(unpacket_traits<PacketMul2Xd>::size);
  idx = __riscv_vsll_vx_u64m2(__riscv_vand_vx_u64m2(idx, 0xfffffffffffffffcu, unpacket_traits<PacketMul2Xd>::size), 1,
                              unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vloxei64_v_f64m2(from, idx, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<double>(double* to, const PacketMul2Xd& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_f64m2(to, from, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const PacketMul2Xd& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_f64m2(to, from, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul2Xd pgather<double, PacketMul2Xd>(const double* from, Index stride) {
  return __riscv_vlse64_v_f64m2(from, stride * sizeof(double), unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<double, PacketMul2Xd>(double* to, const PacketMul2Xd& from, Index stride) {
  __riscv_vsse64(to, stride * sizeof(double), from, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE double pfirst<PacketMul2Xd>(const PacketMul2Xd& a) {
  return __riscv_vfmv_f_s_f64m2_f64(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd psqrt(const PacketMul2Xd& a) {
  return __riscv_vfsqrt_v_f64m2(a, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd print<PacketMul2Xd>(const PacketMul2Xd& a) {
  const PacketMul2Xd limit = pset1<PacketMul2Xd>(static_cast<double>(1ull << 52));
  const PacketMul2Xd abs_a = pabs(a);

  PacketMask32 mask = __riscv_vmfne_vv_f64m2_b32(a, a, unpacket_traits<PacketMul2Xd>::size);
  const PacketMul2Xd x = __riscv_vfadd_vv_f64m2_tumu(mask, a, a, a, unpacket_traits<PacketMul2Xd>::size);
  const PacketMul2Xd new_x = __riscv_vfcvt_f_x_v_f64m2(
      __riscv_vfcvt_x_f_v_i64m2(a, unpacket_traits<PacketMul2Xd>::size), unpacket_traits<PacketMul2Xd>::size);

  mask = __riscv_vmflt_vv_f64m2_b32(abs_a, limit, unpacket_traits<PacketMul2Xd>::size);
  PacketMul2Xd signed_x = __riscv_vfsgnj_vv_f64m2(new_x, x, unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vmerge_vvm_f64m2(x, signed_x, mask, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pfloor<PacketMul2Xd>(const PacketMul2Xd& a) {
  PacketMul2Xd tmp = print<PacketMul2Xd>(a);
  // If greater, subtract one.
  PacketMask32 mask = __riscv_vmflt_vv_f64m2_b32(a, tmp, unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vfsub_vf_f64m2_tumu(mask, tmp, tmp, 1.0, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd preverse(const PacketMul2Xd& a) {
  PacketMul2Xul idx =
      __riscv_vrsub_vx_u64m2(__riscv_vid_v_u64m2(unpacket_traits<PacketMul2Xd>::size),
                             unpacket_traits<PacketMul2Xd>::size - 1, unpacket_traits<PacketMul2Xd>::size);
  return __riscv_vrgather_vv_f64m2(a, idx, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pfrexp<PacketMul2Xd>(const PacketMul2Xd& a, PacketMul2Xd& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE double predux<PacketMul2Xd>(const PacketMul2Xd& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f64m2_f64m1(
      a, __riscv_vfmv_v_f_f64m1(0.0, unpacket_traits<PacketMul2Xd>::size / 2), unpacket_traits<PacketMul2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_mul<PacketMul2Xd>(const PacketMul2Xd& a) {
  return predux_mul<PacketMul1Xd>(__riscv_vfmul_vv_f64m1(__riscv_vget_v_f64m2_f64m1(a, 0), __riscv_vget_v_f64m2_f64m1(a, 1),
                                                     unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_min<PacketMul2Xd>(const PacketMul2Xd& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f64m2_f64m1(
                        a,
                        __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(),
                                               unpacket_traits<PacketMul2Xd>::size / 2),
                        unpacket_traits<PacketMul2Xd>::size)),
                    (std::numeric_limits<double>::max)());
}

template <>
EIGEN_STRONG_INLINE double predux_max<PacketMul2Xd>(const PacketMul2Xd& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f64m2_f64m1(
                        a,
                        __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(),
                                               unpacket_traits<PacketMul2Xd>::size / 2),
                        unpacket_traits<PacketMul2Xd>::size)),
                    -(std::numeric_limits<double>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul2Xd, N>& kernel) {
  double buffer[unpacket_traits<PacketMul2Xd>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(double), kernel.packet[i], unpacket_traits<PacketMul2Xd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_f64m2(&buffer[i * unpacket_traits<PacketMul2Xd>::size], unpacket_traits<PacketMul2Xd>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pldexp<PacketMul2Xd>(const PacketMul2Xd& a, const PacketMul2Xd& exponent) {
  return pldexp_generic(a, exponent);
}

template <typename Packet = PacketMul4Xd>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul4Xd>::value && (unpacket_traits<PacketMul4Xd>::size % 8) == 0,
                        PacketMul2Xd>::type
predux_half_dowto4(const PacketMul4Xd& a) {
  return __riscv_vfadd_vv_f64m2(__riscv_vget_v_f64m4_f64m2(a, 0), __riscv_vget_v_f64m4_f64m2(a, 1),
                                unpacket_traits<PacketMul2Xd>::size);
}

template <typename Packet = PacketMul2Xd>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul2Xd>::value && (unpacket_traits<PacketMul2Xd>::size % 8) == 0,
                        PacketMul1Xd>::type
predux_half_dowto4(const PacketMul2Xd& a) {
  return __riscv_vfadd_vv_f64m1(__riscv_vget_v_f64m2_f64m1(a, 0), __riscv_vget_v_f64m2_f64m1(a, 1),
                                unpacket_traits<PacketMul1Xd>::size);
}

/********************************* PacketMul2Xs ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pset1<PacketMul2Xs>(const numext::int16_t& from) {
  return __riscv_vmv_v_x_i16m2(from, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs plset<PacketMul2Xs>(const numext::int16_t& a) {
  PacketMul2Xs idx = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vid_v_u16m2(unpacket_traits<PacketMul2Xs>::size));
  return __riscv_vadd_vx_i16m2(idx, a, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pzero<PacketMul2Xs>(const PacketMul2Xs& /*a*/) {
  return __riscv_vmv_v_x_i16m2(0, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs padd<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vadd_vv_i16m2(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs psub<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pnegate(const PacketMul2Xs& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pconj(const PacketMul2Xs& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pmul<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pdiv<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pmadd(const PacketMul2Xs& a, const PacketMul2Xs& b, const PacketMul2Xs& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pmsub(const PacketMul2Xs& a, const PacketMul2Xs& b, const PacketMul2Xs& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pnmadd(const PacketMul2Xs& a, const PacketMul2Xs& b, const PacketMul2Xs& c) {
  return __riscv_vnmsub_vv_i16m2(a, b, c, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pnmsub(const PacketMul2Xs& a, const PacketMul2Xs& b, const PacketMul2Xs& c) {
  return __riscv_vnmsub_vv_i16m2(a, b, pnegate(c), unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pmin<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pmax<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pcmp_le<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  PacketMask8 mask = __riscv_vmsle_vv_i16m2_b8(a, b, unpacket_traits<PacketMul2Xs>::size);
  return __riscv_vmerge_vxm_i16m2(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pcmp_lt<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  PacketMask8 mask = __riscv_vmslt_vv_i16m2_b8(a, b, unpacket_traits<PacketMul2Xs>::size);
  return __riscv_vmerge_vxm_i16m2(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pcmp_eq<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  PacketMask8 mask = __riscv_vmseq_vv_i16m2_b8(a, b, unpacket_traits<PacketMul2Xs>::size);
  return __riscv_vmerge_vxm_i16m2(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs ptrue<PacketMul2Xs>(const PacketMul2Xs& /*a*/) {
  return __riscv_vmv_v_x_i16m2(static_cast<unsigned short>(0xffffu), unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pand<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vand_vv_i16m2(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs por<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vor_vv_i16m2(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pxor<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vxor_vv_i16m2(a, b, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pandnot<PacketMul2Xs>(const PacketMul2Xs& a, const PacketMul2Xs& b) {
  return __riscv_vand_vv_i16m2(a, __riscv_vnot_v_i16m2(b, unpacket_traits<PacketMul2Xs>::size),
                               unpacket_traits<PacketMul2Xs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xs parithmetic_shift_right(PacketMul2Xs a) {
  return __riscv_vsra_vx_i16m2(a, N, unpacket_traits<PacketMul2Xs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xs plogical_shift_right(PacketMul2Xs a) {
  return __riscv_vreinterpret_i16m2(
      __riscv_vsrl_vx_u16m2(__riscv_vreinterpret_u16m2(a), N, unpacket_traits<PacketMul2Xs>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul2Xs plogical_shift_left(PacketMul2Xs a) {
  return __riscv_vsll_vx_i16m2(a, N, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pload<PacketMul2Xs>(const numext::int16_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle16_v_i16m2(from, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs ploadu<PacketMul2Xs>(const numext::int16_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle16_v_i16m2(from, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs ploaddup<PacketMul2Xs>(const numext::int16_t* from) {
  PacketMul2Xsu idx = __riscv_vid_v_u16m2(unpacket_traits<PacketMul2Xs>::size);
  idx = __riscv_vand_vx_u16m2(idx, 0xfffeu, unpacket_traits<PacketMul2Xs>::size);
  // idx = 0 0 sizeof(int16_t) sizeof(int16_t) 2*sizeof(int16_t) 2*sizeof(int16_t) ...
  return __riscv_vloxei16_v_i16m2(from, idx, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs ploadquad<PacketMul2Xs>(const numext::int16_t* from) {
  PacketMul2Xsu idx = __riscv_vid_v_u16m2(unpacket_traits<PacketMul2Xs>::size);
  idx = __riscv_vsrl_vx_u16m2(__riscv_vand_vx_u16m2(idx, 0xfffcu, unpacket_traits<PacketMul2Xs>::size), 1,
                              unpacket_traits<PacketMul2Xs>::size);
  return __riscv_vloxei16_v_i16m2(from, idx, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int16_t>(numext::int16_t* to, const PacketMul2Xs& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse16_v_i16m2(to, from, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int16_t>(numext::int16_t* to, const PacketMul2Xs& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse16_v_i16m2(to, from, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul2Xs pgather<numext::int16_t, PacketMul2Xs>(const numext::int16_t* from,
                                                                             Index stride) {
  return __riscv_vlse16_v_i16m2(from, stride * sizeof(numext::int16_t), unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int16_t, PacketMul2Xs>(numext::int16_t* to, const PacketMul2Xs& from,
                                                                      Index stride) {
  __riscv_vsse16(to, stride * sizeof(numext::int16_t), from, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t pfirst<PacketMul2Xs>(const PacketMul2Xs& a) {
  return __riscv_vmv_x_s_i16m2_i16(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs preverse(const PacketMul2Xs& a) {
  PacketMul2Xsu idx =
      __riscv_vrsub_vx_u16m2(__riscv_vid_v_u16m2(unpacket_traits<PacketMul2Xs>::size),
                             unpacket_traits<PacketMul2Xs>::size - 1, unpacket_traits<PacketMul2Xs>::size);
  return __riscv_vrgather_vv_i16m2(a, idx, unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pabs(const PacketMul2Xs& a) {
  PacketMul2Xs mask = __riscv_vsra_vx_i16m2(a, 15, unpacket_traits<PacketMul2Xs>::size);
  return __riscv_vsub_vv_i16m2(__riscv_vxor_vv_i16m2(a, mask, unpacket_traits<PacketMul2Xs>::size), mask,
                               unpacket_traits<PacketMul2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux<PacketMul2Xs>(const PacketMul2Xs& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i16m2_i16m1(
      a, __riscv_vmv_v_x_i16m1(0, unpacket_traits<PacketMul2Xs>::size / 2), unpacket_traits<PacketMul2Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_mul<PacketMul2Xs>(const PacketMul2Xs& a) {
  return predux_mul<PacketMul1Xs>(__riscv_vmul_vv_i16m1(__riscv_vget_v_i16m2_i16m1(a, 0), __riscv_vget_v_i16m2_i16m1(a, 1),
                                                    unpacket_traits<PacketMul1Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_min<PacketMul2Xs>(const PacketMul2Xs& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i16m2_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::max)(), unpacket_traits<PacketMul2Xs>::size / 2),
      unpacket_traits<PacketMul2Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_max<PacketMul2Xs>(const PacketMul2Xs& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i16m2_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::min)(), unpacket_traits<PacketMul2Xs>::size / 2),
      unpacket_traits<PacketMul2Xs>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul2Xs, N>& kernel) {
  numext::int16_t buffer[unpacket_traits<PacketMul2Xs>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse16(&buffer[i], N * sizeof(numext::int16_t), kernel.packet[i], unpacket_traits<PacketMul2Xs>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle16_v_i16m2(&buffer[i * unpacket_traits<PacketMul2Xs>::size], unpacket_traits<PacketMul2Xs>::size);
  }
}

template <typename Packet = PacketMul4Xs>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul4Xs>::value && (unpacket_traits<PacketMul4Xs>::size % 8) == 0,
                        PacketMul2Xs>::type
predux_half_dowto4(const PacketMul4Xs& a) {
  return __riscv_vadd_vv_i16m2(__riscv_vget_v_i16m4_i16m2(a, 0), __riscv_vget_v_i16m4_i16m2(a, 1),
                               unpacket_traits<PacketMul2Xs>::size);
}

template <typename Packet = PacketMul2Xs>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, PacketMul2Xs>::value && (unpacket_traits<PacketMul2Xs>::size % 8) == 0,
                        PacketMul1Xs>::type
predux_half_dowto4(const PacketMul2Xs& a) {
  return __riscv_vadd_vv_i16m1(__riscv_vget_v_i16m2_i16m1(a, 0), __riscv_vget_v_i16m2_i16m1(a, 1),
                               unpacket_traits<PacketMul1Xs>::size);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET2_MATH_RVV10_H
