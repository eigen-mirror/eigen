// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET4_MATH_RVV10_H
#define EIGEN_PACKET4_MATH_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

/********************************* PacketMul4Xi ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pset1<PacketMul4Xi>(const numext::int32_t& from) {
  return __riscv_vmv_v_x_i32m4(from, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi plset<PacketMul4Xi>(const numext::int32_t& a) {
  PacketMul4Xi idx = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xi>::size));
  return __riscv_vadd_vx_i32m4(idx, a, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pzero<PacketMul4Xi>(const PacketMul4Xi& /*a*/) {
  return __riscv_vmv_v_x_i32m4(0, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi padd<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vadd_vv_i32m4(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi psub<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pnegate(const PacketMul4Xi& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pconj(const PacketMul4Xi& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pmul<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pdiv<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pmadd(const PacketMul4Xi& a, const PacketMul4Xi& b, const PacketMul4Xi& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pmsub(const PacketMul4Xi& a, const PacketMul4Xi& b, const PacketMul4Xi& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pnmadd(const PacketMul4Xi& a, const PacketMul4Xi& b, const PacketMul4Xi& c) {
  return __riscv_vnmsub_vv_i32m4(a, b, c, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pnmsub(const PacketMul4Xi& a, const PacketMul4Xi& b, const PacketMul4Xi& c) {
  return __riscv_vnmsub_vv_i32m4(a, b, pnegate(c), unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pmin<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pmax<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pcmp_le<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  PacketMask8 mask = __riscv_vmsle_vv_i32m4_b8(a, b, unpacket_traits<PacketMul4Xi>::size);
  return __riscv_vmerge_vxm_i32m4(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pcmp_lt<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  PacketMask8 mask = __riscv_vmslt_vv_i32m4_b8(a, b, unpacket_traits<PacketMul4Xi>::size);
  return __riscv_vmerge_vxm_i32m4(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pcmp_eq<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  PacketMask8 mask = __riscv_vmseq_vv_i32m4_b8(a, b, unpacket_traits<PacketMul4Xi>::size);
  return __riscv_vmerge_vxm_i32m4(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi ptrue<PacketMul4Xi>(const PacketMul4Xi& /*a*/) {
  return __riscv_vmv_v_x_i32m4(0xffffffffu, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pand<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vand_vv_i32m4(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi por<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vor_vv_i32m4(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pxor<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vxor_vv_i32m4(a, b, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pandnot<PacketMul4Xi>(const PacketMul4Xi& a, const PacketMul4Xi& b) {
  return __riscv_vand_vv_i32m4(a, __riscv_vnot_v_i32m4(b, unpacket_traits<PacketMul4Xi>::size),
                               unpacket_traits<PacketMul4Xi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xi parithmetic_shift_right(PacketMul4Xi a) {
  return __riscv_vsra_vx_i32m4(a, N, unpacket_traits<PacketMul4Xi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xi plogical_shift_right(PacketMul4Xi a) {
  return __riscv_vreinterpret_i32m4(
      __riscv_vsrl_vx_u32m4(__riscv_vreinterpret_u32m4(a), N, unpacket_traits<PacketMul4Xi>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xi plogical_shift_left(PacketMul4Xi a) {
  return __riscv_vsll_vx_i32m4(a, N, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pload<PacketMul4Xi>(const numext::int32_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_i32m4(from, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi ploadu<PacketMul4Xi>(const numext::int32_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_i32m4(from, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi ploaddup<PacketMul4Xi>(const numext::int32_t* from) {
  PacketMul4Xu idx = __riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xi>::size);
  idx = __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(idx, 0xfffffffeu, unpacket_traits<PacketMul4Xi>::size), 1,
                              unpacket_traits<PacketMul4Xi>::size);
  // idx = 0 0 sizeof(int32_t) sizeof(int32_t) 2*sizeof(int32_t) 2*sizeof(int32_t) ...
  return __riscv_vloxei32_v_i32m4(from, idx, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi ploadquad<PacketMul4Xi>(const numext::int32_t* from) {
  PacketMul4Xu idx = __riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xi>::size);
  idx = __riscv_vand_vx_u32m4(idx, 0xfffffffcu, unpacket_traits<PacketMul4Xi>::size);
  return __riscv_vloxei32_v_i32m4(from, idx, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int32_t>(numext::int32_t* to, const PacketMul4Xi& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_i32m4(to, from, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int32_t>(numext::int32_t* to, const PacketMul4Xi& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_i32m4(to, from, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul4Xi pgather<numext::int32_t, PacketMul4Xi>(const numext::int32_t* from,
                                                                             Index stride) {
  return __riscv_vlse32_v_i32m4(from, stride * sizeof(numext::int32_t), unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int32_t, PacketMul4Xi>(numext::int32_t* to, const PacketMul4Xi& from,
                                                                      Index stride) {
  __riscv_vsse32(to, stride * sizeof(numext::int32_t), from, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t pfirst<PacketMul4Xi>(const PacketMul4Xi& a) {
  return __riscv_vmv_x_s_i32m4_i32(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi preverse(const PacketMul4Xi& a) {
  PacketMul4Xu idx =
      __riscv_vrsub_vx_u32m4(__riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xi>::size),
                             unpacket_traits<PacketMul4Xi>::size - 1, unpacket_traits<PacketMul4Xi>::size);
  return __riscv_vrgather_vv_i32m4(a, idx, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pabs(const PacketMul4Xi& a) {
  PacketMul4Xi mask = __riscv_vsra_vx_i32m4(a, 31, unpacket_traits<PacketMul4Xi>::size);
  return __riscv_vsub_vv_i32m4(__riscv_vxor_vv_i32m4(a, mask, unpacket_traits<PacketMul4Xi>::size), mask,
                               unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux<PacketMul4Xi>(const PacketMul4Xi& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i32m4_i32m1(
      a, __riscv_vmv_v_x_i32m1(0, unpacket_traits<PacketMul4Xi>::size / 4), unpacket_traits<PacketMul4Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_mul<PacketMul4Xi>(const PacketMul4Xi& a) {
  PacketMul1Xi half1 = __riscv_vmul_vv_i32m1(__riscv_vget_v_i32m4_i32m1(a, 0), __riscv_vget_v_i32m4_i32m1(a, 1),
                                         unpacket_traits<PacketMul1Xi>::size);
  PacketMul1Xi half2 = __riscv_vmul_vv_i32m1(__riscv_vget_v_i32m4_i32m1(a, 2), __riscv_vget_v_i32m4_i32m1(a, 3),
                                         unpacket_traits<PacketMul1Xi>::size);
  return predux_mul<PacketMul1Xi>(__riscv_vmul_vv_i32m1(half1, half2, unpacket_traits<PacketMul1Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_min<PacketMul4Xi>(const PacketMul4Xi& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i32m4_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::max)(), unpacket_traits<PacketMul4Xi>::size / 4),
      unpacket_traits<PacketMul4Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_max<PacketMul4Xi>(const PacketMul4Xi& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i32m4_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::min)(), unpacket_traits<PacketMul4Xi>::size / 4),
      unpacket_traits<PacketMul4Xi>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul4Xi, N>& kernel) {
  numext::int32_t buffer[unpacket_traits<PacketMul4Xi>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(numext::int32_t), kernel.packet[i], unpacket_traits<PacketMul4Xi>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_i32m4(&buffer[i * unpacket_traits<PacketMul4Xi>::size], unpacket_traits<PacketMul4Xi>::size);
  }
}

/********************************* PacketMul4Xf ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul4Xf ptrue<PacketMul4Xf>(const PacketMul4Xf& /*a*/) {
  return __riscv_vreinterpret_f32m4(__riscv_vmv_v_x_u32m4(0xffffffffu, unpacket_traits<PacketMul4Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pzero<PacketMul4Xf>(const PacketMul4Xf& /*a*/) {
  return __riscv_vfmv_v_f_f32m4(0.0f, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pabs(const PacketMul4Xf& a) {
  return __riscv_vfabs_v_f32m4(a, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pset1<PacketMul4Xf>(const float& from) {
  return __riscv_vfmv_v_f_f32m4(from, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pset1frombits<PacketMul4Xf>(numext::uint32_t from) {
  return __riscv_vreinterpret_f32m4(__riscv_vmv_v_x_u32m4(from, unpacket_traits<PacketMul4Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf plset<PacketMul4Xf>(const float& a) {
  PacketMul4Xf idx = __riscv_vfcvt_f_x_v_f32m4(
      __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xi>::size)),
      unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vfadd_vf_f32m4(idx, a, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf padd<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vfadd_vv_f32m4(a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf psub<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vfsub_vv_f32m4(a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pnegate(const PacketMul4Xf& a) {
  return __riscv_vfneg_v_f32m4(a, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pconj(const PacketMul4Xf& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmul<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vfmul_vv_f32m4(a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pdiv<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vfdiv_vv_f32m4(a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmadd(const PacketMul4Xf& a, const PacketMul4Xf& b, const PacketMul4Xf& c) {
  return __riscv_vfmadd_vv_f32m4(a, b, c, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmsub(const PacketMul4Xf& a, const PacketMul4Xf& b, const PacketMul4Xf& c) {
  return __riscv_vfmsub_vv_f32m4(a, b, c, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pnmadd(const PacketMul4Xf& a, const PacketMul4Xf& b, const PacketMul4Xf& c) {
  return __riscv_vfnmsub_vv_f32m4(a, b, c, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pnmsub(const PacketMul4Xf& a, const PacketMul4Xf& b, const PacketMul4Xf& c) {
  return __riscv_vfnmadd_vv_f32m4(a, b, c, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmin<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  PacketMul4Xf nans =
      __riscv_vfmv_v_f_f32m4((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul4Xf>::size);
  PacketMask8 mask = __riscv_vmfeq_vv_f32m4_b8(a, a, unpacket_traits<PacketMul4Xf>::size);
  PacketMask8 mask2 = __riscv_vmfeq_vv_f32m4_b8(b, b, unpacket_traits<PacketMul4Xf>::size);
  mask = __riscv_vmand_mm_b8(mask, mask2, unpacket_traits<PacketMul4Xf>::size);

  return __riscv_vfmin_vv_f32m4_tumu(mask, nans, a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmin<PropagateNaN, PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return pmin<PacketMul4Xf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmin<PropagateNumbers, PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vfmin_vv_f32m4(a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmax<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  PacketMul4Xf nans =
      __riscv_vfmv_v_f_f32m4((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul4Xf>::size);
  PacketMask8 mask = __riscv_vmfeq_vv_f32m4_b8(a, a, unpacket_traits<PacketMul4Xf>::size);
  PacketMask8 mask2 = __riscv_vmfeq_vv_f32m4_b8(b, b, unpacket_traits<PacketMul4Xf>::size);
  mask = __riscv_vmand_mm_b8(mask, mask2, unpacket_traits<PacketMul4Xf>::size);

  return __riscv_vfmax_vv_f32m4_tumu(mask, nans, a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmax<PropagateNaN, PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return pmax<PacketMul4Xf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pmax<PropagateNumbers, PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vfmax_vv_f32m4(a, b, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcmp_le<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  PacketMask8 mask = __riscv_vmfle_vv_f32m4_b8(a, b, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vmerge_vvm_f32m4(pzero<PacketMul4Xf>(a), ptrue<PacketMul4Xf>(a), mask,
                                  unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcmp_lt<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  PacketMask8 mask = __riscv_vmflt_vv_f32m4_b8(a, b, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vmerge_vvm_f32m4(pzero<PacketMul4Xf>(a), ptrue<PacketMul4Xf>(a), mask,
                                  unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcmp_eq<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  PacketMask8 mask = __riscv_vmfeq_vv_f32m4_b8(a, b, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vmerge_vvm_f32m4(pzero<PacketMul4Xf>(a), ptrue<PacketMul4Xf>(a), mask,
                                  unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcmp_lt_or_nan<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  PacketMask8 mask = __riscv_vmfge_vv_f32m4_b8(a, b, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vfmerge_vfm_f32m4(ptrue<PacketMul4Xf>(a), 0.0f, mask, unpacket_traits<PacketMul4Xf>::size);
}

// Logical Operations are not supported for float, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketMul4Xf pand<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(a),
                                                                  __riscv_vreinterpret_v_f32m4_u32m4(b),
                                                                  unpacket_traits<PacketMul4Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf por<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(a),
                                                                 __riscv_vreinterpret_v_f32m4_u32m4(b),
                                                                 unpacket_traits<PacketMul4Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pxor<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vxor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(a),
                                                                  __riscv_vreinterpret_v_f32m4_u32m4(b),
                                                                  unpacket_traits<PacketMul4Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pandnot<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& b) {
  return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vv_u32m4(
      __riscv_vreinterpret_v_f32m4_u32m4(a),
      __riscv_vnot_v_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(b), unpacket_traits<PacketMul4Xf>::size),
      unpacket_traits<PacketMul4Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pload<PacketMul4Xf>(const float* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_f32m4(from, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf ploadu<PacketMul4Xf>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_f32m4(from, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf ploaddup<PacketMul4Xf>(const float* from) {
  PacketMul4Xu idx = __riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xf>::size);
  idx = __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(idx, 0xfffffffeu, unpacket_traits<PacketMul4Xf>::size), 1,
                              unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vloxei32_v_f32m4(from, idx, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf ploadquad<PacketMul4Xf>(const float* from) {
  PacketMul4Xu idx = __riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xf>::size);
  idx = __riscv_vand_vx_u32m4(idx, 0xfffffffcu, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vloxei32_v_f32m4(from, idx, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const PacketMul4Xf& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_f32m4(to, from, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const PacketMul4Xf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_f32m4(to, from, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul4Xf pgather<float, PacketMul4Xf>(const float* from, Index stride) {
  return __riscv_vlse32_v_f32m4(from, stride * sizeof(float), unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, PacketMul4Xf>(float* to, const PacketMul4Xf& from, Index stride) {
  __riscv_vsse32(to, stride * sizeof(float), from, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE float pfirst<PacketMul4Xf>(const PacketMul4Xf& a) {
  return __riscv_vfmv_f_s_f32m4_f32(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf psqrt(const PacketMul4Xf& a) {
  return __riscv_vfsqrt_v_f32m4(a, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf print<PacketMul4Xf>(const PacketMul4Xf& a) {
  const PacketMul4Xf limit = pset1<PacketMul4Xf>(static_cast<float>(1 << 23));
  const PacketMul4Xf abs_a = pabs(a);

  PacketMask8 mask = __riscv_vmfne_vv_f32m4_b8(a, a, unpacket_traits<PacketMul4Xf>::size);
  const PacketMul4Xf x = __riscv_vfadd_vv_f32m4_tumu(mask, a, a, a, unpacket_traits<PacketMul4Xf>::size);
  const PacketMul4Xf new_x = __riscv_vfcvt_f_x_v_f32m4(
      __riscv_vfcvt_x_f_v_i32m4(a, unpacket_traits<PacketMul4Xf>::size), unpacket_traits<PacketMul4Xf>::size);

  mask = __riscv_vmflt_vv_f32m4_b8(abs_a, limit, unpacket_traits<PacketMul4Xf>::size);
  PacketMul4Xf signed_x = __riscv_vfsgnj_vv_f32m4(new_x, x, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vmerge_vvm_f32m4(x, signed_x, mask, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pfloor<PacketMul4Xf>(const PacketMul4Xf& a) {
  PacketMul4Xf tmp = print<PacketMul4Xf>(a);
  // If greater, subtract one.
  PacketMask8 mask = __riscv_vmflt_vv_f32m4_b8(a, tmp, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vfsub_vf_f32m4_tumu(mask, tmp, tmp, 1.0f, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf preverse(const PacketMul4Xf& a) {
  PacketMul4Xu idx =
      __riscv_vrsub_vx_u32m4(__riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xf>::size),
                             unpacket_traits<PacketMul4Xf>::size - 1, unpacket_traits<PacketMul4Xf>::size);
  return __riscv_vrgather_vv_f32m4(a, idx, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pfrexp<PacketMul4Xf>(const PacketMul4Xf& a, PacketMul4Xf& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE float predux<PacketMul4Xf>(const PacketMul4Xf& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f32m4_f32m1(
      a, __riscv_vfmv_v_f_f32m1(0.0, unpacket_traits<PacketMul4Xf>::size / 4), unpacket_traits<PacketMul4Xf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_mul<PacketMul4Xf>(const PacketMul4Xf& a) {
  PacketMul1Xf half1 = __riscv_vfmul_vv_f32m1(__riscv_vget_v_f32m4_f32m1(a, 0), __riscv_vget_v_f32m4_f32m1(a, 1),
                                          unpacket_traits<PacketMul1Xf>::size);
  PacketMul1Xf half2 = __riscv_vfmul_vv_f32m1(__riscv_vget_v_f32m4_f32m1(a, 2), __riscv_vget_v_f32m4_f32m1(a, 3),
                                          unpacket_traits<PacketMul1Xf>::size);
  return predux_mul<PacketMul1Xf>(__riscv_vfmul_vv_f32m1(half1, half2, unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketMul4Xf>(const PacketMul4Xf& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f32m4_f32m1(
                        a,
                        __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(),
                                               unpacket_traits<PacketMul4Xf>::size / 4),
                        unpacket_traits<PacketMul4Xf>::size)),
                    (std::numeric_limits<float>::max)());
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketMul4Xf>(const PacketMul4Xf& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f32m4_f32m1(
                        a,
                        __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(),
                                               unpacket_traits<PacketMul4Xf>::size / 4),
                        unpacket_traits<PacketMul4Xf>::size)),
                    -(std::numeric_limits<float>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul4Xf, N>& kernel) {
  float buffer[unpacket_traits<PacketMul4Xf>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(float), kernel.packet[i], unpacket_traits<PacketMul4Xf>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_f32m4(&buffer[i * unpacket_traits<PacketMul4Xf>::size], unpacket_traits<PacketMul4Xf>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pldexp<PacketMul4Xf>(const PacketMul4Xf& a, const PacketMul4Xf& exponent) {
  return pldexp_generic(a, exponent);
}

/********************************* PacketMul4Xl ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pset1<PacketMul4Xl>(const numext::int64_t& from) {
  return __riscv_vmv_v_x_i64m4(from, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl plset<PacketMul4Xl>(const numext::int64_t& a) {
  PacketMul4Xl idx = __riscv_vreinterpret_v_u64m4_i64m4(__riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xl>::size));
  return __riscv_vadd_vx_i64m4(idx, a, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pzero<PacketMul4Xl>(const PacketMul4Xl& /*a*/) {
  return __riscv_vmv_v_x_i64m4(0, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl padd<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vadd_vv_i64m4(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl psub<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pnegate(const PacketMul4Xl& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pconj(const PacketMul4Xl& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pmul<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pdiv<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pmadd(const PacketMul4Xl& a, const PacketMul4Xl& b, const PacketMul4Xl& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pmsub(const PacketMul4Xl& a, const PacketMul4Xl& b, const PacketMul4Xl& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pnmadd(const PacketMul4Xl& a, const PacketMul4Xl& b, const PacketMul4Xl& c) {
  return __riscv_vnmsub_vv_i64m4(a, b, c, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pnmsub(const PacketMul4Xl& a, const PacketMul4Xl& b, const PacketMul4Xl& c) {
  return __riscv_vnmsub_vv_i64m4(a, b, pnegate(c), unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pmin<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pmax<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pcmp_le<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  PacketMask16 mask = __riscv_vmsle_vv_i64m4_b16(a, b, unpacket_traits<PacketMul4Xl>::size);
  return __riscv_vmerge_vxm_i64m4(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pcmp_lt<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  PacketMask16 mask = __riscv_vmslt_vv_i64m4_b16(a, b, unpacket_traits<PacketMul4Xl>::size);
  return __riscv_vmerge_vxm_i64m4(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pcmp_eq<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  PacketMask16 mask = __riscv_vmseq_vv_i64m4_b16(a, b, unpacket_traits<PacketMul4Xl>::size);
  return __riscv_vmerge_vxm_i64m4(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl ptrue<PacketMul4Xl>(const PacketMul4Xl& /*a*/) {
  return __riscv_vmv_v_x_i64m4(0xffffffffffffffffu, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pand<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vand_vv_i64m4(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl por<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vor_vv_i64m4(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pxor<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vxor_vv_i64m4(a, b, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pandnot<PacketMul4Xl>(const PacketMul4Xl& a, const PacketMul4Xl& b) {
  return __riscv_vand_vv_i64m4(a, __riscv_vnot_v_i64m4(b, unpacket_traits<PacketMul4Xl>::size),
                               unpacket_traits<PacketMul4Xl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xl parithmetic_shift_right(PacketMul4Xl a) {
  return __riscv_vsra_vx_i64m4(a, N, unpacket_traits<PacketMul4Xl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xl plogical_shift_right(PacketMul4Xl a) {
  return __riscv_vreinterpret_i64m4(
      __riscv_vsrl_vx_u64m4(__riscv_vreinterpret_u64m4(a), N, unpacket_traits<PacketMul4Xl>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xl plogical_shift_left(PacketMul4Xl a) {
  return __riscv_vsll_vx_i64m4(a, N, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pload<PacketMul4Xl>(const numext::int64_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_i64m4(from, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl ploadu<PacketMul4Xl>(const numext::int64_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_i64m4(from, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl ploaddup<PacketMul4Xl>(const numext::int64_t* from) {
  PacketMul4Xul idx = __riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xl>::size);
  idx = __riscv_vsll_vx_u64m4(__riscv_vand_vx_u64m4(idx, 0xfffffffffffffffeu, unpacket_traits<PacketMul4Xl>::size), 2,
                              unpacket_traits<PacketMul4Xl>::size);
  // idx = 0 0 sizeof(int64_t) sizeof(int64_t) 2*sizeof(int64_t) 2*sizeof(int64_t) ...
  return __riscv_vloxei64_v_i64m4(from, idx, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl ploadquad<PacketMul4Xl>(const numext::int64_t* from) {
  PacketMul4Xul idx = __riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xl>::size);
  idx = __riscv_vsll_vx_u64m4(__riscv_vand_vx_u64m4(idx, 0xfffffffffffffffcu, unpacket_traits<PacketMul4Xl>::size), 1,
                              unpacket_traits<PacketMul4Xl>::size);
  return __riscv_vloxei64_v_i64m4(from, idx, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int64_t>(numext::int64_t* to, const PacketMul4Xl& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_i64m4(to, from, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int64_t>(numext::int64_t* to, const PacketMul4Xl& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_i64m4(to, from, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul4Xl pgather<numext::int64_t, PacketMul4Xl>(const numext::int64_t* from,
                                                                             Index stride) {
  return __riscv_vlse64_v_i64m4(from, stride * sizeof(numext::int64_t), unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int64_t, PacketMul4Xl>(numext::int64_t* to, const PacketMul4Xl& from,
                                                                      Index stride) {
  __riscv_vsse64(to, stride * sizeof(numext::int64_t), from, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t pfirst<PacketMul4Xl>(const PacketMul4Xl& a) {
  return __riscv_vmv_x_s_i64m4_i64(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl preverse(const PacketMul4Xl& a) {
  PacketMul4Xul idx =
      __riscv_vrsub_vx_u64m4(__riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xl>::size),
                             unpacket_traits<PacketMul4Xl>::size - 1, unpacket_traits<PacketMul4Xl>::size);
  return __riscv_vrgather_vv_i64m4(a, idx, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pabs(const PacketMul4Xl& a) {
  PacketMul4Xl mask = __riscv_vsra_vx_i64m4(a, 63, unpacket_traits<PacketMul4Xl>::size);
  return __riscv_vsub_vv_i64m4(__riscv_vxor_vv_i64m4(a, mask, unpacket_traits<PacketMul4Xl>::size), mask,
                               unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux<PacketMul4Xl>(const PacketMul4Xl& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i64m4_i64m1(
      a, __riscv_vmv_v_x_i64m1(0, unpacket_traits<PacketMul4Xl>::size / 4), unpacket_traits<PacketMul4Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_mul<PacketMul4Xl>(const PacketMul4Xl& a) {
  PacketMul1Xl half1 = __riscv_vmul_vv_i64m1(__riscv_vget_v_i64m4_i64m1(a, 0), __riscv_vget_v_i64m4_i64m1(a, 1),
                                         unpacket_traits<PacketMul1Xl>::size);
  PacketMul1Xl half2 = __riscv_vmul_vv_i64m1(__riscv_vget_v_i64m4_i64m1(a, 2), __riscv_vget_v_i64m4_i64m1(a, 3),
                                         unpacket_traits<PacketMul1Xl>::size);
  return predux_mul<PacketMul1Xl>(__riscv_vmul_vv_i64m1(half1, half2, unpacket_traits<PacketMul1Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_min<PacketMul4Xl>(const PacketMul4Xl& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i64m4_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::max)(), unpacket_traits<PacketMul4Xl>::size / 4),
      unpacket_traits<PacketMul4Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_max<PacketMul4Xl>(const PacketMul4Xl& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i64m4_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::min)(), unpacket_traits<PacketMul4Xl>::size / 4),
      unpacket_traits<PacketMul4Xl>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul4Xl, N>& kernel) {
  numext::int64_t buffer[unpacket_traits<PacketMul4Xl>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(numext::int64_t), kernel.packet[i], unpacket_traits<PacketMul4Xl>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_i64m4(&buffer[i * unpacket_traits<PacketMul4Xl>::size], unpacket_traits<PacketMul4Xl>::size);
  }
}

/********************************* PacketMul4Xd ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul4Xd ptrue<PacketMul4Xd>(const PacketMul4Xd& /*a*/) {
  return __riscv_vreinterpret_f64m4(__riscv_vmv_v_x_u64m4(0xffffffffffffffffu, unpacket_traits<PacketMul4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pzero<PacketMul4Xd>(const PacketMul4Xd& /*a*/) {
  return __riscv_vfmv_v_f_f64m4(0.0, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pabs(const PacketMul4Xd& a) {
  return __riscv_vfabs_v_f64m4(a, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pset1<PacketMul4Xd>(const double& from) {
  return __riscv_vfmv_v_f_f64m4(from, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pset1frombits<PacketMul4Xd>(numext::uint64_t from) {
  return __riscv_vreinterpret_f64m4(__riscv_vmv_v_x_u64m4(from, unpacket_traits<PacketMul4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd plset<PacketMul4Xd>(const double& a) {
  PacketMul4Xd idx = __riscv_vfcvt_f_x_v_f64m4(
      __riscv_vreinterpret_v_u64m4_i64m4(__riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xi>::size)),
      unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vfadd_vf_f64m4(idx, a, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd padd<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vfadd_vv_f64m4(a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd psub<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vfsub_vv_f64m4(a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pnegate(const PacketMul4Xd& a) {
  return __riscv_vfneg_v_f64m4(a, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pconj(const PacketMul4Xd& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmul<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vfmul_vv_f64m4(a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pdiv<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vfdiv_vv_f64m4(a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmadd(const PacketMul4Xd& a, const PacketMul4Xd& b, const PacketMul4Xd& c) {
  return __riscv_vfmadd_vv_f64m4(a, b, c, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmsub(const PacketMul4Xd& a, const PacketMul4Xd& b, const PacketMul4Xd& c) {
  return __riscv_vfmsub_vv_f64m4(a, b, c, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pnmadd(const PacketMul4Xd& a, const PacketMul4Xd& b, const PacketMul4Xd& c) {
  return __riscv_vfnmsub_vv_f64m4(a, b, c, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pnmsub(const PacketMul4Xd& a, const PacketMul4Xd& b, const PacketMul4Xd& c) {
  return __riscv_vfnmadd_vv_f64m4(a, b, c, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmin<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  PacketMul4Xd nans =
      __riscv_vfmv_v_f_f64m4((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul4Xd>::size);
  PacketMask16 mask = __riscv_vmfeq_vv_f64m4_b16(a, a, unpacket_traits<PacketMul4Xd>::size);
  PacketMask16 mask2 = __riscv_vmfeq_vv_f64m4_b16(b, b, unpacket_traits<PacketMul4Xd>::size);
  mask = __riscv_vmand_mm_b16(mask, mask2, unpacket_traits<PacketMul4Xd>::size);

  return __riscv_vfmin_vv_f64m4_tumu(mask, nans, a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmin<PropagateNaN, PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return pmin<PacketMul4Xd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmin<PropagateNumbers, PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vfmin_vv_f64m4(a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmax<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  PacketMul4Xd nans =
      __riscv_vfmv_v_f_f64m4((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul4Xd>::size);
  PacketMask16 mask = __riscv_vmfeq_vv_f64m4_b16(a, a, unpacket_traits<PacketMul4Xd>::size);
  PacketMask16 mask2 = __riscv_vmfeq_vv_f64m4_b16(b, b, unpacket_traits<PacketMul4Xd>::size);
  mask = __riscv_vmand_mm_b16(mask, mask2, unpacket_traits<PacketMul4Xd>::size);

  return __riscv_vfmax_vv_f64m4_tumu(mask, nans, a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmax<PropagateNaN, PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return pmax<PacketMul4Xd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pmax<PropagateNumbers, PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vfmax_vv_f64m4(a, b, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcmp_le<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  PacketMask16 mask = __riscv_vmfle_vv_f64m4_b16(a, b, unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vmerge_vvm_f64m4(pzero<PacketMul4Xd>(a), ptrue<PacketMul4Xd>(a), mask,
                                  unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcmp_lt<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  PacketMask16 mask = __riscv_vmflt_vv_f64m4_b16(a, b, unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vmerge_vvm_f64m4(pzero<PacketMul4Xd>(a), ptrue<PacketMul4Xd>(a), mask,
                                  unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcmp_eq<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  PacketMask16 mask = __riscv_vmfeq_vv_f64m4_b16(a, b, unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vmerge_vvm_f64m4(pzero<PacketMul4Xd>(a), ptrue<PacketMul4Xd>(a), mask,
                                  unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcmp_lt_or_nan<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  PacketMask16 mask = __riscv_vmfge_vv_f64m4_b16(a, b, unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vfmerge_vfm_f64m4(ptrue<PacketMul4Xd>(a), 0.0, mask, unpacket_traits<PacketMul4Xd>::size);
}

// Logical Operations are not supported for double, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketMul4Xd pand<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(a),
                                                                  __riscv_vreinterpret_v_f64m4_u64m4(b),
                                                                  unpacket_traits<PacketMul4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd por<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(a),
                                                                 __riscv_vreinterpret_v_f64m4_u64m4(b),
                                                                 unpacket_traits<PacketMul4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pxor<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vxor_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(a),
                                                                  __riscv_vreinterpret_v_f64m4_u64m4(b),
                                                                  unpacket_traits<PacketMul4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pandnot<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& b) {
  return __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vv_u64m4(
      __riscv_vreinterpret_v_f64m4_u64m4(a),
      __riscv_vnot_v_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(b), unpacket_traits<PacketMul4Xd>::size),
      unpacket_traits<PacketMul4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pload<PacketMul4Xd>(const double* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_f64m4(from, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd ploadu<PacketMul4Xd>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_f64m4(from, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd ploaddup<PacketMul4Xd>(const double* from) {
  PacketMul4Xul idx = __riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xd>::size);
  idx = __riscv_vsll_vx_u64m4(__riscv_vand_vx_u64m4(idx, 0xfffffffffffffffeu, unpacket_traits<PacketMul4Xd>::size), 2,
                              unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vloxei64_v_f64m4(from, idx, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd ploadquad<PacketMul4Xd>(const double* from) {
  PacketMul4Xul idx = __riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xd>::size);
  idx = __riscv_vsll_vx_u64m4(__riscv_vand_vx_u64m4(idx, 0xfffffffffffffffcu, unpacket_traits<PacketMul4Xd>::size), 1,
                              unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vloxei64_v_f64m4(from, idx, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<double>(double* to, const PacketMul4Xd& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_f64m4(to, from, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const PacketMul4Xd& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_f64m4(to, from, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul4Xd pgather<double, PacketMul4Xd>(const double* from, Index stride) {
  return __riscv_vlse64_v_f64m4(from, stride * sizeof(double), unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<double, PacketMul4Xd>(double* to, const PacketMul4Xd& from, Index stride) {
  __riscv_vsse64(to, stride * sizeof(double), from, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE double pfirst<PacketMul4Xd>(const PacketMul4Xd& a) {
  return __riscv_vfmv_f_s_f64m4_f64(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd psqrt(const PacketMul4Xd& a) {
  return __riscv_vfsqrt_v_f64m4(a, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd print<PacketMul4Xd>(const PacketMul4Xd& a) {
  const PacketMul4Xd limit = pset1<PacketMul4Xd>(static_cast<double>(1ull << 52));
  const PacketMul4Xd abs_a = pabs(a);

  PacketMask16 mask = __riscv_vmfne_vv_f64m4_b16(a, a, unpacket_traits<PacketMul4Xd>::size);
  const PacketMul4Xd x = __riscv_vfadd_vv_f64m4_tumu(mask, a, a, a, unpacket_traits<PacketMul4Xd>::size);
  const PacketMul4Xd new_x = __riscv_vfcvt_f_x_v_f64m4(
      __riscv_vfcvt_x_f_v_i64m4(a, unpacket_traits<PacketMul4Xd>::size), unpacket_traits<PacketMul4Xd>::size);

  mask = __riscv_vmflt_vv_f64m4_b16(abs_a, limit, unpacket_traits<PacketMul4Xd>::size);
  PacketMul4Xd signed_x = __riscv_vfsgnj_vv_f64m4(new_x, x, unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vmerge_vvm_f64m4(x, signed_x, mask, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pfloor<PacketMul4Xd>(const PacketMul4Xd& a) {
  PacketMul4Xd tmp = print<PacketMul4Xd>(a);
  // If greater, subtract one.
  PacketMask16 mask = __riscv_vmflt_vv_f64m4_b16(a, tmp, unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vfsub_vf_f64m4_tumu(mask, tmp, tmp, 1.0, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd preverse(const PacketMul4Xd& a) {
  PacketMul4Xul idx =
      __riscv_vrsub_vx_u64m4(__riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xd>::size),
                             unpacket_traits<PacketMul4Xd>::size - 1, unpacket_traits<PacketMul4Xd>::size);
  return __riscv_vrgather_vv_f64m4(a, idx, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pfrexp<PacketMul4Xd>(const PacketMul4Xd& a, PacketMul4Xd& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE double predux<PacketMul4Xd>(const PacketMul4Xd& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f64m4_f64m1(
      a, __riscv_vfmv_v_f_f64m1(0.0, unpacket_traits<PacketMul4Xd>::size / 4), unpacket_traits<PacketMul4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_mul<PacketMul4Xd>(const PacketMul4Xd& a) {
  PacketMul1Xd half1 = __riscv_vfmul_vv_f64m1(__riscv_vget_v_f64m4_f64m1(a, 0), __riscv_vget_v_f64m4_f64m1(a, 1),
                                          unpacket_traits<PacketMul1Xd>::size);
  PacketMul1Xd half2 = __riscv_vfmul_vv_f64m1(__riscv_vget_v_f64m4_f64m1(a, 2), __riscv_vget_v_f64m4_f64m1(a, 3),
                                          unpacket_traits<PacketMul1Xd>::size);
  return predux_mul<PacketMul1Xd>(__riscv_vfmul_vv_f64m1(half1, half2, unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_min<PacketMul4Xd>(const PacketMul4Xd& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f64m4_f64m1(
                        a,
                        __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(),
                                               unpacket_traits<PacketMul4Xd>::size / 4),
                        unpacket_traits<PacketMul4Xd>::size)),
                    (std::numeric_limits<double>::max)());
}

template <>
EIGEN_STRONG_INLINE double predux_max<PacketMul4Xd>(const PacketMul4Xd& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f64m4_f64m1(
                        a,
                        __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(),
                                               unpacket_traits<PacketMul4Xd>::size / 4),
                        unpacket_traits<PacketMul4Xd>::size)),
                    -(std::numeric_limits<double>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul4Xd, N>& kernel) {
  double buffer[unpacket_traits<PacketMul4Xd>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(double), kernel.packet[i], unpacket_traits<PacketMul4Xd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_f64m4(&buffer[i * unpacket_traits<PacketMul4Xd>::size], unpacket_traits<PacketMul4Xd>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pldexp<PacketMul4Xd>(const PacketMul4Xd& a, const PacketMul4Xd& exponent) {
  return pldexp_generic(a, exponent);
}

/********************************* PacketMul4Xs ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pset1<PacketMul4Xs>(const numext::int16_t& from) {
  return __riscv_vmv_v_x_i16m4(from, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs plset<PacketMul4Xs>(const numext::int16_t& a) {
  PacketMul4Xs idx = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vid_v_u16m4(unpacket_traits<PacketMul4Xs>::size));
  return __riscv_vadd_vx_i16m4(idx, a, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pzero<PacketMul4Xs>(const PacketMul4Xs& /*a*/) {
  return __riscv_vmv_v_x_i16m4(0, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs padd<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vadd_vv_i16m4(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs psub<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pnegate(const PacketMul4Xs& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pconj(const PacketMul4Xs& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pmul<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pdiv<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pmadd(const PacketMul4Xs& a, const PacketMul4Xs& b, const PacketMul4Xs& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pmsub(const PacketMul4Xs& a, const PacketMul4Xs& b, const PacketMul4Xs& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pnmadd(const PacketMul4Xs& a, const PacketMul4Xs& b, const PacketMul4Xs& c) {
  return __riscv_vnmsub_vv_i16m4(a, b, c, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pnmsub(const PacketMul4Xs& a, const PacketMul4Xs& b, const PacketMul4Xs& c) {
  return __riscv_vnmsub_vv_i16m4(a, b, pnegate(c), unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pmin<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pmax<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pcmp_le<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  PacketMask4 mask = __riscv_vmsle_vv_i16m4_b4(a, b, unpacket_traits<PacketMul4Xs>::size);
  return __riscv_vmerge_vxm_i16m4(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pcmp_lt<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  PacketMask4 mask = __riscv_vmslt_vv_i16m4_b4(a, b, unpacket_traits<PacketMul4Xs>::size);
  return __riscv_vmerge_vxm_i16m4(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pcmp_eq<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  PacketMask4 mask = __riscv_vmseq_vv_i16m4_b4(a, b, unpacket_traits<PacketMul4Xs>::size);
  return __riscv_vmerge_vxm_i16m4(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs ptrue<PacketMul4Xs>(const PacketMul4Xs& /*a*/) {
  return __riscv_vmv_v_x_i16m4(static_cast<unsigned short>(0xffffu), unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pand<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vand_vv_i16m4(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs por<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vor_vv_i16m4(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pxor<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vxor_vv_i16m4(a, b, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pandnot<PacketMul4Xs>(const PacketMul4Xs& a, const PacketMul4Xs& b) {
  return __riscv_vand_vv_i16m4(a, __riscv_vnot_v_i16m4(b, unpacket_traits<PacketMul4Xs>::size),
                               unpacket_traits<PacketMul4Xs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xs parithmetic_shift_right(PacketMul4Xs a) {
  return __riscv_vsra_vx_i16m4(a, N, unpacket_traits<PacketMul4Xs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xs plogical_shift_right(PacketMul4Xs a) {
  return __riscv_vreinterpret_i16m4(
      __riscv_vsrl_vx_u16m4(__riscv_vreinterpret_u16m4(a), N, unpacket_traits<PacketMul4Xs>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul4Xs plogical_shift_left(PacketMul4Xs a) {
  return __riscv_vsll_vx_i16m4(a, N, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pload<PacketMul4Xs>(const numext::int16_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle16_v_i16m4(from, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs ploadu<PacketMul4Xs>(const numext::int16_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle16_v_i16m4(from, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs ploaddup<PacketMul4Xs>(const numext::int16_t* from) {
  PacketMul4Xsu idx = __riscv_vid_v_u16m4(unpacket_traits<PacketMul4Xs>::size);
  idx = __riscv_vand_vx_u16m4(idx, 0xfffeu, unpacket_traits<PacketMul4Xs>::size);
  // idx = 0 0 sizeof(int16_t) sizeof(int16_t) 2*sizeof(int16_t) 2*sizeof(int16_t) ...
  return __riscv_vloxei16_v_i16m4(from, idx, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs ploadquad<PacketMul4Xs>(const numext::int16_t* from) {
  PacketMul4Xsu idx = __riscv_vid_v_u16m4(unpacket_traits<PacketMul4Xs>::size);
  idx = __riscv_vsrl_vx_u16m4(__riscv_vand_vx_u16m4(idx, 0xfffcu, unpacket_traits<PacketMul4Xs>::size), 1,
                              unpacket_traits<PacketMul4Xs>::size);
  return __riscv_vloxei16_v_i16m4(from, idx, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int16_t>(numext::int16_t* to, const PacketMul4Xs& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse16_v_i16m4(to, from, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int16_t>(numext::int16_t* to, const PacketMul4Xs& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse16_v_i16m4(to, from, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul4Xs pgather<numext::int16_t, PacketMul4Xs>(const numext::int16_t* from,
                                                                             Index stride) {
  return __riscv_vlse16_v_i16m4(from, stride * sizeof(numext::int16_t), unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int16_t, PacketMul4Xs>(numext::int16_t* to, const PacketMul4Xs& from,
                                                                      Index stride) {
  __riscv_vsse16(to, stride * sizeof(numext::int16_t), from, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t pfirst<PacketMul4Xs>(const PacketMul4Xs& a) {
  return __riscv_vmv_x_s_i16m4_i16(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs preverse(const PacketMul4Xs& a) {
  PacketMul4Xsu idx =
      __riscv_vrsub_vx_u16m4(__riscv_vid_v_u16m4(unpacket_traits<PacketMul4Xs>::size),
                             unpacket_traits<PacketMul4Xs>::size - 1, unpacket_traits<PacketMul4Xs>::size);
  return __riscv_vrgather_vv_i16m4(a, idx, unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pabs(const PacketMul4Xs& a) {
  PacketMul4Xs mask = __riscv_vsra_vx_i16m4(a, 15, unpacket_traits<PacketMul4Xs>::size);
  return __riscv_vsub_vv_i16m4(__riscv_vxor_vv_i16m4(a, mask, unpacket_traits<PacketMul4Xs>::size), mask,
                               unpacket_traits<PacketMul4Xs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux<PacketMul4Xs>(const PacketMul4Xs& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i16m4_i16m1(
      a, __riscv_vmv_v_x_i16m1(0, unpacket_traits<PacketMul4Xs>::size / 4), unpacket_traits<PacketMul4Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_mul<PacketMul4Xs>(const PacketMul4Xs& a) {
  PacketMul1Xs half1 = __riscv_vmul_vv_i16m1(__riscv_vget_v_i16m4_i16m1(a, 0), __riscv_vget_v_i16m4_i16m1(a, 1),
                                         unpacket_traits<PacketMul1Xs>::size);
  PacketMul1Xs half2 = __riscv_vmul_vv_i16m1(__riscv_vget_v_i16m4_i16m1(a, 2), __riscv_vget_v_i16m4_i16m1(a, 3),
                                         unpacket_traits<PacketMul1Xs>::size);
  return predux_mul<PacketMul1Xs>(__riscv_vmul_vv_i16m1(half1, half2, unpacket_traits<PacketMul1Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_min<PacketMul4Xs>(const PacketMul4Xs& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i16m4_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::max)(), unpacket_traits<PacketMul4Xs>::size / 4),
      unpacket_traits<PacketMul4Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_max<PacketMul4Xs>(const PacketMul4Xs& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i16m4_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::min)(), unpacket_traits<PacketMul4Xs>::size / 4),
      unpacket_traits<PacketMul4Xs>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul4Xs, N>& kernel) {
  numext::int16_t buffer[unpacket_traits<PacketMul4Xs>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse16(&buffer[i], N * sizeof(numext::int16_t), kernel.packet[i], unpacket_traits<PacketMul4Xs>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle16_v_i16m4(&buffer[i * unpacket_traits<PacketMul4Xs>::size], unpacket_traits<PacketMul4Xs>::size);
  }
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET4_MATH_RVV10_H
