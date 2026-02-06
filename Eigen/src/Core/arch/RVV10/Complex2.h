// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Chip Kerchner <ckerchner@tenstorrent.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX2_RVV10_H
#define EIGEN_COMPLEX2_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/********************************* float32 ************************************/

template <>
EIGEN_STRONG_INLINE Packet2Xcf pcast<Packet4Xf, Packet2Xcf>(const Packet4Xf& a) {
  return Packet2Xcf(a);
}

template <>
EIGEN_STRONG_INLINE Packet4Xf pcast<Packet2Xcf, Packet4Xf>(const Packet2Xcf& a) {
  return a.v; 
}

EIGEN_STRONG_INLINE Packet4Xul __riscv_vreinterpret_v_f32m4_u64m4(const Packet4Xf& a) {
  return __riscv_vreinterpret_v_u32m4_u64m4(__riscv_vreinterpret_v_f32m4_u32m4(a));
}

EIGEN_STRONG_INLINE Packet4Xl __riscv_vreinterpret_v_f32m4_i64m4(const Packet4Xf& a) {
  return __riscv_vreinterpret_v_u64m4_i64m4(__riscv_vreinterpret_v_u32m4_u64m4(__riscv_vreinterpret_v_f32m4_u32m4(a)));
}

EIGEN_STRONG_INLINE Packet4Xf __riscv_vreinterpret_v_i64m4_f32m4(const Packet4Xl& a) {
  return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vreinterpret_v_u64m4_u32m4(__riscv_vreinterpret_v_i64m4_u64m4(a)));
}

EIGEN_STRONG_INLINE void prealimag2(const Packet2Xcf& a, Packet4Xf& real, Packet4Xf& imag) {
  const PacketMask8 mask = __riscv_vreinterpret_v_i8m1_b8(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xc>::size));
  Packet4Xu res = __riscv_vreinterpret_v_f32m4_u32m4(a.v);
  real = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vslide1up_vx_u32m4_tumu(mask, res, res, 0,
      unpacket_traits<Packet4Xi>::size));
  imag = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vslide1down_vx_u32m4_tumu(__riscv_vmnot_m_b8(mask,
      unpacket_traits<Packet1Xc>::size), res, res, 0, unpacket_traits<Packet4Xi>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pset1<Packet2Xcf>(const std::complex<float>& from) {
  const numext::int64_t from2 = *reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(&from));
  Packet4Xf res = __riscv_vreinterpret_v_i64m4_f32m4(pset1<Packet4Xl>(from2));
  return Packet2Xcf(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf padd<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  return Packet2Xcf(padd<Packet4Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf psub<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  return Packet2Xcf(psub<Packet4Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pnegate(const Packet2Xcf& a) {
  return Packet2Xcf(pnegate<Packet4Xf>(a.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pconj(const Packet2Xcf& a) {
  return Packet2Xcf(__riscv_vreinterpret_v_u64m4_f32m4(__riscv_vxor_vx_u64m4(
      __riscv_vreinterpret_v_f32m4_u64m4(a.v), 0x8000000000000000ull, unpacket_traits<Packet4Xl>::size)));
}

// Can use __riscv_vror_vx_u64m4 if zvbb is present
template <>
EIGEN_STRONG_INLINE Packet2Xcf pcplxflip<Packet2Xcf>(const Packet2Xcf& a) {
  Packet4Xu res = __riscv_vreinterpret_v_f32m4_u32m4(a.v);
  const PacketMask8 mask = __riscv_vreinterpret_v_i8m1_b8(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xc>::size));
  Packet4Xu data = __riscv_vslide1down_vx_u32m4(res, 0, unpacket_traits<Packet4Xi>::size);
  Packet4Xf res2 = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vslide1up_vx_u32m4_tumu(mask, data, res, 0,
      unpacket_traits<Packet4Xi>::size));
  return Packet2Xcf(res2);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pmul<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  Packet4Xf real, imag;
  prealimag2(a, real, imag);
  return Packet2Xcf(pmadd<Packet4Xf>(imag, pcplxflip<Packet2Xcf>(pconj<Packet2Xcf>(b)).v,
     pmul<Packet4Xf>(real, b.v)));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pmadd<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b, const Packet2Xcf& c) {
  Packet4Xf real, imag;
  prealimag2(a, real, imag);
  return Packet2Xcf(pmadd<Packet4Xf>(imag, pcplxflip<Packet2Xcf>(pconj<Packet2Xcf>(b)).v,
     pmadd<Packet4Xf>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pmsub<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b, const Packet2Xcf& c) {
  Packet4Xf real, imag;
  prealimag2(a, real, imag);
  return Packet2Xcf(pmadd<Packet4Xf>(imag, pcplxflip<Packet2Xcf>(pconj<Packet2Xcf>(b)).v,
     pmsub<Packet4Xf>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pcmp_eq(const Packet2Xcf& a, const Packet2Xcf& b) {
  Packet4Xi c = __riscv_vundefined_i32m4();
  PacketMask8 mask = __riscv_vmfeq_vv_f32m4_b8(a.v, b.v, unpacket_traits<Packet4Xf>::size);
  Packet4Xl res = __riscv_vreinterpret_v_i32m4_i64m4(__riscv_vmerge_vvm_i32m4(
      pzero<Packet4Xi>(c), ptrue<Packet4Xi>(c), mask, unpacket_traits<Packet4Xi>::size));
  Packet4Xf res2 = __riscv_vreinterpret_v_i64m4_f32m4(__riscv_vsra_vx_i64m4(__riscv_vand_vv_i64m4(
    __riscv_vsll_vx_i64m4(res, 32, unpacket_traits<Packet4Xl>::size), res, unpacket_traits<Packet4Xl>::size), 32,
    unpacket_traits<Packet4Xl>::size));
  return Packet2Xcf(res2);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pand<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  return Packet2Xcf(pand<Packet4Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf por<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  return Packet2Xcf(por<Packet4Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pxor<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  return Packet2Xcf(pxor<Packet4Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pandnot<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  return Packet2Xcf(pandnot<Packet4Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pload<Packet2Xcf>(const std::complex<float>* from) {
  Packet4Xf res = pload<Packet4Xf>(reinterpret_cast<const float *>(from));
  EIGEN_DEBUG_ALIGNED_LOAD return Packet2Xcf(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf ploadu<Packet2Xcf>(const std::complex<float>* from) {
  Packet4Xf res = ploadu<Packet4Xf>(reinterpret_cast<const float *>(from));
  EIGEN_DEBUG_UNALIGNED_LOAD return Packet2Xcf(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf ploaddup<Packet2Xcf>(const std::complex<float>* from) {
  Packet4Xl res = ploaddup<Packet4Xl>(reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)));
  return Packet2Xcf(__riscv_vreinterpret_v_i64m4_f32m4(res));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf ploadquad<Packet2Xcf>(const std::complex<float>* from) {
  Packet4Xl res = ploadquad<Packet4Xl>(reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)));
  return Packet2Xcf(__riscv_vreinterpret_v_i64m4_f32m4(res));
}

template <>
EIGEN_STRONG_INLINE void pstore<std::complex<float> >(std::complex<float>* to, const Packet2Xcf& from) {
  EIGEN_DEBUG_ALIGNED_STORE pstore<float>(reinterpret_cast<float *>(to), from.v);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float>* to, const Packet2Xcf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE pstoreu<float>(reinterpret_cast<float *>(to), from.v);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet2Xcf pgather<std::complex<float>, Packet2Xcf>(const std::complex<float>* from,
                                                                           Index stride) {
  return Packet2Xcf(__riscv_vreinterpret_v_i64m4_f32m4(pgather<int64_t, Packet4Xl>(
      reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)), stride)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pscatter<std::complex<float>, Packet2Xcf>(std::complex<float>* to, const Packet2Xcf& from,
                                                                       Index stride) {
  pscatter<int64_t, Packet4Xl>(reinterpret_cast<numext::int64_t *>(reinterpret_cast<void *>(to)), __riscv_vreinterpret_v_f32m4_i64m4(from.v),
       stride);
}

template <>
EIGEN_STRONG_INLINE std::complex<float> pfirst<Packet2Xcf>(const Packet2Xcf& a) {
  numext::int64_t res = pfirst<Packet4Xl>(__riscv_vreinterpret_v_f32m4_i64m4(a.v));
  return numext::bit_cast<std::complex<float>>(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf preverse(const Packet2Xcf& a) {
  return Packet2Xcf(__riscv_vreinterpret_v_i64m4_f32m4(preverse<Packet4Xl>(__riscv_vreinterpret_v_f32m4_i64m4(a.v))));
}

template <>
EIGEN_STRONG_INLINE std::complex<float> predux<Packet2Xcf>(const Packet2Xcf& a) {
  Packet4Xl res = __riscv_vreinterpret_v_f32m4_i64m4(a.v);
  Packet4Xf real = __riscv_vreinterpret_v_i64m4_f32m4(__riscv_vand_vx_i64m4(res, 0x00000000ffffffffull,
      unpacket_traits<Packet4Xl>::size));
  Packet4Xf imag = __riscv_vreinterpret_v_i64m4_f32m4(__riscv_vand_vx_i64m4(res, 0xffffffff00000000ull,
      unpacket_traits<Packet4Xl>::size));
  return std::complex<float>(predux<Packet4Xf>(real), predux<Packet4Xf>(imag));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pdiv<Packet2Xcf>(const Packet2Xcf& a, const Packet2Xcf& b) {
  return pdiv_complex(a, b);
}

template <int N>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void ptranspose(PacketBlock<Packet2Xcf, N>& kernel) {
  numext::int64_t buffer[unpacket_traits<Packet4Xl>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(numext::int64_t), __riscv_vreinterpret_v_f32m4_i64m4(kernel.packet[i].v),
        unpacket_traits<Packet4Xl>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] = Packet2Xcf(__riscv_vreinterpret_v_i64m4_f32m4(
        __riscv_vle64_v_i64m4(&buffer[i * unpacket_traits<Packet4Xl>::size], unpacket_traits<Packet4Xl>::size)));
  }
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf psqrt<Packet2Xcf>(const Packet2Xcf& a) {
  return psqrt_complex(a);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf plog<Packet2Xcf>(const Packet2Xcf& a) {
  return plog_complex(a);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcf pexp<Packet2Xcf>(const Packet2Xcf& a) {
  return pexp_complex(a);
}

#ifndef USE_LMUL2_ONLY
template <typename Packet = Packet2Xcf>
EIGEN_STRONG_INLINE Packet1Xcf predux_half(const Packet2Xcf& a) {
  return Packet1Xcf(__riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(a.v, 0), __riscv_vget_v_f32m4_f32m2(a.v, 1),
                                unpacket_traits<Packet2Xf>::size));
}
#endif

EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(Packet2Xcf, Packet4Xf)

/********************************* double ************************************/

template <>
EIGEN_STRONG_INLINE Packet2Xcd pcast<Packet4Xd, Packet2Xcd>(const Packet4Xd& a) {
  return Packet2Xcd(a);
}

template <>
EIGEN_STRONG_INLINE Packet4Xd pcast<Packet2Xcd, Packet4Xd>(const Packet2Xcd& a) {
  return a.v;
}

EIGEN_STRONG_INLINE void prealimag2(const Packet2Xcd& a, Packet4Xd& real, Packet4Xd& imag) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xc>::size));
  real = __riscv_vfslide1up_vf_f64m4_tumu(mask, a.v, a.v, 0.0, unpacket_traits<Packet4Xd>::size);
  imag = __riscv_vfslide1down_vf_f64m4_tumu(__riscv_vmnot_m_b16(mask, unpacket_traits<Packet1Xs>::size),
      a.v, a.v, 0.0, unpacket_traits<Packet4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pset1<Packet2Xcd>(const std::complex<double>& from) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xc>::size));
  Packet4Xd res = __riscv_vmerge_vvm_f64m4(pset1<Packet4Xd>(from.real()), pset1<Packet4Xd>(from.imag()),
      mask, unpacket_traits<Packet4Xd>::size);
  return Packet2Xcd(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd padd<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  return Packet2Xcd(padd<Packet4Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd psub<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  return Packet2Xcd(psub<Packet4Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pnegate(const Packet2Xcd& a) {
  return Packet2Xcd(pnegate<Packet4Xd>(a.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pconj(const Packet2Xcd& a) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xc>::size));
  return Packet2Xcd(__riscv_vfsgnjn_vv_f64m4_tumu(mask, a.v, a.v, a.v, unpacket_traits<Packet4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pcplxflip<Packet2Xcd>(const Packet2Xcd& a) {
  Packet4Xul res = __riscv_vreinterpret_v_f64m4_u64m4(a.v);
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xc>::size));
  Packet4Xul data = __riscv_vslide1down_vx_u64m4(res, 0, unpacket_traits<Packet4Xl>::size);
  Packet4Xd res2 = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vslide1up_vx_u64m4_tumu(mask, data, res, 0,
      unpacket_traits<Packet4Xl>::size));
  return Packet2Xcd(res2);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pmul<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  Packet4Xd real, imag;
  prealimag2(a, real, imag);
  return Packet2Xcd(pmadd<Packet4Xd>(imag, pcplxflip<Packet2Xcd>(pconj<Packet2Xcd>(b)).v,
     pmul<Packet4Xd>(real, b.v)));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pmadd<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b, const Packet2Xcd& c) {
  Packet4Xd real, imag;
  prealimag2(a, real, imag);
  return Packet2Xcd(pmadd<Packet4Xd>(imag, pcplxflip<Packet2Xcd>(pconj<Packet2Xcd>(b)).v,
     pmadd<Packet4Xd>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pmsub<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b, const Packet2Xcd& c) {
  Packet4Xd real, imag;
  prealimag2(a, real, imag);
  return Packet2Xcd(pmadd<Packet4Xd>(imag, pcplxflip<Packet2Xcd>(pconj<Packet2Xcd>(b)).v,
     pmsub<Packet4Xd>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pcmp_eq(const Packet2Xcd& a, const Packet2Xcd& b) {
  Packet4Xl c = __riscv_vundefined_i64m4();
  Packet1Xsu mask = __riscv_vreinterpret_v_b16_u16m1(__riscv_vmfeq_vv_f64m4_b16(a.v, b.v,
    unpacket_traits<Packet4Xd>::size));
  Packet1Xsu mask_r = __riscv_vsrl_vx_u16m1(__riscv_vand_vx_u16m1(mask, static_cast<short>(0xaaaa),
    unpacket_traits<Packet1Xs>::size), 1, unpacket_traits<Packet1Xs>::size);
  mask = __riscv_vand_vv_u16m1(mask, mask_r, unpacket_traits<Packet1Xs>::size);
  mask = __riscv_vor_vv_u16m1(__riscv_vsll_vx_u16m1(mask, 1, unpacket_traits<Packet1Xs>::size),
    mask, unpacket_traits<Packet1Xs>::size);
  Packet4Xd res = __riscv_vreinterpret_v_i64m4_f64m4(__riscv_vmerge_vvm_i64m4(pzero<Packet4Xl>(c),
    ptrue<Packet4Xl>(c), __riscv_vreinterpret_v_u16m1_b16(mask), unpacket_traits<Packet4Xl>::size));
  return Packet2Xcd(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pand<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  return Packet2Xcd(pand<Packet4Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd por<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  return Packet2Xcd(por<Packet4Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pxor<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  return Packet2Xcd(pxor<Packet4Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pandnot<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  return Packet2Xcd(pandnot<Packet4Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pload<Packet2Xcd>(const std::complex<double>* from) {
  Packet4Xd res = pload<Packet4Xd>(reinterpret_cast<const double *>(from));
  EIGEN_DEBUG_ALIGNED_LOAD return Packet2Xcd(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd ploadu<Packet2Xcd>(const std::complex<double>* from) {
  Packet4Xd res = ploadu<Packet4Xd>(reinterpret_cast<const double *>(from));
  EIGEN_DEBUG_UNALIGNED_LOAD return Packet2Xcd(res);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd ploaddup<Packet2Xcd>(const std::complex<double>* from) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0x66),
      unpacket_traits<Packet1Xc>::size));
  Packet4Xul idx1 = __riscv_vsrl_vx_u64m4(__riscv_vid_v_u64m4(unpacket_traits<Packet4Xd>::size), 1,
      unpacket_traits<Packet4Xd>::size);
  Packet4Xul idx2 = __riscv_vxor_vx_u64m4_tumu(mask, idx1, idx1, 1, unpacket_traits<Packet4Xl>::size);
  return Packet2Xcd(__riscv_vrgather_vv_f64m4(pload<Packet4Xd>(reinterpret_cast<const double *>(from)), idx2,
       unpacket_traits<Packet4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd ploadquad<Packet2Xcd>(const std::complex<double>* from) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0x5a),
      unpacket_traits<Packet1Xc>::size));
  Packet4Xul idx1 = __riscv_vsrl_vx_u64m4(__riscv_vid_v_u64m4(unpacket_traits<Packet4Xd>::size), 2,
      unpacket_traits<Packet4Xd>::size);
  Packet4Xul idx2 = __riscv_vxor_vx_u64m4_tumu(mask, idx1, idx1, 1, unpacket_traits<Packet4Xl>::size);
  return Packet2Xcd(__riscv_vrgather_vv_f64m4(pload<Packet4Xd>(reinterpret_cast<const double *>(from)), idx2,
       unpacket_traits<Packet4Xd>::size));
}

template <>
EIGEN_STRONG_INLINE void pstore<std::complex<double> >(std::complex<double>* to, const Packet2Xcd& from) {
  EIGEN_DEBUG_ALIGNED_STORE pstore<double>(reinterpret_cast<double *>(to), from.v);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double>* to, const Packet2Xcd& from) {
  EIGEN_DEBUG_UNALIGNED_STORE pstoreu<double>(reinterpret_cast<double *>(to), from.v);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet2Xcd pgather<std::complex<double>, Packet2Xcd>(const std::complex<double>* from,
                                                                            Index stride) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0x55),
      unpacket_traits<Packet1Xc>::size));
  const double *from2 = reinterpret_cast<const double *>(from);
  Packet4Xd res = __riscv_vundefined_f64m4();
  res = __riscv_vlse64_v_f64m4_tumu(mask, res,
      &from2[0 - (0 * stride)], stride * sizeof(double), unpacket_traits<Packet4Xd>::size);
  res = __riscv_vlse64_v_f64m4_tumu(__riscv_vmnot_m_b16(mask, unpacket_traits<Packet1Xs>::size), res,
      &from2[1 - (1 * stride)], stride * sizeof(double), unpacket_traits<Packet4Xd>::size);
  return Packet2Xcd(res);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pscatter<std::complex<double>, Packet2Xcd>(std::complex<double>* to, const Packet2Xcd& from,
                                                                        Index stride) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0x55),
      unpacket_traits<Packet1Xc>::size));
  double *to2 = reinterpret_cast<double *>(to);
  __riscv_vsse64_v_f64m4_m(mask, &to2[0 - (0 * stride)],
      stride * sizeof(double), from.v, unpacket_traits<Packet4Xd>::size);
  __riscv_vsse64_v_f64m4_m(__riscv_vmnot_m_b16(mask, unpacket_traits<Packet1Xs>::size), &to2[1 - (1 * stride)],
      stride * sizeof(double), from.v, unpacket_traits<Packet4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> pfirst<Packet2Xcd>(const Packet2Xcd& a) {
  double real = pfirst<Packet4Xd>(a.v);
  double imag = pfirst<Packet4Xd>(__riscv_vfslide1down_vf_f64m4(a.v, 0.0, unpacket_traits<Packet4Xd>::size));
  return std::complex<double>(real, imag);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd preverse(const Packet2Xcd& a) {
  Packet4Xul idx = __riscv_vxor_vx_u64m4(__riscv_vid_v_u64m4(unpacket_traits<Packet4Xl>::size),
      unpacket_traits<Packet4Xl>::size - 2, unpacket_traits<Packet4Xl>::size);
  Packet4Xd res = __riscv_vrgather_vv_f64m4(a.v, idx, unpacket_traits<Packet4Xd>::size);
  return Packet2Xcd(res);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> predux<Packet2Xcd>(const Packet2Xcd& a) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xc>::size));
  Packet4Xl res = __riscv_vreinterpret_v_f64m4_i64m4(a.v);
  Packet4Xd real = __riscv_vreinterpret_v_i64m4_f64m4(__riscv_vand_vx_i64m4_tumu(mask, res, res, 0,
      unpacket_traits<Packet4Xl>::size));
  Packet4Xd imag = __riscv_vreinterpret_v_i64m4_f64m4(__riscv_vand_vx_i64m4_tumu(
      __riscv_vmnot_m_b16(mask, unpacket_traits<Packet1Xs>::size), res, res, 0, unpacket_traits<Packet4Xl>::size));
  return std::complex<double>(predux<Packet4Xd>(real), predux<Packet4Xd>(imag));
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd pdiv<Packet2Xcd>(const Packet2Xcd& a, const Packet2Xcd& b) {
  return pdiv_complex(a, b);
}

template <int N>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void ptranspose(PacketBlock<Packet2Xcd, N>& kernel) {
  double buffer[unpacket_traits<Packet4Xd>::size * N];
  int i = 0;

  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0x55),
      unpacket_traits<Packet1Xc>::size));

  for (i = 0; i < N; i++) {
    __riscv_vsse64_v_f64m4_m(mask,
      &buffer[(i * 2) - (0 * N) + 0], N * sizeof(double), kernel.packet[i].v, unpacket_traits<Packet4Xd>::size);
    __riscv_vsse64_v_f64m4_m(__riscv_vmnot_m_b16(mask, unpacket_traits<Packet1Xs>::size),
      &buffer[(i * 2) - (1 * N) + 1], N * sizeof(double), kernel.packet[i].v, unpacket_traits<Packet4Xd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] = Packet2Xcd(__riscv_vle64_v_f64m4(&buffer[i * unpacket_traits<Packet4Xd>::size],
        unpacket_traits<Packet4Xd>::size));
  }
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd psqrt<Packet2Xcd>(const Packet2Xcd& a) {
  return psqrt_complex(a);
}

template <>
EIGEN_STRONG_INLINE Packet2Xcd plog<Packet2Xcd>(const Packet2Xcd& a) {
  return plog_complex(a);
}

#if 0 // bug with pexp_complex for complex<double>
template <>
EIGEN_STRONG_INLINE Packet2Xcd pexp<Packet2Xcd>(const Packet2Xcd& a) {
  return pexp_complex(a);
}
#endif

#ifndef USE_LMUL2_ONLY
template <typename Packet = Packet2Xcd>
EIGEN_STRONG_INLINE Packet1Xcd predux_half(const Packet2Xcd& a) {
  return Packet1Xcd(__riscv_vfadd_vv_f64m2(__riscv_vget_v_f64m4_f64m2(a.v, 0), __riscv_vget_v_f64m4_f64m2(a.v, 1),
                                unpacket_traits<Packet2Xd>::size));
}
#endif

EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(Packet2Xcd, Packet4Xd)

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_COMPLEX2_RVV10_H
