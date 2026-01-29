// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
// Copyright (C) 2026 Chip Kerchner <ckerchner@tenstorrent.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_RVV10_H
#define EIGEN_COMPLEX_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/********************************* float32 ************************************/

struct PacketXcf {
  EIGEN_STRONG_INLINE PacketXcf() {}
  EIGEN_STRONG_INLINE explicit PacketXcf(const Packet2Xf& a) : v(a) {}

  Packet2Xf v;
};

template <>
struct packet_traits<std::complex<float>> : default_packet_traits {
  typedef PacketXcf type;
  typedef PacketXcf half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasSqrt = 1,
    HasLog = 1,
    HasExp = 1,
    HasSign = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasSetLinear = 0
  };
};

template <>
struct unpacket_traits<PacketXcf> : default_unpacket_traits {
  typedef std::complex<float> type;
  typedef PacketXcf half;
  typedef Packet2Xf as_real;
  enum {
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true
  };
};

typedef PacketXcf Packet2cf;

template <>
EIGEN_STRONG_INLINE PacketXcf pcast<Packet2Xf, PacketXcf>(const Packet2Xf& a) {
  return PacketXcf(a);
}

template <>
EIGEN_STRONG_INLINE Packet2Xf pcast<PacketXcf, Packet2Xf>(const PacketXcf& a) {
  return a.v; 
}

EIGEN_STRONG_INLINE Packet2Xul __riscv_vreinterpret_v_f32m2_u64m2(const Packet2Xf& a) {
  return __riscv_vreinterpret_v_u32m2_u64m2(__riscv_vreinterpret_v_f32m2_u32m2(a));
}

EIGEN_STRONG_INLINE Packet2Xl __riscv_vreinterpret_v_f32m2_i64m2(const Packet2Xf& a) {
  return __riscv_vreinterpret_v_u64m2_i64m2(__riscv_vreinterpret_v_u32m2_u64m2(__riscv_vreinterpret_v_f32m2_u32m2(a)));
}

EIGEN_STRONG_INLINE Packet2Xf __riscv_vreinterpret_v_i64m2_f32m2(const Packet2Xl& a) {
  return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vreinterpret_v_u64m2_u32m2(__riscv_vreinterpret_v_i64m2_u64m2(a)));
}

EIGEN_STRONG_INLINE void prealimag2(const PacketXcf& a, Packet2Xf& real, Packet2Xf& imag) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i8m1_b16(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xs>::size * 2));
  real = __riscv_vfslide1up_vf_f32m2_tumu(mask, a.v, a.v, 0.0f, unpacket_traits<Packet2Xf>::size);
  imag = __riscv_vfslide1down_vf_f32m2_tumu(__riscv_vmnot_m_b16(mask, unpacket_traits<Packet1Xs>::size),
      a.v, a.v, 0.0f, unpacket_traits<Packet2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXcf pset1<PacketXcf>(const std::complex<float>& from) {
  const numext::int64_t from2 = *reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(&from));
  Packet2Xf res = __riscv_vreinterpret_v_i64m2_f32m2(pset1<Packet2Xl>(from2));
  return PacketXcf(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcf padd<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(padd<Packet2Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf psub<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(psub<Packet2Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pnegate(const PacketXcf& a) {
  return PacketXcf(pnegate<Packet2Xf>(a.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pconj(const PacketXcf& a) {
  return PacketXcf(__riscv_vreinterpret_v_u64m2_f32m2(__riscv_vxor_vx_u64m2(
      __riscv_vreinterpret_v_f32m2_u64m2(a.v), 0x8000000000000000ull, unpacket_traits<Packet2Xl>::size)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pcplxflip<PacketXcf>(const PacketXcf& a) {
  Packet2Xul res = __riscv_vreinterpret_v_f32m2_u64m2(a.v);
  Packet2Xf res2 = __riscv_vreinterpret_v_u64m2_f32m2(__riscv_vor_vv_u64m2(
    __riscv_vsll_vx_u64m2(res, 32, unpacket_traits<Packet2Xl>::size),
    __riscv_vsrl_vx_u64m2(res, 32, unpacket_traits<Packet2Xl>::size), unpacket_traits<Packet2Xl>::size));
  return PacketXcf(res2);
}

template <>
EIGEN_STRONG_INLINE PacketXcf pmul<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  Packet2Xf real, imag;
  prealimag2(a, real, imag);
  return PacketXcf(pmadd<Packet2Xf>(imag, pcplxflip<PacketXcf>(pconj<PacketXcf>(b)).v,
     pmul<Packet2Xf>(real, b.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pmadd<PacketXcf>(const PacketXcf& a, const PacketXcf& b, const PacketXcf& c) {
  Packet2Xf real, imag;
  prealimag2(a, real, imag);
  return PacketXcf(pmadd<Packet2Xf>(imag, pcplxflip<PacketXcf>(pconj<PacketXcf>(b)).v,
     pmadd<Packet2Xf>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pmsub<PacketXcf>(const PacketXcf& a, const PacketXcf& b, const PacketXcf& c) {
  Packet2Xf real, imag;
  prealimag2(a, real, imag);
  return PacketXcf(pmadd<Packet2Xf>(imag, pcplxflip<PacketXcf>(pconj<PacketXcf>(b)).v,
     pmsub<Packet2Xf>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pcmp_eq(const PacketXcf& a, const PacketXcf& b) {
  Packet2Xi c = __riscv_vundefined_i32m2();
  PacketMask16 mask = __riscv_vmfeq_vv_f32m2_b16(a.v, b.v, unpacket_traits<Packet2Xf>::size);
  Packet2Xl res = __riscv_vreinterpret_v_i32m2_i64m2(__riscv_vmerge_vvm_i32m2(
      pzero<Packet2Xi>(c), ptrue<Packet2Xi>(c), mask, unpacket_traits<Packet2Xi>::size));
  Packet2Xf res2 = __riscv_vreinterpret_v_i64m2_f32m2(__riscv_vsra_vx_i64m2(__riscv_vand_vv_i64m2(
    __riscv_vsll_vx_i64m2(res, 32, unpacket_traits<Packet2Xl>::size), res, unpacket_traits<Packet2Xl>::size), 32,
    unpacket_traits<Packet2Xl>::size));
  return PacketXcf(res2);
}

template <>
EIGEN_STRONG_INLINE PacketXcf pand<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(pand<Packet2Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf por<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(por<Packet2Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pxor<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(pxor<Packet2Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pandnot<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(pandnot<Packet2Xf>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pload<PacketXcf>(const std::complex<float>* from) {
  Packet2Xf res = pload<Packet2Xf>(reinterpret_cast<const float *>(from));
  EIGEN_DEBUG_ALIGNED_LOAD return PacketXcf(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploadu<PacketXcf>(const std::complex<float>* from) {
  Packet2Xf res = ploadu<Packet2Xf>(reinterpret_cast<const float *>(from));
  EIGEN_DEBUG_UNALIGNED_LOAD return PacketXcf(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploaddup<PacketXcf>(const std::complex<float>* from) {
  Packet2Xl res = ploaddup<Packet2Xl>(reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)));
  return PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(res));
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploadquad<PacketXcf>(const std::complex<float>* from) {
  Packet2Xl res = ploadquad<Packet2Xl>(reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)));
  return PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(res));
}

template <>
EIGEN_STRONG_INLINE void pstore<std::complex<float> >(std::complex<float>* to, const PacketXcf& from) {
  EIGEN_DEBUG_ALIGNED_STORE pstore<float>(reinterpret_cast<float *>(to), from.v);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float>* to, const PacketXcf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE pstoreu<float>(reinterpret_cast<float *>(to), from.v);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketXcf pgather<std::complex<float>, PacketXcf>(const std::complex<float>* from,
                                                                           Index stride) {
  return PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(pgather<int64_t, Packet2Xl>(
      reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)), stride)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pscatter<std::complex<float>, PacketXcf>(std::complex<float>* to, const PacketXcf& from,
                                                                       Index stride) {
  pscatter<int64_t, Packet2Xl>(reinterpret_cast<numext::int64_t *>(reinterpret_cast<void *>(to)), __riscv_vreinterpret_v_f32m2_i64m2(from.v),
       stride);
}

template <>
EIGEN_STRONG_INLINE std::complex<float> pfirst<PacketXcf>(const PacketXcf& a) {
  numext::int64_t res = pfirst<Packet2Xl>(__riscv_vreinterpret_v_f32m2_i64m2(a.v));
  return numext::bit_cast<std::complex<float>>(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcf preverse(const PacketXcf& a) {
  return PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(preverse<Packet2Xl>(__riscv_vreinterpret_v_f32m2_i64m2(a.v))));
}

template <>
EIGEN_STRONG_INLINE std::complex<float> predux<PacketXcf>(const PacketXcf& a) {
  Packet2Xl res = __riscv_vreinterpret_v_f32m2_i64m2(a.v);
  Packet2Xf real = __riscv_vreinterpret_v_i64m2_f32m2(__riscv_vand_vx_i64m2(res, 0x00000000ffffffffull,
      unpacket_traits<Packet2Xl>::size));
  Packet2Xf imag = __riscv_vreinterpret_v_i64m2_f32m2(__riscv_vand_vx_i64m2(res, 0xffffffff00000000ull,
      unpacket_traits<Packet2Xl>::size));
  return std::complex<float>(predux<Packet2Xf>(real), predux<Packet2Xf>(imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pdiv<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return pdiv_complex(a, b);
}

template <int N>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void ptranspose(PacketBlock<PacketXcf, N>& kernel) {
  numext::int64_t buffer[unpacket_traits<Packet2Xl>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(numext::int64_t), __riscv_vreinterpret_v_f32m2_i64m2(kernel.packet[i].v),
        unpacket_traits<Packet2Xl>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] = PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(
        __riscv_vle64_v_i64m2(&buffer[i * unpacket_traits<Packet2Xl>::size], unpacket_traits<Packet2Xl>::size)));
  }
}

template <>
EIGEN_STRONG_INLINE PacketXcf psqrt<PacketXcf>(const PacketXcf& a) {
  return psqrt_complex(a);
}

template <>
EIGEN_STRONG_INLINE PacketXcf plog<PacketXcf>(const PacketXcf& a) {
  return plog_complex(a);
}

template <>
EIGEN_STRONG_INLINE PacketXcf pexp<PacketXcf>(const PacketXcf& a) {
  return pexp_complex(a);
}

template <>
struct conj_helper<Packet2Xf, PacketXcf, false, false> {
  EIGEN_STRONG_INLINE PacketXcf pmadd(const Packet2Xf& x, const PacketXcf& y, const PacketXcf& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcf pmsub(const Packet2Xf& x, const PacketXcf& y, const PacketXcf& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcf pmul(const Packet2Xf& x, const PacketXcf& y) const {
    return PacketXcf(Eigen::internal::pmul<Packet2Xf>(x, pcast<PacketXcf, Packet2Xf>(y)));
  }
};

template <>
struct conj_helper<PacketXcf, Packet2Xf, false, false> {
  EIGEN_STRONG_INLINE PacketXcf pmadd(const PacketXcf& x, const Packet2Xf& y, const PacketXcf& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcf pmsub(const PacketXcf& x, const Packet2Xf& y, const PacketXcf& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcf pmul(const PacketXcf& x, const Packet2Xf& y) const {
    return PacketXcf(Eigen::internal::pmul<Packet2Xf>(pcast<PacketXcf, Packet2Xf>(x), y));
  }
};

/********************************* double ************************************/

struct PacketXcd {
  EIGEN_STRONG_INLINE PacketXcd() {}
  EIGEN_STRONG_INLINE explicit PacketXcd(const Packet2Xd& a) : v(a) {}

  Packet2Xd v;
};

template <>
struct packet_traits<std::complex<double>> : default_packet_traits {
  typedef PacketXcd type;
  typedef PacketXcd half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasSqrt = 1,
    HasLog = 1,
#if 0 // bug with pexp_complex for complex<double>
    HasExp = 1,
#else
    HasExp = 0,
#endif
    HasSign = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasSetLinear = 0
  };
};

template <>
struct unpacket_traits<PacketXcd> : default_unpacket_traits {
  typedef std::complex<double> type;
  typedef PacketXcd half;
  typedef Packet2Xd as_real;
  enum {
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true
  };
};

typedef PacketXcd Packet1cd;

template <>
EIGEN_STRONG_INLINE PacketXcd pcast<Packet2Xd, PacketXcd>(const Packet2Xd& a) {
  return PacketXcd(a);
}

template <>
EIGEN_STRONG_INLINE Packet2Xd pcast<PacketXcd, Packet2Xd>(const PacketXcd& a) {
  return a.v;
}

EIGEN_STRONG_INLINE void prealimag2(const PacketXcd& a, Packet2Xd& real, Packet2Xd& imag) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xi>::size * 4));
  real = __riscv_vfslide1up_vf_f64m2_tumu(mask, a.v, a.v, 0.0, unpacket_traits<Packet2Xd>::size);
  imag = __riscv_vfslide1down_vf_f64m2_tumu(__riscv_vmnot_m_b32(mask, unpacket_traits<Packet1Xs>::size),
      a.v, a.v, 0.0, unpacket_traits<Packet2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXcd pset1<PacketXcd>(const std::complex<double>& from) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xi>::size * 4));
  Packet2Xd res = __riscv_vmerge_vvm_f64m2(pset1<Packet2Xd>(from.real()), pset1<Packet2Xd>(from.imag()),
      mask, unpacket_traits<Packet2Xd>::size);
  return PacketXcd(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd padd<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(padd<Packet2Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcd psub<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(psub<Packet2Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pnegate(const PacketXcd& a) {
  return PacketXcd(pnegate<Packet2Xd>(a.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pconj(const PacketXcd& a) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xi>::size * 4));
  return PacketXcd(__riscv_vfsgnjn_vv_f64m2_tumu(mask, a.v, a.v, a.v, unpacket_traits<Packet2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pcplxflip<PacketXcd>(const PacketXcd& a) {
  Packet2Xul idx = __riscv_vxor_vx_u64m2(__riscv_vid_v_u64m2(unpacket_traits<Packet2Xl>::size), 1,
      unpacket_traits<Packet2Xl>::size);
  Packet2Xd res = __riscv_vrgather_vv_f64m2(a.v, idx, unpacket_traits<Packet2Xd>::size);
  return PacketXcd(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmul<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  Packet2Xd real, imag;
  prealimag2(a, real, imag);
  return PacketXcd(pmadd<Packet2Xd>(imag, pcplxflip<PacketXcd>(pconj<PacketXcd>(b)).v,
     pmul<Packet2Xd>(real, b.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmadd<PacketXcd>(const PacketXcd& a, const PacketXcd& b, const PacketXcd& c) {
  Packet2Xd real, imag;
  prealimag2(a, real, imag);
  return PacketXcd(pmadd<Packet2Xd>(imag, pcplxflip<PacketXcd>(pconj<PacketXcd>(b)).v,
     pmadd<Packet2Xd>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmsub<PacketXcd>(const PacketXcd& a, const PacketXcd& b, const PacketXcd& c) {
  Packet2Xd real, imag;
  prealimag2(a, real, imag);
  return PacketXcd(pmadd<Packet2Xd>(imag, pcplxflip<PacketXcd>(pconj<PacketXcd>(b)).v,
     pmsub<Packet2Xd>(real, b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pcmp_eq(const PacketXcd& a, const PacketXcd& b) {
  Packet2Xl c = __riscv_vundefined_i64m2();
  Packet1Xu mask = __riscv_vreinterpret_v_b32_u32m1(__riscv_vmfeq_vv_f64m2_b32(a.v, b.v,
    unpacket_traits<Packet2Xd>::size));
  Packet1Xu mask_r = __riscv_vsrl_vx_u32m1(__riscv_vand_vx_u32m1(mask, 0xaaaaaaaa,
    unpacket_traits<Packet1Xi>::size), 1, unpacket_traits<Packet1Xi>::size);
  mask = __riscv_vand_vv_u32m1(mask, mask_r, unpacket_traits<Packet1Xi>::size);
  mask = __riscv_vor_vv_u32m1(__riscv_vsll_vx_u32m1(mask, 1, unpacket_traits<Packet1Xi>::size),
    mask, unpacket_traits<Packet1Xi>::size);
  Packet2Xd res = __riscv_vreinterpret_v_i64m2_f64m2(__riscv_vmerge_vvm_i64m2(pzero<Packet2Xl>(c),
    ptrue<Packet2Xl>(c), __riscv_vreinterpret_v_u32m1_b32(mask), unpacket_traits<Packet2Xl>::size));
  return PacketXcd(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd pand<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pand<Packet2Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcd por<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(por<Packet2Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pxor<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pxor<Packet2Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pandnot<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pandnot<Packet2Xd>(a.v, b.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pload<PacketXcd>(const std::complex<double>* from) {
  Packet2Xd res = pload<Packet2Xd>(reinterpret_cast<const double *>(from));
  EIGEN_DEBUG_ALIGNED_LOAD return PacketXcd(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploadu<PacketXcd>(const std::complex<double>* from) {
  Packet2Xd res = ploadu<Packet2Xd>(reinterpret_cast<const double *>(from));
  EIGEN_DEBUG_UNALIGNED_LOAD return PacketXcd(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploaddup<PacketXcd>(const std::complex<double>* from) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0x66),
      unpacket_traits<Packet1Xi>::size * 4));
  Packet2Xul idx1 = __riscv_vsrl_vx_u64m2(__riscv_vid_v_u64m2(unpacket_traits<Packet2Xd>::size), 1,
      unpacket_traits<Packet2Xd>::size);
  Packet2Xul idx2 = __riscv_vxor_vx_u64m2_tumu(mask, idx1, idx1, 1, unpacket_traits<Packet2Xl>::size);
  return PacketXcd(__riscv_vrgather_vv_f64m2(pload<Packet2Xd>(reinterpret_cast<const double *>(from)), idx2,
       unpacket_traits<Packet2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploadquad<PacketXcd>(const std::complex<double>* from) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0x5a),
      unpacket_traits<Packet1Xi>::size * 4));
  Packet2Xul idx1 = __riscv_vsrl_vx_u64m2(__riscv_vid_v_u64m2(unpacket_traits<Packet2Xd>::size), 2,
      unpacket_traits<Packet2Xd>::size);
  Packet2Xul idx2 = __riscv_vxor_vx_u64m2_tumu(mask, idx1, idx1, 1, unpacket_traits<Packet2Xl>::size);
  return PacketXcd(__riscv_vrgather_vv_f64m2(pload<Packet2Xd>(reinterpret_cast<const double *>(from)), idx2,
       unpacket_traits<Packet2Xd>::size));
}

template <>
EIGEN_STRONG_INLINE void pstore<std::complex<double> >(std::complex<double>* to, const PacketXcd& from) {
  EIGEN_DEBUG_ALIGNED_STORE pstore<double>(reinterpret_cast<double *>(to), from.v);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double>* to, const PacketXcd& from) {
  EIGEN_DEBUG_UNALIGNED_STORE pstoreu<double>(reinterpret_cast<double *>(to), from.v);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketXcd pgather<std::complex<double>, PacketXcd>(const std::complex<double>* from,
                                                                            Index stride) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0x55),
      unpacket_traits<Packet1Xi>::size * 4));
  const double *from2 = reinterpret_cast<const double *>(from);
  Packet2Xd res = __riscv_vundefined_f64m2();
  res = __riscv_vlse64_v_f64m2_tumu(mask, res,
      &from2[0 - (0 * stride)], stride * sizeof(double), unpacket_traits<Packet2Xd>::size);
  res = __riscv_vlse64_v_f64m2_tumu(__riscv_vmnot_m_b32(mask, unpacket_traits<Packet1Xi>::size), res,
      &from2[1 - (1 * stride)], stride * sizeof(double), unpacket_traits<Packet2Xd>::size);
  return PacketXcd(res);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pscatter<std::complex<double>, PacketXcd>(std::complex<double>* to, const PacketXcd& from,
                                                                        Index stride) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0x55),
      unpacket_traits<Packet1Xi>::size * 4));
  double *to2 = reinterpret_cast<double *>(to);
  __riscv_vsse64_v_f64m2_m(mask, &to2[0 - (0 * stride)],
      stride * sizeof(double), from.v, unpacket_traits<Packet2Xd>::size);
  __riscv_vsse64_v_f64m2_m(__riscv_vmnot_m_b32(mask, unpacket_traits<Packet1Xi>::size), &to2[1 - (1 * stride)],
      stride * sizeof(double), from.v, unpacket_traits<Packet2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> pfirst<PacketXcd>(const PacketXcd& a) {
  double real = pfirst<Packet2Xd>(a.v);
  double imag = pfirst<Packet2Xd>(__riscv_vfslide1down_vf_f64m2(a.v, 0.0, unpacket_traits<Packet2Xd>::size));
  return std::complex<double>(real, imag);
}

template <>
EIGEN_STRONG_INLINE PacketXcd preverse(const PacketXcd& a) {
  Packet2Xul idx = __riscv_vxor_vx_u64m2(__riscv_vid_v_u64m2(unpacket_traits<Packet2Xl>::size),
      unpacket_traits<Packet2Xl>::size - 2, unpacket_traits<Packet2Xl>::size);
  Packet2Xd res = __riscv_vrgather_vv_f64m2(a.v, idx, unpacket_traits<Packet2Xd>::size);
  return PacketXcd(res);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> predux<PacketXcd>(const PacketXcd& a) {
  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0xaa),
      unpacket_traits<Packet1Xi>::size * 4));
  Packet2Xl res = __riscv_vreinterpret_v_f64m2_i64m2(a.v);
  Packet2Xd real = __riscv_vreinterpret_v_i64m2_f64m2(__riscv_vand_vx_i64m2_tumu(mask, res, res, 0,
      unpacket_traits<Packet2Xl>::size));
  Packet2Xd imag = __riscv_vreinterpret_v_i64m2_f64m2(__riscv_vand_vx_i64m2_tumu(
      __riscv_vmnot_m_b32(mask, unpacket_traits<Packet1Xs>::size), res, res, 0, unpacket_traits<Packet2Xl>::size));
  return std::complex<double>(predux<Packet2Xd>(real), predux<Packet2Xd>(imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pdiv<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return pdiv_complex(a, b);
}

template <int N>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void ptranspose(PacketBlock<PacketXcd, N>& kernel) {
  double buffer[unpacket_traits<Packet2Xd>::size * N];
  int i = 0;

  const PacketMask32 mask = __riscv_vreinterpret_v_i8m1_b32(__riscv_vmv_v_x_i8m1(static_cast<char>(0x55),
      unpacket_traits<Packet1Xi>::size * 4));

  for (i = 0; i < N; i++) {
    __riscv_vsse64_v_f64m2_m(mask,
      &buffer[(i * 2) - (0 * N) + 0], N * sizeof(double), kernel.packet[i].v, unpacket_traits<Packet2Xd>::size);
    __riscv_vsse64_v_f64m2_m(__riscv_vmnot_m_b32(mask, unpacket_traits<Packet1Xs>::size),
      &buffer[(i * 2) - (1 * N) + 1], N * sizeof(double), kernel.packet[i].v, unpacket_traits<Packet2Xd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] = PacketXcd(__riscv_vle64_v_f64m2(&buffer[i * unpacket_traits<Packet2Xd>::size],
        unpacket_traits<Packet2Xd>::size));
  }
}

template <>
EIGEN_STRONG_INLINE PacketXcd psqrt<PacketXcd>(const PacketXcd& a) {
  return psqrt_complex(a);
}

template <>
EIGEN_STRONG_INLINE PacketXcd plog<PacketXcd>(const PacketXcd& a) {
  return plog_complex(a);
}

#if 0 // bug with pexp_complex for complex<double>
template <>
EIGEN_STRONG_INLINE PacketXcd pexp<PacketXcd>(const PacketXcd& a) {
  return pexp_complex(a);
}
#endif

template <>
struct conj_helper<Packet2Xd, PacketXcd, false, false> {
  EIGEN_STRONG_INLINE PacketXcd pmadd(const Packet2Xd& x, const PacketXcd& y, const PacketXcd& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcd pmsub(const Packet2Xd& x, const PacketXcd& y, const PacketXcd& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcd pmul(const Packet2Xd& x, const PacketXcd& y) const {
    return PacketXcd(Eigen::internal::pmul<Packet2Xd>(x, pcast<PacketXcd, Packet2Xd>(y)));
  }
};

template <>
struct conj_helper<PacketXcd, Packet2Xd, false, false> {
  EIGEN_STRONG_INLINE PacketXcd pmadd(const PacketXcd& x, const Packet2Xd& y, const PacketXcd& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcd pmsub(const PacketXcd& x, const Packet2Xd& y, const PacketXcd& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcd pmul(const PacketXcd& x, const Packet2Xd& y) const {
    return PacketXcd(Eigen::internal::pmul<Packet2Xd>(pcast<PacketXcd, Packet2Xd>(x), y));
  }
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_RVV10_H
