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

EIGEN_STRONG_INLINE Packet1Xf preal(const PacketXcf& a) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vncvt_x_x_w_u32m1(__riscv_vreinterpret_v_f32m2_u64m2(a.v),
      unpacket_traits<Packet2Xl>::size));
}

EIGEN_STRONG_INLINE Packet1Xf pimag(const PacketXcf& a) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vnsrl_wx_u32m1(__riscv_vreinterpret_v_f32m2_u64m2(a.v), 32,
      unpacket_traits<Packet2Xl>::size));
}

EIGEN_STRONG_INLINE Packet2Xf preal2(const PacketXcf& a) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i16m1_b16(__riscv_vmv_v_x_i16m1(0xaaaa,
      unpacket_traits<Packet1Xs>::size));
  return __riscv_vfslide1up_vf_f32m2_tumu(mask, a.v, a.v, 0.0f, unpacket_traits<Packet2Xf>::size);
}

EIGEN_STRONG_INLINE Packet2Xf pimag2(const PacketXcf& a) {
  const PacketMask16 mask = __riscv_vreinterpret_v_i16m1_b16(__riscv_vmv_v_x_i16m1(0x5555,
      unpacket_traits<Packet1Xs>::size));
  return __riscv_vfslide1down_vf_f32m2_tumu(mask, a.v, a.v, 0.0f, unpacket_traits<Packet2Xf>::size);
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
  return PacketXcf(pmadd<Packet2Xf>(pimag2(a), pcplxflip<PacketXcf>(pconj<PacketXcf>(b)).v,
     pmul<Packet2Xf>(preal2(a), b.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pmadd<PacketXcf>(const PacketXcf& a, const PacketXcf& b, const PacketXcf& c) {
  return PacketXcf(pmadd<Packet2Xf>(pimag2(a), pcplxflip<PacketXcf>(pconj<PacketXcf>(b)).v,
     pmadd<Packet2Xf>(preal2(a), b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pmsub<PacketXcf>(const PacketXcf& a, const PacketXcf& b, const PacketXcf& c) {
  return PacketXcf(pmadd<Packet2Xf>(pimag2(a), pcplxflip<PacketXcf>(pconj<PacketXcf>(b)).v,
     pmsub<Packet2Xf>(preal2(a), b.v, c.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pcmp_eq(const PacketXcf& a, const PacketXcf& b) {
  Packet2Xl c = __riscv_vundefined_i64m2();
  PacketMask32 mask_r = __riscv_vmfeq_vv_f32m1_b32(preal(a), preal(b), unpacket_traits<Packet1Xf>::size);
  PacketMask32 mask_i = __riscv_vmfeq_vv_f32m1_b32(pimag(a), pimag(b), unpacket_traits<Packet1Xf>::size);
  return PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(__riscv_vmerge_vvm_i64m2(pzero<Packet2Xl>(c), ptrue<Packet2Xl>(c),
      __riscv_vmand_mm_b32(mask_r, mask_i, unpacket_traits<Packet1Xf>::size), unpacket_traits<Packet2Xl>::size)));
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
  Packet2Xl res = pload<Packet2Xl>(reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)));
  EIGEN_DEBUG_ALIGNED_LOAD return PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(res));
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploadu<PacketXcf>(const std::complex<float>* from) {
  Packet2Xl res = ploadu<Packet2Xl>(reinterpret_cast<const numext::int64_t *>(reinterpret_cast<const void *>(from)));
  EIGEN_DEBUG_UNALIGNED_LOAD return PacketXcf(__riscv_vreinterpret_v_i64m2_f32m2(res));
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
  const Packet2Xl res = __riscv_vreinterpret_v_f32m2_i64m2(from.v);
  EIGEN_DEBUG_ALIGNED_STORE pstore<numext::int64_t>(reinterpret_cast<numext::int64_t *>(reinterpret_cast<void *>(to)), res);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float>* to, const PacketXcf& from) {
  const Packet2Xl res = __riscv_vreinterpret_v_f32m2_i64m2(from.v);
  EIGEN_DEBUG_UNALIGNED_STORE pstoreu<numext::int64_t>(reinterpret_cast<numext::int64_t *>(reinterpret_cast<void *>(to)), res);
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
  float real = pfirst<Packet2Xf>(a.v);
  float imag = pfirst<Packet2Xf>(__riscv_vfslide1down_vf_f32m2(a.v, 0.0f, unpacket_traits<Packet2Xi>::size));
  return std::complex<float>(real, imag);
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

#if 0
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

template <>
EIGEN_STRONG_INLINE PacketXcd pset1<PacketXcd>(const std::complex<double>& from) {
  Packet1Xf res;
  res = __riscv_vreinterpret_v_u64m1_u32m1(pset1<Packet1Xl>((const long *)&from));
  return PacketXcd(res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd padd<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(padd<Packet1Xd>(a.real, b.real), padd<Packet1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd psub<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(psub<Packet1Xd>(a.real, b.real), psub<Packet1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pnegate(const PacketXcd& a) {
  return PacketXcd(pnegate<Packet1Xd>(a.real), pnegate<Packet1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pconj(const PacketXcd& a) {
  return PacketXcd(
      a.real, __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vxor_vx_u64m1(
                  __riscv_vreinterpret_v_f64m1_u64m1(a.imag), 0x8000000000000000, unpacket_traits<Packet1Xd>::size)));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmul<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  Packet1Xd v1 = pmul<Packet1Xd>(a.real, b.real);
  Packet1Xd v2 = pmul<Packet1Xd>(a.imag, b.imag);
  Packet1Xd v3 = pmul<Packet1Xd>(a.real, b.imag);
  Packet1Xd v4 = pmul<Packet1Xd>(a.imag, b.real);
  return PacketXcd(psub<Packet1Xd>(v1, v2), padd<Packet1Xd>(v3, v4));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmadd<PacketXcd>(const PacketXcd& a, const PacketXcd& b, const PacketXcd& c) {
  Packet1Xd v1 = pmadd<Packet1Xd>(a.real, b.real, c.real);
  Packet1Xd v2 = pmul<Packet1Xd>(a.imag, b.imag);
  Packet1Xd v3 = pmadd<Packet1Xd>(a.real, b.imag, c.imag);
  Packet1Xd v4 = pmul<Packet1Xd>(a.imag, b.real);
  return PacketXcd(psub<Packet1Xd>(v1, v2), padd<Packet1Xd>(v3, v4));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmsub<PacketXcd>(const PacketXcd& a, const PacketXcd& b, const PacketXcd& c) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXcd pcmp_eq(const PacketXcd& a, const PacketXcd& b) {
  PacketMask64 eq_both = pand<PacketMask64>(pcmp_eq_mask(a.real, b.real), pcmp_eq_mask(a.imag, b.imag));
  Packet1Xd res = pselect(eq_both, ptrue<Packet1Xd>(a.real), pzero<Packet1Xd>(a.real));
  return PacketXcd(res, res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd pand<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pand<Packet1Xd>(a.real, b.real), pand<Packet1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd por<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(por<Packet1Xd>(a.real, b.real), por<Packet1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pxor<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pxor<Packet1Xd>(a.real, b.real), pxor<Packet1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pandnot<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pandnot<Packet1Xd>(a.real, b.real), pandnot<Packet1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pload<PacketXcd>(const std::complex<double>* from) {
  vfloat64m1x2_t res = __riscv_vlseg2e64_v_f64m1x2((const double*)from, unpacket_traits<Packet1Xd>::size);
  EIGEN_DEBUG_ALIGNED_LOAD return PacketXcd(__riscv_vget_v_f64m1x2_f64m1(res, 0), __riscv_vget_v_f64m1x2_f64m1(res, 1));
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploadu<PacketXcd>(const std::complex<double>* from) {
  vfloat64m1x2_t res = __riscv_vlseg2e64_v_f64m1x2((const double*)from, unpacket_traits<Packet1Xd>::size);
  EIGEN_DEBUG_UNALIGNED_LOAD return PacketXcd(__riscv_vget_v_f64m1x2_f64m1(res, 0),
                                              __riscv_vget_v_f64m1x2_f64m1(res, 1));
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploaddup<PacketXcd>(const std::complex<double>* from) {
  Packet1Xul real_idx = __riscv_vid_v_u64m1(unpacket_traits<Packet1Xd>::size);
  real_idx =
      __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(real_idx, 0xfffffffffffffffeu, unpacket_traits<Packet1Xd>::size), 3,
                            unpacket_traits<Packet1Xd>::size);
  Packet1Xul imag_idx = __riscv_vadd_vx_u64m1(real_idx, sizeof(double), unpacket_traits<Packet1Xd>::size);
  // real_idx = 0 0 2*sizeof(double) 2*sizeof(double) 4*sizeof(double) 4*sizeof(double) ...
  return PacketXcd(__riscv_vloxei64_v_f64m1((const double*)from, real_idx, unpacket_traits<Packet1Xd>::size),
                   __riscv_vloxei64_v_f64m1((const double*)from, imag_idx, unpacket_traits<Packet1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploadquad<PacketXcd>(const std::complex<double>* from) {
  Packet1Xul real_idx = __riscv_vid_v_u64m1(unpacket_traits<Packet1Xd>::size);
  real_idx =
      __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(real_idx, 0xfffffffffffffffcu, unpacket_traits<Packet1Xd>::size), 2,
                            unpacket_traits<Packet1Xd>::size);
  Packet1Xul imag_idx = __riscv_vadd_vx_u64m1(real_idx, sizeof(double), unpacket_traits<Packet1Xd>::size);
  // real_idx = 0 0 2*sizeof(double) 2*sizeof(double) 4*sizeof(double) 4*sizeof(double) ...
  return PacketXcd(__riscv_vloxei64_v_f64m1((const double*)from, real_idx, unpacket_traits<Packet1Xd>::size),
                   __riscv_vloxei64_v_f64m1((const double*)from, imag_idx, unpacket_traits<Packet1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE void pstore<std::complex<double> >(std::complex<double>* to, const PacketXcd& from) {
  vfloat64m1x2_t vx2 = __riscv_vundefined_f64m1x2();
  vx2 = __riscv_vset_v_f64m1_f64m1x2(vx2, 0, from.real);
  vx2 = __riscv_vset_v_f64m1_f64m1x2(vx2, 1, from.imag);
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vsseg2e64_v_f64m1x2((double*)to, vx2, unpacket_traits<PacketXcd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double>* to, const PacketXcd& from) {
  vfloat64m1x2_t vx2 = __riscv_vundefined_f64m1x2();
  vx2 = __riscv_vset_v_f64m1_f64m1x2(vx2, 0, from.real);
  vx2 = __riscv_vset_v_f64m1_f64m1x2(vx2, 1, from.imag);
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vsseg2e64_v_f64m1x2((double*)to, vx2, unpacket_traits<Packet1Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketXcd pgather<std::complex<double>, PacketXcd>(const std::complex<double>* from,
                                                                            Index stride) {
  vfloat64m1x2_t res =
      __riscv_vlsseg2e64_v_f64m1x2((const double*)from, 2 * stride * sizeof(double), unpacket_traits<Packet1Xd>::size);
  return PacketXcd(__riscv_vget_v_f64m1x2_f64m1(res, 0), __riscv_vget_v_f64m1x2_f64m1(res, 1));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pscatter<std::complex<double>, PacketXcd>(std::complex<double>* to, const PacketXcd& from,
                                                                        Index stride) {
  vfloat64m1x2_t from_rvv_type = __riscv_vundefined_f64m1x2();
  from_rvv_type = __riscv_vset_v_f64m1_f64m1x2(from_rvv_type, 0, from.real);
  from_rvv_type = __riscv_vset_v_f64m1_f64m1x2(from_rvv_type, 1, from.imag);
  __riscv_vssseg2e64_v_f64m1x2((double*)to, 2 * stride * sizeof(double), from_rvv_type,
                               unpacket_traits<Packet1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> pfirst<PacketXcd>(const PacketXcd& a) {
  return std::complex<double>(pfirst<Packet1Xd>(a.real), pfirst<Packet1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd preverse(const PacketXcd& a) {
  return PacketXcd(preverse<Packet1Xd>(a.real), preverse<Packet1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pcplxflip<PacketXcd>(const PacketXcd& a) {
  return PacketXcd(a.imag, a.real);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> predux<PacketXcd>(const PacketXcd& a) {
  return std::complex<double>(predux<Packet1Xd>(a.real), predux<Packet1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pdiv<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  PacketXcd b_conj = pconj<PacketXcd>(b);
  PacketXcd dividend = pmul<PacketXcd>(a, b_conj);
  Packet1Xd divider = psub<Packet1Xd>(pmul<Packet1Xd>(b.real, b_conj.real), pmul<Packet1Xd>(b.imag, b_conj.imag));
  return PacketXcd(pdiv<Packet1Xd>(dividend.real, divider), pdiv<Packet1Xd>(dividend.imag, divider));
}

template <int N>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void ptranspose(PacketBlock<PacketXcd, N>& kernel) {
  double buffer_real[unpacket_traits<Packet1Xd>::size * N];
  double buffer_imag[unpacket_traits<Packet1Xd>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer_real[i], N * sizeof(double), kernel.packet[i].real, unpacket_traits<Packet1Xd>::size);
    __riscv_vsse64(&buffer_imag[i], N * sizeof(double), kernel.packet[i].imag, unpacket_traits<Packet1Xd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i].real =
        __riscv_vle64_v_f64m1(&buffer_real[i * unpacket_traits<Packet1Xd>::size], unpacket_traits<Packet1Xd>::size);
    kernel.packet[i].imag =
        __riscv_vle64_v_f64m1(&buffer_imag[i * unpacket_traits<Packet1Xd>::size], unpacket_traits<Packet1Xd>::size);
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

template <>
EIGEN_STRONG_INLINE PacketXcd pexp<PacketXcd>(const PacketXcd& a) {
  return pexp_complex(a);
}

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
#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_RVV10_H
