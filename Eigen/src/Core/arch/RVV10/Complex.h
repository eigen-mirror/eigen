// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
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
  EIGEN_STRONG_INLINE explicit PacketXcf(const PacketMul1Xf& _real, const PacketMul1Xf& _imag) : real(_real), imag(_imag) {}
  EIGEN_STRONG_INLINE explicit PacketXcf(const PacketMul2Xf& a)
      : real(__riscv_vget_v_f32m2_f32m1(a, 0)), imag(__riscv_vget_v_f32m2_f32m1(a, 1)) {}
  PacketMul1Xf real;
  PacketMul1Xf imag;
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
    HasSign = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasLog = 0,
    HasSetLinear = 0
  };
};

template <>
struct unpacket_traits<PacketXcf> {
  typedef std::complex<float> type;
  typedef PacketXcf half;
  typedef PacketMul2Xf as_real;
  enum {
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE PacketXcf pcast<PacketMul2Xf, PacketXcf>(const PacketMul2Xf& a) {
  return PacketXcf(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcast<PacketXcf, PacketMul2Xf>(const PacketXcf& a) {
  return __riscv_vcreate_v_f32m1_f32m2(a.real, a.imag);
}

template <>
EIGEN_STRONG_INLINE PacketXcf pset1<PacketXcf>(const std::complex<float>& from) {
  PacketMul1Xf real = pset1<PacketMul1Xf>(from.real());
  PacketMul1Xf imag = pset1<PacketMul1Xf>(from.imag());
  return PacketXcf(real, imag);
}

template <>
EIGEN_STRONG_INLINE PacketXcf padd<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(padd<PacketMul1Xf>(a.real, b.real), padd<PacketMul1Xf>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf psub<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(psub<PacketMul1Xf>(a.real, b.real), psub<PacketMul1Xf>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pnegate(const PacketXcf& a) {
  return PacketXcf(pnegate<PacketMul1Xf>(a.real), pnegate<PacketMul1Xf>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pconj(const PacketXcf& a) {
  return PacketXcf(
      a.real, __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vxor_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a.imag),
                                                                       0x80000000, unpacket_traits<PacketMul1Xf>::size)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pmul<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  PacketMul1Xf v1 = pmul<PacketMul1Xf>(a.real, b.real);
  PacketMul1Xf v2 = pmul<PacketMul1Xf>(a.imag, b.imag);
  PacketMul1Xf v3 = pmul<PacketMul1Xf>(a.real, b.imag);
  PacketMul1Xf v4 = pmul<PacketMul1Xf>(a.imag, b.real);
  return PacketXcf(psub<PacketMul1Xf>(v1, v2), padd<PacketMul1Xf>(v3, v4));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pmadd<PacketXcf>(const PacketXcf& a, const PacketXcf& b, const PacketXcf& c) {
  PacketMul1Xf v1 = pmadd<PacketMul1Xf>(a.real, b.real, c.real);
  PacketMul1Xf v2 = pmul<PacketMul1Xf>(a.imag, b.imag);
  PacketMul1Xf v3 = pmadd<PacketMul1Xf>(a.real, b.imag, c.imag);
  PacketMul1Xf v4 = pmul<PacketMul1Xf>(a.imag, b.real);
  return PacketXcf(psub<PacketMul1Xf>(v1, v2), padd<PacketMul1Xf>(v3, v4));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pcmp_eq(const PacketXcf& a, const PacketXcf& b) {
  PacketMask32 eq_both = pand<PacketMask32>(pcmp_eq_mask(a.real, b.real), pcmp_eq_mask(a.imag, b.imag));
  PacketMul1Xf res = pselect(eq_both, ptrue<PacketMul1Xf>(a.real), pzero<PacketMul1Xf>(a.real));
  return PacketXcf(res, res);
}

template <>
EIGEN_STRONG_INLINE PacketXcf pand<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(pand<PacketMul1Xf>(a.real, b.real), pand<PacketMul1Xf>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf por<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(por<PacketMul1Xf>(a.real, b.real), por<PacketMul1Xf>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pxor<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(pxor<PacketMul1Xf>(a.real, b.real), pxor<PacketMul1Xf>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pandnot<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  return PacketXcf(pandnot<PacketMul1Xf>(a.real, b.real), pandnot<PacketMul1Xf>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pload<PacketXcf>(const std::complex<float>* from) {
  vfloat32m1x2_t res = __riscv_vlseg2e32_v_f32m1x2((const float*)from, unpacket_traits<PacketMul1Xf>::size);
  EIGEN_DEBUG_ALIGNED_LOAD return PacketXcf(__riscv_vget_v_f32m1x2_f32m1(res, 0), __riscv_vget_v_f32m1x2_f32m1(res, 1));
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploadu<PacketXcf>(const std::complex<float>* from) {
  vfloat32m1x2_t res = __riscv_vlseg2e32_v_f32m1x2((const float*)from, unpacket_traits<PacketMul1Xf>::size);
  EIGEN_DEBUG_UNALIGNED_LOAD return PacketXcf(__riscv_vget_v_f32m1x2_f32m1(res, 0),
                                              __riscv_vget_v_f32m1x2_f32m1(res, 1));
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploaddup<PacketXcf>(const std::complex<float>* from) {
  PacketMul1Xu real_idx = __riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xf>::size);
  real_idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(real_idx, 0xfffffffeu, unpacket_traits<PacketMul1Xf>::size), 2,
                                   unpacket_traits<PacketMul1Xf>::size);
  PacketMul1Xu imag_idx = __riscv_vadd_vx_u32m1(real_idx, sizeof(float), unpacket_traits<PacketMul1Xf>::size);
  // real_idx = 0 0 2*sizeof(float) 2*sizeof(float) 4*sizeof(float) 4*sizeof(float) ...
  return PacketXcf(__riscv_vloxei32_v_f32m1((const float*)from, real_idx, unpacket_traits<PacketMul1Xf>::size),
                   __riscv_vloxei32_v_f32m1((const float*)from, imag_idx, unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploadquad<PacketXcf>(const std::complex<float>* from) {
  PacketMul1Xu real_idx = __riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xf>::size);
  real_idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(real_idx, 0xfffffffcu, unpacket_traits<PacketMul1Xf>::size), 1,
                                   unpacket_traits<PacketMul1Xf>::size);
  PacketMul1Xu imag_idx = __riscv_vadd_vx_u32m1(real_idx, sizeof(float), unpacket_traits<PacketMul1Xf>::size);
  // real_idx = 0 0 2*sizeof(float) 2*sizeof(float) 4*sizeof(float) 4*sizeof(float) ...
  return PacketXcf(__riscv_vloxei32_v_f32m1((const float*)from, real_idx, unpacket_traits<PacketMul1Xf>::size),
                   __riscv_vloxei32_v_f32m1((const float*)from, imag_idx, unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE void pstore<std::complex<float> >(std::complex<float>* to, const PacketXcf& from) {
  vfloat32m1x2_t vx2 = __riscv_vundefined_f32m1x2();
  vx2 = __riscv_vset_v_f32m1_f32m1x2(vx2, 0, from.real);
  vx2 = __riscv_vset_v_f32m1_f32m1x2(vx2, 1, from.imag);
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vsseg2e32_v_f32m1x2((float*)to, vx2, unpacket_traits<PacketXcf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float>* to, const PacketXcf& from) {
  vfloat32m1x2_t vx2 = __riscv_vundefined_f32m1x2();
  vx2 = __riscv_vset_v_f32m1_f32m1x2(vx2, 0, from.real);
  vx2 = __riscv_vset_v_f32m1_f32m1x2(vx2, 1, from.imag);
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vsseg2e32_v_f32m1x2((float*)to, vx2, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXcf pgather<std::complex<float>, PacketXcf>(const std::complex<float>* from,
                                                                           Index stride) {
  vfloat32m1x2_t res =
      __riscv_vlsseg2e32_v_f32m1x2((const float*)from, 2 * stride * sizeof(float), unpacket_traits<PacketMul1Xf>::size);
  return PacketXcf(__riscv_vget_v_f32m1x2_f32m1(res, 0), __riscv_vget_v_f32m1x2_f32m1(res, 1));
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<std::complex<float>, PacketXcf>(std::complex<float>* to, const PacketXcf& from,
                                                                       Index stride) {
  vfloat32m1x2_t from_rvv_type = __riscv_vundefined_f32m1x2();
  from_rvv_type = __riscv_vset_v_f32m1_f32m1x2(from_rvv_type, 0, from.real);
  from_rvv_type = __riscv_vset_v_f32m1_f32m1x2(from_rvv_type, 1, from.imag);
  __riscv_vssseg2e32_v_f32m1x2((float*)to, 2 * stride * sizeof(float), from_rvv_type, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE std::complex<float> pfirst<PacketXcf>(const PacketXcf& a) {
  return std::complex<float>(pfirst<PacketMul1Xf>(a.real), pfirst<PacketMul1Xf>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf preverse(const PacketXcf& a) {
  return PacketXcf(preverse<PacketMul1Xf>(a.real), preverse<PacketMul1Xf>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pcplxflip<PacketXcf>(const PacketXcf& a) {
  return PacketXcf(a.imag, a.real);
}

template <>
EIGEN_STRONG_INLINE std::complex<float> predux<PacketXcf>(const PacketXcf& a) {
  return std::complex<float>(predux<PacketMul1Xf>(a.real), predux<PacketMul1Xf>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pdiv<PacketXcf>(const PacketXcf& a, const PacketXcf& b) {
  PacketXcf b_conj = pconj<PacketXcf>(b);
  PacketXcf dividend = pmul<PacketXcf>(a, b_conj);
  PacketMul1Xf divider = psub<PacketMul1Xf>(pmul<PacketMul1Xf>(b.real, b_conj.real), pmul<PacketMul1Xf>(b.imag, b_conj.imag));
  return PacketXcf(pdiv<PacketMul1Xf>(dividend.real, divider), pdiv<PacketMul1Xf>(dividend.imag, divider));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXcf, N>& kernel) {
  float buffer_real[unpacket_traits<PacketMul1Xf>::size * N];
  float buffer_imag[unpacket_traits<PacketMul1Xf>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer_real[i], N * sizeof(float), kernel.packet[i].real, unpacket_traits<PacketMul1Xf>::size);
    __riscv_vsse32(&buffer_imag[i], N * sizeof(float), kernel.packet[i].imag, unpacket_traits<PacketMul1Xf>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i].real =
        __riscv_vle32_v_f32m1(&buffer_real[i * unpacket_traits<PacketMul1Xf>::size], unpacket_traits<PacketMul1Xf>::size);
    kernel.packet[i].imag =
        __riscv_vle32_v_f32m1(&buffer_imag[i * unpacket_traits<PacketMul1Xf>::size], unpacket_traits<PacketMul1Xf>::size);
  }
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet psqrt_complex_rvv(const Packet& a) {
  typedef typename unpacket_traits<Packet>::type Scalar;
  typedef typename Scalar::value_type RealScalar;
  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename unpacket_traits<RealPacket>::packet_mask PacketMask;

  // Computes the principal sqrt of the complex numbers in the input.
  //
  // For example, for packets containing 2 complex numbers stored in
  // [real0, real1, imag0, imag1] format
  //    a = [a0, a1] = [x0, x1, y0, y1],
  // where x0 = real(a0), y0 = imag(a0) etc., this function returns
  //    b = [b0, b1] = [u0, u1, v0, v1],
  // such that b0^2 = a0, b1^2 = a1.
  //
  // To derive the formula for the complex square roots, let's consider the equation for
  // a single complex square root of the number x + i*y. We want to find real numbers
  // u and v such that
  //    (u + i*v)^2 = x + i*y  <=>
  //    u^2 - v^2 + i*2*u*v = x + i*v.
  // By equating the real and imaginary parts we get:
  //    u^2 - v^2 = x
  //    2*u*v = y.
  //
  // For x >= 0, this has the numerically stable solution
  //    u = sqrt(0.5 * (x + sqrt(x^2 + y^2)))
  //    v = 0.5 * (y / u)
  // and for x < 0,
  //    v = sign(y) * sqrt(0.5 * (-x + sqrt(x^2 + y^2)))
  //    u = 0.5 * (y / v)
  //
  //  To avoid unnecessary over- and underflow, we compute sqrt(x^2 + y^2) as
  //     l = max(|x|, |y|) * sqrt(1 + (min(|x|, |y|) / max(|x|, |y|))^2) ,

  // In the following, without lack of generality, we have annotated the code, assuming
  // that the input is a packet of 2 complex numbers.
  //
  // Step 1. Compute l = [l0, l1], where
  //    l0 = sqrt(x0^2 + y0^2),  l1 = sqrt(x1^2 + y1^2)
  // To avoid over- and underflow, we use the stable formula for each hypotenuse
  //    l0 = (min0 == 0 ? max0 : max0 * sqrt(1 + (min0/max0)**2)),
  // where max0 = max(|x0|, |y0|), min0 = min(|x0|, |y0|), and similarly for l1.

  Packet a_abs = Packet(pabs(a.real), pabs(a.imag));
  RealPacket a_max = pmax(a_abs.real, a_abs.imag);
  RealPacket a_min = pmin(a_abs.real, a_abs.imag);

  PacketMask a_min_zero_mask = pcmp_eq_mask(a_min, pzero(a_min));
  PacketMask a_max_zero_mask = pcmp_eq_mask(a_max, pzero(a_max));
  RealPacket r = pdiv(a_min, a_max);

  const RealPacket cst_one = pset1<RealPacket>(RealScalar(1));
  const RealPacket cst_true = ptrue<RealPacket>(cst_one);
  RealPacket l = pmul(a_max, psqrt(padd(cst_one, pmul(r, r))));
  // Set l to a_max if a_min is zero.
  l = pselect(a_min_zero_mask, a_max, l);

  // Step 2. Compute [rho0, rho1], where
  // rho0 = sqrt(0.5 * (l0 + |x0|)), rho1 =  sqrt(0.5 * (l1 + |x1|))
  // We don't care about the imaginary parts computed here. They will be overwritten later.
  const RealPacket cst_half = pset1<RealPacket>(RealScalar(0.5));
  RealPacket rho = psqrt(pmul(cst_half, padd(a_abs.real, l)));

  // Step 3. Compute [rho0, rho1, eta0, eta1], where
  // eta0 = (y0 / rho0) / 2, and eta1 = (y1 / rho1) / 2.
  // set eta = 0 of input is 0 + i0.
  RealPacket eta = pselect(a_max_zero_mask, pzero<RealPacket>(cst_one), pmul(cst_half, pdiv(a.imag, rho)));
  // Compute result for inputs with positive real part.
  Packet positive_real_result = Packet(rho, eta);

  // Step 4. Compute solution for inputs with negative real part:
  //         [|eta0| |eta1|, sign(y0)*rho0, sign(y1)*rho1]
  const RealPacket cst_imag_sign_mask = pset1<RealPacket>(RealScalar(-0.0));
  RealPacket imag_signs = pand(a.imag, cst_imag_sign_mask);
  Packet negative_real_result = Packet(pabs(eta), por(rho, imag_signs));

  // Step 5. Select solution branch based on the sign of the real parts.
  PacketMask negative_real_mask_half = pcmp_lt_mask(a.real, pzero(a.real));
  Packet result = Packet(pselect(negative_real_mask_half, negative_real_result.real, positive_real_result.real),
                         pselect(negative_real_mask_half, negative_real_result.imag, positive_real_result.imag));

  // Step 6. Handle special cases for infinities:
  // * If z is (x,+∞), the result is (+∞,+∞) even if x is NaN
  // * If z is (x,-∞), the result is (+∞,-∞) even if x is NaN
  // * If z is (-∞,y), the result is (0*|y|,+∞) for finite or NaN y
  // * If z is (+∞,y), the result is (+∞,0*|y|) for finite or NaN y
  const RealPacket cst_pos_inf = pset1<RealPacket>(NumTraits<RealScalar>::infinity());
  PacketMask is_real_inf = pcmp_eq_mask(a_abs.real, cst_pos_inf);
  // prepare packet of (+∞,0*|y|) or (0*|y|,+∞), depending on the sign of the infinite real part.
  const Packet cst_one_zero = pset1<Packet>(Scalar(RealScalar(1.0), RealScalar(0.0)));
  Packet real_inf_result = Packet(pmul(a_abs.real, cst_one_zero.real), pmul(a_abs.imag, cst_one_zero.imag));
  real_inf_result = Packet(pselect(negative_real_mask_half, real_inf_result.imag, real_inf_result.real),
                           pselect(negative_real_mask_half, real_inf_result.real, real_inf_result.imag));
  // prepare packet of (+∞,+∞) or (+∞,-∞), depending on the sign of the infinite imaginary part.
  PacketMask is_imag_inf = pcmp_eq_mask(a_abs.imag, cst_pos_inf);
  // unless otherwise specified, if either the real or imaginary component is nan, the entire result is nan
  result = Packet(pselect(pcmp_eq_mask(result.real, result.real), result.real, cst_true),
                  pselect(pcmp_eq_mask(result.imag, result.imag), result.imag, cst_true));

  result = Packet(pselect(is_real_inf, real_inf_result.real, result.real),
                  pselect(is_real_inf, real_inf_result.imag, result.imag));

  return Packet(pselect(is_imag_inf, cst_pos_inf, result.real), pselect(is_imag_inf, a.imag, result.imag));
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet plog_complex_rvv(const Packet& x) {
  typedef typename unpacket_traits<Packet>::type Scalar;
  typedef typename Scalar::value_type RealScalar;
  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename unpacket_traits<RealPacket>::packet_mask PacketMask;

  // log(sqrt(a^2 + b^2)), atan2(b, a)
  RealPacket xlogr = plog(psqrt(padd(pmul<RealPacket>(x.real, x.real), pmul<RealPacket>(x.imag, x.imag))));
  RealPacket ximg = patan2(x.imag, x.real);

  const RealPacket cst_pos_inf = pset1<RealPacket>(NumTraits<RealScalar>::infinity());
  RealPacket r_abs = pabs(x.real);
  RealPacket i_abs = pabs(x.imag);
  PacketMask is_r_pos_inf = pcmp_eq_mask(r_abs, cst_pos_inf);
  PacketMask is_i_pos_inf = pcmp_eq_mask(i_abs, cst_pos_inf);
  PacketMask is_any_inf = por(is_r_pos_inf, is_i_pos_inf);
  RealPacket xreal = pselect(is_any_inf, cst_pos_inf, xlogr);

  return Packet(xreal, ximg);
}

template <>
EIGEN_STRONG_INLINE PacketXcf psqrt<PacketXcf>(const PacketXcf& a) {
  return psqrt_complex_rvv<PacketXcf>(a);
}

template <>
EIGEN_STRONG_INLINE PacketXcf plog<PacketXcf>(const PacketXcf& a) {
  return plog_complex_rvv<PacketXcf>(a);
}

template <>
struct conj_helper<PacketMul2Xf, PacketXcf, false, false> {
  EIGEN_STRONG_INLINE PacketXcf pmadd(const PacketMul2Xf& x, const PacketXcf& y, const PacketXcf& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcf pmsub(const PacketMul2Xf& x, const PacketXcf& y, const PacketXcf& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcf pmul(const PacketMul2Xf& x, const PacketXcf& y) const {
    return PacketXcf(Eigen::internal::pmul<PacketMul2Xf>(x, pcast<PacketXcf, PacketMul2Xf>(y)));
  }
};

template <>
struct conj_helper<PacketXcf, PacketMul2Xf, false, false> {
  EIGEN_STRONG_INLINE PacketXcf pmadd(const PacketXcf& x, const PacketMul2Xf& y, const PacketXcf& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcf pmsub(const PacketXcf& x, const PacketMul2Xf& y, const PacketXcf& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcf pmul(const PacketXcf& x, const PacketMul2Xf& y) const {
    return PacketXcf(Eigen::internal::pmul<PacketMul2Xf>(pcast<PacketXcf, PacketMul2Xf>(x), y));
  }
};

/********************************* double ************************************/

struct PacketXcd {
  EIGEN_STRONG_INLINE PacketXcd() {}
  EIGEN_STRONG_INLINE explicit PacketXcd(const PacketMul1Xd& _real, const PacketMul1Xd& _imag) : real(_real), imag(_imag) {}
  EIGEN_STRONG_INLINE explicit PacketXcd(const PacketMul2Xd& a)
      : real(__riscv_vget_v_f64m2_f64m1(a, 0)), imag(__riscv_vget_v_f64m2_f64m1(a, 1)) {}
  PacketMul1Xd real;
  PacketMul1Xd imag;
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
    HasSign = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasLog = 0,
    HasSetLinear = 0
  };
};

template <>
struct unpacket_traits<PacketXcd> {
  typedef std::complex<double> type;
  typedef PacketXcd half;
  typedef PacketMul2Xd as_real;
  enum {
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE PacketXcd pcast<PacketMul2Xd, PacketXcd>(const PacketMul2Xd& a) {
  return PacketXcd(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcast<PacketXcd, PacketMul2Xd>(const PacketXcd& a) {
  return __riscv_vcreate_v_f64m1_f64m2(a.real, a.imag);
}

template <>
EIGEN_STRONG_INLINE PacketXcd pset1<PacketXcd>(const std::complex<double>& from) {
  PacketMul1Xd real = pset1<PacketMul1Xd>(from.real());
  PacketMul1Xd imag = pset1<PacketMul1Xd>(from.imag());
  return PacketXcd(real, imag);
}

template <>
EIGEN_STRONG_INLINE PacketXcd padd<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(padd<PacketMul1Xd>(a.real, b.real), padd<PacketMul1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd psub<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(psub<PacketMul1Xd>(a.real, b.real), psub<PacketMul1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pnegate(const PacketXcd& a) {
  return PacketXcd(pnegate<PacketMul1Xd>(a.real), pnegate<PacketMul1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pconj(const PacketXcd& a) {
  return PacketXcd(
      a.real, __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vxor_vx_u64m1(
                  __riscv_vreinterpret_v_f64m1_u64m1(a.imag), 0x8000000000000000, unpacket_traits<PacketMul1Xd>::size)));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmul<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  PacketMul1Xd v1 = pmul<PacketMul1Xd>(a.real, b.real);
  PacketMul1Xd v2 = pmul<PacketMul1Xd>(a.imag, b.imag);
  PacketMul1Xd v3 = pmul<PacketMul1Xd>(a.real, b.imag);
  PacketMul1Xd v4 = pmul<PacketMul1Xd>(a.imag, b.real);
  return PacketXcd(psub<PacketMul1Xd>(v1, v2), padd<PacketMul1Xd>(v3, v4));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pmadd<PacketXcd>(const PacketXcd& a, const PacketXcd& b, const PacketXcd& c) {
  PacketMul1Xd v1 = pmadd<PacketMul1Xd>(a.real, b.real, c.real);
  PacketMul1Xd v2 = pmul<PacketMul1Xd>(a.imag, b.imag);
  PacketMul1Xd v3 = pmadd<PacketMul1Xd>(a.real, b.imag, c.imag);
  PacketMul1Xd v4 = pmul<PacketMul1Xd>(a.imag, b.real);
  return PacketXcd(psub<PacketMul1Xd>(v1, v2), padd<PacketMul1Xd>(v3, v4));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pcmp_eq(const PacketXcd& a, const PacketXcd& b) {
  PacketMask64 eq_both = pand<PacketMask64>(pcmp_eq_mask(a.real, b.real), pcmp_eq_mask(a.imag, b.imag));
  PacketMul1Xd res = pselect(eq_both, ptrue<PacketMul1Xd>(a.real), pzero<PacketMul1Xd>(a.real));
  return PacketXcd(res, res);
}

template <>
EIGEN_STRONG_INLINE PacketXcd pand<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pand<PacketMul1Xd>(a.real, b.real), pand<PacketMul1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd por<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(por<PacketMul1Xd>(a.real, b.real), por<PacketMul1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pxor<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pxor<PacketMul1Xd>(a.real, b.real), pxor<PacketMul1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pandnot<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  return PacketXcd(pandnot<PacketMul1Xd>(a.real, b.real), pandnot<PacketMul1Xd>(a.imag, b.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pload<PacketXcd>(const std::complex<double>* from) {
  vfloat64m1x2_t res = __riscv_vlseg2e64_v_f64m1x2((const double*)from, unpacket_traits<PacketMul1Xd>::size);
  EIGEN_DEBUG_ALIGNED_LOAD return PacketXcd(__riscv_vget_v_f64m1x2_f64m1(res, 0), __riscv_vget_v_f64m1x2_f64m1(res, 1));
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploadu<PacketXcd>(const std::complex<double>* from) {
  vfloat64m1x2_t res = __riscv_vlseg2e64_v_f64m1x2((const double*)from, unpacket_traits<PacketMul1Xd>::size);
  EIGEN_DEBUG_UNALIGNED_LOAD return PacketXcd(__riscv_vget_v_f64m1x2_f64m1(res, 0),
                                              __riscv_vget_v_f64m1x2_f64m1(res, 1));
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploaddup<PacketXcd>(const std::complex<double>* from) {
  PacketMul1Xul real_idx = __riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xd>::size);
  real_idx =
      __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(real_idx, 0xfffffffffffffffeu, unpacket_traits<PacketMul1Xd>::size), 3,
                            unpacket_traits<PacketMul1Xd>::size);
  PacketMul1Xul imag_idx = __riscv_vadd_vx_u64m1(real_idx, sizeof(double), unpacket_traits<PacketMul1Xd>::size);
  // real_idx = 0 0 2*sizeof(double) 2*sizeof(double) 4*sizeof(double) 4*sizeof(double) ...
  return PacketXcd(__riscv_vloxei64_v_f64m1((const double*)from, real_idx, unpacket_traits<PacketMul1Xd>::size),
                   __riscv_vloxei64_v_f64m1((const double*)from, imag_idx, unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXcd ploadquad<PacketXcd>(const std::complex<double>* from) {
  PacketMul1Xul real_idx = __riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xd>::size);
  real_idx =
      __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(real_idx, 0xfffffffffffffffcu, unpacket_traits<PacketMul1Xd>::size), 2,
                            unpacket_traits<PacketMul1Xd>::size);
  PacketMul1Xul imag_idx = __riscv_vadd_vx_u64m1(real_idx, sizeof(double), unpacket_traits<PacketMul1Xd>::size);
  // real_idx = 0 0 2*sizeof(double) 2*sizeof(double) 4*sizeof(double) 4*sizeof(double) ...
  return PacketXcd(__riscv_vloxei64_v_f64m1((const double*)from, real_idx, unpacket_traits<PacketMul1Xd>::size),
                   __riscv_vloxei64_v_f64m1((const double*)from, imag_idx, unpacket_traits<PacketMul1Xd>::size));
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
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vsseg2e64_v_f64m1x2((double*)to, vx2, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXcd pgather<std::complex<double>, PacketXcd>(const std::complex<double>* from,
                                                                            Index stride) {
  vfloat64m1x2_t res =
      __riscv_vlsseg2e64_v_f64m1x2((const double*)from, 2 * stride * sizeof(double), unpacket_traits<PacketMul1Xd>::size);
  return PacketXcd(__riscv_vget_v_f64m1x2_f64m1(res, 0), __riscv_vget_v_f64m1x2_f64m1(res, 1));
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<std::complex<double>, PacketXcd>(std::complex<double>* to, const PacketXcd& from,
                                                                        Index stride) {
  vfloat64m1x2_t from_rvv_type = __riscv_vundefined_f64m1x2();
  from_rvv_type = __riscv_vset_v_f64m1_f64m1x2(from_rvv_type, 0, from.real);
  from_rvv_type = __riscv_vset_v_f64m1_f64m1x2(from_rvv_type, 1, from.imag);
  __riscv_vssseg2e64_v_f64m1x2((double*)to, 2 * stride * sizeof(double), from_rvv_type,
                               unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> pfirst<PacketXcd>(const PacketXcd& a) {
  return std::complex<double>(pfirst<PacketMul1Xd>(a.real), pfirst<PacketMul1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd preverse(const PacketXcd& a) {
  return PacketXcd(preverse<PacketMul1Xd>(a.real), preverse<PacketMul1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pcplxflip<PacketXcd>(const PacketXcd& a) {
  return PacketXcd(a.imag, a.real);
}

template <>
EIGEN_STRONG_INLINE std::complex<double> predux<PacketXcd>(const PacketXcd& a) {
  return std::complex<double>(predux<PacketMul1Xd>(a.real), predux<PacketMul1Xd>(a.imag));
}

template <>
EIGEN_STRONG_INLINE PacketXcd pdiv<PacketXcd>(const PacketXcd& a, const PacketXcd& b) {
  PacketXcd b_conj = pconj<PacketXcd>(b);
  PacketXcd dividend = pmul<PacketXcd>(a, b_conj);
  PacketMul1Xd divider = psub<PacketMul1Xd>(pmul<PacketMul1Xd>(b.real, b_conj.real), pmul<PacketMul1Xd>(b.imag, b_conj.imag));
  return PacketXcd(pdiv<PacketMul1Xd>(dividend.real, divider), pdiv<PacketMul1Xd>(dividend.imag, divider));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXcd, N>& kernel) {
  double buffer_real[unpacket_traits<PacketMul1Xd>::size * N];
  double buffer_imag[unpacket_traits<PacketMul1Xd>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer_real[i], N * sizeof(double), kernel.packet[i].real, unpacket_traits<PacketMul1Xd>::size);
    __riscv_vsse64(&buffer_imag[i], N * sizeof(double), kernel.packet[i].imag, unpacket_traits<PacketMul1Xd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i].real =
        __riscv_vle64_v_f64m1(&buffer_real[i * unpacket_traits<PacketMul1Xd>::size], unpacket_traits<PacketMul1Xd>::size);
    kernel.packet[i].imag =
        __riscv_vle64_v_f64m1(&buffer_imag[i * unpacket_traits<PacketMul1Xd>::size], unpacket_traits<PacketMul1Xd>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketXcd psqrt<PacketXcd>(const PacketXcd& a) {
  return psqrt_complex_rvv<PacketXcd>(a);
}

template <>
EIGEN_STRONG_INLINE PacketXcd plog<PacketXcd>(const PacketXcd& a) {
  return plog_complex_rvv<PacketXcd>(a);
}

template <>
struct conj_helper<PacketMul2Xd, PacketXcd, false, false> {
  EIGEN_STRONG_INLINE PacketXcd pmadd(const PacketMul2Xd& x, const PacketXcd& y, const PacketXcd& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcd pmsub(const PacketMul2Xd& x, const PacketXcd& y, const PacketXcd& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcd pmul(const PacketMul2Xd& x, const PacketXcd& y) const {
    return PacketXcd(Eigen::internal::pmul<PacketMul2Xd>(x, pcast<PacketXcd, PacketMul2Xd>(y)));
  }
};

template <>
struct conj_helper<PacketXcd, PacketMul2Xd, false, false> {
  EIGEN_STRONG_INLINE PacketXcd pmadd(const PacketXcd& x, const PacketMul2Xd& y, const PacketXcd& c) const {
    return padd(c, this->pmul(x, y));
  }
  EIGEN_STRONG_INLINE PacketXcd pmsub(const PacketXcd& x, const PacketMul2Xd& y, const PacketXcd& c) const {
    return psub(this->pmul(x, y), c);
  }
  EIGEN_STRONG_INLINE PacketXcd pmul(const PacketXcd& x, const PacketMul2Xd& y) const {
    return PacketXcd(Eigen::internal::pmul<PacketMul2Xd>(pcast<PacketXcd, PacketMul2Xd>(x), y));
  }
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_RVV10_H
