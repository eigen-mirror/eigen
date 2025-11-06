// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_RVV10_H
#define EIGEN_PACKET_MATH_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {
#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif

#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32

template <typename Scalar, std::size_t VectorLength, std::size_t VectorLMul>
struct rvv_packet_size_selector {
  enum { size = VectorLength * VectorLMul / (sizeof(Scalar) * CHAR_BIT) };
};

template <std::size_t VectorLength, std::size_t VectorLMul>
struct rvv_packet_alignment_selector {
  enum {
    alignment =
        (VectorLength * VectorLMul) >= 1024
            ? Aligned128
            : ((VectorLength * VectorLMul) >= 512 ? Aligned64
                                                  : ((VectorLength * VectorLMul) >= 256 ? Aligned32 : Aligned16))
  };
};

typedef vbool64_t PacketMask64;
typedef vbool32_t PacketMask32;
typedef vbool16_t PacketMask16;
typedef vbool8_t PacketMask8;
typedef vbool4_t PacketMask4;

/********************************* int32 **************************************/
typedef eigen_packet_wrapper<vint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 0> PacketMul1Xi;
typedef eigen_packet_wrapper<vuint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 1> PacketMul1Xu;

typedef eigen_packet_wrapper<vint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 2>
    PacketMul2Xi;
typedef eigen_packet_wrapper<vuint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 3>
    PacketMul2Xu;

typedef eigen_packet_wrapper<vint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 4>
    PacketMul4Xi;
typedef eigen_packet_wrapper<vuint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 5>
    PacketMul4Xu;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef PacketMul1Xi PacketXi;
typedef PacketMul1Xu PacketXu;

template <>
struct packet_traits<numext::int32_t> : default_packet_traits {
  typedef PacketMul1Xi type;
  typedef PacketMul1Xi half;  // Half not implemented yet
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef PacketMul2Xi PacketXi;
typedef PacketMul2Xu PacketXu;

template <>
struct packet_traits<numext::int32_t> : default_packet_traits {
  typedef PacketMul2Xi type;
  typedef PacketMul1Xi half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL, 2>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef PacketMul4Xi PacketXi;
typedef PacketMul4Xu PacketXu;

template <>
struct packet_traits<numext::int32_t> : default_packet_traits {
  typedef PacketMul4Xi type;
  typedef PacketMul2Xi half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL, 4>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};
#endif

template <>
struct unpacket_traits<PacketMul1Xi> {
  typedef numext::int32_t type;
  typedef PacketMul1Xi half;  // Half not yet implemented
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 1>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul2Xi> {
  typedef numext::int32_t type;
  typedef PacketMul1Xi half;
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL, 2>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul4Xi> {
  typedef numext::int32_t type;
  typedef PacketMul2Xi half;
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV64_RVV_VL, 4>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 4>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int32_t>(const numext::int32_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

/********************************* PacketMul1Xi ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pset1<PacketMul1Xi>(const numext::int32_t& from) {
  return __riscv_vmv_v_x_i32m1(from, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi plset<PacketMul1Xi>(const numext::int32_t& a) {
  PacketMul1Xi idx = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xi>::size));
  return __riscv_vadd_vx_i32m1(idx, a, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pzero<PacketMul1Xi>(const PacketMul1Xi& /*a*/) {
  return __riscv_vmv_v_x_i32m1(0, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi padd<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vadd_vv_i32m1(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi psub<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pnegate(const PacketMul1Xi& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pconj(const PacketMul1Xi& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pmul<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pdiv<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pmadd(const PacketMul1Xi& a, const PacketMul1Xi& b, const PacketMul1Xi& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pmsub(const PacketMul1Xi& a, const PacketMul1Xi& b, const PacketMul1Xi& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pnmadd(const PacketMul1Xi& a, const PacketMul1Xi& b, const PacketMul1Xi& c) {
  return __riscv_vnmsub_vv_i32m1(a, b, c, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pnmsub(const PacketMul1Xi& a, const PacketMul1Xi& b, const PacketMul1Xi& c) {
  return __riscv_vnmsub_vv_i32m1(a, b, pnegate(c), unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pmin<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pmax<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pcmp_le<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  PacketMask32 mask = __riscv_vmsle_vv_i32m1_b32(a, b, unpacket_traits<PacketMul1Xi>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pcmp_lt<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  PacketMask32 mask = __riscv_vmslt_vv_i32m1_b32(a, b, unpacket_traits<PacketMul1Xi>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pcmp_eq<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  PacketMask32 mask = __riscv_vmseq_vv_i32m1_b32(a, b, unpacket_traits<PacketMul1Xi>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi ptrue<PacketMul1Xi>(const PacketMul1Xi& /*a*/) {
  return __riscv_vmv_v_x_i32m1(0xffffffffu, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pand<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vand_vv_i32m1(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi por<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vor_vv_i32m1(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pxor<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vxor_vv_i32m1(a, b, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pandnot<PacketMul1Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vand_vv_i32m1(a, __riscv_vnot_v_i32m1(b, unpacket_traits<PacketMul1Xi>::size),
                               unpacket_traits<PacketMul1Xi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xi parithmetic_shift_right(PacketMul1Xi a) {
  return __riscv_vsra_vx_i32m1(a, N, unpacket_traits<PacketMul1Xi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xi plogical_shift_right(PacketMul1Xi a) {
  return __riscv_vreinterpret_i32m1(
      __riscv_vsrl_vx_u32m1(__riscv_vreinterpret_u32m1(a), N, unpacket_traits<PacketMul1Xi>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xi plogical_shift_left(PacketMul1Xi a) {
  return __riscv_vsll_vx_i32m1(a, N, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pload<PacketMul1Xi>(const numext::int32_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_i32m1(from, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi ploadu<PacketMul1Xi>(const numext::int32_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_i32m1(from, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi ploaddup<PacketMul1Xi>(const numext::int32_t* from) {
  PacketMul1Xu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xi>::size);
  idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(idx, 0xfffffffeu, unpacket_traits<PacketMul1Xi>::size), 1,
                              unpacket_traits<PacketMul1Xi>::size);
  // idx = 0 0 sizeof(int32_t) sizeof(int32_t) 2*sizeof(int32_t) 2*sizeof(int32_t) ...
  return __riscv_vloxei32_v_i32m1(from, idx, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi ploadquad<PacketMul1Xi>(const numext::int32_t* from) {
  PacketMul1Xu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xi>::size);
  idx = __riscv_vand_vx_u32m1(idx, 0xfffffffcu, unpacket_traits<PacketMul1Xi>::size);
  return __riscv_vloxei32_v_i32m1(from, idx, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int32_t>(numext::int32_t* to, const PacketMul1Xi& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_i32m1(to, from, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int32_t>(numext::int32_t* to, const PacketMul1Xi& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_i32m1(to, from, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul1Xi pgather<numext::int32_t, PacketMul1Xi>(const numext::int32_t* from, Index stride) {
  return __riscv_vlse32_v_i32m1(from, stride * sizeof(numext::int32_t), unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int32_t, PacketMul1Xi>(numext::int32_t* to, const PacketMul1Xi& from,
                                                                  Index stride) {
  __riscv_vsse32(to, stride * sizeof(numext::int32_t), from, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t pfirst<PacketMul1Xi>(const PacketMul1Xi& a) {
  return __riscv_vmv_x_s_i32m1_i32(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi preverse(const PacketMul1Xi& a) {
  PacketMul1Xu idx = __riscv_vrsub_vx_u32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xi>::size),
                                        unpacket_traits<PacketMul1Xi>::size - 1, unpacket_traits<PacketMul1Xi>::size);
  return __riscv_vrgather_vv_i32m1(a, idx, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pabs(const PacketMul1Xi& a) {
  PacketMul1Xi mask = __riscv_vsra_vx_i32m1(a, 31, unpacket_traits<PacketMul1Xi>::size);
  return __riscv_vsub_vv_i32m1(__riscv_vxor_vv_i32m1(a, mask, unpacket_traits<PacketMul1Xi>::size), mask,
                               unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux<PacketMul1Xi>(const PacketMul1Xi& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i32m1_i32m1(a, __riscv_vmv_v_x_i32m1(0, unpacket_traits<PacketMul1Xi>::size),
                                                      unpacket_traits<PacketMul1Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_mul<PacketMul1Xi>(const PacketMul1Xi& a) {
  // Multiply the vector by its reverse
  PacketMul1Xi prod = __riscv_vmul_vv_i32m1(preverse(a), a, unpacket_traits<PacketMul1Xi>::size);
  PacketMul1Xi half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_i32m1(prod, 8, unpacket_traits<PacketMul1Xi>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketMul1Xi>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_i32m1(prod, 4, unpacket_traits<PacketMul1Xi>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketMul1Xi>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_i32m1(prod, 2, unpacket_traits<PacketMul1Xi>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketMul1Xi>::size);
  }
  // Last reduction
  half_prod = __riscv_vslidedown_vx_i32m1(prod, 1, unpacket_traits<PacketMul1Xi>::size);
  prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketMul1Xi>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_min<PacketMul1Xi>(const PacketMul1Xi& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i32m1_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::max)(), unpacket_traits<PacketMul1Xi>::size),
      unpacket_traits<PacketMul1Xi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_max<PacketMul1Xi>(const PacketMul1Xi& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i32m1_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::min)(), unpacket_traits<PacketMul1Xi>::size),
      unpacket_traits<PacketMul1Xi>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul1Xi, N>& kernel) {
  numext::int32_t buffer[unpacket_traits<PacketMul1Xi>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(numext::int32_t), kernel.packet[i], unpacket_traits<PacketMul1Xi>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_i32m1(&buffer[i * unpacket_traits<PacketMul1Xi>::size], unpacket_traits<PacketMul1Xi>::size);
  }
}

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

/********************************* float32 ************************************/

typedef eigen_packet_wrapper<vfloat32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 6> PacketMul1Xf;
typedef eigen_packet_wrapper<vfloat32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 7>
    PacketMul2Xf;
typedef eigen_packet_wrapper<vfloat32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 8>
    PacketMul4Xf;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef PacketMul1Xf PacketXf;

template <>
struct packet_traits<float> : default_packet_traits {
  typedef PacketMul1Xf type;
  typedef PacketMul1Xf half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef PacketMul2Xf PacketXf;

template <>
struct packet_traits<float> : default_packet_traits {
  typedef PacketMul2Xf type;
  typedef PacketMul1Xf half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 2>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef PacketMul4Xf PacketXf;

template <>
struct packet_traits<float> : default_packet_traits {
  typedef PacketMul4Xf type;
  typedef PacketMul2Xf half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 4>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};
#endif

template <>
struct unpacket_traits<PacketMul1Xf> {
  typedef float type;
  typedef PacketMul1Xf half;  // Half not yet implemented
  typedef PacketMul1Xi integer_packet;
  typedef numext::uint8_t mask_t;
  typedef PacketMask32 packet_mask;

  enum {
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 1>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul2Xf> {
  typedef float type;
  typedef PacketMul1Xf half;
  typedef PacketMul2Xi integer_packet;
  typedef numext::uint8_t mask_t;
  typedef PacketMask16 packet_mask;

  enum {
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 2>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul4Xf> {
  typedef float type;
  typedef PacketMul2Xf half;
  typedef PacketMul4Xi integer_packet;
  typedef numext::uint8_t mask_t;
  typedef PacketMask8 packet_mask;

  enum {
    size = rvv_packet_size_selector<float, EIGEN_RISCV64_RVV_VL, 4>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 4>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

/********************************* PacketMul1Xf ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul1Xf ptrue<PacketMul1Xf>(const PacketMul1Xf& /*a*/) {
  return __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(0xffffffffu, unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pzero<PacketMul1Xf>(const PacketMul1Xf& /*a*/) {
  return __riscv_vfmv_v_f_f32m1(0.0f, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pabs(const PacketMul1Xf& a) {
  return __riscv_vfabs_v_f32m1(a, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pset1<PacketMul1Xf>(const float& from) {
  return __riscv_vfmv_v_f_f32m1(from, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pset1frombits<PacketMul1Xf>(numext::uint32_t from) {
  return __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(from, unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf plset<PacketMul1Xf>(const float& a) {
  PacketMul1Xf idx = __riscv_vfcvt_f_x_v_f32m1(
      __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xi>::size)),
      unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vfadd_vf_f32m1(idx, a, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf padd<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vfadd_vv_f32m1(a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf psub<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vfsub_vv_f32m1(a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pnegate(const PacketMul1Xf& a) {
  return __riscv_vfneg_v_f32m1(a, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pconj(const PacketMul1Xf& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmul<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vfmul_vv_f32m1(a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pdiv<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vfdiv_vv_f32m1(a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmadd(const PacketMul1Xf& a, const PacketMul1Xf& b, const PacketMul1Xf& c) {
  return __riscv_vfmadd_vv_f32m1(a, b, c, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmsub(const PacketMul1Xf& a, const PacketMul1Xf& b, const PacketMul1Xf& c) {
  return __riscv_vfmsub_vv_f32m1(a, b, c, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pnmadd(const PacketMul1Xf& a, const PacketMul1Xf& b, const PacketMul1Xf& c) {
  return __riscv_vfnmsub_vv_f32m1(a, b, c, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pnmsub(const PacketMul1Xf& a, const PacketMul1Xf& b, const PacketMul1Xf& c) {
  return __riscv_vfnmadd_vv_f32m1(a, b, c, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmin<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  PacketMul1Xf nans = __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul1Xf>::size);
  PacketMask32 mask = __riscv_vmfeq_vv_f32m1_b32(a, a, unpacket_traits<PacketMul1Xf>::size);
  PacketMask32 mask2 = __riscv_vmfeq_vv_f32m1_b32(b, b, unpacket_traits<PacketMul1Xf>::size);
  mask = __riscv_vmand_mm_b32(mask, mask2, unpacket_traits<PacketMul1Xf>::size);

  return __riscv_vfmin_vv_f32m1_tumu(mask, nans, a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmin<PropagateNaN, PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return pmin<PacketMul1Xf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmin<PropagateNumbers, PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vfmin_vv_f32m1(a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmax<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  PacketMul1Xf nans = __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul1Xf>::size);
  PacketMask32 mask = __riscv_vmfeq_vv_f32m1_b32(a, a, unpacket_traits<PacketMul1Xf>::size);
  PacketMask32 mask2 = __riscv_vmfeq_vv_f32m1_b32(b, b, unpacket_traits<PacketMul1Xf>::size);
  mask = __riscv_vmand_mm_b32(mask, mask2, unpacket_traits<PacketMul1Xf>::size);

  return __riscv_vfmax_vv_f32m1_tumu(mask, nans, a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmax<PropagateNaN, PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return pmax<PacketMul1Xf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pmax<PropagateNumbers, PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vfmax_vv_f32m1(a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pcmp_le<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  PacketMask32 mask = __riscv_vmfle_vv_f32m1_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketMul1Xf>(a), ptrue<PacketMul1Xf>(a), mask, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pcmp_lt<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  PacketMask32 mask = __riscv_vmflt_vv_f32m1_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketMul1Xf>(a), ptrue<PacketMul1Xf>(a), mask, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pcmp_eq<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  PacketMask32 mask = __riscv_vmfeq_vv_f32m1_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketMul1Xf>(a), ptrue<PacketMul1Xf>(a), mask, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pcmp_lt_or_nan<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  PacketMask32 mask = __riscv_vmfge_vv_f32m1_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vfmerge_vfm_f32m1(ptrue<PacketMul1Xf>(a), 0.0f, mask, unpacket_traits<PacketMul1Xf>::size);
}

// Logical Operations are not supported for float, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketMul1Xf pand<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf por<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pxor<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vxor_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pandnot<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a),
      __riscv_vnot_v_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketMul1Xf>::size),
      unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pload<PacketMul1Xf>(const float* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_f32m1(from, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf ploadu<PacketMul1Xf>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_f32m1(from, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf ploaddup<PacketMul1Xf>(const float* from) {
  PacketMul1Xu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xf>::size);
  idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(idx, 0xfffffffeu, unpacket_traits<PacketMul1Xf>::size), 1,
                              unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vloxei32_v_f32m1(from, idx, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf ploadquad<PacketMul1Xf>(const float* from) {
  PacketMul1Xu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xf>::size);
  idx = __riscv_vand_vx_u32m1(idx, 0xfffffffcu, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vloxei32_v_f32m1(from, idx, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const PacketMul1Xf& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_f32m1(to, from, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const PacketMul1Xf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_f32m1(to, from, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul1Xf pgather<float, PacketMul1Xf>(const float* from, Index stride) {
  return __riscv_vlse32_v_f32m1(from, stride * sizeof(float), unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, PacketMul1Xf>(float* to, const PacketMul1Xf& from, Index stride) {
  __riscv_vsse32(to, stride * sizeof(float), from, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE float pfirst<PacketMul1Xf>(const PacketMul1Xf& a) {
  return __riscv_vfmv_f_s_f32m1_f32(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf psqrt(const PacketMul1Xf& a) {
  return __riscv_vfsqrt_v_f32m1(a, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf print<PacketMul1Xf>(const PacketMul1Xf& a) {
  const PacketMul1Xf limit = pset1<PacketMul1Xf>(static_cast<float>(1 << 23));
  const PacketMul1Xf abs_a = pabs(a);

  PacketMask32 mask = __riscv_vmfne_vv_f32m1_b32(a, a, unpacket_traits<PacketMul1Xf>::size);
  const PacketMul1Xf x = __riscv_vfadd_vv_f32m1_tumu(mask, a, a, a, unpacket_traits<PacketMul1Xf>::size);
  const PacketMul1Xf new_x = __riscv_vfcvt_f_x_v_f32m1(__riscv_vfcvt_x_f_v_i32m1(a, unpacket_traits<PacketMul1Xf>::size),
                                                   unpacket_traits<PacketMul1Xf>::size);

  mask = __riscv_vmflt_vv_f32m1_b32(abs_a, limit, unpacket_traits<PacketMul1Xf>::size);
  PacketMul1Xf signed_x = __riscv_vfsgnj_vv_f32m1(new_x, x, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vmerge_vvm_f32m1(x, signed_x, mask, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pfloor<PacketMul1Xf>(const PacketMul1Xf& a) {
  PacketMul1Xf tmp = print<PacketMul1Xf>(a);
  // If greater, subtract one.
  PacketMask32 mask = __riscv_vmflt_vv_f32m1_b32(a, tmp, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vfsub_vf_f32m1_tumu(mask, tmp, tmp, 1.0f, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf preverse(const PacketMul1Xf& a) {
  PacketMul1Xu idx = __riscv_vrsub_vx_u32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketMul1Xf>::size),
                                        unpacket_traits<PacketMul1Xf>::size - 1, unpacket_traits<PacketMul1Xf>::size);
  return __riscv_vrgather_vv_f32m1(a, idx, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pfrexp<PacketMul1Xf>(const PacketMul1Xf& a, PacketMul1Xf& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE float predux<PacketMul1Xf>(const PacketMul1Xf& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f32m1_f32m1(
      a, __riscv_vfmv_v_f_f32m1(0.0, unpacket_traits<PacketMul1Xf>::size), unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_mul<PacketMul1Xf>(const PacketMul1Xf& a) {
  // Multiply the vector by its reverse
  PacketMul1Xf prod = __riscv_vfmul_vv_f32m1(preverse(a), a, unpacket_traits<PacketMul1Xf>::size);
  PacketMul1Xf half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_f32m1(prod, 8, unpacket_traits<PacketMul1Xf>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketMul1Xf>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_f32m1(prod, 4, unpacket_traits<PacketMul1Xf>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketMul1Xf>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_f32m1(prod, 2, unpacket_traits<PacketMul1Xf>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketMul1Xf>::size);
  }
  // Last reduction
  half_prod = __riscv_vslidedown_vx_f32m1(prod, 1, unpacket_traits<PacketMul1Xf>::size);
  prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketMul1Xf>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketMul1Xf>(const PacketMul1Xf& a) {
  return (
      std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f32m1_f32m1(
                    a,
                    __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul1Xf>::size),
                    unpacket_traits<PacketMul1Xf>::size)),
                (std::numeric_limits<float>::max)());
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketMul1Xf>(const PacketMul1Xf& a) {
  return (
      std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f32m1_f32m1(
                    a,
                    __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul1Xf>::size),
                    unpacket_traits<PacketMul1Xf>::size)),
                -(std::numeric_limits<float>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul1Xf, N>& kernel) {
  float buffer[unpacket_traits<PacketMul1Xf>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(float), kernel.packet[i], unpacket_traits<PacketMul1Xf>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_f32m1(&buffer[i * unpacket_traits<PacketMul1Xf>::size], unpacket_traits<PacketMul1Xf>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pldexp<PacketMul1Xf>(const PacketMul1Xf& a, const PacketMul1Xf& exponent) {
  return pldexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE PacketMask32 por(const PacketMask32& a, const PacketMask32& b) {
  return __riscv_vmor_mm_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMask32 pand(const PacketMask32& a, const PacketMask32& b) {
  return __riscv_vmand_mm_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
}

EIGEN_STRONG_INLINE PacketMask32 pcmp_eq_mask(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vmfeq_vv_f32m1_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
}

EIGEN_STRONG_INLINE PacketMask32 pcmp_lt_mask(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vmflt_vv_f32m1_b32(a, b, unpacket_traits<PacketMul1Xf>::size);
}

EIGEN_STRONG_INLINE PacketMul1Xf pselect(const PacketMask32& mask, const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vmerge_vvm_f32m1(b, a, mask, unpacket_traits<PacketMul1Xf>::size);
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

/********************************* int64 **************************************/

typedef eigen_packet_wrapper<vint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 9> PacketMul1Xl;
typedef eigen_packet_wrapper<vuint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 10> PacketMul1Xul;

typedef eigen_packet_wrapper<vint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 11>
    PacketMul2Xl;
typedef eigen_packet_wrapper<vuint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 12>
    PacketMul2Xul;

typedef eigen_packet_wrapper<vint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 13>
    PacketMul4Xl;
typedef eigen_packet_wrapper<vuint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 14>
    PacketMul4Xul;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef PacketMul1Xl PacketXl;
typedef PacketMul1Xul PacketXul;

template <>
struct packet_traits<numext::int64_t> : default_packet_traits {
  typedef PacketMul1Xl type;
  typedef PacketMul1Xl half;  // Half not implemented yet
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int64_t, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef PacketMul2Xl PacketXl;
typedef PacketMul2Xul PacketXul;

template <>
struct packet_traits<numext::int64_t> : default_packet_traits {
  typedef PacketMul2Xl type;
  typedef PacketMul1Xl half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int64_t, EIGEN_RISCV64_RVV_VL, 2>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef PacketMul4Xl PacketXl;
typedef PacketMul4Xul PacketXul;

template <>
struct packet_traits<numext::int64_t> : default_packet_traits {
  typedef PacketMul4Xl type;
  typedef PacketMul2Xl half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int64_t, EIGEN_RISCV64_RVV_VL, 4>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};
#endif

template <>
struct unpacket_traits<PacketMul1Xl> {
  typedef numext::int64_t type;
  typedef PacketMul1Xl half;  // Half not yet implemented
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int64_t, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 1>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul2Xl> {
  typedef numext::int64_t type;
  typedef PacketMul1Xl half;
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int64_t, EIGEN_RISCV64_RVV_VL, 2>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul4Xl> {
  typedef numext::int64_t type;
  typedef PacketMul2Xl half;
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int64_t, EIGEN_RISCV64_RVV_VL, 4>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 4>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int64_t>(const numext::int64_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

/********************************* PacketMul1Xl ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pset1<PacketMul1Xl>(const numext::int64_t& from) {
  return __riscv_vmv_v_x_i64m1(from, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl plset<PacketMul1Xl>(const numext::int64_t& a) {
  PacketMul1Xl idx = __riscv_vreinterpret_v_u64m1_i64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xl>::size));
  return __riscv_vadd_vx_i64m1(idx, a, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pzero<PacketMul1Xl>(const PacketMul1Xl& /*a*/) {
  return __riscv_vmv_v_x_i64m1(0, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl padd<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vadd_vv_i64m1(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl psub<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pnegate(const PacketMul1Xl& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pconj(const PacketMul1Xl& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pmul<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pdiv<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pmadd(const PacketMul1Xl& a, const PacketMul1Xl& b, const PacketMul1Xl& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pmsub(const PacketMul1Xl& a, const PacketMul1Xl& b, const PacketMul1Xl& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pnmadd(const PacketMul1Xl& a, const PacketMul1Xl& b, const PacketMul1Xl& c) {
  return __riscv_vnmsub_vv_i64m1(a, b, c, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pnmsub(const PacketMul1Xl& a, const PacketMul1Xl& b, const PacketMul1Xl& c) {
  return __riscv_vnmsub_vv_i64m1(a, b, pnegate(c), unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pmin<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pmax<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pcmp_le<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  PacketMask64 mask = __riscv_vmsle_vv_i64m1_b64(a, b, unpacket_traits<PacketMul1Xl>::size);
  return __riscv_vmerge_vxm_i64m1(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pcmp_lt<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  PacketMask64 mask = __riscv_vmslt_vv_i64m1_b64(a, b, unpacket_traits<PacketMul1Xl>::size);
  return __riscv_vmerge_vxm_i64m1(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pcmp_eq<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  PacketMask64 mask = __riscv_vmseq_vv_i64m1_b64(a, b, unpacket_traits<PacketMul1Xl>::size);
  return __riscv_vmerge_vxm_i64m1(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl ptrue<PacketMul1Xl>(const PacketMul1Xl& /*a*/) {
  return __riscv_vmv_v_x_i64m1(0xffffffffffffffffu, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pand<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vand_vv_i64m1(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl por<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vor_vv_i64m1(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pxor<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vxor_vv_i64m1(a, b, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pandnot<PacketMul1Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vand_vv_i64m1(a, __riscv_vnot_v_i64m1(b, unpacket_traits<PacketMul1Xl>::size),
                               unpacket_traits<PacketMul1Xl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xl parithmetic_shift_right(PacketMul1Xl a) {
  return __riscv_vsra_vx_i64m1(a, N, unpacket_traits<PacketMul1Xl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xl plogical_shift_right(PacketMul1Xl a) {
  return __riscv_vreinterpret_i64m1(
      __riscv_vsrl_vx_u64m1(__riscv_vreinterpret_u64m1(a), N, unpacket_traits<PacketMul1Xl>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xl plogical_shift_left(PacketMul1Xl a) {
  return __riscv_vsll_vx_i64m1(a, N, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pload<PacketMul1Xl>(const numext::int64_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_i64m1(from, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl ploadu<PacketMul1Xl>(const numext::int64_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_i64m1(from, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl ploaddup<PacketMul1Xl>(const numext::int64_t* from) {
  PacketMul1Xul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xl>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffeu, unpacket_traits<PacketMul1Xl>::size), 2,
                              unpacket_traits<PacketMul1Xl>::size);
  // idx = 0 0 sizeof(int64_t) sizeof(int64_t) 2*sizeof(int64_t) 2*sizeof(int64_t) ...
  return __riscv_vloxei64_v_i64m1(from, idx, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl ploadquad<PacketMul1Xl>(const numext::int64_t* from) {
  PacketMul1Xul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xl>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffcu, unpacket_traits<PacketMul1Xl>::size), 1,
                              unpacket_traits<PacketMul1Xl>::size);
  ;
  return __riscv_vloxei64_v_i64m1(from, idx, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int64_t>(numext::int64_t* to, const PacketMul1Xl& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_i64m1(to, from, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int64_t>(numext::int64_t* to, const PacketMul1Xl& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_i64m1(to, from, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul1Xl pgather<numext::int64_t, PacketMul1Xl>(const numext::int64_t* from, Index stride) {
  return __riscv_vlse64_v_i64m1(from, stride * sizeof(numext::int64_t), unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int64_t, PacketMul1Xl>(numext::int64_t* to, const PacketMul1Xl& from,
                                                                  Index stride) {
  __riscv_vsse64(to, stride * sizeof(numext::int64_t), from, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t pfirst<PacketMul1Xl>(const PacketMul1Xl& a) {
  return __riscv_vmv_x_s_i64m1_i64(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl preverse(const PacketMul1Xl& a) {
  PacketMul1Xul idx = __riscv_vrsub_vx_u64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xl>::size),
                                         unpacket_traits<PacketMul1Xl>::size - 1, unpacket_traits<PacketMul1Xl>::size);
  return __riscv_vrgather_vv_i64m1(a, idx, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pabs(const PacketMul1Xl& a) {
  PacketMul1Xl mask = __riscv_vsra_vx_i64m1(a, 63, unpacket_traits<PacketMul1Xl>::size);
  return __riscv_vsub_vv_i64m1(__riscv_vxor_vv_i64m1(a, mask, unpacket_traits<PacketMul1Xl>::size), mask,
                               unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux<PacketMul1Xl>(const PacketMul1Xl& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i64m1_i64m1(a, __riscv_vmv_v_x_i64m1(0, unpacket_traits<PacketMul1Xl>::size),
                                                      unpacket_traits<PacketMul1Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_mul<PacketMul1Xl>(const PacketMul1Xl& a) {
  // Multiply the vector by its reverse
  PacketMul1Xl prod = __riscv_vmul_vv_i64m1(preverse(a), a, unpacket_traits<PacketMul1Xl>::size);
  PacketMul1Xl half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_i64m1(prod, 4, unpacket_traits<PacketMul1Xl>::size);
    prod = __riscv_vmul_vv_i64m1(prod, half_prod, unpacket_traits<PacketMul1Xl>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_i64m1(prod, 2, unpacket_traits<PacketMul1Xl>::size);
    prod = __riscv_vmul_vv_i64m1(prod, half_prod, unpacket_traits<PacketMul1Xl>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_i64m1(prod, 1, unpacket_traits<PacketMul1Xl>::size);
    prod = __riscv_vmul_vv_i64m1(prod, half_prod, unpacket_traits<PacketMul1Xl>::size);
  }

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_min<PacketMul1Xl>(const PacketMul1Xl& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i64m1_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::max)(), unpacket_traits<PacketMul1Xl>::size),
      unpacket_traits<PacketMul1Xl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_max<PacketMul1Xl>(const PacketMul1Xl& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i64m1_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::min)(), unpacket_traits<PacketMul1Xl>::size),
      unpacket_traits<PacketMul1Xl>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul1Xl, N>& kernel) {
  numext::int64_t buffer[unpacket_traits<PacketMul1Xl>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(numext::int64_t), kernel.packet[i], unpacket_traits<PacketMul1Xl>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_i64m1(&buffer[i * unpacket_traits<PacketMul1Xl>::size], unpacket_traits<PacketMul1Xl>::size);
  }
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

/********************************* double ************************************/

typedef eigen_packet_wrapper<vfloat64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 15> PacketMul1Xd;
typedef eigen_packet_wrapper<vfloat64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 16>
    PacketMul2Xd;
typedef eigen_packet_wrapper<vfloat64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 17>
    PacketMul4Xd;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef PacketMul1Xd PacketXd;

template <>
struct packet_traits<double> : default_packet_traits {
  typedef PacketMul1Xd type;
  typedef PacketMul1Xd half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef PacketMul2Xd PacketXd;

template <>
struct packet_traits<double> : default_packet_traits {
  typedef PacketMul2Xd type;
  typedef PacketMul1Xd half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 2>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef PacketMul4Xd PacketXd;

template <>
struct packet_traits<double> : default_packet_traits {
  typedef PacketMul4Xd type;
  typedef PacketMul2Xd half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 4>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1
  };
};
#endif

template <>
struct unpacket_traits<PacketMul1Xd> {
  typedef double type;
  typedef PacketMul1Xd half;  // Half not yet implemented
  typedef PacketMul1Xl integer_packet;
  typedef numext::uint8_t mask_t;
  typedef PacketMask64 packet_mask;

  enum {
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 1>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul2Xd> {
  typedef double type;
  typedef PacketMul1Xd half;
  typedef PacketMul2Xl integer_packet;
  typedef numext::uint8_t mask_t;
  typedef PacketMask32 packet_mask;

  enum {
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 2>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul4Xd> {
  typedef double type;
  typedef PacketMul2Xd half;
  typedef PacketMul4Xl integer_packet;
  typedef numext::uint8_t mask_t;
  typedef PacketMask16 packet_mask;

  enum {
    size = rvv_packet_size_selector<double, EIGEN_RISCV64_RVV_VL, 4>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 4>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

/********************************* PacketMul1Xd ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul1Xd ptrue<PacketMul1Xd>(const PacketMul1Xd& /*a*/) {
  return __riscv_vreinterpret_f64m1(__riscv_vmv_v_x_u64m1(0xffffffffffffffffu, unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pzero<PacketMul1Xd>(const PacketMul1Xd& /*a*/) {
  return __riscv_vfmv_v_f_f64m1(0.0, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pabs(const PacketMul1Xd& a) {
  return __riscv_vfabs_v_f64m1(a, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pset1<PacketMul1Xd>(const double& from) {
  return __riscv_vfmv_v_f_f64m1(from, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pset1frombits<PacketMul1Xd>(numext::uint64_t from) {
  return __riscv_vreinterpret_f64m1(__riscv_vmv_v_x_u64m1(from, unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd plset<PacketMul1Xd>(const double& a) {
  PacketMul1Xd idx = __riscv_vfcvt_f_x_v_f64m1(
      __riscv_vreinterpret_v_u64m1_i64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xl>::size)),
      unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vfadd_vf_f64m1(idx, a, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd padd<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vfadd_vv_f64m1(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd psub<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vfsub_vv_f64m1(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pnegate(const PacketMul1Xd& a) {
  return __riscv_vfneg_v_f64m1(a, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pconj(const PacketMul1Xd& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmul<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vfmul_vv_f64m1(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pdiv<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vfdiv_vv_f64m1(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmadd(const PacketMul1Xd& a, const PacketMul1Xd& b, const PacketMul1Xd& c) {
  return __riscv_vfmadd_vv_f64m1(a, b, c, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmsub(const PacketMul1Xd& a, const PacketMul1Xd& b, const PacketMul1Xd& c) {
  return __riscv_vfmsub_vv_f64m1(a, b, c, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pnmadd(const PacketMul1Xd& a, const PacketMul1Xd& b, const PacketMul1Xd& c) {
  return __riscv_vfnmsub_vv_f64m1(a, b, c, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pnmsub(const PacketMul1Xd& a, const PacketMul1Xd& b, const PacketMul1Xd& c) {
  return __riscv_vfnmadd_vv_f64m1(a, b, c, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmin<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  PacketMul1Xd nans = __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul1Xd>::size);
  PacketMask64 mask = __riscv_vmfeq_vv_f64m1_b64(a, a, unpacket_traits<PacketMul1Xd>::size);
  PacketMask64 mask2 = __riscv_vmfeq_vv_f64m1_b64(b, b, unpacket_traits<PacketMul1Xd>::size);
  mask = __riscv_vmand_mm_b64(mask, mask2, unpacket_traits<PacketMul1Xd>::size);

  return __riscv_vfmin_vv_f64m1_tumu(mask, nans, a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmin<PropagateNaN, PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return pmin<PacketMul1Xd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmin<PropagateNumbers, PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vfmin_vv_f64m1(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmax<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  PacketMul1Xd nans = __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul1Xd>::size);
  PacketMask64 mask = __riscv_vmfeq_vv_f64m1_b64(a, a, unpacket_traits<PacketMul1Xd>::size);
  PacketMask64 mask2 = __riscv_vmfeq_vv_f64m1_b64(b, b, unpacket_traits<PacketMul1Xd>::size);
  mask = __riscv_vmand_mm_b64(mask, mask2, unpacket_traits<PacketMul1Xd>::size);

  return __riscv_vfmax_vv_f64m1_tumu(mask, nans, a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmax<PropagateNaN, PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return pmax<PacketMul1Xd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pmax<PropagateNumbers, PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vfmax_vv_f64m1(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pcmp_le<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  PacketMask64 mask = __riscv_vmfle_vv_f64m1_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vmerge_vvm_f64m1(pzero<PacketMul1Xd>(a), ptrue<PacketMul1Xd>(a), mask, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pcmp_lt<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  PacketMask64 mask = __riscv_vmflt_vv_f64m1_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vmerge_vvm_f64m1(pzero<PacketMul1Xd>(a), ptrue<PacketMul1Xd>(a), mask, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pcmp_eq<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  PacketMask64 mask = __riscv_vmfeq_vv_f64m1_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vmerge_vvm_f64m1(pzero<PacketMul1Xd>(a), ptrue<PacketMul1Xd>(a), mask, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pcmp_lt_or_nan<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  PacketMask64 mask = __riscv_vmfge_vv_f64m1_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vfmerge_vfm_f64m1(ptrue<PacketMul1Xd>(a), 0.0, mask, unpacket_traits<PacketMul1Xd>::size);
}

// Logical Operations are not supported for double, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketMul1Xd pand<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a), __riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd por<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a), __riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pxor<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vxor_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a), __riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pandnot<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a),
      __riscv_vnot_v_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketMul1Xd>::size),
      unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pload<PacketMul1Xd>(const double* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_f64m1(from, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd ploadu<PacketMul1Xd>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_f64m1(from, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd ploaddup<PacketMul1Xd>(const double* from) {
  PacketMul1Xul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xd>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffeu, unpacket_traits<PacketMul1Xd>::size), 2,
                              unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vloxei64_v_f64m1(from, idx, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd ploadquad<PacketMul1Xd>(const double* from) {
  PacketMul1Xul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xd>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffcu, unpacket_traits<PacketMul1Xd>::size), 1,
                              unpacket_traits<PacketMul1Xd>::size);
  ;
  return __riscv_vloxei64_v_f64m1(from, idx, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<double>(double* to, const PacketMul1Xd& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_f64m1(to, from, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const PacketMul1Xd& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_f64m1(to, from, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul1Xd pgather<double, PacketMul1Xd>(const double* from, Index stride) {
  return __riscv_vlse64_v_f64m1(from, stride * sizeof(double), unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<double, PacketMul1Xd>(double* to, const PacketMul1Xd& from, Index stride) {
  __riscv_vsse64(to, stride * sizeof(double), from, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE double pfirst<PacketMul1Xd>(const PacketMul1Xd& a) {
  return __riscv_vfmv_f_s_f64m1_f64(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd psqrt(const PacketMul1Xd& a) {
  return __riscv_vfsqrt_v_f64m1(a, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd print<PacketMul1Xd>(const PacketMul1Xd& a) {
  const PacketMul1Xd limit = pset1<PacketMul1Xd>(static_cast<double>(1ull << 52));
  const PacketMul1Xd abs_a = pabs(a);

  PacketMask64 mask = __riscv_vmfne_vv_f64m1_b64(a, a, unpacket_traits<PacketMul1Xd>::size);
  const PacketMul1Xd x = __riscv_vfadd_vv_f64m1_tumu(mask, a, a, a, unpacket_traits<PacketMul1Xd>::size);
  const PacketMul1Xd new_x = __riscv_vfcvt_f_x_v_f64m1(__riscv_vfcvt_x_f_v_i64m1(a, unpacket_traits<PacketMul1Xd>::size),
                                                   unpacket_traits<PacketMul1Xd>::size);

  mask = __riscv_vmflt_vv_f64m1_b64(abs_a, limit, unpacket_traits<PacketMul1Xd>::size);
  PacketMul1Xd signed_x = __riscv_vfsgnj_vv_f64m1(new_x, x, unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vmerge_vvm_f64m1(x, signed_x, mask, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pfloor<PacketMul1Xd>(const PacketMul1Xd& a) {
  PacketMul1Xd tmp = print<PacketMul1Xd>(a);
  // If greater, subtract one.
  PacketMask64 mask = __riscv_vmflt_vv_f64m1_b64(a, tmp, unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vfsub_vf_f64m1_tumu(mask, tmp, tmp, 1.0, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd preverse(const PacketMul1Xd& a) {
  PacketMul1Xul idx = __riscv_vrsub_vx_u64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketMul1Xd>::size),
                                         unpacket_traits<PacketMul1Xd>::size - 1, unpacket_traits<PacketMul1Xd>::size);
  return __riscv_vrgather_vv_f64m1(a, idx, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pfrexp<PacketMul1Xd>(const PacketMul1Xd& a, PacketMul1Xd& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE double predux<PacketMul1Xd>(const PacketMul1Xd& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f64m1_f64m1(
      a, __riscv_vfmv_v_f_f64m1(0.0, unpacket_traits<PacketMul1Xd>::size), unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_mul<PacketMul1Xd>(const PacketMul1Xd& a) {
  // Multiply the vector by its reverse
  PacketMul1Xd prod = __riscv_vfmul_vv_f64m1(preverse(a), a, unpacket_traits<PacketMul1Xd>::size);
  PacketMul1Xd half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_f64m1(prod, 4, unpacket_traits<PacketMul1Xd>::size);
    prod = __riscv_vfmul_vv_f64m1(prod, half_prod, unpacket_traits<PacketMul1Xd>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_f64m1(prod, 2, unpacket_traits<PacketMul1Xd>::size);
    prod = __riscv_vfmul_vv_f64m1(prod, half_prod, unpacket_traits<PacketMul1Xd>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_f64m1(prod, 1, unpacket_traits<PacketMul1Xd>::size);
    prod = __riscv_vfmul_vv_f64m1(prod, half_prod, unpacket_traits<PacketMul1Xd>::size);
  }

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE double predux_min<PacketMul1Xd>(const PacketMul1Xd& a) {
  return (
      std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f64m1_f64m1(
                    a,
                    __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul1Xd>::size),
                    unpacket_traits<PacketMul1Xd>::size)),
                (std::numeric_limits<double>::max)());
}

template <>
EIGEN_STRONG_INLINE double predux_max<PacketMul1Xd>(const PacketMul1Xd& a) {
  return (
      std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f64m1_f64m1(
                    a,
                    __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul1Xd>::size),
                    unpacket_traits<PacketMul1Xd>::size)),
                -(std::numeric_limits<double>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul1Xd, N>& kernel) {
  double buffer[unpacket_traits<PacketMul1Xd>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(double), kernel.packet[i], unpacket_traits<PacketMul1Xd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_f64m1(&buffer[i * unpacket_traits<PacketMul1Xd>::size], unpacket_traits<PacketMul1Xd>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pldexp<PacketMul1Xd>(const PacketMul1Xd& a, const PacketMul1Xd& exponent) {
  return pldexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE PacketMask64 por(const PacketMask64& a, const PacketMask64& b) {
  return __riscv_vmor_mm_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMask64 pandnot(const PacketMask64& a, const PacketMask64& b) {
  return __riscv_vmor_mm_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMask64 pand(const PacketMask64& a, const PacketMask64& b) {
  return __riscv_vmand_mm_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
}

EIGEN_STRONG_INLINE PacketMask64 pcmp_eq_mask(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vmfeq_vv_f64m1_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
}

EIGEN_STRONG_INLINE PacketMask64 pcmp_lt_mask(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vmflt_vv_f64m1_b64(a, b, unpacket_traits<PacketMul1Xd>::size);
}

EIGEN_STRONG_INLINE PacketMul1Xd pselect(const PacketMask64& mask, const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vmerge_vvm_f64m1(b, a, mask, unpacket_traits<PacketMul1Xd>::size);
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

/********************************* short **************************************/

typedef eigen_packet_wrapper<vint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 18> PacketMul1Xs;
typedef eigen_packet_wrapper<vuint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 19> PacketMul1Xsu;

typedef eigen_packet_wrapper<vint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 20>
    PacketMul2Xs;
typedef eigen_packet_wrapper<vuint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 21>
    PacketMul2Xsu;

typedef eigen_packet_wrapper<vint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 22>
    PacketMul4Xs;
typedef eigen_packet_wrapper<vuint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 23>
    PacketMul4Xsu;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef PacketMul1Xs PacketXs;
typedef PacketMul1Xsu PacketXsu;

template <>
struct packet_traits<numext::int16_t> : default_packet_traits {
  typedef PacketMul1Xs type;
  typedef PacketMul1Xs half;  // Half not implemented yet
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int16_t, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
typedef PacketMul2Xs PacketXs;
typedef PacketMul2Xsu PacketXsu;

template <>
struct packet_traits<numext::int16_t> : default_packet_traits {
  typedef PacketMul2Xs type;
  typedef PacketMul1Xs half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int16_t, EIGEN_RISCV64_RVV_VL, 2>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};

#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
typedef PacketMul4Xs PacketXs;
typedef PacketMul4Xsu PacketXsu;

template <>
struct packet_traits<numext::int16_t> : default_packet_traits {
  typedef PacketMul4Xs type;
  typedef PacketMul2Xs half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int16_t, EIGEN_RISCV64_RVV_VL, 4>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0
  };
};
#endif

template <>
struct unpacket_traits<PacketMul1Xs> {
  typedef numext::int16_t type;
  typedef PacketMul1Xs half;  // Half not yet implemented
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int16_t, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 1>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul2Xs> {
  typedef numext::int16_t type;
  typedef PacketMul1Xs half;
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int16_t, EIGEN_RISCV64_RVV_VL, 2>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<PacketMul4Xs> {
  typedef numext::int16_t type;
  typedef PacketMul2Xs half;
  typedef numext::uint8_t mask_t;
  enum {
    size = rvv_packet_size_selector<numext::int16_t, EIGEN_RISCV64_RVV_VL, 4>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 4>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int16_t>(const numext::int16_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

/********************************* PacketMul1Xs ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pset1<PacketMul1Xs>(const numext::int16_t& from) {
  return __riscv_vmv_v_x_i16m1(from, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs plset<PacketMul1Xs>(const numext::int16_t& a) {
  PacketMul1Xs idx = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vid_v_u16m1(unpacket_traits<PacketMul1Xs>::size));
  return __riscv_vadd_vx_i16m1(idx, a, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pzero<PacketMul1Xs>(const PacketMul1Xs& /*a*/) {
  return __riscv_vmv_v_x_i16m1(0, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs padd<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vadd_vv_i16m1(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs psub<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pnegate(const PacketMul1Xs& a) {
  return __riscv_vneg(a, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pconj(const PacketMul1Xs& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pmul<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pdiv<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pmadd(const PacketMul1Xs& a, const PacketMul1Xs& b, const PacketMul1Xs& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pmsub(const PacketMul1Xs& a, const PacketMul1Xs& b, const PacketMul1Xs& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pnmadd(const PacketMul1Xs& a, const PacketMul1Xs& b, const PacketMul1Xs& c) {
  return __riscv_vnmsub_vv_i16m1(a, b, c, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pnmsub(const PacketMul1Xs& a, const PacketMul1Xs& b, const PacketMul1Xs& c) {
  return __riscv_vnmsub_vv_i16m1(a, b, pnegate(c), unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pmin<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pmax<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pcmp_le<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  PacketMask16 mask = __riscv_vmsle_vv_i16m1_b16(a, b, unpacket_traits<PacketMul1Xs>::size);
  return __riscv_vmerge_vxm_i16m1(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pcmp_lt<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  PacketMask16 mask = __riscv_vmslt_vv_i16m1_b16(a, b, unpacket_traits<PacketMul1Xs>::size);
  return __riscv_vmerge_vxm_i16m1(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pcmp_eq<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  PacketMask16 mask = __riscv_vmseq_vv_i16m1_b16(a, b, unpacket_traits<PacketMul1Xs>::size);
  return __riscv_vmerge_vxm_i16m1(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs ptrue<PacketMul1Xs>(const PacketMul1Xs& /*a*/) {
  return __riscv_vmv_v_x_i16m1(static_cast<unsigned short>(0xffffu), unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pand<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vand_vv_i16m1(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs por<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vor_vv_i16m1(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pxor<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vxor_vv_i16m1(a, b, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pandnot<PacketMul1Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vand_vv_i16m1(a, __riscv_vnot_v_i16m1(b, unpacket_traits<PacketMul1Xs>::size),
                               unpacket_traits<PacketMul1Xs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xs parithmetic_shift_right(PacketMul1Xs a) {
  return __riscv_vsra_vx_i16m1(a, N, unpacket_traits<PacketMul1Xs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xs plogical_shift_right(PacketMul1Xs a) {
  return __riscv_vreinterpret_i16m1(
      __riscv_vsrl_vx_u16m1(__riscv_vreinterpret_u16m1(a), N, unpacket_traits<PacketMul1Xs>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketMul1Xs plogical_shift_left(PacketMul1Xs a) {
  return __riscv_vsll_vx_i16m1(a, N, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pload<PacketMul1Xs>(const numext::int16_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle16_v_i16m1(from, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs ploadu<PacketMul1Xs>(const numext::int16_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle16_v_i16m1(from, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs ploaddup<PacketMul1Xs>(const numext::int16_t* from) {
  PacketMul1Xsu idx = __riscv_vid_v_u16m1(unpacket_traits<PacketMul1Xs>::size);
  idx = __riscv_vand_vx_u16m1(idx, 0xfffeu, unpacket_traits<PacketMul1Xs>::size);
  // idx = 0 0 sizeof(int16_t) sizeof(int16_t) 2*sizeof(int16_t) 2*sizeof(int16_t) ...
  return __riscv_vloxei16_v_i16m1(from, idx, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs ploadquad<PacketMul1Xs>(const numext::int16_t* from) {
  PacketMul1Xsu idx = __riscv_vid_v_u16m1(unpacket_traits<PacketMul1Xs>::size);
  idx = __riscv_vsrl_vx_u16m1(__riscv_vand_vx_u16m1(idx, 0xfffcu, unpacket_traits<PacketMul1Xs>::size), 1,
                              unpacket_traits<PacketMul1Xs>::size);
  return __riscv_vloxei16_v_i16m1(from, idx, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int16_t>(numext::int16_t* to, const PacketMul1Xs& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse16_v_i16m1(to, from, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int16_t>(numext::int16_t* to, const PacketMul1Xs& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse16_v_i16m1(to, from, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketMul1Xs pgather<numext::int16_t, PacketMul1Xs>(const numext::int16_t* from, Index stride) {
  return __riscv_vlse16_v_i16m1(from, stride * sizeof(numext::int16_t), unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int16_t, PacketMul1Xs>(numext::int16_t* to, const PacketMul1Xs& from,
                                                                  Index stride) {
  __riscv_vsse16(to, stride * sizeof(numext::int16_t), from, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t pfirst<PacketMul1Xs>(const PacketMul1Xs& a) {
  return __riscv_vmv_x_s_i16m1_i16(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs preverse(const PacketMul1Xs& a) {
  PacketMul1Xsu idx = __riscv_vrsub_vx_u16m1(__riscv_vid_v_u16m1(unpacket_traits<PacketMul1Xs>::size),
                                         unpacket_traits<PacketMul1Xs>::size - 1, unpacket_traits<PacketMul1Xs>::size);
  return __riscv_vrgather_vv_i16m1(a, idx, unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xs pabs(const PacketMul1Xs& a) {
  PacketMul1Xs mask = __riscv_vsra_vx_i16m1(a, 15, unpacket_traits<PacketMul1Xs>::size);
  return __riscv_vsub_vv_i16m1(__riscv_vxor_vv_i16m1(a, mask, unpacket_traits<PacketMul1Xs>::size), mask,
                               unpacket_traits<PacketMul1Xs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux<PacketMul1Xs>(const PacketMul1Xs& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i16m1_i16m1(a, __riscv_vmv_v_x_i16m1(0, unpacket_traits<PacketMul1Xs>::size),
                                                      unpacket_traits<PacketMul1Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_mul<PacketMul1Xs>(const PacketMul1Xs& a) {
  // Multiply the vector by its reverse
  PacketMul1Xs prod = __riscv_vmul_vv_i16m1(preverse(a), a, unpacket_traits<PacketMul1Xs>::size);
  PacketMul1Xs half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_i16m1(prod, 16, unpacket_traits<PacketMul1Xs>::size);
    prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketMul1Xs>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_i16m1(prod, 8, unpacket_traits<PacketMul1Xs>::size);
    prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketMul1Xs>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_i16m1(prod, 4, unpacket_traits<PacketMul1Xs>::size);
    prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketMul1Xs>::size);
  }
  // Last reduction
  half_prod = __riscv_vslidedown_vx_i16m1(prod, 2, unpacket_traits<PacketMul1Xs>::size);
  prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketMul1Xs>::size);

  half_prod = __riscv_vslidedown_vx_i16m1(prod, 1, unpacket_traits<PacketMul1Xs>::size);
  prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketMul1Xs>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_min<PacketMul1Xs>(const PacketMul1Xs& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i16m1_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::max)(), unpacket_traits<PacketMul1Xs>::size),
      unpacket_traits<PacketMul1Xs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_max<PacketMul1Xs>(const PacketMul1Xs& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i16m1_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::min)(), unpacket_traits<PacketMul1Xs>::size),
      unpacket_traits<PacketMul1Xs>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketMul1Xs, N>& kernel) {
  numext::int16_t buffer[unpacket_traits<PacketMul1Xs>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse16(&buffer[i], N * sizeof(numext::int16_t), kernel.packet[i], unpacket_traits<PacketMul1Xs>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle16_v_i16m1(&buffer[i * unpacket_traits<PacketMul1Xs>::size], unpacket_traits<PacketMul1Xs>::size);
  }
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

#endif  // EIGEN_PACKET_MATH_RVV10_H
