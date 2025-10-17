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
typedef eigen_packet_wrapper<vint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 0> PacketXi;
typedef eigen_packet_wrapper<vuint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 1> PacketXu;

typedef eigen_packet_wrapper<vint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 2> PacketMul2Xi;
typedef eigen_packet_wrapper<vuint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 3> PacketMul2Xu;

typedef eigen_packet_wrapper<vint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 4> PacketMul4Xi;
typedef eigen_packet_wrapper<vuint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 5> PacketMul4Xu;

template <>
struct packet_traits<numext::int32_t> : default_packet_traits {
  typedef PacketXi type;
  typedef PacketXi half;  // Half not implemented yet
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

template <>
struct packet_traits<numext::int32_t, 2> : default_packet_traits {
  typedef PacketMul2Xi type;
  typedef PacketXi half;
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

template <>
struct packet_traits<numext::int32_t, 4> : default_packet_traits {
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

template <>
struct unpacket_traits<PacketXi> {
  typedef numext::int32_t type;
  typedef PacketXi half;  // Half not yet implemented
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
  typedef PacketXi half;
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

/********************************* PacketXi ************************************/

template <>
EIGEN_STRONG_INLINE PacketXi pset1<PacketXi>(const numext::int32_t& from) {
  return __riscv_vmv_v_x_i32m1(from, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi plset<PacketXi>(const numext::int32_t& a) {
  PacketXi idx = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketXi>::size));
  return __riscv_vadd_vx_i32m1(idx, a, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pzero<PacketXi>(const PacketXi& /*a*/) {
  return __riscv_vmv_v_x_i32m1(0, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi padd<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vadd_vv_i32m1(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi psub<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pnegate(const PacketXi& a) {
  return __riscv_vneg(a, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pconj(const PacketXi& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXi pmul<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pdiv<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmadd(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmsub(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pnmadd(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vnmsub_vv_i32m1(a, b, c, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pnmsub(const PacketXi& a, const PacketXi& b, const PacketXi& c) {
  return __riscv_vnmsub_vv_i32m1(a, b, pnegate(c), unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmin<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmax<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_le<PacketXi>(const PacketXi& a, const PacketXi& b) {
  PacketMask32 mask = __riscv_vmsle_vv_i32m1_b32(a, b, unpacket_traits<PacketXi>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_lt<PacketXi>(const PacketXi& a, const PacketXi& b) {
  PacketMask32 mask = __riscv_vmslt_vv_i32m1_b32(a, b, unpacket_traits<PacketXi>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_eq<PacketXi>(const PacketXi& a, const PacketXi& b) {
  PacketMask32 mask = __riscv_vmseq_vv_i32m1_b32(a, b, unpacket_traits<PacketXi>::size);
  return __riscv_vmerge_vxm_i32m1(pzero(a), 0xffffffff, mask, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ptrue<PacketXi>(const PacketXi& /*a*/) {
  return __riscv_vmv_v_x_i32m1(0xffffffffu, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pand<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vand_vv_i32m1(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi por<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vor_vv_i32m1(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pxor<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vxor_vv_i32m1(a, b, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pandnot<PacketXi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vand_vv_i32m1(a, __riscv_vnot_v_i32m1(b, unpacket_traits<PacketXi>::size),
                               unpacket_traits<PacketXi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXi parithmetic_shift_right(PacketXi a) {
  return __riscv_vsra_vx_i32m1(a, N, unpacket_traits<PacketXi>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXi plogical_shift_right(PacketXi a) {
  return __riscv_vreinterpret_i32m1(
      __riscv_vsrl_vx_u32m1(__riscv_vreinterpret_u32m1(a), N, unpacket_traits<PacketXi>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketXi plogical_shift_left(PacketXi a) {
  return __riscv_vsll_vx_i32m1(a, N, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pload<PacketXi>(const numext::int32_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_i32m1(from, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploadu<PacketXi>(const numext::int32_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_i32m1(from, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploaddup<PacketXi>(const numext::int32_t* from) {
  PacketXu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketXi>::size);
  idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(idx, 0xfffffffeu, unpacket_traits<PacketXi>::size), 1,
                              unpacket_traits<PacketXi>::size);
  // idx = 0 0 sizeof(int32_t) sizeof(int32_t) 2*sizeof(int32_t) 2*sizeof(int32_t) ...
  return __riscv_vloxei32_v_i32m1(from, idx, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploadquad<PacketXi>(const numext::int32_t* from) {
  PacketXu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketXi>::size);
  idx = __riscv_vand_vx_u32m1(idx, 0xfffffffcu, unpacket_traits<PacketXi>::size);
  return __riscv_vloxei32_v_i32m1(from, idx, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int32_t>(numext::int32_t* to, const PacketXi& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_i32m1(to, from, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int32_t>(numext::int32_t* to, const PacketXi& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_i32m1(to, from, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXi pgather<numext::int32_t, PacketXi>(const numext::int32_t* from, Index stride) {
  return __riscv_vlse32_v_i32m1(from, stride * sizeof(numext::int32_t), unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int32_t, PacketXi>(numext::int32_t* to, const PacketXi& from,
                                                                  Index stride) {
  __riscv_vsse32(to, stride * sizeof(numext::int32_t), from, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t pfirst<PacketXi>(const PacketXi& a) {
  return __riscv_vmv_x_s_i32m1_i32(a);
}

template <>
EIGEN_STRONG_INLINE PacketXi preverse(const PacketXi& a) {
  PacketXu idx = __riscv_vrsub_vx_u32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketXi>::size),
                                        unpacket_traits<PacketXi>::size - 1, unpacket_traits<PacketXi>::size);
  return __riscv_vrgather_vv_i32m1(a, idx, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pabs(const PacketXi& a) {
  PacketXi mask = __riscv_vsra_vx_i32m1(a, 31, unpacket_traits<PacketXi>::size);
  return __riscv_vsub_vv_i32m1(__riscv_vxor_vv_i32m1(a, mask, unpacket_traits<PacketXi>::size), mask,
                               unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux<PacketXi>(const PacketXi& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i32m1_i32m1(a, __riscv_vmv_v_x_i32m1(0, unpacket_traits<PacketXi>::size),
                                                      unpacket_traits<PacketXi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_mul<PacketXi>(const PacketXi& a) {
  // Multiply the vector by its reverse
  PacketXi prod = __riscv_vmul_vv_i32m1(preverse(a), a, unpacket_traits<PacketXi>::size);
  PacketXi half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_i32m1(prod, 8, unpacket_traits<PacketXi>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketXi>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_i32m1(prod, 4, unpacket_traits<PacketXi>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketXi>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_i32m1(prod, 2, unpacket_traits<PacketXi>::size);
    prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketXi>::size);
  }
  // Last reduction
  half_prod = __riscv_vslidedown_vx_i32m1(prod, 1, unpacket_traits<PacketXi>::size);
  prod = __riscv_vmul_vv_i32m1(prod, half_prod, unpacket_traits<PacketXi>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_min<PacketXi>(const PacketXi& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i32m1_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::max)(), unpacket_traits<PacketXi>::size),
      unpacket_traits<PacketXi>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_max<PacketXi>(const PacketXi& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i32m1_i32m1(
      a, __riscv_vmv_v_x_i32m1((std::numeric_limits<numext::int32_t>::min)(), unpacket_traits<PacketXi>::size),
      unpacket_traits<PacketXi>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXi, N>& kernel) {
  numext::int32_t buffer[unpacket_traits<PacketXi>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(numext::int32_t), kernel.packet[i], unpacket_traits<PacketXi>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_i32m1(&buffer[i * unpacket_traits<PacketXi>::size], unpacket_traits<PacketXi>::size);
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
  PacketXi half1 = __riscv_vmul_vv_i32m1(__riscv_vget_v_i32m4_i32m1(a, 0), __riscv_vget_v_i32m4_i32m1(a, 1),
                                         unpacket_traits<PacketXi>::size);
  PacketXi half2 = __riscv_vmul_vv_i32m1(__riscv_vget_v_i32m4_i32m1(a, 2), __riscv_vget_v_i32m4_i32m1(a, 3),
                                         unpacket_traits<PacketXi>::size);
  return predux_mul<PacketXi>(__riscv_vmul_vv_i32m1(half1, half2, unpacket_traits<PacketXi>::size));
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
  return predux_mul<PacketXi>(__riscv_vmul_vv_i32m1(__riscv_vget_v_i32m2_i32m1(a, 0), __riscv_vget_v_i32m2_i32m1(a, 1),
                                                    unpacket_traits<PacketXi>::size));
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
                            PacketXi>::type
    predux_half_dowto4(const PacketMul2Xi& a) {
  return __riscv_vadd_vv_i32m1(__riscv_vget_v_i32m2_i32m1(a, 0), __riscv_vget_v_i32m2_i32m1(a, 1),
                               unpacket_traits<PacketXi>::size);
}

/********************************* float32 ************************************/

typedef eigen_packet_wrapper<vfloat32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 6> PacketXf;
typedef eigen_packet_wrapper<vfloat32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 7> PacketMul2Xf;
typedef eigen_packet_wrapper<vfloat32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 8> PacketMul4Xf;

template <>
struct packet_traits<float> : default_packet_traits {
  typedef PacketXf type;
  typedef PacketXf half;

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
    HasRound = 0,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

template <>
struct packet_traits<float, 2> : default_packet_traits {
  typedef PacketMul2Xf type;
  typedef PacketXf half;

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
    HasRound = 0,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

template <>
struct packet_traits<float, 4> : default_packet_traits {
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
    HasRound = 0,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

template <>
struct unpacket_traits<PacketXf> {
  typedef float type;
  typedef PacketXf half;  // Half not yet implemented
  typedef PacketXi integer_packet;
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
  typedef PacketXf half;
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

/********************************* PacketXf ************************************/

template <>
EIGEN_STRONG_INLINE PacketXf ptrue<PacketXf>(const PacketXf& /*a*/) {
  return __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(0xffffffffu, unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pzero<PacketXf>(const PacketXf& /*a*/) {
  return __riscv_vfmv_v_f_f32m1(0.0f, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pabs(const PacketXf& a) {
  return __riscv_vfabs_v_f32m1(a, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pset1<PacketXf>(const float& from) {
  return __riscv_vfmv_v_f_f32m1(from, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pset1frombits<PacketXf>(numext::uint32_t from) {
  return __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(from, unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf plset<PacketXf>(const float& a) {
  PacketXf idx =
      __riscv_vfcvt_f_x_v_f32m1(__riscv_vreinterpret_v_u32m1_i32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketXi>::size)), unpacket_traits<PacketXf>::size);
  return __riscv_vfadd_vf_f32m1(idx, a, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf padd<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfadd_vv_f32m1(a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf psub<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfsub_vv_f32m1(a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pnegate(const PacketXf& a) {
  return __riscv_vfneg_v_f32m1(a, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pconj(const PacketXf& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXf pmul<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfmul_vv_f32m1(a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pdiv<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfdiv_vv_f32m1(a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmadd(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfmadd_vv_f32m1(a, b, c, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmsub(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfmsub_vv_f32m1(a, b, c, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pnmadd(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfnmsub_vv_f32m1(a, b, c, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pnmsub(const PacketXf& a, const PacketXf& b, const PacketXf& c) {
  return __riscv_vfnmadd_vv_f32m1(a, b, c, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketXf nans = __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketXf>::size);
  PacketMask32 mask = __riscv_vmfeq_vv_f32m1_b32(a, a, unpacket_traits<PacketXf>::size);
  PacketMask32 mask2 = __riscv_vmfeq_vv_f32m1_b32(b, b, unpacket_traits<PacketXf>::size);
  mask = __riscv_vmand_mm_b32(mask, mask2, unpacket_traits<PacketXf>::size);

  return __riscv_vfmin_vv_f32m1_tumu(mask, nans, a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PropagateNaN, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return pmin<PacketXf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PropagateNumbers, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfmin_vv_f32m1(a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketXf nans = __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketXf>::size);
  PacketMask32 mask = __riscv_vmfeq_vv_f32m1_b32(a, a, unpacket_traits<PacketXf>::size);
  PacketMask32 mask2 = __riscv_vmfeq_vv_f32m1_b32(b, b, unpacket_traits<PacketXf>::size);
  mask = __riscv_vmand_mm_b32(mask, mask2, unpacket_traits<PacketXf>::size);

  return __riscv_vfmax_vv_f32m1_tumu(mask, nans, a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PropagateNaN, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return pmax<PacketXf>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PropagateNumbers, PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vfmax_vv_f32m1(a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_le<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask32 mask = __riscv_vmfle_vv_f32m1_b32(a, b, unpacket_traits<PacketXf>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketXf>(a), ptrue<PacketXf>(a), mask, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_lt<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask32 mask = __riscv_vmflt_vv_f32m1_b32(a, b, unpacket_traits<PacketXf>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketXf>(a), ptrue<PacketXf>(a), mask, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_eq<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask32 mask = __riscv_vmfeq_vv_f32m1_b32(a, b, unpacket_traits<PacketXf>::size);
  return __riscv_vmerge_vvm_f32m1(pzero<PacketXf>(a), ptrue<PacketXf>(a), mask, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_lt_or_nan<PacketXf>(const PacketXf& a, const PacketXf& b) {
  PacketMask32 mask = __riscv_vmfge_vv_f32m1_b32(a, b, unpacket_traits<PacketXf>::size);
  return __riscv_vfmerge_vfm_f32m1(ptrue<PacketXf>(a), 0.0f, mask, unpacket_traits<PacketXf>::size);
}

// Logical Operations are not supported for float, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketXf pand<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf por<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pxor<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vxor_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pandnot<PacketXf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(
      __riscv_vreinterpret_v_f32m1_u32m1(a),
      __riscv_vnot_v_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(b), unpacket_traits<PacketXf>::size),
      unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXf pload<PacketXf>(const float* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_f32m1(from, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploadu<PacketXf>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_f32m1(from, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploaddup<PacketXf>(const float* from) {
  PacketXu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketXf>::size);
  idx = __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(idx, 0xfffffffeu, unpacket_traits<PacketXf>::size), 1,
                              unpacket_traits<PacketXf>::size);
  return __riscv_vloxei32_v_f32m1(from, idx, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploadquad<PacketXf>(const float* from) {
  PacketXu idx = __riscv_vid_v_u32m1(unpacket_traits<PacketXf>::size);
  idx = __riscv_vand_vx_u32m1(idx, 0xfffffffcu, unpacket_traits<PacketXf>::size);
  return __riscv_vloxei32_v_f32m1(from, idx, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const PacketXf& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_f32m1(to, from, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const PacketXf& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_f32m1(to, from, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXf pgather<float, PacketXf>(const float* from, Index stride) {
  return __riscv_vlse32_v_f32m1(from, stride * sizeof(float), unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, PacketXf>(float* to, const PacketXf& from, Index stride) {
  __riscv_vsse32(to, stride * sizeof(float), from, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE float pfirst<PacketXf>(const PacketXf& a) {
  return __riscv_vfmv_f_s_f32m1_f32(a);
}

template <>
EIGEN_STRONG_INLINE PacketXf psqrt(const PacketXf& a) {
  return __riscv_vfsqrt_v_f32m1(a, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf print<PacketXf>(const PacketXf& a) {
  const PacketXf limit = pset1<PacketXf>(static_cast<float>(1 << 23));
  const PacketXf abs_a = pabs(a);

  PacketMask32 mask = __riscv_vmfne_vv_f32m1_b32(a, a, unpacket_traits<PacketXf>::size);
  const PacketXf x = __riscv_vfadd_vv_f32m1_tumu(mask, a, a, a, unpacket_traits<PacketXf>::size);
  const PacketXf new_x = __riscv_vfcvt_f_x_v_f32m1(__riscv_vfcvt_x_f_v_i32m1(a, unpacket_traits<PacketXf>::size),
                                                   unpacket_traits<PacketXf>::size);

  mask = __riscv_vmflt_vv_f32m1_b32(abs_a, limit, unpacket_traits<PacketXf>::size);
  PacketXf signed_x = __riscv_vfsgnj_vv_f32m1(new_x, x, unpacket_traits<PacketXf>::size);
  return __riscv_vmerge_vvm_f32m1(x, signed_x, mask, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pfloor<PacketXf>(const PacketXf& a) {
  PacketXf tmp = print<PacketXf>(a);
  // If greater, subtract one.
  PacketMask32 mask = __riscv_vmflt_vv_f32m1_b32(a, tmp, unpacket_traits<PacketXf>::size);
  return __riscv_vfsub_vf_f32m1_tumu(mask, tmp, tmp, 1.0f, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf preverse(const PacketXf& a) {
  PacketXu idx = __riscv_vrsub_vx_u32m1(__riscv_vid_v_u32m1(unpacket_traits<PacketXf>::size),
                                        unpacket_traits<PacketXf>::size - 1, unpacket_traits<PacketXf>::size);
  return __riscv_vrgather_vv_f32m1(a, idx, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pfrexp<PacketXf>(const PacketXf& a, PacketXf& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE float predux<PacketXf>(const PacketXf& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f32m1_f32m1(
      a, __riscv_vfmv_v_f_f32m1(0.0, unpacket_traits<PacketXf>::size), unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_mul<PacketXf>(const PacketXf& a) {
  // Multiply the vector by its reverse
  PacketXf prod = __riscv_vfmul_vv_f32m1(preverse(a), a, unpacket_traits<PacketXf>::size);
  PacketXf half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_f32m1(prod, 8, unpacket_traits<PacketXf>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketXf>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_f32m1(prod, 4, unpacket_traits<PacketXf>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketXf>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_f32m1(prod, 2, unpacket_traits<PacketXf>::size);
    prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketXf>::size);
  }
  // Last reduction
  half_prod = __riscv_vslidedown_vx_f32m1(prod, 1, unpacket_traits<PacketXf>::size);
  prod = __riscv_vfmul_vv_f32m1(prod, half_prod, unpacket_traits<PacketXf>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketXf>(const PacketXf& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f32m1_f32m1(
      a, __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketXf>::size),
      unpacket_traits<PacketXf>::size)), (std::numeric_limits<float>::max)());
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketXf>(const PacketXf& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f32m1_f32m1(
      a, __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketXf>::size),
      unpacket_traits<PacketXf>::size)), -(std::numeric_limits<float>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXf, N>& kernel) {
  float buffer[unpacket_traits<PacketXf>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse32(&buffer[i], N * sizeof(float), kernel.packet[i], unpacket_traits<PacketXf>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle32_v_f32m1(&buffer[i * unpacket_traits<PacketXf>::size], unpacket_traits<PacketXf>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketXf pldexp<PacketXf>(const PacketXf& a, const PacketXf& exponent) {
  return pldexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE PacketMask32 por(const PacketMask32& a, const PacketMask32& b) {
  return __riscv_vmor_mm_b32(a, b, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMask32 pand(const PacketMask32& a, const PacketMask32& b) {
  return __riscv_vmand_mm_b32(a, b, unpacket_traits<PacketXf>::size);
}

EIGEN_STRONG_INLINE PacketMask32 pcmp_eq_mask(const PacketXf& a, const PacketXf& b) {
  return __riscv_vmfeq_vv_f32m1_b32(a, b, unpacket_traits<PacketXf>::size);
}

EIGEN_STRONG_INLINE PacketMask32 pcmp_lt_mask(const PacketXf& a, const PacketXf& b) {
  return __riscv_vmflt_vv_f32m1_b32(a, b, unpacket_traits<PacketXf>::size);
}

EIGEN_STRONG_INLINE PacketXf pselect(const PacketMask32& mask, const PacketXf& a, const PacketXf& b) {
  return __riscv_vmerge_vvm_f32m1(b, a, mask, unpacket_traits<PacketXf>::size);
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
  PacketMul4Xf idx = __riscv_vfcvt_f_x_v_f32m4(__riscv_vreinterpret_v_u32m4_i32m4(__riscv_vid_v_u32m4(unpacket_traits<PacketMul4Xi>::size)),
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
  PacketXf half1 = __riscv_vfmul_vv_f32m1(__riscv_vget_v_f32m4_f32m1(a, 0), __riscv_vget_v_f32m4_f32m1(a, 1),
                                          unpacket_traits<PacketXf>::size);
  PacketXf half2 = __riscv_vfmul_vv_f32m1(__riscv_vget_v_f32m4_f32m1(a, 2), __riscv_vget_v_f32m4_f32m1(a, 3),
                                          unpacket_traits<PacketXf>::size);
  return predux_mul<PacketXf>(__riscv_vfmul_vv_f32m1(half1, half2, unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketMul4Xf>(const PacketMul4Xf& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f32m4_f32m1(
      a, __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul4Xf>::size / 4),
      unpacket_traits<PacketMul4Xf>::size)), (std::numeric_limits<float>::max)());
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketMul4Xf>(const PacketMul4Xf& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f32m4_f32m1(
      a, __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul4Xf>::size / 4),
      unpacket_traits<PacketMul4Xf>::size)), -(std::numeric_limits<float>::max)());
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
  PacketMul2Xf idx = __riscv_vfcvt_f_x_v_f32m2(__riscv_vreinterpret_v_u32m2_i32m2(__riscv_vid_v_u32m2(unpacket_traits<PacketMul4Xi>::size)),
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
  return predux_mul<PacketXf>(__riscv_vfmul_vv_f32m1(__riscv_vget_v_f32m2_f32m1(a, 0), __riscv_vget_v_f32m2_f32m1(a, 1),
                                                     unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketMul2Xf>(const PacketMul2Xf& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f32m2_f32m1(
      a, __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul2Xf>::size / 2),
      unpacket_traits<PacketMul2Xf>::size)), (std::numeric_limits<float>::max)());
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketMul2Xf>(const PacketMul2Xf& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f32m2_f32m1(
      a, __riscv_vfmv_v_f_f32m1((std::numeric_limits<float>::quiet_NaN)(), unpacket_traits<PacketMul2Xf>::size / 2),
      unpacket_traits<PacketMul2Xf>::size)), -(std::numeric_limits<float>::max)());
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
                            PacketXf>::type
    predux_half_dowto4(const PacketMul2Xf& a) {
  return __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(a, 0), __riscv_vget_v_f32m2_f32m1(a, 1),
                                unpacket_traits<PacketXf>::size);
}

/********************************* int64 **************************************/

typedef eigen_packet_wrapper<vint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 9> PacketXl;
typedef eigen_packet_wrapper<vuint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 10> PacketXul;

typedef eigen_packet_wrapper<vint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 11> PacketMul2Xl;
typedef eigen_packet_wrapper<vuint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 12> PacketMul2Xul;

typedef eigen_packet_wrapper<vint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 13> PacketMul4Xl;
typedef eigen_packet_wrapper<vuint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 14> PacketMul4Xul;

template <>
struct packet_traits<numext::int64_t> : default_packet_traits {
  typedef PacketXl type;
  typedef PacketXl half;  // Half not implemented yet
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

template <>
struct packet_traits<numext::int64_t, 2> : default_packet_traits {
  typedef PacketMul2Xl type;
  typedef PacketXl half;
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

template <>
struct packet_traits<numext::int64_t, 4> : default_packet_traits {
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

template <>
struct unpacket_traits<PacketXl> {
  typedef numext::int64_t type;
  typedef PacketXl half;  // Half not yet implemented
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
  typedef PacketXl half;
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

/********************************* PacketXl ************************************/

template <>
EIGEN_STRONG_INLINE PacketXl pset1<PacketXl>(const numext::int64_t& from) {
  return __riscv_vmv_v_x_i64m1(from, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl plset<PacketXl>(const numext::int64_t& a) {
  PacketXl idx = __riscv_vreinterpret_v_u64m1_i64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketXl>::size));
  return __riscv_vadd_vx_i64m1(idx, a, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pzero<PacketXl>(const PacketXl& /*a*/) {
  return __riscv_vmv_v_x_i64m1(0, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl padd<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vadd_vv_i64m1(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl psub<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pnegate(const PacketXl& a) {
  return __riscv_vneg(a, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pconj(const PacketXl& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXl pmul<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pdiv<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pmadd(const PacketXl& a, const PacketXl& b, const PacketXl& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pmsub(const PacketXl& a, const PacketXl& b, const PacketXl& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pnmadd(const PacketXl& a, const PacketXl& b, const PacketXl& c) {
  return __riscv_vnmsub_vv_i64m1(a, b, c, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pnmsub(const PacketXl& a, const PacketXl& b, const PacketXl& c) {
  return __riscv_vnmsub_vv_i64m1(a, b, pnegate(c), unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pmin<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pmax<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pcmp_le<PacketXl>(const PacketXl& a, const PacketXl& b) {
  PacketMask64 mask = __riscv_vmsle_vv_i64m1_b64(a, b, unpacket_traits<PacketXl>::size);
  return __riscv_vmerge_vxm_i64m1(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pcmp_lt<PacketXl>(const PacketXl& a, const PacketXl& b) {
  PacketMask64 mask = __riscv_vmslt_vv_i64m1_b64(a, b, unpacket_traits<PacketXl>::size);
  return __riscv_vmerge_vxm_i64m1(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pcmp_eq<PacketXl>(const PacketXl& a, const PacketXl& b) {
  PacketMask64 mask = __riscv_vmseq_vv_i64m1_b64(a, b, unpacket_traits<PacketXl>::size);
  return __riscv_vmerge_vxm_i64m1(pzero(a), 0xffffffffffffffff, mask, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl ptrue<PacketXl>(const PacketXl& /*a*/) {
  return __riscv_vmv_v_x_i64m1(0xffffffffffffffffu, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pand<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vand_vv_i64m1(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl por<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vor_vv_i64m1(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pxor<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vxor_vv_i64m1(a, b, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pandnot<PacketXl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vand_vv_i64m1(a, __riscv_vnot_v_i64m1(b, unpacket_traits<PacketXl>::size),
                               unpacket_traits<PacketXl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXl parithmetic_shift_right(PacketXl a) {
  return __riscv_vsra_vx_i64m1(a, N, unpacket_traits<PacketXl>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXl plogical_shift_right(PacketXl a) {
  return __riscv_vreinterpret_i64m1(
      __riscv_vsrl_vx_u64m1(__riscv_vreinterpret_u64m1(a), N, unpacket_traits<PacketXl>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketXl plogical_shift_left(PacketXl a) {
  return __riscv_vsll_vx_i64m1(a, N, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pload<PacketXl>(const numext::int64_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_i64m1(from, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl ploadu<PacketXl>(const numext::int64_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_i64m1(from, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl ploaddup<PacketXl>(const numext::int64_t* from) {
  PacketXul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketXl>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffeu, unpacket_traits<PacketXl>::size), 2,
                              unpacket_traits<PacketXl>::size);
  // idx = 0 0 sizeof(int64_t) sizeof(int64_t) 2*sizeof(int64_t) 2*sizeof(int64_t) ...
  return __riscv_vloxei64_v_i64m1(from, idx, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl ploadquad<PacketXl>(const numext::int64_t* from) {
  PacketXul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketXl>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffcu, unpacket_traits<PacketXl>::size), 1,
                              unpacket_traits<PacketXl>::size);
  ;
  return __riscv_vloxei64_v_i64m1(from, idx, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int64_t>(numext::int64_t* to, const PacketXl& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_i64m1(to, from, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int64_t>(numext::int64_t* to, const PacketXl& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_i64m1(to, from, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXl pgather<numext::int64_t, PacketXl>(const numext::int64_t* from, Index stride) {
  return __riscv_vlse64_v_i64m1(from, stride * sizeof(numext::int64_t), unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int64_t, PacketXl>(numext::int64_t* to, const PacketXl& from,
                                                                  Index stride) {
  __riscv_vsse64(to, stride * sizeof(numext::int64_t), from, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t pfirst<PacketXl>(const PacketXl& a) {
  return __riscv_vmv_x_s_i64m1_i64(a);
}

template <>
EIGEN_STRONG_INLINE PacketXl preverse(const PacketXl& a) {
  PacketXul idx = __riscv_vrsub_vx_u64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketXl>::size),
                                         unpacket_traits<PacketXl>::size - 1, unpacket_traits<PacketXl>::size);
  return __riscv_vrgather_vv_i64m1(a, idx, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pabs(const PacketXl& a) {
  PacketXl mask = __riscv_vsra_vx_i64m1(a, 63, unpacket_traits<PacketXl>::size);
  return __riscv_vsub_vv_i64m1(__riscv_vxor_vv_i64m1(a, mask, unpacket_traits<PacketXl>::size), mask,
                               unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux<PacketXl>(const PacketXl& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i64m1_i64m1(a, __riscv_vmv_v_x_i64m1(0, unpacket_traits<PacketXl>::size),
                                                      unpacket_traits<PacketXl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_mul<PacketXl>(const PacketXl& a) {
  // Multiply the vector by its reverse
  PacketXl prod = __riscv_vmul_vv_i64m1(preverse(a), a, unpacket_traits<PacketXl>::size);
  PacketXl half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_i64m1(prod, 4, unpacket_traits<PacketXl>::size);
    prod = __riscv_vmul_vv_i64m1(prod, half_prod, unpacket_traits<PacketXl>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_i64m1(prod, 2, unpacket_traits<PacketXl>::size);
    prod = __riscv_vmul_vv_i64m1(prod, half_prod, unpacket_traits<PacketXl>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_i64m1(prod, 1, unpacket_traits<PacketXl>::size);
    prod = __riscv_vmul_vv_i64m1(prod, half_prod, unpacket_traits<PacketXl>::size);
  }

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_min<PacketXl>(const PacketXl& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i64m1_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::max)(), unpacket_traits<PacketXl>::size),
      unpacket_traits<PacketXl>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int64_t predux_max<PacketXl>(const PacketXl& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i64m1_i64m1(
      a, __riscv_vmv_v_x_i64m1((std::numeric_limits<numext::int64_t>::min)(), unpacket_traits<PacketXl>::size),
      unpacket_traits<PacketXl>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXl, N>& kernel) {
  numext::int64_t buffer[unpacket_traits<PacketXl>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(numext::int64_t), kernel.packet[i], unpacket_traits<PacketXl>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_i64m1(&buffer[i * unpacket_traits<PacketXl>::size], unpacket_traits<PacketXl>::size);
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
  PacketXl half1 = __riscv_vmul_vv_i64m1(__riscv_vget_v_i64m4_i64m1(a, 0), __riscv_vget_v_i64m4_i64m1(a, 1),
                                         unpacket_traits<PacketXl>::size);
  PacketXl half2 = __riscv_vmul_vv_i64m1(__riscv_vget_v_i64m4_i64m1(a, 2), __riscv_vget_v_i64m4_i64m1(a, 3),
                                         unpacket_traits<PacketXl>::size);
  return predux_mul<PacketXl>(__riscv_vmul_vv_i64m1(half1, half2, unpacket_traits<PacketXl>::size));
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
  return predux_mul<PacketXl>(__riscv_vmul_vv_i64m1(__riscv_vget_v_i64m2_i64m1(a, 0), __riscv_vget_v_i64m2_i64m1(a, 1),
                                                    unpacket_traits<PacketXl>::size));
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
                            PacketXl>::type
    predux_half_dowto4(const PacketMul2Xl& a) {
  return __riscv_vadd_vv_i64m1(__riscv_vget_v_i64m2_i64m1(a, 0), __riscv_vget_v_i64m2_i64m1(a, 1),
                               unpacket_traits<PacketXl>::size);
}

/********************************* double ************************************/

typedef eigen_packet_wrapper<vfloat64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 15> PacketXd;
typedef eigen_packet_wrapper<vfloat64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 16> PacketMul2Xd;
typedef eigen_packet_wrapper<vfloat64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 17> PacketMul4Xd;

template <>
struct packet_traits<double> : default_packet_traits {
  typedef PacketXd type;
  typedef PacketXd half;

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
    HasRound = 0,

    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1
  };
};

template <>
struct packet_traits<double, 2> : default_packet_traits {
  typedef PacketMul2Xd type;
  typedef PacketXd half;

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
    HasRound = 0,

    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1
  };
};

template <>
struct packet_traits<double, 4> : default_packet_traits {
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
    HasRound = 0,

    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1
  };
};

template <>
struct unpacket_traits<PacketXd> {
  typedef double type;
  typedef PacketXd half;  // Half not yet implemented
  typedef PacketXl integer_packet;
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
  typedef PacketXd half;
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

/********************************* PacketXd ************************************/

template <>
EIGEN_STRONG_INLINE PacketXd ptrue<PacketXd>(const PacketXd& /*a*/) {
  return __riscv_vreinterpret_f64m1(__riscv_vmv_v_x_u64m1(0xffffffffffffffffu, unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXd pzero<PacketXd>(const PacketXd& /*a*/) {
  return __riscv_vfmv_v_f_f64m1(0.0, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pabs(const PacketXd& a) {
  return __riscv_vfabs_v_f64m1(a, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pset1<PacketXd>(const double& from) {
  return __riscv_vfmv_v_f_f64m1(from, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pset1frombits<PacketXd>(numext::uint64_t from) {
  return __riscv_vreinterpret_f64m1(__riscv_vmv_v_x_u64m1(from, unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXd plset<PacketXd>(const double& a) {
  PacketXd idx =
      __riscv_vfcvt_f_x_v_f64m1(__riscv_vreinterpret_v_u64m1_i64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketXl>::size)), unpacket_traits<PacketXd>::size);
  return __riscv_vfadd_vf_f64m1(idx, a, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd padd<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vfadd_vv_f64m1(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd psub<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vfsub_vv_f64m1(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pnegate(const PacketXd& a) {
  return __riscv_vfneg_v_f64m1(a, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pconj(const PacketXd& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXd pmul<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vfmul_vv_f64m1(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pdiv<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vfdiv_vv_f64m1(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmadd(const PacketXd& a, const PacketXd& b, const PacketXd& c) {
  return __riscv_vfmadd_vv_f64m1(a, b, c, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmsub(const PacketXd& a, const PacketXd& b, const PacketXd& c) {
  return __riscv_vfmsub_vv_f64m1(a, b, c, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pnmadd(const PacketXd& a, const PacketXd& b, const PacketXd& c) {
  return __riscv_vfnmsub_vv_f64m1(a, b, c, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pnmsub(const PacketXd& a, const PacketXd& b, const PacketXd& c) {
  return __riscv_vfnmadd_vv_f64m1(a, b, c, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmin<PacketXd>(const PacketXd& a, const PacketXd& b) {
  PacketXd nans = __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketXd>::size);
  PacketMask64 mask = __riscv_vmfeq_vv_f64m1_b64(a, a, unpacket_traits<PacketXd>::size);
  PacketMask64 mask2 = __riscv_vmfeq_vv_f64m1_b64(b, b, unpacket_traits<PacketXd>::size);
  mask = __riscv_vmand_mm_b64(mask, mask2, unpacket_traits<PacketXd>::size);

  return __riscv_vfmin_vv_f64m1_tumu(mask, nans, a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmin<PropagateNaN, PacketXd>(const PacketXd& a, const PacketXd& b) {
  return pmin<PacketXd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmin<PropagateNumbers, PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vfmin_vv_f64m1(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmax<PacketXd>(const PacketXd& a, const PacketXd& b) {
  PacketXd nans = __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketXd>::size);
  PacketMask64 mask = __riscv_vmfeq_vv_f64m1_b64(a, a, unpacket_traits<PacketXd>::size);
  PacketMask64 mask2 = __riscv_vmfeq_vv_f64m1_b64(b, b, unpacket_traits<PacketXd>::size);
  mask = __riscv_vmand_mm_b64(mask, mask2, unpacket_traits<PacketXd>::size);

  return __riscv_vfmax_vv_f64m1_tumu(mask, nans, a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmax<PropagateNaN, PacketXd>(const PacketXd& a, const PacketXd& b) {
  return pmax<PacketXd>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXd pmax<PropagateNumbers, PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vfmax_vv_f64m1(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pcmp_le<PacketXd>(const PacketXd& a, const PacketXd& b) {
  PacketMask64 mask = __riscv_vmfle_vv_f64m1_b64(a, b, unpacket_traits<PacketXd>::size);
  return __riscv_vmerge_vvm_f64m1(pzero<PacketXd>(a), ptrue<PacketXd>(a), mask, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pcmp_lt<PacketXd>(const PacketXd& a, const PacketXd& b) {
  PacketMask64 mask = __riscv_vmflt_vv_f64m1_b64(a, b, unpacket_traits<PacketXd>::size);
  return __riscv_vmerge_vvm_f64m1(pzero<PacketXd>(a), ptrue<PacketXd>(a), mask, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pcmp_eq<PacketXd>(const PacketXd& a, const PacketXd& b) {
  PacketMask64 mask = __riscv_vmfeq_vv_f64m1_b64(a, b, unpacket_traits<PacketXd>::size);
  return __riscv_vmerge_vvm_f64m1(pzero<PacketXd>(a), ptrue<PacketXd>(a), mask, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pcmp_lt_or_nan<PacketXd>(const PacketXd& a, const PacketXd& b) {
  PacketMask64 mask = __riscv_vmfge_vv_f64m1_b64(a, b, unpacket_traits<PacketXd>::size);
  return __riscv_vfmerge_vfm_f64m1(ptrue<PacketXd>(a), 0.0, mask, unpacket_traits<PacketXd>::size);
}

// Logical Operations are not supported for double, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketXd pand<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a), __riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXd por<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a), __riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXd pxor<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vxor_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a), __riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXd pandnot<PacketXd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(
      __riscv_vreinterpret_v_f64m1_u64m1(a),
      __riscv_vnot_v_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(b), unpacket_traits<PacketXd>::size),
      unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXd pload<PacketXd>(const double* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle64_v_f64m1(from, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd ploadu<PacketXd>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle64_v_f64m1(from, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd ploaddup<PacketXd>(const double* from) {
  PacketXul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketXd>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffeu, unpacket_traits<PacketXd>::size), 2,
                              unpacket_traits<PacketXd>::size);
  return __riscv_vloxei64_v_f64m1(from, idx, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd ploadquad<PacketXd>(const double* from) {
  PacketXul idx = __riscv_vid_v_u64m1(unpacket_traits<PacketXd>::size);
  idx = __riscv_vsll_vx_u64m1(__riscv_vand_vx_u64m1(idx, 0xfffffffffffffffcu, unpacket_traits<PacketXd>::size), 1,
                              unpacket_traits<PacketXd>::size);
  ;
  return __riscv_vloxei64_v_f64m1(from, idx, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<double>(double* to, const PacketXd& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse64_v_f64m1(to, from, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const PacketXd& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse64_v_f64m1(to, from, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXd pgather<double, PacketXd>(const double* from, Index stride) {
  return __riscv_vlse64_v_f64m1(from, stride * sizeof(double), unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<double, PacketXd>(double* to, const PacketXd& from, Index stride) {
  __riscv_vsse64(to, stride * sizeof(double), from, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE double pfirst<PacketXd>(const PacketXd& a) {
  return __riscv_vfmv_f_s_f64m1_f64(a);
}

template <>
EIGEN_STRONG_INLINE PacketXd psqrt(const PacketXd& a) {
  return __riscv_vfsqrt_v_f64m1(a, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd print<PacketXd>(const PacketXd& a) {
  const PacketXd limit = pset1<PacketXd>(static_cast<double>(1ull << 52));
  const PacketXd abs_a = pabs(a);

  PacketMask64 mask = __riscv_vmfne_vv_f64m1_b64(a, a, unpacket_traits<PacketXd>::size);
  const PacketXd x = __riscv_vfadd_vv_f64m1_tumu(mask, a, a, a, unpacket_traits<PacketXd>::size);
  const PacketXd new_x = __riscv_vfcvt_f_x_v_f64m1(__riscv_vfcvt_x_f_v_i64m1(a, unpacket_traits<PacketXd>::size),
                                                   unpacket_traits<PacketXd>::size);

  mask = __riscv_vmflt_vv_f64m1_b64(abs_a, limit, unpacket_traits<PacketXd>::size);
  PacketXd signed_x = __riscv_vfsgnj_vv_f64m1(new_x, x, unpacket_traits<PacketXd>::size);
  return __riscv_vmerge_vvm_f64m1(x, signed_x, mask, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pfloor<PacketXd>(const PacketXd& a) {
  PacketXd tmp = print<PacketXd>(a);
  // If greater, subtract one.
  PacketMask64 mask = __riscv_vmflt_vv_f64m1_b64(a, tmp, unpacket_traits<PacketXd>::size);
  return __riscv_vfsub_vf_f64m1_tumu(mask, tmp, tmp, 1.0, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd preverse(const PacketXd& a) {
  PacketXul idx = __riscv_vrsub_vx_u64m1(__riscv_vid_v_u64m1(unpacket_traits<PacketXd>::size),
                                         unpacket_traits<PacketXd>::size - 1, unpacket_traits<PacketXd>::size);
  return __riscv_vrgather_vv_f64m1(a, idx, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd pfrexp<PacketXd>(const PacketXd& a, PacketXd& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE double predux<PacketXd>(const PacketXd& a) {
  return __riscv_vfmv_f(__riscv_vfredusum_vs_f64m1_f64m1(
      a, __riscv_vfmv_v_f_f64m1(0.0, unpacket_traits<PacketXd>::size), unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_mul<PacketXd>(const PacketXd& a) {
  // Multiply the vector by its reverse
  PacketXd prod = __riscv_vfmul_vv_f64m1(preverse(a), a, unpacket_traits<PacketXd>::size);
  PacketXd half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_f64m1(prod, 4, unpacket_traits<PacketXd>::size);
    prod = __riscv_vfmul_vv_f64m1(prod, half_prod, unpacket_traits<PacketXd>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_f64m1(prod, 2, unpacket_traits<PacketXd>::size);
    prod = __riscv_vfmul_vv_f64m1(prod, half_prod, unpacket_traits<PacketXd>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_f64m1(prod, 1, unpacket_traits<PacketXd>::size);
    prod = __riscv_vfmul_vv_f64m1(prod, half_prod, unpacket_traits<PacketXd>::size);
  }

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE double predux_min<PacketXd>(const PacketXd& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f64m1_f64m1(
      a, __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketXd>::size),
      unpacket_traits<PacketXd>::size)), (std::numeric_limits<double>::max)());
}

template <>
EIGEN_STRONG_INLINE double predux_max<PacketXd>(const PacketXd& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f64m1_f64m1(
      a, __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketXd>::size),
      unpacket_traits<PacketXd>::size)), -(std::numeric_limits<double>::max)());
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXd, N>& kernel) {
  double buffer[unpacket_traits<PacketXd>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse64(&buffer[i], N * sizeof(double), kernel.packet[i], unpacket_traits<PacketXd>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle64_v_f64m1(&buffer[i * unpacket_traits<PacketXd>::size], unpacket_traits<PacketXd>::size);
  }
}

template <>
EIGEN_STRONG_INLINE PacketXd pldexp<PacketXd>(const PacketXd& a, const PacketXd& exponent) {
  return pldexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE PacketMask64 por(const PacketMask64& a, const PacketMask64& b) {
  return __riscv_vmor_mm_b64(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMask64 pandnot(const PacketMask64& a, const PacketMask64& b) {
  return __riscv_vmor_mm_b64(a, b, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMask64 pand(const PacketMask64& a, const PacketMask64& b) {
  return __riscv_vmand_mm_b64(a, b, unpacket_traits<PacketXd>::size);
}

EIGEN_STRONG_INLINE PacketMask64 pcmp_eq_mask(const PacketXd& a, const PacketXd& b) {
  return __riscv_vmfeq_vv_f64m1_b64(a, b, unpacket_traits<PacketXd>::size);
}

EIGEN_STRONG_INLINE PacketMask64 pcmp_lt_mask(const PacketXd& a, const PacketXd& b) {
  return __riscv_vmflt_vv_f64m1_b64(a, b, unpacket_traits<PacketXd>::size);
}

EIGEN_STRONG_INLINE PacketXd pselect(const PacketMask64& mask, const PacketXd& a, const PacketXd& b) {
  return __riscv_vmerge_vvm_f64m1(b, a, mask, unpacket_traits<PacketXd>::size);
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
  PacketMul4Xd idx = __riscv_vfcvt_f_x_v_f64m4(__riscv_vreinterpret_v_u64m4_i64m4(__riscv_vid_v_u64m4(unpacket_traits<PacketMul4Xi>::size)),
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
  PacketXd half1 = __riscv_vfmul_vv_f64m1(__riscv_vget_v_f64m4_f64m1(a, 0), __riscv_vget_v_f64m4_f64m1(a, 1),
                                          unpacket_traits<PacketXd>::size);
  PacketXd half2 = __riscv_vfmul_vv_f64m1(__riscv_vget_v_f64m4_f64m1(a, 2), __riscv_vget_v_f64m4_f64m1(a, 3),
                                          unpacket_traits<PacketXd>::size);
  return predux_mul<PacketXd>(__riscv_vfmul_vv_f64m1(half1, half2, unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_min<PacketMul4Xd>(const PacketMul4Xd& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f64m4_f64m1(
      a, __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul4Xd>::size / 4),
      unpacket_traits<PacketMul4Xd>::size)), (std::numeric_limits<double>::max)());
}

template <>
EIGEN_STRONG_INLINE double predux_max<PacketMul4Xd>(const PacketMul4Xd& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f64m4_f64m1(
      a, __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul4Xd>::size / 4),
      unpacket_traits<PacketMul4Xd>::size)), -(std::numeric_limits<double>::max)());
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
  PacketMul2Xd idx = __riscv_vfcvt_f_x_v_f64m2(__riscv_vreinterpret_v_u64m2_i64m2(__riscv_vid_v_u64m2(unpacket_traits<PacketMul4Xi>::size)),
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
  return predux_mul<PacketXd>(__riscv_vfmul_vv_f64m1(__riscv_vget_v_f64m2_f64m1(a, 0), __riscv_vget_v_f64m2_f64m1(a, 1),
                                                     unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE double predux_min<PacketMul2Xd>(const PacketMul2Xd& a) {
  return (std::min)(__riscv_vfmv_f(__riscv_vfredmin_vs_f64m2_f64m1(
      a, __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul2Xd>::size / 2),
      unpacket_traits<PacketMul2Xd>::size)), (std::numeric_limits<double>::max)());
}

template <>
EIGEN_STRONG_INLINE double predux_max<PacketMul2Xd>(const PacketMul2Xd& a) {
  return (std::max)(__riscv_vfmv_f(__riscv_vfredmax_vs_f64m2_f64m1(
      a, __riscv_vfmv_v_f_f64m1((std::numeric_limits<double>::quiet_NaN)(), unpacket_traits<PacketMul2Xd>::size / 2),
      unpacket_traits<PacketMul2Xd>::size)), -(std::numeric_limits<double>::max)());
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
                            PacketXd>::type
    predux_half_dowto4(const PacketMul2Xd& a) {
  return __riscv_vfadd_vv_f64m1(__riscv_vget_v_f64m2_f64m1(a, 0), __riscv_vget_v_f64m2_f64m1(a, 1),
                                unpacket_traits<PacketXd>::size);
}

/********************************* short **************************************/

typedef eigen_packet_wrapper<vint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 18> PacketXs;
typedef eigen_packet_wrapper<vuint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 19> PacketXsu;

typedef eigen_packet_wrapper<vint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 20> PacketMul2Xs;
typedef eigen_packet_wrapper<vuint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 21> PacketMul2Xsu;

typedef eigen_packet_wrapper<vint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 22> PacketMul4Xs;
typedef eigen_packet_wrapper<vuint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 23> PacketMul4Xsu;

template <>
struct packet_traits<numext::int16_t> : default_packet_traits {
  typedef PacketXs type;
  typedef PacketXs half;  // Half not implemented yet
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

template <>
struct packet_traits<numext::int16_t, 2> : default_packet_traits {
  typedef PacketMul2Xs type;
  typedef PacketXs half;
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

template <>
struct packet_traits<numext::int16_t, 4> : default_packet_traits {
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

template <>
struct unpacket_traits<PacketXs> {
  typedef numext::int16_t type;
  typedef PacketXs half;  // Half not yet implemented
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
  typedef PacketXs half;
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

/********************************* PacketXs ************************************/

template <>
EIGEN_STRONG_INLINE PacketXs pset1<PacketXs>(const numext::int16_t& from) {
  return __riscv_vmv_v_x_i16m1(from, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs plset<PacketXs>(const numext::int16_t& a) {
  PacketXs idx = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vid_v_u16m1(unpacket_traits<PacketXs>::size));
  return __riscv_vadd_vx_i16m1(idx, a, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pzero<PacketXs>(const PacketXs& /*a*/) {
  return __riscv_vmv_v_x_i16m1(0, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs padd<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vadd_vv_i16m1(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs psub<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vsub(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pnegate(const PacketXs& a) {
  return __riscv_vneg(a, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pconj(const PacketXs& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXs pmul<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vmul(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pdiv<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vdiv(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pmadd(const PacketXs& a, const PacketXs& b, const PacketXs& c) {
  return __riscv_vmadd(a, b, c, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pmsub(const PacketXs& a, const PacketXs& b, const PacketXs& c) {
  return __riscv_vmadd(a, b, pnegate(c), unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pnmadd(const PacketXs& a, const PacketXs& b, const PacketXs& c) {
  return __riscv_vnmsub_vv_i16m1(a, b, c, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pnmsub(const PacketXs& a, const PacketXs& b, const PacketXs& c) {
  return __riscv_vnmsub_vv_i16m1(a, b, pnegate(c), unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pmin<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vmin(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pmax<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vmax(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pcmp_le<PacketXs>(const PacketXs& a, const PacketXs& b) {
  PacketMask16 mask = __riscv_vmsle_vv_i16m1_b16(a, b, unpacket_traits<PacketXs>::size);
  return __riscv_vmerge_vxm_i16m1(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pcmp_lt<PacketXs>(const PacketXs& a, const PacketXs& b) {
  PacketMask16 mask = __riscv_vmslt_vv_i16m1_b16(a, b, unpacket_traits<PacketXs>::size);
  return __riscv_vmerge_vxm_i16m1(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pcmp_eq<PacketXs>(const PacketXs& a, const PacketXs& b) {
  PacketMask16 mask = __riscv_vmseq_vv_i16m1_b16(a, b, unpacket_traits<PacketXs>::size);
  return __riscv_vmerge_vxm_i16m1(pzero(a), static_cast<short>(0xffff), mask, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs ptrue<PacketXs>(const PacketXs& /*a*/) {
  return __riscv_vmv_v_x_i16m1(static_cast<unsigned short>(0xffffu), unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pand<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vand_vv_i16m1(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs por<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vor_vv_i16m1(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pxor<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vxor_vv_i16m1(a, b, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pandnot<PacketXs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vand_vv_i16m1(a, __riscv_vnot_v_i16m1(b, unpacket_traits<PacketXs>::size),
                               unpacket_traits<PacketXs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXs parithmetic_shift_right(PacketXs a) {
  return __riscv_vsra_vx_i16m1(a, N, unpacket_traits<PacketXs>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXs plogical_shift_right(PacketXs a) {
  return __riscv_vreinterpret_i16m1(
      __riscv_vsrl_vx_u16m1(__riscv_vreinterpret_u16m1(a), N, unpacket_traits<PacketXs>::size));
}

template <int N>
EIGEN_STRONG_INLINE PacketXs plogical_shift_left(PacketXs a) {
  return __riscv_vsll_vx_i16m1(a, N, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pload<PacketXs>(const numext::int16_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle16_v_i16m1(from, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs ploadu<PacketXs>(const numext::int16_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle16_v_i16m1(from, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs ploaddup<PacketXs>(const numext::int16_t* from) {
  PacketXsu idx = __riscv_vid_v_u16m1(unpacket_traits<PacketXs>::size);
  idx = __riscv_vand_vx_u16m1(idx, 0xfffeu, unpacket_traits<PacketXs>::size);
  // idx = 0 0 sizeof(int16_t) sizeof(int16_t) 2*sizeof(int16_t) 2*sizeof(int16_t) ...
  return __riscv_vloxei16_v_i16m1(from, idx, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs ploadquad<PacketXs>(const numext::int16_t* from) {
  PacketXsu idx = __riscv_vid_v_u16m1(unpacket_traits<PacketXs>::size);
  idx = __riscv_vsrl_vx_u16m1(__riscv_vand_vx_u16m1(idx, 0xfffcu, unpacket_traits<PacketXs>::size), 1,
                              unpacket_traits<PacketXs>::size);
  return __riscv_vloxei16_v_i16m1(from, idx, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int16_t>(numext::int16_t* to, const PacketXs& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse16_v_i16m1(to, from, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int16_t>(numext::int16_t* to, const PacketXs& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse16_v_i16m1(to, from, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXs pgather<numext::int16_t, PacketXs>(const numext::int16_t* from, Index stride) {
  return __riscv_vlse16_v_i16m1(from, stride * sizeof(numext::int16_t), unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int16_t, PacketXs>(numext::int16_t* to, const PacketXs& from,
                                                                  Index stride) {
  __riscv_vsse16(to, stride * sizeof(numext::int16_t), from, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t pfirst<PacketXs>(const PacketXs& a) {
  return __riscv_vmv_x_s_i16m1_i16(a);
}

template <>
EIGEN_STRONG_INLINE PacketXs preverse(const PacketXs& a) {
  PacketXsu idx = __riscv_vrsub_vx_u16m1(__riscv_vid_v_u16m1(unpacket_traits<PacketXs>::size),
                                         unpacket_traits<PacketXs>::size - 1, unpacket_traits<PacketXs>::size);
  return __riscv_vrgather_vv_i16m1(a, idx, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pabs(const PacketXs& a) {
  PacketXs mask = __riscv_vsra_vx_i16m1(a, 15, unpacket_traits<PacketXs>::size);
  return __riscv_vsub_vv_i16m1(__riscv_vxor_vv_i16m1(a, mask, unpacket_traits<PacketXs>::size), mask,
                               unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux<PacketXs>(const PacketXs& a) {
  return __riscv_vmv_x(__riscv_vredsum_vs_i16m1_i16m1(a, __riscv_vmv_v_x_i16m1(0, unpacket_traits<PacketXs>::size),
                                                      unpacket_traits<PacketXs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_mul<PacketXs>(const PacketXs& a) {
  // Multiply the vector by its reverse
  PacketXs prod = __riscv_vmul_vv_i16m1(preverse(a), a, unpacket_traits<PacketXs>::size);
  PacketXs half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_i16m1(prod, 16, unpacket_traits<PacketXs>::size);
    prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketXs>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_i16m1(prod, 8, unpacket_traits<PacketXs>::size);
    prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketXs>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_i16m1(prod, 4, unpacket_traits<PacketXs>::size);
    prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketXs>::size);
  }
  // Last reduction
  half_prod = __riscv_vslidedown_vx_i16m1(prod, 2, unpacket_traits<PacketXs>::size);
  prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketXs>::size);

  half_prod = __riscv_vslidedown_vx_i16m1(prod, 1, unpacket_traits<PacketXs>::size);
  prod = __riscv_vmul_vv_i16m1(prod, half_prod, unpacket_traits<PacketXs>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_min<PacketXs>(const PacketXs& a) {
  return __riscv_vmv_x(__riscv_vredmin_vs_i16m1_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::max)(), unpacket_traits<PacketXs>::size),
      unpacket_traits<PacketXs>::size));
}

template <>
EIGEN_STRONG_INLINE numext::int16_t predux_max<PacketXs>(const PacketXs& a) {
  return __riscv_vmv_x(__riscv_vredmax_vs_i16m1_i16m1(
      a, __riscv_vmv_v_x_i16m1((std::numeric_limits<numext::int16_t>::min)(), unpacket_traits<PacketXs>::size),
      unpacket_traits<PacketXs>::size));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXs, N>& kernel) {
  numext::int16_t buffer[unpacket_traits<PacketXs>::size * N] = {0};
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse16(&buffer[i], N * sizeof(numext::int16_t), kernel.packet[i], unpacket_traits<PacketXs>::size);
  }
  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle16_v_i16m1(&buffer[i * unpacket_traits<PacketXs>::size], unpacket_traits<PacketXs>::size);
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
  PacketXs half1 = __riscv_vmul_vv_i16m1(__riscv_vget_v_i16m4_i16m1(a, 0), __riscv_vget_v_i16m4_i16m1(a, 1),
                                         unpacket_traits<PacketXs>::size);
  PacketXs half2 = __riscv_vmul_vv_i16m1(__riscv_vget_v_i16m4_i16m1(a, 2), __riscv_vget_v_i16m4_i16m1(a, 3),
                                         unpacket_traits<PacketXs>::size);
  return predux_mul<PacketXs>(__riscv_vmul_vv_i16m1(half1, half2, unpacket_traits<PacketXs>::size));
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
  return predux_mul<PacketXs>(__riscv_vmul_vv_i16m1(__riscv_vget_v_i16m2_i16m1(a, 0), __riscv_vget_v_i16m2_i16m1(a, 1),
                                                    unpacket_traits<PacketXs>::size));
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
                            PacketXs>::type
    predux_half_dowto4(const PacketMul2Xs& a) {
  return __riscv_vadd_vv_i16m1(__riscv_vget_v_i16m2_i16m1(a, 0), __riscv_vget_v_i16m2_i16m1(a, 1),
                               unpacket_traits<PacketXs>::size);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET_MATH_RVV10_H
