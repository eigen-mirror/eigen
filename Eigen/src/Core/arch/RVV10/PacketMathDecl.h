// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
// Copyright (C) 2025 Chip Kerchner <ckerchner@tenstorrent.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_PACKET_MATH_RVV10_DECL_H
#define EIGEN_PACKET_MATH_RVV10_DECL_H

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

using PacketMask64 = vbool64_t;
using PacketMask32 = vbool32_t;
using PacketMask16 = vbool16_t;
using PacketMask8 = vbool8_t;
using PacketMask4 = vbool4_t;

template <typename Packet>
struct rvv_half_packet {
  using type = Packet;
};

template <typename Packet>
using rvv_half_packet_t = typename rvv_half_packet<Packet>::type;

template <typename Scalar, typename Packet, std::size_t VectorLMul>
struct rvv_default_unpacket_traits {
  using type = Scalar;
  using half = rvv_half_packet_t<Packet>;
  using mask_t = numext::uint8_t;
  enum {
    size = rvv_packet_size_selector<Scalar, EIGEN_RISCV64_RVV_VL, VectorLMul>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, VectorLMul>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

/********************************* short **************************************/

using Packet1Xs = eigen_packet_wrapper<vint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 18>;
using Packet1Xsu = eigen_packet_wrapper<vuint16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 19>;

using Packet2Xs = eigen_packet_wrapper<vint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 20>;
using Packet2Xsu =
    eigen_packet_wrapper<vuint16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 21>;

using Packet4Xs = eigen_packet_wrapper<vint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 22>;
using Packet4Xsu =
    eigen_packet_wrapper<vuint16m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 23>;

template <>
struct rvv_half_packet<Packet2Xs> {
  using type = Packet1Xs;
};
template <>
struct rvv_half_packet<Packet4Xs> {
  using type = Packet2Xs;
};

template <>
struct unpacket_traits<Packet1Xs> : rvv_default_unpacket_traits<numext::int16_t, Packet1Xs, 1> {};

template <>
struct unpacket_traits<Packet2Xs> : rvv_default_unpacket_traits<numext::int16_t, Packet2Xs, 2> {};

template <>
struct unpacket_traits<Packet4Xs> : rvv_default_unpacket_traits<numext::int16_t, Packet4Xs, 4> {};

/********************************* int32 **************************************/
using Packet1Xi = eigen_packet_wrapper<vint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 0>;
using Packet1Xu = eigen_packet_wrapper<vuint32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 1>;

using Packet2Xi = eigen_packet_wrapper<vint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 2>;
using Packet2Xu = eigen_packet_wrapper<vuint32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 3>;

using Packet4Xi = eigen_packet_wrapper<vint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 4>;
using Packet4Xu = eigen_packet_wrapper<vuint32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 5>;

template <>
struct rvv_half_packet<Packet2Xi> {
  using type = Packet1Xi;
};
template <>
struct rvv_half_packet<Packet4Xi> {
  using type = Packet2Xi;
};

template <>
struct unpacket_traits<Packet1Xi> : rvv_default_unpacket_traits<numext::int32_t, Packet1Xi, 1> {};
template <>
struct unpacket_traits<Packet2Xi> : rvv_default_unpacket_traits<numext::int32_t, Packet2Xi, 2> {};
template <>
struct unpacket_traits<Packet4Xi> : rvv_default_unpacket_traits<numext::int32_t, Packet4Xi, 4> {};

/********************************* int64 **************************************/

using Packet1Xl = eigen_packet_wrapper<vint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 9>;
using Packet1Xul = eigen_packet_wrapper<vuint64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 10>;

using Packet2Xl = eigen_packet_wrapper<vint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 11>;
using Packet2Xul =
    eigen_packet_wrapper<vuint64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 12>;

using Packet4Xl = eigen_packet_wrapper<vint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 13>;
using Packet4Xul =
    eigen_packet_wrapper<vuint64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 14>;

template <>
struct rvv_half_packet<Packet2Xl> {
  using type = Packet1Xl;
};
template <>
struct rvv_half_packet<Packet4Xl> {
  using type = Packet2Xl;
};

template <>
struct unpacket_traits<Packet1Xl> : rvv_default_unpacket_traits<numext::int64_t, Packet1Xl, 1> {};

template <>
struct unpacket_traits<Packet2Xl> : rvv_default_unpacket_traits<numext::int64_t, Packet2Xl, 2> {};

template <>
struct unpacket_traits<Packet4Xl> : rvv_default_unpacket_traits<numext::int64_t, Packet4Xl, 4> {};

/********************************* float32 ************************************/

using Packet1Xf = eigen_packet_wrapper<vfloat32m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 6>;
using Packet2Xf =
    eigen_packet_wrapper<vfloat32m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 7>;
using Packet4Xf =
    eigen_packet_wrapper<vfloat32m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 8>;

template <>
struct rvv_half_packet<Packet2Xf> {
  using type = Packet1Xf;
};
template <>
struct rvv_half_packet<Packet4Xf> {
  using type = Packet2Xf;
};

template <>
struct unpacket_traits<Packet1Xf> : rvv_default_unpacket_traits<float, Packet1Xf, 1> {
  using integer_packet = Packet1Xi;
  using packet_mask = PacketMask32;
};

template <>
struct unpacket_traits<Packet2Xf> : rvv_default_unpacket_traits<float, Packet2Xf, 2> {
  using integer_packet = Packet2Xi;
  using packet_mask = PacketMask16;
};

template <>
struct unpacket_traits<Packet4Xf> : rvv_default_unpacket_traits<float, Packet4Xf, 4> {
  using integer_packet = Packet4Xi;
  using packet_mask = PacketMask8;
};

/********************************* double ************************************/

using Packet1Xd = eigen_packet_wrapper<vfloat64m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 15>;
using Packet2Xd =
    eigen_packet_wrapper<vfloat64m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 16>;
using Packet4Xd =
    eigen_packet_wrapper<vfloat64m4_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 4))), 17>;

template <>
struct rvv_half_packet<Packet2Xd> {
  using type = Packet1Xd;
};
template <>
struct rvv_half_packet<Packet4Xd> {
  using type = Packet2Xd;
};

template <>
struct unpacket_traits<Packet1Xd> : rvv_default_unpacket_traits<double, Packet1Xd, 1> {
  using integer_packet = Packet1Xl;
  using packet_mask = PacketMask64;
};

template <>
struct unpacket_traits<Packet2Xd> : rvv_default_unpacket_traits<double, Packet2Xd, 2> {
  using integer_packet = Packet2Xl;
  using packet_mask = PacketMask32;
};

template <>
struct unpacket_traits<Packet4Xd> : rvv_default_unpacket_traits<double, Packet4Xd, 4> {
  using integer_packet = Packet4Xl;
  using packet_mask = PacketMask16;
};

/********************************* char ************************************/

using Packet1Xc = eigen_packet_wrapper<vint8m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 28>;

template <>
struct unpacket_traits<Packet1Xc> : rvv_default_unpacket_traits<numext::int8_t, Packet1Xc, 1> {};

/********************************* default **************************************/

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
using PacketXs = Packet1Xs;
using PacketXsu = Packet1Xsu;
using PacketXi = Packet1Xi;
using PacketXu = Packet1Xu;
using PacketXl = Packet1Xl;
using PacketXul = Packet1Xul;
using PacketXf = Packet1Xf;
using PacketXd = Packet1Xd;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 2
using PacketXs = Packet2Xs;
using PacketXsu = Packet2Xsu;
using PacketXi = Packet2Xi;
using PacketXu = Packet2Xu;
using PacketXl = Packet2Xl;
using PacketXul = Packet2Xul;
using PacketXf = Packet2Xf;
using PacketXd = Packet2Xd;
#elif EIGEN_RISCV64_DEFAULT_LMUL == 4
using PacketXs = Packet4Xs;
using PacketXsu = Packet4Xsu;
using PacketXi = Packet4Xi;
using PacketXu = Packet4Xu;
using PacketXl = Packet4Xl;
using PacketXul = Packet4Xul;
using PacketXf = Packet4Xf;
using PacketXd = Packet4Xd;
#endif

template <typename Scalar, typename Packet>
struct rvv_default_packet_traits : default_packet_traits {
  using type = Packet;
  using half = rvv_half_packet_t<type>;

  enum {
    size = unpacket_traits<Packet>::size,
    Vectorizable = 1,
    AlignedOnScalar = 1,
    HasCmp = 1
  };
};

template <typename Scalar, typename Packet>
struct rvv_default_float_packet_traits : rvv_default_packet_traits<Scalar, Packet> {
  enum {
    HasDiv = 1,
    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasTan = EIGEN_FAST_MATH,
    HasSinh = 1,
    HasCosh = 1,
    HasASinh = 1,
    HasACosh = 1,
    HasATan = 1,
    HasATanh = 1,
    HasLog = 1,
    HasExp = 1,
    HasLog10 = 1,
    HasLog1p = 1,
    HasExpm1 = 1,
    HasPow = 1,
    HasSqrt = 1,
    HasCbrt = 1,
    HasRsqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH,
    HasErfc = EIGEN_FAST_MATH
  };
};

template <>
struct packet_traits<numext::int16_t> : rvv_default_packet_traits<numext::int16_t, PacketXs> {};

template <>
struct packet_traits<numext::int32_t> : rvv_default_packet_traits<numext::int32_t, PacketXi> {};

template <>
struct packet_traits<numext::int64_t> : rvv_default_packet_traits<numext::int64_t, PacketXl> {};

template <>
struct packet_traits<float> : rvv_default_float_packet_traits<float, PacketXf> {};

template <>
struct packet_traits<double> : rvv_default_float_packet_traits<double, PacketXd> {};

/********************************* prefetch **************************************/

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int16_t>(const numext::int16_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int32_t>(const numext::int32_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int64_t>(const numext::int64_t* addr) {
#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  __builtin_prefetch(addr);
#endif
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET_MATH_RVV10_DECL_H
