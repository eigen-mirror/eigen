// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_COMPLEX_SVE_H
#define EIGEN_COMPLEX_SVE_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// A complex packet is one real packet holding interleaved {re, im} pairs. That
// layout is not a choice: GenericPacketMathComplex.h and ConjHelper.h both index
// the member `v` and construct a complex packet from a real one.
//
// std::complex<double> is not vectorized here yet. plog_complex and pexp_complex
// reach plog and pexp for PacketXd, which need a 64-bit integer packet this
// backend does not define, and packetmath instantiates plog for any vectorizable
// complex type regardless of HasLog -- so complex<double> has to wait for the
// double transcendentals rather than ship with the flags turned off.
struct PacketXcf {
  EIGEN_STRONG_INLINE PacketXcf() {}
  EIGEN_STRONG_INLINE explicit PacketXcf(const PacketXf& a) : v(a) {}
  PacketXf v;
};

template <>
struct packet_traits<std::complex<float>> : default_packet_traits {
  typedef PacketXcf type;
  typedef PacketXcf half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = sve_packet_size_selector<std::complex<float>, EIGEN_ARM64_SVE_VL>::size,

    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasArg = 0,

    HasSqrt = 1,
    HasLog = 1,
    // pexp_complex evaluates exp(a+bi) as e^a * (cos b + i sin b), and psin for
    // PacketXf returns |sin| under -ffast-math with GCC (13 through 15): the
    // magnitude is right but the sign is dropped, so exp of any value with a
    // negative imaginary part comes out conjugated. clang is unaffected, as is
    // NEON under either compiler, and every primitive psincos_float uses checks
    // out in isolation -- a verbatim copy of its body in another translation
    // unit gives the right answer with the same flags. Until that is resolved,
    // leave exp to the scalar path. See
    // https://gitlab.com/libeigen/eigen/-/issues/3132.
    HasExp = 0
  };
};

template <>
struct unpacket_traits<PacketXcf> {
  typedef std::complex<float> type;
  typedef PacketXcf half;
  typedef PacketXf as_real;
  enum {
    size = sve_packet_size_selector<std::complex<float>, EIGEN_ARM64_SVE_VL>::size,
    alignment = sve_packet_alignment_selector<EIGEN_ARM64_SVE_VL>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

/********************************* complex<float> *****************************/

template <>
EIGEN_STRONG_INLINE PacketXcf pset1<PacketXcf>(const std::complex<float>& from) {
  // {re, im} is one 64-bit lane, so broadcasting the value is a 64-bit dup.
  return PacketXcf(svreinterpret_f32_u64(svdup_n_u64(numext::bit_cast<numext::uint64_t>(from))));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pload<PacketXcf>(const std::complex<float>* from) {
  return PacketXcf(pload<PacketXf>(reinterpret_cast<const float*>(from)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploadu<PacketXcf>(const std::complex<float>* from) {
  return PacketXcf(ploadu<PacketXf>(reinterpret_cast<const float*>(from)));
}

template <>
EIGEN_STRONG_INLINE void pstore<std::complex<float>>(std::complex<float>* to, const PacketXcf& from) {
  pstore(reinterpret_cast<float*>(to), from.v);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<std::complex<float>>(std::complex<float>* to, const PacketXcf& from) {
  pstoreu(reinterpret_cast<float*>(to), from.v);
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploaddup<PacketXcf>(const std::complex<float>* from) {
  // Load the size/2 values this reads into the low half and interleave them
  // with themselves on 64-bit lanes, which moves whole {re, im} pairs. The
  // predicate is exact rather than svptrue -- a wider one would read past the
  // end of the input.
  constexpr uint64_t kHalf = uint64_t(packet_traits<std::complex<float>>::size) / 2;
  const svuint64_t lo =
      svreinterpret_u64_f32(svld1_f32(svwhilelt_b32(uint64_t(0), 2 * kHalf), reinterpret_cast<const float*>(from)));
  return PacketXcf(svreinterpret_f32_u64(svzip1_u64(lo, lo)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf ploadquad<PacketXcf>(const std::complex<float>* from) {
  // As ploaddup, one zip further: size/4 values, each repeated four times. At
  // the smallest vector length size/4 rounds to zero, where one value still
  // has to be read.
  constexpr uint64_t kQuarter = numext::maxi(uint64_t(packet_traits<std::complex<float>>::size) / 4, uint64_t(1));
  svuint64_t lo =
      svreinterpret_u64_f32(svld1_f32(svwhilelt_b32(uint64_t(0), 2 * kQuarter), reinterpret_cast<const float*>(from)));
  lo = svzip1_u64(lo, lo);
  return PacketXcf(svreinterpret_f32_u64(svzip1_u64(lo, lo)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pgather<std::complex<float>, PacketXcf>(const std::complex<float>* from, Index stride) {
  const svuint64_t idx = svindex_u64(0, numext::uint64_t(stride));
  return PacketXcf(svreinterpret_f32_u64(
      svld1_gather_u64index_u64(svptrue_b64(), reinterpret_cast<const numext::uint64_t*>(from), idx)));
}

template <>
EIGEN_STRONG_INLINE void pscatter<std::complex<float>, PacketXcf>(std::complex<float>* to, const PacketXcf& from,
                                                                  Index stride) {
  const svuint64_t idx = svindex_u64(0, numext::uint64_t(stride));
  svst1_scatter_u64index_u64(svptrue_b64(), reinterpret_cast<numext::uint64_t*>(to), idx,
                             svreinterpret_u64_f32(from.v));
}

template <>
EIGEN_STRONG_INLINE std::complex<float> pfirst<PacketXcf>(const PacketXcf& a) {
  // svlasta with no active lane returns lane 0, which is the whole value.
  return numext::bit_cast<std::complex<float>>(svlasta_u64(svpfalse_b(), svreinterpret_u64_f32(a.v)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pconj(const PacketXcf& a) {
  // {re, im} is one 64-bit lane with im in the high half, so flipping bit 63
  // negates the imaginary part alone.
  return PacketXcf(
      svreinterpret_f32_u64(sveor_n_u64_x(svptrue_b64(), svreinterpret_u64_f32(a.v), numext::uint64_t(1) << 63)));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pcplxflip<PacketXcf>(const PacketXcf& a) {
  // Swap the 32-bit halves of every 64-bit lane: {re, im} -> {im, re}.
  return PacketXcf(svreinterpret_f32_u64(svrevw_u64_x(svptrue_b64(), svreinterpret_u64_f32(a.v))));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pdupreal<PacketXcf>(const PacketXcf& a) {
  return PacketXcf(svtrn1_f32(a.v, a.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf pdupimag<PacketXcf>(const PacketXcf& a) {
  return PacketXcf(svtrn2_f32(a.v, a.v));
}

template <>
EIGEN_STRONG_INLINE PacketXcf preverse(const PacketXcf& a) {
  // Reversing 64-bit lanes moves whole complex values and keeps {re, im} paired.
  return PacketXcf(svreinterpret_f32_u64(svrev_u64(svreinterpret_u64_f32(a.v))));
}

template <>
EIGEN_STRONG_INLINE std::complex<float> predux<PacketXcf>(const PacketXcf& a) {
  // Read as a 32-bit predicate, an all-true 64-bit one is exactly the even
  // lanes -- the real parts -- and reversing it gives the odd ones.
  const svbool_t even = svptrue_b64();
  const svbool_t odd = svrev_b32(even);
  return {svaddv_f32(even, a.v), svaddv_f32(odd, a.v)};
}

/********************************* shared *************************************/

// Everything that acts on {re, im} pairs identically forwards to the real packet.
#define EIGEN_SVE_COMPLEX_DELEGATE(PACKET_CPLX)                                                       \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX padd<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {     \
    return PACKET_CPLX(padd(a.v, b.v));                                                               \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX psub<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {     \
    return PACKET_CPLX(psub(a.v, b.v));                                                               \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pnegate(const PACKET_CPLX& a) {                                     \
    return PACKET_CPLX(pnegate(a.v));                                                                 \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pzero<PACKET_CPLX>(const PACKET_CPLX& a) {                          \
    return PACKET_CPLX(pzero(a.v));                                                                   \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pand<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {     \
    return PACKET_CPLX(pand(a.v, b.v));                                                               \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX por<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {      \
    return PACKET_CPLX(por(a.v, b.v));                                                                \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pxor<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {     \
    return PACKET_CPLX(pxor(a.v, b.v));                                                               \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pandnot<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {  \
    return PACKET_CPLX(pandnot(a.v, b.v));                                                            \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pselect<PACKET_CPLX>(const PACKET_CPLX& mask, const PACKET_CPLX& a, \
                                                       const PACKET_CPLX& b) {                        \
    return PACKET_CPLX(pselect(mask.v, a.v, b.v));                                                    \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pmul<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {     \
    return pmul_complex(a, b);                                                                        \
  }                                                                                                   \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pdiv<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {     \
    return pdiv_complex(a, b);                                                                        \
  }                                                                                                   \
  /* A complex value is equal only if both components are, so fold the two */                         \
  /* real-lane results together across each pair. */                                                  \
  template <>                                                                                         \
  EIGEN_STRONG_INLINE PACKET_CPLX pcmp_eq<PACKET_CPLX>(const PACKET_CPLX& a, const PACKET_CPLX& b) {  \
    const PACKET_CPLX t = PACKET_CPLX(pcmp_eq(a.v, b.v));                                             \
    return PACKET_CPLX(pand(pdupreal(t).v, pdupimag(t).v));                                           \
  }

EIGEN_SVE_COMPLEX_DELEGATE(PacketXcf)
#undef EIGEN_SVE_COMPLEX_DELEGATE

EIGEN_INSTANTIATE_COMPLEX_MATH_FUNCS_NO_EXP(PacketXcf)

// A complex value is exactly one 64-bit lane, so transposing complex packets is
// a zip network run on 64-bit elements.
template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXcf, N>& kernel) {
  EIGEN_STATIC_ASSERT((N & (N - 1)) == 0, EIGEN_INTERNAL_ERROR_PLEASE_FILE_A_BUG_REPORT);
  for (int stride = N / 2; stride > 0; stride >>= 1) {
    for (int block = 0; block < N; block += 2 * stride) {
      for (int k = 0; k < stride; ++k) {
        const svuint64_t a = svreinterpret_u64_f32(kernel.packet[block + k].v);
        const svuint64_t b = svreinterpret_u64_f32(kernel.packet[block + k + stride].v);
        kernel.packet[block + k] = PacketXcf(svreinterpret_f32_u64(svzip1_u64(a, b)));
        kernel.packet[block + k + stride] = PacketXcf(svreinterpret_f32_u64(svzip2_u64(a, b)));
      }
    }
  }
}

EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(PacketXcf, PacketXf)

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_SVE_H
