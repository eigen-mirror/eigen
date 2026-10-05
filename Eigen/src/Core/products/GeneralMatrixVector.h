// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_GENERAL_MATRIX_VECTOR_H
#define EIGEN_GENERAL_MATRIX_VECTOR_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// C4804: unsafe use of type 'bool' in operation. Unavoidable in generic code
// instantiated with bool scalars (e.g. += and * on bool).
#if EIGEN_COMP_MSVC
#pragma warning(push)
#pragma warning(disable : 4804)
#endif

namespace Eigen {

namespace internal {

template <typename LhsScalar, typename RhsScalar, int PacketSize_ = GEBPPacketFull>
class gemv_traits {
  using ResScalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;

#define PACKET_DECL_COND_POSTFIX(postfix, name, packet_size)                                               \
  typedef typename packet_conditional<                                                                     \
      packet_size, typename packet_traits<name##Scalar>::type, typename packet_traits<name##Scalar>::half, \
      typename unpacket_traits<typename packet_traits<name##Scalar>::half>::half>::type name##Packet##postfix

  PACKET_DECL_COND_POSTFIX(_, Lhs, PacketSize_);
  PACKET_DECL_COND_POSTFIX(_, Rhs, PacketSize_);
  PACKET_DECL_COND_POSTFIX(_, Res, PacketSize_);
#undef PACKET_DECL_COND_POSTFIX

 public:
  enum {
    Vectorizable = unpacket_traits<LhsPacket_>::vectorizable && unpacket_traits<RhsPacket_>::vectorizable &&
                   int(unpacket_traits<LhsPacket_>::size) == int(unpacket_traits<RhsPacket_>::size),
    LhsPacketSize = Vectorizable ? unpacket_traits<LhsPacket_>::size : 1,
    RhsPacketSize = Vectorizable ? unpacket_traits<RhsPacket_>::size : 1,
    ResPacketSize = Vectorizable ? unpacket_traits<ResPacket_>::size : 1
  };

  using LhsPacket = std::conditional_t<Vectorizable, LhsPacket_, LhsScalar>;
  using RhsPacket = std::conditional_t<Vectorizable, RhsPacket_, RhsScalar>;
  using ResPacket = std::conditional_t<Vectorizable, ResPacket_, ResScalar>;
};

// Whether a mapper's coefficients are addressable and consecutive along the given storage order: the BLAS mappers
// with unit inner stride. Other mappers (e.g. the Tensor contraction input mappers) return coefficients by value.
template <typename Mapper, int Order>
struct gemv_mapper_is_contiguous : std::false_type {};
template <typename Scalar, typename Index, int Order, int Alignment>
struct gemv_mapper_is_contiguous<blas_data_mapper<Scalar, Index, Order, Alignment, 1>, Order> : std::true_type {};
template <typename Scalar, typename Index, int Order>
struct gemv_mapper_is_contiguous<const_blas_data_mapper<Scalar, Index, Order>, Order> : std::true_type {};

// Whether the GEMV kernels finish a partial packet with masked segment loads. Segment loads zero the lanes outside
// the segment, which the row-major kernel relies on before its horizontal reduction.
template <typename LhsPacket, typename RhsPacket, typename ResPacket>
using gemv_use_packet_segment =
    bool_constant<has_packet_segment<ResPacket>::value && std::is_same<LhsPacket, ResPacket>::value &&
                  std::is_same<RhsPacket, ResPacket>::value>;

// Loads count < packet-size consecutive coefficients starting at (i, j), along Order. Only contiguous mappers take
// the masked load; the kernels never select segments for other mappers, whose branch merely has to compile.
template <typename Packet, int Order, typename Mapper,
          bool Contiguous = gemv_mapper_is_contiguous<Mapper, Order>::value>
struct gemv_segment_loader {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet run(const Mapper&, Index, Index, Index) {
    eigen_internal_assert(false && "segment load from a non-contiguous mapper");
    return pzero(Packet{});
  }
};
template <typename Packet, int Order, typename Mapper>
struct gemv_segment_loader<Packet, Order, Mapper, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet run(const Mapper& m, Index i, Index j, Index count) {
    return ploaduSegment<Packet>(&m(i, j), 0, count);
  }
};

/* Optimized col-major matrix * vector product:
 * This algorithm processes the matrix per vertical panels,
 * which are then processed horizontally per chunk of 8*PacketSize x 1 vertical segments.
 *
 * Mixing type logic: C += alpha * A * B
 *  |  A  |  B  |alpha| comments
 *  |real |cplx |cplx | no vectorization
 *  |real |cplx |real | alpha is converted to a cplx when calling the run function, no vectorization
 *  |cplx |real |cplx | invalid, the caller has to do tmp: = A * B; C += alpha*tmp
 *  |cplx |real |real | optimal case, vectorization possible via real-cplx mul
 *
 * The same reasoning applies for the transposed case.
 */
template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, LhsScalar, LhsMapper, ColMajor, ConjugateLhs, RhsScalar, RhsMapper,
                                     ConjugateRhs, Version> {
  using Traits = gemv_traits<LhsScalar, RhsScalar>;
  using HalfTraits = gemv_traits<LhsScalar, RhsScalar, GEBPPacketHalf>;
  using QuarterTraits = gemv_traits<LhsScalar, RhsScalar, GEBPPacketQuarter>;

  using ResScalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;

  using LhsPacket = typename Traits::LhsPacket;
  using RhsPacket = typename Traits::RhsPacket;
  using ResPacket = typename Traits::ResPacket;

  using LhsPacketHalf = typename HalfTraits::LhsPacket;
  using RhsPacketHalf = typename HalfTraits::RhsPacket;
  using ResPacketHalf = typename HalfTraits::ResPacket;

  using LhsPacketQuarter = typename QuarterTraits::LhsPacket;
  using RhsPacketQuarter = typename QuarterTraits::RhsPacket;
  using ResPacketQuarter = typename QuarterTraits::ResPacket;

  EIGEN_DEVICE_FUNC inline static void run(Index rows, Index cols, const LhsMapper& lhs, const RhsMapper& rhs,
                                           ResScalar* res, Index resIncr, RhsScalar alpha);

  // Processes N full packets of rows starting at row i and, if Segment, the next count < ResPacketSize rows as one
  // masked packet, all in a single pass over the columns [j2, jend).
  template <int N, bool Segment = false>
  EIGEN_DEVICE_FUNC static EIGEN_ALWAYS_INLINE void process_rows(
      Index i, Index j2, Index jend, const LhsMapper& lhs, const RhsMapper& rhs, ResScalar* res,
      const ResPacket& palpha, conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>& pcj, Index count = 0);

  // Finishes the rows from i on, full_packets < 10 full packets followed by 0 < count < ResPacketSize rows, in one
  // process_rows pass. Out of line: it runs at most once per column block, and only when rows is not a multiple of
  // the packet size. lhs is taken by value: a reference would make run() keep its copy in memory, which clang fills
  // with a load that cannot be forwarded from the caller's stores.
  EIGEN_DEVICE_FUNC static EIGEN_DONT_INLINE void process_segment_tail(
      std::true_type, Index full_packets, Index i, Index j2, Index jend, LhsMapper lhs, const RhsMapper& rhs,
      ResScalar* res, const ResPacket& palpha, conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>& pcj,
      Index count);
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void process_segment_tail(
      std::false_type, Index, Index, Index, Index, const LhsMapper&, const RhsMapper&, ResScalar*, const ResPacket&,
      conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>&, Index) {}
};

// Integer-sequence helper for col-major GEMV full-packet row blocks.
template <int N>
struct gemv_colmajor_unroller {
  template <typename Packet, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void init_zero_impl(std::integer_sequence<int, K...>, Packet* c) {
    int unused[] = {0, ((c[K] = pzero(Packet{})), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void init_zero(Packet* c) {
    init_zero_impl(std::make_integer_sequence<int, N>{}, c);
  }

  template <typename LhsPacket, int LhsStride, int Alignment, typename AccPacket, typename RhsPacket,
            typename ConjHelper, typename LhsMapper, typename Index, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void madd_impl(std::integer_sequence<int, K...>, AccPacket* c,
                                                              const LhsMapper& lhs, Index i, Index j,
                                                              const RhsPacket& b0, ConjHelper& pcj) {
    int unused[] = {
        0, ((c[K] = pcj.pmadd(lhs.template load<LhsPacket, Alignment>(i + LhsStride * K, j), b0, c[K])), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename LhsPacket, int LhsStride, int Alignment, typename AccPacket, typename RhsPacket,
            typename ConjHelper, typename LhsMapper, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void madd(AccPacket* c, const LhsMapper& lhs, Index i, Index j,
                                                         const RhsPacket& b0, ConjHelper& pcj) {
    madd_impl<LhsPacket, LhsStride, Alignment>(std::make_integer_sequence<int, N>{}, c, lhs, i, j, b0, pcj);
  }

  template <int K, typename ResPacket, int ResStride, typename ResScalar, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void store_one(const ResPacket* c, ResScalar* res, Index i,
                                                              const ResPacket& palpha) {
    ResScalar* r = res + i + ResStride * K;
    pstoreu(r, pmadd(c[K], palpha, ploadu<ResPacket>(r)));
  }

  template <typename ResPacket, int ResStride, typename ResScalar, typename Index, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void store_impl(std::integer_sequence<int, K...>, const ResPacket* c,
                                                               ResScalar* res, Index i, const ResPacket& palpha) {
    int unused[] = {0, (store_one<K, ResPacket, ResStride>(c, res, i, palpha), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename ResPacket, int ResStride, typename ResScalar, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void store(const ResPacket* c, ResScalar* res, Index i,
                                                          const ResPacket& palpha) {
    store_impl<ResPacket, ResStride>(std::make_integer_sequence<int, N>{}, c, res, i, palpha);
  }
};

// No full packets: the final pass may hold only the masked partial packet.
template <>
struct gemv_colmajor_unroller<0> {
  template <typename Packet>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void init_zero(Packet*) {}

  template <typename LhsPacket, int LhsStride, int Alignment, typename AccPacket, typename RhsPacket,
            typename ConjHelper, typename LhsMapper, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void madd(AccPacket*, const LhsMapper&, Index, Index, const RhsPacket&,
                                                         ConjHelper&) {}

  template <typename ResPacket, int ResStride, typename ResScalar, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void store(const ResPacket*, ResScalar*, Index, const ResPacket&) {}
};

template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
template <int N, bool Segment>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void
general_matrix_vector_product<Index, LhsScalar, LhsMapper, ColMajor, ConjugateLhs, RhsScalar, RhsMapper, ConjugateRhs,
                              Version>::process_rows(Index i, Index j2, Index jend, const LhsMapper& lhs,
                                                     const RhsMapper& rhs, ResScalar* res, const ResPacket& palpha,
                                                     conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>& pcj,
                                                     Index count) {
  enum { LhsAlignment = Unaligned, LhsPacketSize = Traits::LhsPacketSize, ResPacketSize = Traits::ResPacketSize };
  using Unroller = gemv_colmajor_unroller<N>;
  const Index iseg = i + N * ResPacketSize;

  // c_seg accumulates the masked partial packet when Segment is set. It is not part of c: a wider array grows the
  // stack frame GCC estimates for run() and stops it from being inlined.
  ResPacket c[N > 0 ? N : 1];
  Unroller::init_zero(c);
  ResPacket c_seg = pzero(ResPacket{});
  for (Index j = j2; j < jend; ++j) {
    RhsPacket b0 = pset1<RhsPacket>(rhs(j, 0));
    Unroller::template madd<LhsPacket, LhsPacketSize, LhsAlignment>(c, lhs, i, j, b0, pcj);
    EIGEN_IF_CONSTEXPR (Segment) {
      c_seg = pcj.pmadd(gemv_segment_loader<LhsPacket, ColMajor, LhsMapper>::run(lhs, iseg, j, count), b0, c_seg);
    }
  }
  Unroller::template store<ResPacket, ResPacketSize>(c, res, i, palpha);
  EIGEN_IF_CONSTEXPR (Segment) {
    pstoreuSegment(res + iseg, pmadd(c_seg, palpha, ploaduSegment<ResPacket>(res + iseg, 0, count)), 0, count);
  }
}

template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void general_matrix_vector_product<
    Index, LhsScalar, LhsMapper, ColMajor, ConjugateLhs, RhsScalar, RhsMapper, ConjugateRhs,
    Version>::process_segment_tail(std::true_type, Index full_packets, Index i, Index j2, Index jend, LhsMapper lhs,
                                   const RhsMapper& rhs, ResScalar* res, const ResPacket& palpha,
                                   conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>& pcj, Index count) {
#define EIGEN_GEMV_PROCESS_ROW(n)                                          \
  case n:                                                                  \
    process_rows<n, true>(i, j2, jend, lhs, rhs, res, palpha, pcj, count); \
    break
  switch (full_packets) {
    EIGEN_GEMV_PROCESS_ROW(0);
    EIGEN_GEMV_PROCESS_ROW(1);
    EIGEN_GEMV_PROCESS_ROW(2);
    EIGEN_GEMV_PROCESS_ROW(3);
    EIGEN_GEMV_PROCESS_ROW(4);
    EIGEN_GEMV_PROCESS_ROW(5);
    EIGEN_GEMV_PROCESS_ROW(6);
    EIGEN_GEMV_PROCESS_ROW(7);
    EIGEN_GEMV_PROCESS_ROW(8);
    EIGEN_GEMV_PROCESS_ROW(9);
    default:
      eigen_internal_assert(false);
      break;
  }
#undef EIGEN_GEMV_PROCESS_ROW
}

template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DEVICE_FUNC inline void
general_matrix_vector_product<Index, LhsScalar, LhsMapper, ColMajor, ConjugateLhs, RhsScalar, RhsMapper, ConjugateRhs,
                              Version>::run(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs,
                                            ResScalar* res, Index resIncr, RhsScalar alpha) {
  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr == 1);

  // BLAS contract: if alpha == 0, the result is unchanged (and lhs/rhs need not be read).
  if (numext::is_exactly_zero(alpha)) return;

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);

  conj_helper<LhsScalar, RhsScalar, ConjugateLhs, ConjugateRhs> cj;
  conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs> pcj;
  conj_helper<LhsPacketHalf, RhsPacketHalf, ConjugateLhs, ConjugateRhs> pcj_half;
  conj_helper<LhsPacketQuarter, RhsPacketQuarter, ConjugateLhs, ConjugateRhs> pcj_quarter;

  const Index lhsStride = lhs.stride();
  // LhsAlignment stays Unaligned; enabling aligned reads would require
  // propagating the Mapper's Alignment through the run() template, and on
  // modern x86 aligned/unaligned packet loads are equivalent anyway.
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = Traits::ResPacketSize,
    ResPacketSizeHalf = HalfTraits::ResPacketSize,
    ResPacketSizeQuarter = QuarterTraits::ResPacketSize,
    LhsPacketSize = Traits::LhsPacketSize,
    HasHalf = (int)ResPacketSizeHalf < (int)ResPacketSize,
    HasQuarter = (int)ResPacketSizeQuarter < (int)ResPacketSizeHalf,
    UseSegment = gemv_use_packet_segment<LhsPacket, RhsPacket, ResPacket>::value &&
                 gemv_mapper_is_contiguous<LhsMapper, ColMajor>::value
  };

  using UnsignedIndex = std::make_unsigned_t<Index>;
  // With segments, the count trailing rows of a partial packet take one masked pass together with the full packets
  // before them. The 8-packet blocks then stop while at least 2 full packets remain, so that pass is never a lone
  // packet whose accumulator chain is bound by the FMA latency. Exact multiples of the packet size keep the 4, 3, 2
  // and 1-packet passes.
  const Index count = UseSegment ? Index(UnsignedIndex(rows) % ResPacketSize) : Index(0);
  const Index n8 = rows - (count > 0 ? 10 : 8) * ResPacketSize + 1;
  const Index n4 = rows - 4 * ResPacketSize + 1;
  const Index n3 = rows - 3 * ResPacketSize + 1;
  const Index n2 = rows - 2 * ResPacketSize + 1;
  const Index n1 = rows - 1 * ResPacketSize + 1;
  const Index n_half = rows - 1 * ResPacketSizeHalf + 1;
  const Index n_quarter = rows - 1 * ResPacketSizeQuarter + 1;

  // Choose block_cols so that one column slice of the LHS roughly fits in L1.
  // When it does not, fall back to a smaller batch to keep cache pressure down.
  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);
  const Index block_cols =
      cols < 128 ? cols : (lhsStride * Index(sizeof(LhsScalar)) < Index(l1) ? Index(16) : Index(4));
  ResPacket palpha = pset1<ResPacket>(alpha);
  ResPacketHalf palpha_half = pset1<ResPacketHalf>(alpha);
  ResPacketQuarter palpha_quarter = pset1<ResPacketQuarter>(alpha);

  for (Index j2 = 0; j2 < cols; j2 += block_cols) {
    Index jend = numext::mini(j2 + block_cols, cols);
    Index i = 0;
    for (; i < n8; i += ResPacketSize * 8) process_rows<8>(i, j2, jend, lhs, rhs, res, palpha, pcj);
    if (count > 0) {
      process_segment_tail(bool_constant<UseSegment>(), Index(UnsignedIndex(rows - i) / ResPacketSize), i, j2, jend,
                           lhs, rhs, res, palpha, pcj, count);
    } else {
#define EIGEN_GEMV_PROCESS_ROW(k)                             \
  if (i < n##k) {                                             \
    process_rows<k>(i, j2, jend, lhs, rhs, res, palpha, pcj); \
    i += ResPacketSize * (k);                                 \
  }                                                           \
  static_assert(true, "Trailing semicolon required")
      EIGEN_GEMV_PROCESS_ROW(4);
      EIGEN_GEMV_PROCESS_ROW(3);
      EIGEN_GEMV_PROCESS_ROW(2);
      EIGEN_GEMV_PROCESS_ROW(1);
#undef EIGEN_GEMV_PROCESS_ROW
      EIGEN_IF_CONSTEXPR (HasHalf) {
        if (i < n_half) {
          ResPacketHalf c0 = pzero(ResPacketHalf{});
          for (Index j = j2; j < jend; j += 1) {
            RhsPacketHalf b0 = pset1<RhsPacketHalf>(rhs(j, 0));
            c0 = pcj_half.pmadd(lhs.template load<LhsPacketHalf, LhsAlignment>(i + 0, j), b0, c0);
          }
          pstoreu(res + i + ResPacketSizeHalf * 0,
                  pmadd(c0, palpha_half, ploadu<ResPacketHalf>(res + i + ResPacketSizeHalf * 0)));
          i += ResPacketSizeHalf;
        }
      }
      EIGEN_IF_CONSTEXPR (HasQuarter) {
        if (i < n_quarter) {
          ResPacketQuarter c0 = pzero(ResPacketQuarter{});
          for (Index j = j2; j < jend; j += 1) {
            RhsPacketQuarter b0 = pset1<RhsPacketQuarter>(rhs(j, 0));
            c0 = pcj_quarter.pmadd(lhs.template load<LhsPacketQuarter, LhsAlignment>(i + 0, j), b0, c0);
          }
          pstoreu(res + i + ResPacketSizeQuarter * 0,
                  pmadd(c0, palpha_quarter, ploadu<ResPacketQuarter>(res + i + ResPacketSizeQuarter * 0)));
          i += ResPacketSizeQuarter;
        }
      }
      for (; i < rows; ++i) {
        ResScalar c0(0);
        for (Index j = j2; j < jend; j += 1) c0 += cj.pmul(lhs(i, j), rhs(j, 0));
        res[i] += alpha * c0;
      }
    }
  }
}

/* Optimized row-major matrix * vector product:
 * This algorithm processes 4 rows at once that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 *
 * Mixing type logic:
 *  - alpha is always a complex (or converted to a complex)
 *  - no vectorization
 */
template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, LhsScalar, LhsMapper, RowMajor, ConjugateLhs, RhsScalar, RhsMapper,
                                     ConjugateRhs, Version> {
  using Traits = gemv_traits<LhsScalar, RhsScalar>;
  using HalfTraits = gemv_traits<LhsScalar, RhsScalar, GEBPPacketHalf>;
  using QuarterTraits = gemv_traits<LhsScalar, RhsScalar, GEBPPacketQuarter>;

  using ResScalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;

  using LhsPacket = typename Traits::LhsPacket;
  using RhsPacket = typename Traits::RhsPacket;
  using ResPacket = typename Traits::ResPacket;

  using LhsPacketHalf = typename HalfTraits::LhsPacket;
  using RhsPacketHalf = typename HalfTraits::RhsPacket;
  using ResPacketHalf = typename HalfTraits::ResPacket;

  using LhsPacketQuarter = typename QuarterTraits::LhsPacket;
  using RhsPacketQuarter = typename QuarterTraits::RhsPacket;
  using ResPacketQuarter = typename QuarterTraits::ResPacket;

  EIGEN_DEVICE_FUNC static inline void run(Index rows, Index cols, const LhsMapper& lhs, const RhsMapper& rhs,
                                           ResScalar* res, Index resIncr, ResScalar alpha);

  // Specialized path for when cols < full packet size.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static void run_small_cols(Index rows, Index cols, const LhsMapper& lhs,
                                                                   const RhsMapper& rhs, ResScalar* res, Index resIncr,
                                                                   ResScalar alpha);

  // Templated helper that processes N rows in run_small_cols. N is a compile-time
  // constant; row-dimension unrolling is done inside flat helper loops.
  template <int N>
  EIGEN_DEVICE_FUNC static EIGEN_ALWAYS_INLINE void process_rows_small_cols(Index i, Index cols, const LhsMapper& lhs,
                                                                            const RhsMapper& rhs, ResScalar* res,
                                                                            Index resIncr, ResScalar alpha,
                                                                            Index halfColBlockEnd,
                                                                            Index quarterColBlockEnd);
};

template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DEVICE_FUNC inline void
general_matrix_vector_product<Index, LhsScalar, LhsMapper, RowMajor, ConjugateLhs, RhsScalar, RhsMapper, ConjugateRhs,
                              Version>::run(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs,
                                            ResScalar* res, Index resIncr, ResScalar alpha) {
  // BLAS contract: if alpha == 0, the result is unchanged (and lhs/rhs need not be read).
  if (numext::is_exactly_zero(alpha)) return;

  // When cols < full packet size, the main vectorized loops are empty.
  // Use the sub-packet helper only when half or quarter packets can do useful work;
  // otherwise it would just duplicate the scalar cleanup.
  enum {
    LhsPacketSize_ = Traits::LhsPacketSize,
    MinUsefulCols_ =
        ((int)QuarterTraits::LhsPacketSize < (int)HalfTraits::LhsPacketSize)
            ? (int)QuarterTraits::LhsPacketSize
            : (((int)HalfTraits::LhsPacketSize < (int)Traits::LhsPacketSize) ? (int)HalfTraits::LhsPacketSize
                                                                             : (int)Traits::LhsPacketSize),
    HasSubPackets_ = (int)MinUsefulCols_ < (int)LhsPacketSize_,
    UseSegment_ = gemv_use_packet_segment<LhsPacket, RhsPacket, ResPacket>::value &&
                  gemv_mapper_is_contiguous<LhsMapper, RowMajor>::value &&
                  gemv_mapper_is_contiguous<RhsMapper, ColMajor>::value,
    // With segments, one masked full packet per row beats a half packet followed by two or more scalar columns.
    SmallColsEnd_ = UseSegment_ ? (int)HalfTraits::LhsPacketSize + 2 : (int)LhsPacketSize_
  };
  EIGEN_IF_CONSTEXPR (HasSubPackets_) {
    if (cols >= MinUsefulCols_) {
      if (cols < SmallColsEnd_ && cols < LhsPacketSize_) {
        run_small_cols(rows, cols, alhs, rhs, res, resIncr, alpha);
        return;
      }
    }
  }

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);

  eigen_internal_assert(rhs.stride() == 1);
  conj_helper<LhsScalar, RhsScalar, ConjugateLhs, ConjugateRhs> cj;
  conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs> pcj;
  conj_helper<LhsPacketHalf, RhsPacketHalf, ConjugateLhs, ConjugateRhs> pcj_half;
  conj_helper<LhsPacketQuarter, RhsPacketQuarter, ConjugateLhs, ConjugateRhs> pcj_quarter;

  // Disable the 8-row inner unroll once a single column slice no longer fits in L1; with very
  // large LHS strides each unrolled iteration evicts the previously-loaded rows from cache.
  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);
  const Index n8 = lhs.stride() * Index(sizeof(LhsScalar)) > Index(l1) ? 0 : rows - 7;
  const Index n4 = rows - 3;
  const Index n2 = rows - 1;

  // LhsAlignment stays Unaligned; enabling aligned reads would require
  // propagating the Mapper's Alignment through the run() template, and on
  // modern x86 aligned/unaligned packet loads are equivalent anyway.
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = Traits::ResPacketSize,
    ResPacketSizeHalf = HalfTraits::ResPacketSize,
    ResPacketSizeQuarter = QuarterTraits::ResPacketSize,
    LhsPacketSize = Traits::LhsPacketSize,
    LhsPacketSizeHalf = HalfTraits::LhsPacketSize,
    LhsPacketSizeQuarter = QuarterTraits::LhsPacketSize,
    HasHalf = (int)ResPacketSizeHalf < (int)ResPacketSize,
    HasQuarter = (int)ResPacketSizeQuarter < (int)ResPacketSizeHalf,
    UseSegment = UseSegment_
  };

  using UnsignedIndex = std::make_unsigned_t<Index>;
  const Index fullColBlockEnd = LhsPacketSize * (UnsignedIndex(cols) / LhsPacketSize);
  const Index halfColBlockEnd = LhsPacketSizeHalf * (UnsignedIndex(cols) / LhsPacketSizeHalf);
  const Index quarterColBlockEnd = LhsPacketSizeQuarter * (UnsignedIndex(cols) / LhsPacketSizeQuarter);
  // With segments, the last cols - fullColBlockEnd < LhsPacketSize columns take one masked packet per row and the
  // scalar column loops below are empty.
  const Index segmentCount = cols - fullColBlockEnd;
  const Index scalarColStart = UseSegment ? cols : fullColBlockEnd;
  using LhsSegmentLoader = gemv_segment_loader<LhsPacket, RowMajor, LhsMapper>;
  using RhsSegmentLoader = gemv_segment_loader<RhsPacket, ColMajor, RhsMapper>;

  Index i = 0;
  for (; i < n8; i += 8) {
    ResPacket c0 = pzero(ResPacket{}), c1 = pzero(ResPacket{}), c2 = pzero(ResPacket{}), c3 = pzero(ResPacket{}),
              c4 = pzero(ResPacket{}), c5 = pzero(ResPacket{}), c6 = pzero(ResPacket{}), c7 = pzero(ResPacket{});

    for (Index j = 0; j < fullColBlockEnd; j += LhsPacketSize) {
      RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);

      c0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, c0);
      c1 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 1, j), b0, c1);
      c2 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 2, j), b0, c2);
      c3 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 3, j), b0, c3);
      c4 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 4, j), b0, c4);
      c5 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 5, j), b0, c5);
      c6 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 6, j), b0, c6);
      c7 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 7, j), b0, c7);
    }
    if (UseSegment && segmentCount > 0) {
      RhsPacket b0 = RhsSegmentLoader::run(rhs, fullColBlockEnd, 0, segmentCount);
      c0 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 0, fullColBlockEnd, segmentCount), b0, c0);
      c1 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 1, fullColBlockEnd, segmentCount), b0, c1);
      c2 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 2, fullColBlockEnd, segmentCount), b0, c2);
      c3 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 3, fullColBlockEnd, segmentCount), b0, c3);
      c4 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 4, fullColBlockEnd, segmentCount), b0, c4);
      c5 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 5, fullColBlockEnd, segmentCount), b0, c5);
      c6 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 6, fullColBlockEnd, segmentCount), b0, c6);
      c7 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 7, fullColBlockEnd, segmentCount), b0, c7);
    }
    ResScalar cc0 = predux(c0);
    ResScalar cc1 = predux(c1);
    ResScalar cc2 = predux(c2);
    ResScalar cc3 = predux(c3);
    ResScalar cc4 = predux(c4);
    ResScalar cc5 = predux(c5);
    ResScalar cc6 = predux(c6);
    ResScalar cc7 = predux(c7);

    for (Index j = scalarColStart; j < cols; ++j) {
      RhsScalar b0 = rhs(j, 0);

      cc0 += cj.pmul(lhs(i + 0, j), b0);
      cc1 += cj.pmul(lhs(i + 1, j), b0);
      cc2 += cj.pmul(lhs(i + 2, j), b0);
      cc3 += cj.pmul(lhs(i + 3, j), b0);
      cc4 += cj.pmul(lhs(i + 4, j), b0);
      cc5 += cj.pmul(lhs(i + 5, j), b0);
      cc6 += cj.pmul(lhs(i + 6, j), b0);
      cc7 += cj.pmul(lhs(i + 7, j), b0);
    }
    res[(i + 0) * resIncr] += alpha * cc0;
    res[(i + 1) * resIncr] += alpha * cc1;
    res[(i + 2) * resIncr] += alpha * cc2;
    res[(i + 3) * resIncr] += alpha * cc3;
    res[(i + 4) * resIncr] += alpha * cc4;
    res[(i + 5) * resIncr] += alpha * cc5;
    res[(i + 6) * resIncr] += alpha * cc6;
    res[(i + 7) * resIncr] += alpha * cc7;
  }
  for (; i < n4; i += 4) {
    ResPacket c0 = pzero(ResPacket{}), c1 = pzero(ResPacket{}), c2 = pzero(ResPacket{}), c3 = pzero(ResPacket{});

    for (Index j = 0; j < fullColBlockEnd; j += LhsPacketSize) {
      RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);

      c0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, c0);
      c1 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 1, j), b0, c1);
      c2 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 2, j), b0, c2);
      c3 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 3, j), b0, c3);
    }
    if (UseSegment && segmentCount > 0) {
      RhsPacket b0 = RhsSegmentLoader::run(rhs, fullColBlockEnd, 0, segmentCount);
      c0 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 0, fullColBlockEnd, segmentCount), b0, c0);
      c1 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 1, fullColBlockEnd, segmentCount), b0, c1);
      c2 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 2, fullColBlockEnd, segmentCount), b0, c2);
      c3 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 3, fullColBlockEnd, segmentCount), b0, c3);
    }
    ResScalar cc0 = predux(c0);
    ResScalar cc1 = predux(c1);
    ResScalar cc2 = predux(c2);
    ResScalar cc3 = predux(c3);

    for (Index j = scalarColStart; j < cols; ++j) {
      RhsScalar b0 = rhs(j, 0);

      cc0 += cj.pmul(lhs(i + 0, j), b0);
      cc1 += cj.pmul(lhs(i + 1, j), b0);
      cc2 += cj.pmul(lhs(i + 2, j), b0);
      cc3 += cj.pmul(lhs(i + 3, j), b0);
    }
    res[(i + 0) * resIncr] += alpha * cc0;
    res[(i + 1) * resIncr] += alpha * cc1;
    res[(i + 2) * resIncr] += alpha * cc2;
    res[(i + 3) * resIncr] += alpha * cc3;
  }
  for (; i < n2; i += 2) {
    ResPacket c0 = pzero(ResPacket{}), c1 = pzero(ResPacket{});

    for (Index j = 0; j < fullColBlockEnd; j += LhsPacketSize) {
      RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);

      c0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, c0);
      c1 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 1, j), b0, c1);
    }
    if (UseSegment && segmentCount > 0) {
      RhsPacket b0 = RhsSegmentLoader::run(rhs, fullColBlockEnd, 0, segmentCount);
      c0 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 0, fullColBlockEnd, segmentCount), b0, c0);
      c1 = pcj.pmadd(LhsSegmentLoader::run(lhs, i + 1, fullColBlockEnd, segmentCount), b0, c1);
    }
    ResScalar cc0 = predux(c0);
    ResScalar cc1 = predux(c1);

    for (Index j = scalarColStart; j < cols; ++j) {
      RhsScalar b0 = rhs(j, 0);

      cc0 += cj.pmul(lhs(i + 0, j), b0);
      cc1 += cj.pmul(lhs(i + 1, j), b0);
    }
    res[(i + 0) * resIncr] += alpha * cc0;
    res[(i + 1) * resIncr] += alpha * cc1;
  }
  for (; i < rows; ++i) {
    ResPacket c0 = pzero(ResPacket{});
    ResPacketHalf c0_h = pzero(ResPacketHalf{});
    ResPacketQuarter c0_q = pzero(ResPacketQuarter{});

    for (Index j = 0; j < fullColBlockEnd; j += LhsPacketSize) {
      RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);
      c0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i, j), b0, c0);
    }
    if (UseSegment && segmentCount > 0) {
      RhsPacket b0 = RhsSegmentLoader::run(rhs, fullColBlockEnd, 0, segmentCount);
      c0 = pcj.pmadd(LhsSegmentLoader::run(lhs, i, fullColBlockEnd, segmentCount), b0, c0);
    }
    ResScalar cc0 = predux(c0);
    EIGEN_IF_CONSTEXPR (HasHalf && !UseSegment) {
      for (Index j = fullColBlockEnd; j < halfColBlockEnd; j += LhsPacketSizeHalf) {
        RhsPacketHalf b0 = rhs.template load<RhsPacketHalf, Unaligned>(j, 0);
        c0_h = pcj_half.pmadd(lhs.template load<LhsPacketHalf, LhsAlignment>(i, j), b0, c0_h);
      }
      cc0 += predux(c0_h);
    }
    EIGEN_IF_CONSTEXPR (HasQuarter && !UseSegment) {
      for (Index j = halfColBlockEnd; j < quarterColBlockEnd; j += LhsPacketSizeQuarter) {
        RhsPacketQuarter b0 = rhs.template load<RhsPacketQuarter, Unaligned>(j, 0);
        c0_q = pcj_quarter.pmadd(lhs.template load<LhsPacketQuarter, LhsAlignment>(i, j), b0, c0_q);
      }
      cc0 += predux(c0_q);
    }
    for (Index j = UseSegment ? cols : quarterColBlockEnd; j < cols; ++j) {
      cc0 += cj.pmul(lhs(i, j), rhs(j, 0));
    }
    res[i * resIncr] += alpha * cc0;
  }
}

// Integer-sequence helper for process_rows_small_cols.
template <int N>
struct gemv_small_cols_unroller {
  template <typename LhsPacket, typename AccPacket, int Alignment, typename RhsType, typename ConjHelper,
            typename LhsMapper, typename Index, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void madd_impl(std::integer_sequence<int, K...>, AccPacket* acc,
                                                              const LhsMapper& lhs, Index i, Index j, const RhsType& b0,
                                                              ConjHelper& pcj) {
    int unused[] = {0, ((acc[K] = pcj.pmadd(lhs.template load<LhsPacket, Alignment>(i + K, j), b0, acc[K])), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename LhsPacket, typename AccPacket, int Alignment, typename RhsType, typename ConjHelper,
            typename LhsMapper, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void madd(AccPacket* acc, const LhsMapper& lhs, Index i, Index j,
                                                         const RhsType& b0, ConjHelper& pcj) {
    madd_impl<LhsPacket, AccPacket, Alignment>(std::make_integer_sequence<int, N>{}, acc, lhs, i, j, b0, pcj);
  }

  template <typename ResScalar, typename RhsScalar, typename ConjHelper, typename LhsMapper, typename Index, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scalar_madd_impl(std::integer_sequence<int, K...>, ResScalar* cc,
                                                                     const LhsMapper& lhs, Index i, Index j,
                                                                     const RhsScalar& b0, ConjHelper& cj) {
    int unused[] = {0, ((cc[K] += cj.pmul(lhs(i + K, j), b0)), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename ResScalar, typename RhsScalar, typename ConjHelper, typename LhsMapper, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void scalar_madd(ResScalar* cc, const LhsMapper& lhs, Index i, Index j,
                                                                const RhsScalar& b0, ConjHelper& cj) {
    scalar_madd_impl(std::make_integer_sequence<int, N>{}, cc, lhs, i, j, b0, cj);
  }

  template <typename Scalar, typename Packet, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void predux_accum_impl(std::integer_sequence<int, K...>, Scalar* cc,
                                                                      const Packet* acc) {
    int unused[] = {0, ((cc[K] += predux(acc[K])), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename Scalar, typename Packet>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void predux_accum(Scalar* cc, const Packet* acc) {
    predux_accum_impl(std::make_integer_sequence<int, N>{}, cc, acc);
  }

  template <typename Packet, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void init_zero_impl(std::integer_sequence<int, K...>, Packet* acc) {
    int unused[] = {0, ((acc[K] = pzero(Packet{})), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void init_zero(Packet* acc) {
    init_zero_impl(std::make_integer_sequence<int, N>{}, acc);
  }

  template <typename Scalar, typename Index, int... K>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void write_result_impl(std::integer_sequence<int, K...>, Scalar* res,
                                                                      Index resIncr, Index i, Scalar alpha,
                                                                      const Scalar* cc) {
    int unused[] = {0, ((res[(i + K) * resIncr] += alpha * cc[K]), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  template <typename Scalar, typename Index>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void write_result(Scalar* res, Index resIncr, Index i, Scalar alpha,
                                                                 const Scalar* cc) {
    write_result_impl(std::make_integer_sequence<int, N>{}, res, resIncr, i, alpha, cc);
  }
};

template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
template <int N>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void
general_matrix_vector_product<Index, LhsScalar, LhsMapper, RowMajor, ConjugateLhs, RhsScalar, RhsMapper, ConjugateRhs,
                              Version>::process_rows_small_cols(Index i, Index cols, const LhsMapper& lhs,
                                                                const RhsMapper& rhs, ResScalar* res, Index resIncr,
                                                                ResScalar alpha, Index halfColBlockEnd,
                                                                Index quarterColBlockEnd) {
  conj_helper<LhsScalar, RhsScalar, ConjugateLhs, ConjugateRhs> cj;
  conj_helper<LhsPacketHalf, RhsPacketHalf, ConjugateLhs, ConjugateRhs> pcj_half;
  conj_helper<LhsPacketQuarter, RhsPacketQuarter, ConjugateLhs, ConjugateRhs> pcj_quarter;

  enum {
    LhsAlignment = Unaligned,
    ResPacketSizeHalf = HalfTraits::ResPacketSize,
    ResPacketSizeQuarter = QuarterTraits::ResPacketSize,
    LhsPacketSizeHalf = HalfTraits::LhsPacketSize,
    LhsPacketSizeQuarter = QuarterTraits::LhsPacketSize,
    HasHalf = (int)ResPacketSizeHalf < (int)Traits::ResPacketSize,
    HasQuarter = (int)ResPacketSizeQuarter < (int)ResPacketSizeHalf
  };

  using Unroll = gemv_small_cols_unroller<N>;

  ResScalar cc[N] = {};
  EIGEN_IF_CONSTEXPR (HasHalf) {
    ResPacketHalf h[N];
    Unroll::init_zero(h);
    for (Index j = 0; j < halfColBlockEnd; j += LhsPacketSizeHalf) {
      RhsPacketHalf b0 = rhs.template load<RhsPacketHalf, Unaligned>(j, 0);
      Unroll::template madd<LhsPacketHalf, ResPacketHalf, LhsAlignment>(h, lhs, i, j, b0, pcj_half);
    }
    Unroll::predux_accum(cc, h);
  }
  EIGEN_IF_CONSTEXPR (HasQuarter) {
    ResPacketQuarter q[N];
    Unroll::init_zero(q);
    for (Index j = halfColBlockEnd; j < quarterColBlockEnd; j += LhsPacketSizeQuarter) {
      RhsPacketQuarter b0 = rhs.template load<RhsPacketQuarter, Unaligned>(j, 0);
      Unroll::template madd<LhsPacketQuarter, ResPacketQuarter, LhsAlignment>(q, lhs, i, j, b0, pcj_quarter);
    }
    Unroll::predux_accum(cc, q);
  }
  for (Index j = quarterColBlockEnd; j < cols; ++j) {
    RhsScalar b0 = rhs(j, 0);
    Unroll::scalar_madd(cc, lhs, i, j, b0, cj);
  }
  Unroll::write_result(res, resIncr, i, alpha, cc);
}

template <typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void
general_matrix_vector_product<Index, LhsScalar, LhsMapper, RowMajor, ConjugateLhs, RhsScalar, RhsMapper, ConjugateRhs,
                              Version>::run_small_cols(Index rows, Index cols, const LhsMapper& alhs,
                                                       const RhsMapper& rhs, ResScalar* res, Index resIncr,
                                                       ResScalar alpha) {
  LhsMapper lhs(alhs);
  eigen_internal_assert(rhs.stride() == 1);

  enum {
    LhsPacketSizeHalf = HalfTraits::LhsPacketSize,
    LhsPacketSizeQuarter = QuarterTraits::LhsPacketSize,
  };

  using UnsignedIndex = std::make_unsigned_t<Index>;
  const Index halfColBlockEnd = LhsPacketSizeHalf * (UnsignedIndex(cols) / LhsPacketSizeHalf);
  const Index quarterColBlockEnd = LhsPacketSizeQuarter * (UnsignedIndex(cols) / LhsPacketSizeQuarter);

  // Disable the 8-row inner unroll once a single column slice no longer fits in L1; with very
  // large LHS strides each unrolled iteration evicts the previously-loaded rows from cache.
  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);
  const Index n8 = lhs.stride() * Index(sizeof(LhsScalar)) > Index(l1) ? 0 : rows - 7;
  const Index n4 = rows - 3;
  const Index n2 = rows - 1;

  Index i = 0;
  for (; i < n8; i += 8) {
    process_rows_small_cols<8>(i, cols, lhs, rhs, res, resIncr, alpha, halfColBlockEnd, quarterColBlockEnd);
  }
  // Process remaining groups of 4 rows in case n8 was 0.
  for (; i < n4; i += 4) {
    process_rows_small_cols<4>(i, cols, lhs, rhs, res, resIncr, alpha, halfColBlockEnd, quarterColBlockEnd);
  }
  if (i < n2) {
    process_rows_small_cols<2>(i, cols, lhs, rhs, res, resIncr, alpha, halfColBlockEnd, quarterColBlockEnd);
    i += 2;
  }
  if (i < rows) {
    process_rows_small_cols<1>(i, cols, lhs, rhs, res, resIncr, alpha, halfColBlockEnd, quarterColBlockEnd);
  }
}

}  // end namespace internal

}  // end namespace Eigen

#if EIGEN_COMP_MSVC
#pragma warning(pop)
#endif

#endif  // EIGEN_GENERAL_MATRIX_VECTOR_H
