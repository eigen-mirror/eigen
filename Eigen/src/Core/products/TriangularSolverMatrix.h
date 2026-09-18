// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Modifications Copyright (C) 2022 Intel Corporation
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TRIANGULAR_SOLVER_MATRIX_H
#define EIGEN_TRIANGULAR_SOLVER_MATRIX_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride,
          bool Specialized>
struct trsmKernelL {
  // Generic Implementation of triangular solve for triangular matrix on left and multiple rhs.
  // Handles non-packed matrices.
  //
  // A lower-triangular panel is addressed from its top-left element with indices in [0, size);
  // an upper-triangular one is addressed from its bottom-right element with indices in
  // (-size, 0]. Both origins are elements of the panel, so callers never form a pointer outside
  // the matrix they solve in. The AVX-512 specializations take both by the top-left element and
  // convert when they delegate here.
  static void kernel(Index size, Index otherSize, const Scalar* _tri, Index triStride, Scalar* _other, Index otherIncr,
                     Index otherStride);
};

template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride,
          bool Specialized>
struct trsmKernelR {
  // Generic Implementation of triangular solve for triangular matrix on right and multiple lhs.
  // Handles non-packed matrices.
  static void kernel(Index size, Index otherSize, const Scalar* _tri, Index triStride, Scalar* _other, Index otherIncr,
                     Index otherStride);
};

// The packet lanes are independent right-hand sides. Reserve registers for the
// RHS packets and coefficient broadcasts in addition to the output accumulators.
template <typename Scalar>
struct triangular_solve_packet_traits {
  static constexpr bool Enabled = packet_traits<Scalar>::Vectorizable &&
                                  (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value) &&
                                  std::numeric_limits<Scalar>::is_iec559 && std::numeric_limits<Scalar>::radix == 2;
  static constexpr int PacketSize = packet_traits<Scalar>::size;
  static constexpr int RegisterRows = 4;
  static constexpr int RhsPackets =
      plain_enum_max(1, plain_enum_min(2, (gebp_traits<Scalar, Scalar>::NumberOfRegisters - 2) / (RegisterRows + 1)));
  // Allocation bound, independent of the cache-dependent direct-solve cutoff.
  static constexpr int WorkspaceRows = 128;
#if defined(EIGEN_VECTORIZE_AVX512) && EIGEN_USE_AVX512_TRSM_L_KERNELS
  static constexpr bool UseUnblocked = false;
#else
  static constexpr bool UseUnblocked = Enabled;
#endif

  template <typename Index>
  static EIGEN_STRONG_INLINE bool use_unblocked(Index size, Index cols, std::ptrdiff_t l1) {
    if (!UseUnblocked || size < RegisterRows || size > WorkspaceRows || cols < PacketSize) return false;
    const int packets = cols >= RhsPackets * PacketSize ? RhsPackets : 1;
    // Like GEBP's L1 depth model: the RHS tile, reciprocals, and RegisterRows rows
    // of A are reused along k. Leave half of L1 for streaming and associativity.
    const std::ptrdiff_t rowBytes = (packets * PacketSize + RegisterRows + 1) * sizeof(Scalar);
    const std::ptrdiff_t transposeBytes = PacketSize * PacketSize * sizeof(Scalar);
    return std::ptrdiff_t(size) * rowBytes + transposeBytes <= l1 / 2;
  }
};

template <typename Scalar, typename Index, int Mode, int TriStorageOrder>
struct triangular_solve_packet_kernel {
  using Traits = triangular_solve_packet_traits<Scalar>;
  using Packet = typename packet_traits<Scalar>::type;
  using TriMapper = const_blas_data_mapper<Scalar, Index, TriStorageOrder>;
  static constexpr int PacketSize = Traits::PacketSize;
  static constexpr bool IsLower = (Mode & Lower) != 0;

  template <int RhsPackets>
  static EIGEN_STRONG_INLINE bool all_finite(const PacketBlock<Packet, RhsPackets>& x, bool_constant<true>) {
    // Classify bits: fast-math may assume the arithmetic results are finite.
    using FloatTraits = binary_floating_point_traits<Scalar>;
    bool nonfinite = false;
    for (int p = 0; p < RhsPackets; ++p) {
      Scalar values[PacketSize];
      pstoreu(values, x.packet[p]);
      EIGEN_FAST_MATH_CONSTANT_BARRIER(values);
      for (int c = 0; c < PacketSize; ++c)
        nonfinite |= (FloatTraits::bits(values[c]) & FloatTraits::kExponentMask) == FloatTraits::kExponentMask;
    }
    return !nonfinite;
  }

  // Keep disabled scalar instantiations valid in C++14.
  template <int RhsPackets>
  static EIGEN_STRONG_INLINE bool all_finite(const PacketBlock<Packet, RhsPackets>&, bool_constant<false>) {
    return true;
  }

  template <int RhsPackets>
  static EIGEN_STRONG_INLINE void update(PacketBlock<Packet, RhsPackets>& x, const PacketBlock<Packet, RhsPackets>& y,
                                         Scalar a) {
    const Packet pa = pset1<Packet>(a);
    for (int p = 0; p < RhsPackets; ++p) x.packet[p] = pnmadd(pa, y.packet[p], x.packet[p]);
  }

  template <int RhsPackets>
  static EIGEN_STRONG_INLINE void scale(PacketBlock<Packet, RhsPackets>& x, Scalar a) {
    EIGEN_IF_CONSTEXPR (!(Mode & UnitDiag)) {
      const Packet pa = pset1<Packet>(a);
      for (int p = 0; p < RhsPackets; ++p) x.packet[p] = pmul(x.packet[p], pa);
    }
  }

  template <std::size_t Row, int RhsPackets, std::size_t... Next>
  static EIGEN_STRONG_INLINE void solve_row(PacketBlock<Packet, RhsPackets>* x, const TriMapper& a,
                                            const Scalar* inverse, Index r0, Index step, std::index_sequence<Next...>) {
    const Index row = r0 + Index(Row) * step;
    scale(x[Row], inverse[row]);
    int unroll[] = {0, (update(x[Row + 1 + Next], x[Row], a(r0 + Index(Row + 1 + Next) * step, row)), 0)...};
    EIGEN_UNUSED_VARIABLE(unroll);
  }

  // GCC generates extra instructions when the accumulator indices come from a loop.
  template <int RhsPackets, std::size_t... Rows>
  static EIGEN_STRONG_INLINE void solve_block(Index i, const TriMapper& a, const Scalar* inverse,
                                              PacketBlock<Packet, RhsPackets>* work, Index r0, Index step,
                                              std::index_sequence<Rows...>) {
    PacketBlock<Packet, RhsPackets> x[] = {work[r0 + Index(Rows) * step]...};
    for (Index k = 0; k < i; ++k) {
      const Index c = IsLower ? k : r0 + i - k;
      const PacketBlock<Packet, RhsPackets> y = work[c];
      int unroll[] = {0, (update(x[Rows], y, a(r0 + Index(Rows) * step, c)), 0)...};
      EIGEN_UNUSED_VARIABLE(unroll);
    }
    // The braced expansion orders the dependent solves by row.
    int solve_rows[] = {
        0,
        (solve_row<Rows>(x, a, inverse, r0, step, std::make_index_sequence<Traits::RegisterRows - Rows - 1>{}), 0)...};
    EIGEN_UNUSED_VARIABLE(solve_rows);
    int store_rows[] = {0, (work[r0 + Index(Rows) * step] = x[Rows], 0)...};
    EIGEN_UNUSED_VARIABLE(store_rows);
  }

  template <int RhsPackets>
  static EIGEN_STRONG_INLINE void solve(Index size, const TriMapper& a, const Scalar* inverse, Scalar* other,
                                        Index otherStride) {
    PacketBlock<Packet, RhsPackets> work[Traits::WorkspaceRows];
    for (int p = 0; p < RhsPackets; ++p) {
      Index i = 0;
      for (; i + PacketSize <= size; i += PacketSize) {
        PacketBlock<Packet, PacketSize> block;
        for (int c = 0; c < PacketSize; ++c)
          block.packet[c] = ploadu<Packet>(other + i + (p * PacketSize + c) * otherStride);
        ptranspose(block);
        for (int r = 0; r < PacketSize; ++r) work[i + r].packet[p] = block.packet[r];
      }
      for (; i < size; ++i)
        work[i].packet[p] = pgather<Scalar, Packet>(other + i + p * PacketSize * otherStride, otherStride);
    }
    Index i = 0;
    const Index step = IsLower ? 1 : -1;
    for (; i + Traits::RegisterRows <= size; i += Traits::RegisterRows) {
      const Index r0 = IsLower ? i : size - i - 1;
      // Preserve the four-row schedule that keeps GCC and Clang's hot loops compact.
      EIGEN_IF_CONSTEXPR (Traits::RegisterRows == 4) {
        const Index r1 = r0 + step, r2 = r1 + step, r3 = r2 + step;
        PacketBlock<Packet, RhsPackets> x0 = work[r0], x1 = work[r1], x2 = work[r2], x3 = work[r3];
        for (Index k = 0; k < i; ++k) {
          const Index c = IsLower ? k : size - k - 1;
          const PacketBlock<Packet, RhsPackets> y = work[c];
          update(x0, y, a(r0, c));
          update(x1, y, a(r1, c));
          update(x2, y, a(r2, c));
          update(x3, y, a(r3, c));
        }
        scale(x0, inverse[r0]);
        update(x1, x0, a(r1, r0));
        update(x2, x0, a(r2, r0));
        update(x3, x0, a(r3, r0));
        scale(x1, inverse[r1]);
        update(x2, x1, a(r2, r1));
        update(x3, x1, a(r3, r1));
        scale(x2, inverse[r2]);
        update(x3, x2, a(r3, r2));
        scale(x3, inverse[r3]);
        work[r0] = x0;
        work[r1] = x1;
        work[r2] = x2;
        work[r3] = x3;
      } else {
        solve_block(i, a, inverse, work, r0, step, std::make_index_sequence<Traits::RegisterRows>{});
      }
    }
    for (; i < size; ++i) {
      const Index r = IsLower ? i : size - i - 1;
      PacketBlock<Packet, RhsPackets> x = work[r];
      for (Index k = 0; k < i; ++k) {
        const Index c = IsLower ? k : size - k - 1;
        update(x, work[c], a(r, c));
      }
      scale(x, inverse[r]);
      work[r] = x;
    }
    EIGEN_IF_CONSTEXPR (TriStorageOrder == RowMajor) {
      // Successive RHS updates can overflow before cancellation in a row's dot product.
      // Every later row uses all solved rows, propagating nonfinite lanes to the final row.
      // Retry with the original accumulation order before overwriting any RHS coefficient.
      if (!all_finite(work[IsLower ? size - 1 : 0], bool_constant<Traits::Enabled>{})) {
        const Index origin = IsLower ? 0 : size - 1;
        trsmKernelL<Scalar, Index, Mode, false, TriStorageOrder, 1, false>::kernel(
            size, Index(RhsPackets * PacketSize), &a(origin, origin), a.stride(), other + origin, Index(1),
            otherStride);
        return;
      }
    }
    for (int p = 0; p < RhsPackets; ++p) {
      Index i = 0;
      for (; i + PacketSize <= size; i += PacketSize) {
        PacketBlock<Packet, PacketSize> block;
        for (int r = 0; r < PacketSize; ++r) block.packet[r] = work[i + r].packet[p];
        ptranspose(block);
        for (int c = 0; c < PacketSize; ++c) pstoreu(other + i + (p * PacketSize + c) * otherStride, block.packet[c]);
      }
      for (; i < size; ++i)
        pscatter<Scalar, Packet>(other + i + p * PacketSize * otherStride, work[i].packet[p], otherStride);
    }
  }

  static EIGEN_DONT_INLINE void kernel(Index size, Index cols, const Scalar* tri, Index triStride, Scalar* other,
                                       Index otherStride) {
    eigen_internal_assert(size <= Traits::WorkspaceRows && cols % PacketSize == 0);
    EIGEN_IF_CONSTEXPR (!IsLower) {
      tri -= (size - 1) * (triStride + 1);
      other -= size - 1;
    }
    TriMapper a(tri, triStride);
    Scalar inverse[Traits::WorkspaceRows];
    Map<Vector<Scalar, Dynamic>> mapped(inverse, size);
    EIGEN_IF_CONSTEXPR (Mode & UnitDiag) {
      mapped.setOnes();
    } else {
      const Map<const Vector<Scalar, Dynamic>, Unaligned, InnerStride<Dynamic>> diagonal(
          tri, size, InnerStride<Dynamic>(triStride + 1));
      mapped = diagonal.cwiseInverse();
    }
    Index j = 0;
    EIGEN_IF_CONSTEXPR (Traits::RhsPackets > 1) {
      for (; j + Traits::RhsPackets * PacketSize <= cols; j += Traits::RhsPackets * PacketSize)
        solve<Traits::RhsPackets>(size, a, inverse, other + j * otherStride, otherStride);
    }
    for (; j + PacketSize <= cols; j += PacketSize) solve<1>(size, a, inverse, other + j * otherStride, otherStride);
  }
};

template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride,
          bool Specialized>
EIGEN_STRONG_INLINE void trsmKernelL<Scalar, Index, Mode, Conjugate, TriStorageOrder, OtherInnerStride,
                                     Specialized>::kernel(Index size, Index otherSize, const Scalar* _tri,
                                                          Index triStride, Scalar* _other, Index otherIncr,
                                                          Index otherStride) {
  EIGEN_IF_CONSTEXPR ((Specialized && OtherInnerStride == 1 && triangular_solve_packet_traits<Scalar>::Enabled)) {
    if (size >= triangular_solve_packet_traits<Scalar>::RegisterRows &&
        size <= triangular_solve_packet_traits<Scalar>::WorkspaceRows && otherSize >= packet_traits<Scalar>::size) {
      const Index packetCols = numext::round_down(otherSize, Index(packet_traits<Scalar>::size));
      triangular_solve_packet_kernel<Scalar, Index, Mode, TriStorageOrder>::kernel(size, packetCols, _tri, triStride,
                                                                                   _other, otherStride);
      if (packetCols == otherSize) return;
      otherSize -= packetCols;
      _other += packetCols * otherStride;
    }
  }
  using TriMapper = const_blas_data_mapper<Scalar, Index, TriStorageOrder>;
  using OtherMapper = blas_data_mapper<Scalar, Index, ColMajor, Unaligned, OtherInnerStride>;
  TriMapper tri(_tri, triStride);
  OtherMapper other(_other, otherStride, otherIncr);

  enum { IsLower = (Mode & Lower) == Lower };
  conj_if<Conjugate> conj;

  // tr solve
  for (Index k = 0; k < size; ++k) {
    // TODO: write a small kernel handling this (can be shared with trsv)
    Index i = IsLower ? k : -k;
    Index rs = size - k - 1;  // remaining size
    Index s = TriStorageOrder == RowMajor ? (IsLower ? 0 : i + 1) : IsLower ? i + 1 : i - rs;

    Scalar a = (Mode & UnitDiag) ? Scalar(1) : Scalar(Scalar(1) / conj(tri(i, i)));
    for (Index j = 0; j < otherSize; ++j) {
      EIGEN_IF_CONSTEXPR (TriStorageOrder == RowMajor) {
        Scalar b(0);
        const Scalar* l = &tri(i, s);
        typename OtherMapper::LinearMapper r = other.getLinearMapper(s, j);
        for (Index i3 = 0; i3 < k; ++i3) b += conj(l[i3]) * r(i3);

        other(i, j) = (other(i, j) - b) * a;
      } else {
        Scalar& otherij = other(i, j);
        otherij *= a;
        Scalar b = otherij;
        typename OtherMapper::LinearMapper r = other.getLinearMapper(s, j);
        typename TriMapper::LinearMapper l = tri.getLinearMapper(s, i);
        for (Index i3 = 0; i3 < rs; ++i3) r(i3) -= b * conj(l(i3));
      }
    }
  }
}

template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride,
          bool Specialized>
EIGEN_STRONG_INLINE void trsmKernelR<Scalar, Index, Mode, Conjugate, TriStorageOrder, OtherInnerStride,
                                     Specialized>::kernel(Index size, Index otherSize, const Scalar* _tri,
                                                          Index triStride, Scalar* _other, Index otherIncr,
                                                          Index otherStride) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using LhsMapper = blas_data_mapper<Scalar, Index, ColMajor, Unaligned, OtherInnerStride>;
  using RhsMapper = const_blas_data_mapper<Scalar, Index, TriStorageOrder>;
  LhsMapper lhs(_other, otherStride, otherIncr);
  RhsMapper rhs(_tri, triStride);

  enum { IsLower = (Mode & Lower) == Lower };
  conj_if<Conjugate> conj;

  for (Index k = 0; k < size; ++k) {
    Index j = IsLower ? size - k - 1 : k;

    typename LhsMapper::LinearMapper r = lhs.getLinearMapper(0, j);
    EIGEN_IF_CONSTEXPR (OtherInnerStride == 1 && packet_traits<Scalar>::Vectorizable) {
      using Packet = typename packet_traits<Scalar>::type;
      constexpr Index PS = unpacket_traits<Packet>::size;
      // Unrolled k3 loop by 4 to reduce r load/store traffic.
      Index k3 = 0;
      for (; k3 + 3 < k; k3 += 4) {
        Index col0 = IsLower ? j + 1 + k3 : k3;
        Scalar b0 = conj(rhs(col0, j));
        Scalar b1 = conj(rhs(col0 + 1, j));
        Scalar b2 = conj(rhs(col0 + 2, j));
        Scalar b3 = conj(rhs(col0 + 3, j));
        Packet neg_pb0 = pset1<Packet>(-b0);
        Packet neg_pb1 = pset1<Packet>(-b1);
        Packet neg_pb2 = pset1<Packet>(-b2);
        Packet neg_pb3 = pset1<Packet>(-b3);
        typename LhsMapper::LinearMapper a0 = lhs.getLinearMapper(0, col0);
        typename LhsMapper::LinearMapper a1 = lhs.getLinearMapper(0, col0 + 1);
        typename LhsMapper::LinearMapper a2 = lhs.getLinearMapper(0, col0 + 2);
        typename LhsMapper::LinearMapper a3 = lhs.getLinearMapper(0, col0 + 3);
        Index i = 0;
        for (; i + PS <= otherSize; i += PS) {
          Packet pr = r.template loadPacket<Packet>(i);
          pr = pmadd(a0.template loadPacket<Packet>(i), neg_pb0, pr);
          pr = pmadd(a1.template loadPacket<Packet>(i), neg_pb1, pr);
          pr = pmadd(a2.template loadPacket<Packet>(i), neg_pb2, pr);
          pr = pmadd(a3.template loadPacket<Packet>(i), neg_pb3, pr);
          r.template storePacket<Packet>(i, pr);
        }
        for (; i < otherSize; ++i) {
          r(i) -= a0(i) * b0 + a1(i) * b1 + a2(i) * b2 + a3(i) * b3;
        }
      }
      // Handle remaining k3 iterations with vectorized inner loop.
      for (; k3 < k; ++k3) {
        Scalar b = conj(rhs(IsLower ? j + 1 + k3 : k3, j));
        typename LhsMapper::LinearMapper a = lhs.getLinearMapper(0, IsLower ? j + 1 + k3 : k3);
        Packet neg_pb = pset1<Packet>(-b);
        Index i = 0;
        for (; i + PS <= otherSize; i += PS) {
          Packet pr = r.template loadPacket<Packet>(i);
          pr = pmadd(a.template loadPacket<Packet>(i), neg_pb, pr);
          r.template storePacket<Packet>(i, pr);
        }
        for (; i < otherSize; ++i) r(i) -= a(i) * b;
      }
      // Vectorized diagonal scaling.
      EIGEN_IF_CONSTEXPR ((Mode & UnitDiag) == 0) {
        Scalar inv_rjj = RealScalar(1) / conj(rhs(j, j));
        Packet pinv = pset1<Packet>(inv_rjj);
        Index i = 0;
        for (; i + PS <= otherSize; i += PS) {
          r.template storePacket<Packet>(i, pmul(r.template loadPacket<Packet>(i), pinv));
        }
        for (; i < otherSize; ++i) r(i) *= inv_rjj;
      }
    } else {
      for (Index k3 = 0; k3 < k; ++k3) {
        Scalar b = conj(rhs(IsLower ? j + 1 + k3 : k3, j));
        typename LhsMapper::LinearMapper a = lhs.getLinearMapper(0, IsLower ? j + 1 + k3 : k3);
        for (Index i = 0; i < otherSize; ++i) r(i) -= a(i) * b;
      }
      EIGEN_IF_CONSTEXPR ((Mode & UnitDiag) == 0) {
        Scalar inv_rjj = RealScalar(1) / conj(rhs(j, j));
        for (Index i = 0; i < otherSize; ++i) r(i) *= inv_rjj;
      }
    }
  }
}

// if the rhs is row major, let's transpose the product
template <typename Scalar, typename Index, int Side, int Mode, bool Conjugate, int TriStorageOrder,
          int OtherInnerStride>
struct triangular_solve_matrix<Scalar, Index, Side, Mode, Conjugate, TriStorageOrder, RowMajor, OtherInnerStride> {
  static void run(Index size, Index cols, const Scalar* tri, Index triStride, Scalar* _other, Index otherIncr,
                  Index otherStride, level3_blocking<Scalar, Scalar>& blocking) {
    triangular_solve_matrix<
        Scalar, Index, Side == OnTheLeft ? OnTheRight : OnTheLeft, (Mode & UnitDiag) | ((Mode & Upper) ? Lower : Upper),
        NumTraits<Scalar>::IsComplex && Conjugate, TriStorageOrder == RowMajor ? ColMajor : RowMajor, ColMajor,
        OtherInnerStride>::run(size, cols, tri, triStride, _other, otherIncr, otherStride, blocking);
  }
};

/** \internal Entries of the solved operand a blocked triangular solve may keep in cache while its
 * k-blocks sweep it: a quarter of the L3, which on a multi-die part is the package total of which one
 * core reaches a fraction, and never less than the L2. */
template <typename Scalar>
std::ptrdiff_t triangular_solve_budget(std::ptrdiff_t l2, std::ptrdiff_t l3) {
  return (numext::maxi)(l3 / 4, l2) / std::ptrdiff_t(sizeof(Scalar));
}

/** \internal Columns of the right-hand side a solve on the left takes per panel: the widest multiple
 * of nr whose size x nc entries fit the budget, or all cols. Every panel re-packs the triangle,
 * size^2/2 copies against size^2*nc/2 multiply-adds, and panels narrower than 512 columns measured
 * slower than none, so a right-hand side that would need them is solved whole. */
template <typename Index>
Index triangular_solve_panel_columns(Index size, Index cols, std::ptrdiff_t budget, Index nr) {
  eigen_internal_assert(size > 0);
  const std::ptrdiff_t width = numext::round_down<std::ptrdiff_t>(budget / size, nr);
  return width >= 512 && width < cols ? Index(width) : cols;
}

/** \internal Depth of the k-blocks of a solve against otherSize columns (on the left) or rows (on the
 * right), whose packed buffers hold kc x extent entries. The KcFactor 4 depth SolveTriangular.h blocks
 * with keeps the share of flops in the kc x kc diagonal blocks, which run below gebp's rate, near
 * kc/size. Every k-block also sweeps the operand and packs a slab of the triangle, costs that scale as
 * 1/kc and outweigh that share once the operand exceeds the budget or, for a slab packed one column
 * run at a time (slabRuns), once half the triangle does. The depth then grows toward the
 * single-threaded GEMM depth, no further than size/8 and 160, where deeper diagonal blocks measured
 * slower than the sweeps they save (by up to 10% at the GEMM depth in AVX2 builds), and, when the
 * buffers fit on the stack at the blocking's depth, no further than the depth at which they still do.
 * A caller that preallocated the buffers sized them for the blocking's depth. */
template <typename Scalar, typename Index>
Index triangular_solve_kc(Index size, Index otherSize, Index extent, std::ptrdiff_t budget, bool slabRuns,
                          level3_blocking<Scalar, Scalar>& blocking) {
  EIGEN_UNUSED_VARIABLE(extent);
  const bool deep = std::ptrdiff_t(size) * otherSize > budget || (slabRuns && std::ptrdiff_t(size) * size / 2 > budget);
  if (!deep || blocking.blockA() != nullptr) return blocking.kc();
  Index kc = size, mc = size, nc = otherSize;
  computeProductBlockingSizes<Scalar, Scalar>(kc, mc, nc);
  kc = (numext::mini)(kc, numext::round_down((numext::mini)(size / 8, Index(160)), Index(8)));
#if defined(EIGEN_ALLOCA) && !defined(EIGEN_NO_ALLOCA)
  const std::ptrdiff_t stackKc =
      std::ptrdiff_t(EIGEN_STACK_ALLOCATION_LIMIT) / (std::ptrdiff_t(sizeof(Scalar)) * extent);
  if (blocking.kc() <= stackKc) kc = Index((numext::mini)(std::ptrdiff_t(kc), stackKc));
#endif
  return (numext::maxi)(kc, blocking.kc());
}

/* Optimized triangular solver with multiple right hand side and the triangular matrix on the left
 */
template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride>
struct triangular_solve_matrix<Scalar, Index, OnTheLeft, Mode, Conjugate, TriStorageOrder, ColMajor, OtherInnerStride> {
  static EIGEN_DONT_INLINE void run(Index size, Index otherSize, const Scalar* _tri, Index triStride, Scalar* _other,
                                    Index otherIncr, Index otherStride, level3_blocking<Scalar, Scalar>& blocking);
};

template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride>
EIGEN_DONT_INLINE void triangular_solve_matrix<Scalar, Index, OnTheLeft, Mode, Conjugate, TriStorageOrder, ColMajor,
                                               OtherInnerStride>::run(Index size, Index otherSize, const Scalar* _tri,
                                                                      Index triStride, Scalar* _other, Index otherIncr,
                                                                      Index otherStride,
                                                                      level3_blocking<Scalar, Scalar>& blocking) {
  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);
  EIGEN_IF_CONSTEXPR ((OtherInnerStride == 1 && triangular_solve_packet_traits<Scalar>::Enabled)) {
    using PacketTraits = triangular_solve_packet_traits<Scalar>;
    if (PacketTraits::use_unblocked(size, otherSize, l1)) {
      const Index origin = (Mode & Lower) ? 0 : size - 1;
      trsmKernelL<Scalar, Index, Mode, Conjugate, TriStorageOrder, OtherInnerStride, true>::kernel(
          size, otherSize, _tri + origin * (triStride + 1), triStride, _other + origin, otherIncr, otherStride);
      return;
    }
  }
#if defined(EIGEN_VECTORIZE_AVX512) && defined(EIGEN_USE_AVX512_TRSM_L_KERNELS) && EIGEN_USE_AVX512_TRSM_L_KERNELS && \
    EIGEN_ENABLE_AVX512_NOCOPY_TRSM_L_CUTOFFS
  EIGEN_IF_CONSTEXPR ((OtherInnerStride == 1 &&
                       (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value))) {
    // Very rough cutoffs to determine when to call trsm w/o packing
    // For small problem sizes trsmKernel compiled with clang is generally faster.
    // TODO: Investigate better heuristics for cutoffs.
    double L2Cap = 0.5;  // 50% of L2 size
    if (size < avx512_trsm_cutoff<Scalar>(l2, otherSize, L2Cap)) {
      trsmKernelL<Scalar, Index, Mode, Conjugate, TriStorageOrder, 1, /*Specialized=*/true>::kernel(
          size, otherSize, _tri, triStride, _other, 1, otherStride);
      return;
    }
  }
#endif

  using TriMapper = const_blas_data_mapper<Scalar, Index, TriStorageOrder>;
  using OtherMapper = blas_data_mapper<Scalar, Index, ColMajor, Unaligned, OtherInnerStride>;
  TriMapper tri(_tri, triStride);

  using Traits = gebp_traits<Scalar, Scalar>;

  enum { SmallPanelWidth = plain_enum_max(Traits::mr, Traits::nr), IsLower = (Mode & Lower) == Lower };

  // Every k-block updates the rows of the right-hand side beyond it through gebp, so a right-hand side
  // solved whole streams through the caches size/kc times (issue #3162); column panels that stay in
  // cache keep those sweeps out of memory. This kernel packs the slabs of the triangle by rows, which
  // costs more the deeper they are, so a large triangle alone does not deepen it.
  const std::ptrdiff_t budget = triangular_solve_budget<Scalar>(l2, l3);
  const Index nc = triangular_solve_panel_columns(size, otherSize, budget, Index(Traits::nr));
  const Index mc = (numext::mini)(size, blocking.mc());  // cache block size along the M direction
  // The tr solve below packs up to SmallPanelWidth x kc entries of the triangle into blockA.
  const Index blockARows = (numext::maxi)(mc, Index(SmallPanelWidth));
  // cache block size along the K direction
  const Index kc = triangular_solve_kc<Scalar>(size, otherSize, (numext::maxi)(blockARows, nc), budget,
                                               /*slabRuns=*/false, blocking);

  std::size_t sizeA = kc * blockARows;
  std::size_t sizeB = kc * nc;

  ei_declare_aligned_stack_constructed_variable(Scalar, blockA, sizeA, blocking.blockA());
  ei_declare_aligned_stack_constructed_variable(Scalar, blockB, sizeB, blocking.blockB());

  gebp_kernel<Scalar, Scalar, Index, OtherMapper, Traits::mr, Traits::nr, Conjugate, false> gebp_kernel;
  gemm_pack_lhs<Scalar, Index, TriMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing,
                TriStorageOrder>
      pack_lhs;
  gemm_pack_rhs<Scalar, Index, OtherMapper, Traits::nr, ColMajor, false, true> pack_rhs;

  // the goal here is to subdivide the Rhs panels such that we keep some cache
  // coherence when accessing the rhs elements
  Index subcols = otherSize > 0 ? l2 / (4 * sizeof(Scalar) * numext::maxi<Index>(otherStride, size)) : 0;
  subcols = numext::maxi<Index>((subcols / Traits::nr) * Traits::nr, Traits::nr);

  for (Index j0 = 0; j0 < otherSize; j0 += nc) {
    const Index cols = (numext::mini)(otherSize - j0, nc);
    Scalar* _panel = _other + j0 * otherStride;
    OtherMapper other(_panel, otherStride, otherIncr);

    for (Index k2 = IsLower ? 0 : size; IsLower ? k2 < size : k2 > 0; IsLower ? k2 += kc : k2 -= kc) {
      const Index actual_kc = (numext::mini)(IsLower ? size - k2 : k2, kc);

      // We have selected and packed a big horizontal panel R1 of rhs. Let B be the packed copy of this panel,
      // and R2 the remaining part of rhs. The corresponding vertical panel of lhs is split into
      // A11 (the triangular part) and A21 the remaining rectangular part.
      // Then the high level algorithm is:
      //  - B = R1                    => general block copy (done during the next step)
      //  - R1 = A11^-1 B             => tricky part
      //  - update B from the new R1  => actually this has to be performed continuously during the above step
      //  - R2 -= A21 * B             => GEPP

      // The tricky part: compute R1 = A11^-1 B while updating B from R1
      // The idea is to split A11 into multiple small vertical panels.
      // Each panel can be split into a small triangular part T1k which is processed without optimization,
      // and the remaining small part T2k which is processed using gebp with appropriate block strides
      for (Index j2 = 0; j2 < cols; j2 += subcols) {
        Index actual_cols = (numext::mini)(cols - j2, subcols);
        // for each small vertical panels [T1k^T, T2k^T]^T of lhs
        for (Index k1 = 0; k1 < actual_kc; k1 += SmallPanelWidth) {
          Index actualPanelWidth = numext::mini<Index>(actual_kc - k1, SmallPanelWidth);
          // tr solve
          {
            Index i = IsLower ? k2 + k1 : k2 - k1 - 1;
#if defined(EIGEN_VECTORIZE_AVX512) && defined(EIGEN_USE_AVX512_TRSM_L_KERNELS) && EIGEN_USE_AVX512_TRSM_L_KERNELS
            EIGEN_IF_CONSTEXPR ((OtherInnerStride == 1 &&
                                 (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value))) {
              i = IsLower ? k2 + k1 : k2 - k1 - actualPanelWidth;
            }
#endif
            trsmKernelL<Scalar, Index, Mode, Conjugate, TriStorageOrder, OtherInnerStride,
                        /*Specialized=*/true>::kernel(actualPanelWidth, actual_cols, _tri + i + (i)*triStride,
                                                      triStride, _panel + i * otherIncr + j2 * otherStride, otherIncr,
                                                      otherStride);
          }

          Index lengthTarget = actual_kc - k1 - actualPanelWidth;
          Index startBlock = IsLower ? k2 + k1 : k2 - k1 - actualPanelWidth;
          Index blockBOffset = IsLower ? k1 : lengthTarget;

          // update the respective rows of B from other
          pack_rhs(blockB + actual_kc * j2, other.getSubMapper(startBlock, j2), actualPanelWidth, actual_cols,
                   actual_kc, blockBOffset);

          // GEBP
          if (lengthTarget > 0) {
            Index startTarget = IsLower ? k2 + k1 + actualPanelWidth : k2 - actual_kc;

            pack_lhs(blockA, tri.getSubMapper(startTarget, startBlock), actualPanelWidth, lengthTarget);

            gebp_kernel(other.getSubMapper(startTarget, j2), blockA, blockB + actual_kc * j2, lengthTarget,
                        actualPanelWidth, actual_cols, Scalar(-1), actualPanelWidth, actual_kc, 0, blockBOffset);
          }
        }
      }

      // R2 -= A21 * B => GEPP
      {
        Index start = IsLower ? k2 + kc : 0;
        Index end = IsLower ? size : k2 - kc;
        for (Index i2 = start; i2 < end; i2 += mc) {
          const Index actual_mc = (numext::mini)(mc, end - i2);
          if (actual_mc > 0) {
            pack_lhs(blockA, tri.getSubMapper(i2, IsLower ? k2 : k2 - kc), actual_kc, actual_mc);

            gebp_kernel(other.getSubMapper(i2, 0), blockA, blockB, actual_mc, actual_kc, cols, Scalar(-1), -1, -1, 0,
                        0);
          }
        }
      }
    }
  }
}

/* Optimized triangular solver with multiple left hand sides and the triangular matrix on the right
 */
template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride>
struct triangular_solve_matrix<Scalar, Index, OnTheRight, Mode, Conjugate, TriStorageOrder, ColMajor,
                               OtherInnerStride> {
  static EIGEN_DONT_INLINE void run(Index size, Index otherSize, const Scalar* _tri, Index triStride, Scalar* _other,
                                    Index otherIncr, Index otherStride, level3_blocking<Scalar, Scalar>& blocking);
};

template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride>
EIGEN_DONT_INLINE void triangular_solve_matrix<Scalar, Index, OnTheRight, Mode, Conjugate, TriStorageOrder, ColMajor,
                                               OtherInnerStride>::run(Index size, Index otherSize, const Scalar* _tri,
                                                                      Index triStride, Scalar* _other, Index otherIncr,
                                                                      Index otherStride,
                                                                      level3_blocking<Scalar, Scalar>& blocking) {
  Index rows = otherSize;

  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);

#if defined(EIGEN_VECTORIZE_AVX512) && defined(EIGEN_USE_AVX512_TRSM_R_KERNELS) && EIGEN_USE_AVX512_TRSM_R_KERNELS && \
    EIGEN_ENABLE_AVX512_NOCOPY_TRSM_R_CUTOFFS
  EIGEN_IF_CONSTEXPR ((OtherInnerStride == 1 &&
                       (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value))) {
    // TODO: Investigate better heuristics for cutoffs.
    double L2Cap = 0.5;  // 50% of L2 size
    if (size < avx512_trsm_cutoff<Scalar>(l2, rows, L2Cap)) {
      trsmKernelR<Scalar, Index, Mode, Conjugate, TriStorageOrder, OtherInnerStride, /*Specialized=*/true>::kernel(
          size, rows, _tri, triStride, _other, 1, otherStride);
      return;
    }
  }
#endif

  using LhsMapper = blas_data_mapper<Scalar, Index, ColMajor, Unaligned, OtherInnerStride>;
  using RhsMapper = const_blas_data_mapper<Scalar, Index, TriStorageOrder>;
  LhsMapper lhs(_other, otherStride, otherIncr);
  RhsMapper rhs(_tri, triStride);

  using Traits = gebp_traits<Scalar, Scalar>;
  enum {
    RhsStorageOrder = TriStorageOrder,
    SmallPanelWidth = plain_enum_max(Traits::mr, Traits::nr),
    IsLower = (Mode & Lower) == Lower
  };

  // Every k-block sweeps all rows of the left-hand side through gebp and packs its slab of a
  // column-major triangle one column run at a time, so a large triangle alone deepens this kernel
  // (issue #3162); a row-major triangle is packed by rows, as on the left.
  const std::ptrdiff_t budget = triangular_solve_budget<Scalar>(l2, l3);
  Index mc = (numext::mini)(rows, blocking.mc());  // cache block size along the M direction
  // cache block size along the K direction
  const Index kc = triangular_solve_kc<Scalar>(size, rows, (numext::maxi)(mc, size), budget,
                                               /*slabRuns=*/TriStorageOrder == ColMajor, blocking);
  // blockA packs kc x mc entries of the left-hand side, and rows can far exceed size. Past half the
  // budget a deeper kc takes proportionally fewer rows per pass, so blockA does not outgrow the buffer
  // the blocking chose.
  if (kc > blocking.kc()) {
    const std::ptrdiff_t maxA = (numext::maxi)(std::ptrdiff_t(blocking.kc()) * mc, budget / 2);
    mc = (numext::mini)(mc, (numext::maxi)(Index(Traits::mr), numext::round_down(Index(maxA / kc), Index(Traits::mr))));
  }

  std::size_t sizeA = kc * mc;
  std::size_t sizeB = kc * size;

  ei_declare_aligned_stack_constructed_variable(Scalar, blockA, sizeA, blocking.blockA());
  ei_declare_aligned_stack_constructed_variable(Scalar, blockB, sizeB, blocking.blockB());

  gebp_kernel<Scalar, Scalar, Index, LhsMapper, Traits::mr, Traits::nr, false, Conjugate> gebp_kernel;
  gemm_pack_rhs<Scalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
  gemm_pack_rhs<Scalar, Index, RhsMapper, Traits::nr, RhsStorageOrder, false, true> pack_rhs_panel;
  gemm_pack_lhs<Scalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, ColMajor,
                false, true>
      pack_lhs_panel;

  for (Index k2 = IsLower ? size : 0; IsLower ? k2 > 0 : k2 < size; IsLower ? k2 -= kc : k2 += kc) {
    const Index actual_kc = (numext::mini)(IsLower ? k2 : size - k2, kc);
    Index actual_k2 = IsLower ? k2 - actual_kc : k2;

    Index startPanel = IsLower ? 0 : k2 + actual_kc;
    Index rs = IsLower ? actual_k2 : size - actual_k2 - actual_kc;
    Scalar* geb = blockB + actual_kc * actual_kc;

    if (rs > 0) pack_rhs(geb, rhs.getSubMapper(actual_k2, startPanel), actual_kc, rs);

    // triangular packing (we only pack the panels off the diagonal,
    // neglecting the blocks overlapping the diagonal
    {
      for (Index j2 = 0; j2 < actual_kc; j2 += SmallPanelWidth) {
        Index actualPanelWidth = numext::mini<Index>(actual_kc - j2, SmallPanelWidth);
        Index actual_j2 = actual_k2 + j2;
        Index panelOffset = IsLower ? j2 + actualPanelWidth : 0;
        Index panelLength = IsLower ? actual_kc - j2 - actualPanelWidth : j2;

        if (panelLength > 0)
          pack_rhs_panel(blockB + j2 * actual_kc, rhs.getSubMapper(actual_k2 + panelOffset, actual_j2), panelLength,
                         actualPanelWidth, actual_kc, panelOffset);
      }
    }

    for (Index i2 = 0; i2 < rows; i2 += mc) {
      const Index actual_mc = (numext::mini)(mc, rows - i2);

      // triangular solver kernel
      {
        // for each small block of the diagonal (=> vertical panels of rhs)
        for (Index j2 = IsLower ? (actual_kc - ((actual_kc % SmallPanelWidth) ? Index(actual_kc % SmallPanelWidth)
                                                                              : Index(SmallPanelWidth)))
                                : 0;
             IsLower ? j2 >= 0 : j2 < actual_kc; IsLower ? j2 -= SmallPanelWidth : j2 += SmallPanelWidth) {
          Index actualPanelWidth = numext::mini<Index>(actual_kc - j2, SmallPanelWidth);
          Index absolute_j2 = actual_k2 + j2;
          Index panelOffset = IsLower ? j2 + actualPanelWidth : 0;
          Index panelLength = IsLower ? actual_kc - j2 - actualPanelWidth : j2;

          // GEBP
          if (panelLength > 0) {
            gebp_kernel(lhs.getSubMapper(i2, absolute_j2), blockA, blockB + j2 * actual_kc, actual_mc, panelLength,
                        actualPanelWidth, Scalar(-1), actual_kc, actual_kc,  // strides
                        panelOffset, panelOffset);                           // offsets
          }

          {
            // unblocked triangular solve
            trsmKernelR<Scalar, Index, Mode, Conjugate, TriStorageOrder, OtherInnerStride,
                        /*Specialized=*/true>::kernel(actualPanelWidth, actual_mc,
                                                      _tri + absolute_j2 + absolute_j2 * triStride, triStride,
                                                      _other + i2 * otherIncr + absolute_j2 * otherStride, otherIncr,
                                                      otherStride);
          }
          // pack the just computed part of lhs to A
          pack_lhs_panel(blockA, lhs.getSubMapper(i2, absolute_j2), actualPanelWidth, actual_mc, actual_kc, j2);
        }
      }

      if (rs > 0)
        gebp_kernel(lhs.getSubMapper(i2, startPanel), blockA, geb, actual_mc, actual_kc, rs, Scalar(-1), -1, -1, 0, 0);
    }
  }
}
}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TRIANGULAR_SOLVER_MATRIX_H
