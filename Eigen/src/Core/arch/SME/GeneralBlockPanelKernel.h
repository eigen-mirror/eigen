// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SME_GENERALBLOCKPANELKERNEL_H
#define EIGEN_SME_GENERALBLOCKPANELKERNEL_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

#include <arm_sme.h>

namespace Eigen {
namespace internal {

// ---------------------------------------------------------------------------
// Streaming vector length and tile geometry.
//
// The micro-kernel is organised around a logical kSmeMr x kSmeNr output
// block, packed depth-major (kSmeMr contiguous floats per depth step).  Those
// dimensions are compile-time constants: they feed gebp_traits (cache
// blocking) and the packers, none of which can depend on a runtime value.
//
// The *physical* tiling of that block onto ZA.S tiles, on the other hand, is
// driven by the runtime streaming vector length.  An fp32 ZA.S tile is
// svlw x svlw, where svlw = svcntsw() is the number of 32-bit elements in a
// streaming vector.  There are only 4 ZA.S tiles, so the block is covered by
// up to a 2x2 grid of svlw x svlw tiles, iterated in sub-block passes when the
// grid is smaller than the block (and predicated down to it when larger).
//
// This translation unit must be built without -msve-vector-bits (scalable/VLA
// mode); see the guard in ConfigureVectorization.h for the rationale.
// Everything below derives lane counts/predicates from the runtime svlw; when a
// block matches the tile grid exactly, the micro-kernel additionally switches
// to a hand-scheduled multi-vector-load loop (see sme_process).
// ---------------------------------------------------------------------------

// Logical micro-kernel block (LHS/RHS panel widths): a full 2x2 ZA-tile grid
// at SVL=512; other SVLs tile the block at runtime.  If a future SVL ever
// justifies a larger block, these two constants are the only knobs -- but
// don't grow them speculatively, a doubled block measures slower at SVL=512.
static constexpr int kSmeMr = 32;
static constexpr int kSmeNr = 32;

// min() usable from streaming functions (numext::mini lacks the
// __arm_streaming_compatible attribute).
template <typename T>
static EIGEN_ALWAYS_INLINE T sme_min(T a, T b) __arm_streaming_compatible {
  return a < b ? a : b;
}

// Copy `width` contiguous source columns per depth step into a depth-major
// packed panel of width `width`, for the depth sub-range [k0, k1).  Both dst and
// src are indexed by the absolute depth index k (dst[k*width+off],
// src[k*src_stride+off]); the caller offsets `src` to the region's column base
// and `dst` to the panel base.  Generalised over the runtime svlw: the panel is
// covered in svlw-wide column chunks, each streamed over the depth sub-range.
// The chunk loop is outermost so each chunk's predicate is computed once instead
// of per depth step (the runtime chunk count keeps the compiler from hoisting it
// on its own).  The symm packers reuse this for the diagonal-split direct/
// transposed regions (a contiguous depth sub-range at a depth offset).
template <typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel_range(float* EIGEN_RESTRICT dst, const float* EIGEN_RESTRICT src,
                                                     Index src_stride, Index k0, Index k1, int width) __arm_streaming {
  const int svlw = static_cast<int>(svcntsw());
  for (int off = 0; off < width; off += svlw) {
    const int w = sme_min(width - off, svlw);
    const svbool_t pred = svwhilelt_b32(uint32_t(0), uint32_t(w));
    for (Index k = k0; k < k1; ++k) {
      svst1_f32(pred, &dst[k * width + off], svld1_f32(pred, &src[k * src_stride + off]));
    }
  }
}

// Copy the full depth [0, depth): thin wrapper used by the (non-symm) gemm
// packers, which always pack a whole panel.
template <typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel(float* EIGEN_RESTRICT dst, const float* EIGEN_RESTRICT src,
                                               Index src_stride, Index depth, int width) __arm_streaming {
  sve_copy_panel_range(dst, src, src_stride, Index(0), depth, width);
}

// Transpose-pack `width` source rows into depth-major packed output using ZA's
// 2D store as a free transpose, for the depth sub-range [k0, k1): a svlw x svlw
// block of source (svlw rows x svlw depth) is loaded as horizontal ZA slices,
// then read back as vertical slices, which emits it depth-major.  Row-groups of
// svlw rows are processed two at a time through ZA tiles 0 and 1: ZA is not
// renamed, so a single tile would stall every load pass on the previous read
// pass (write-after-read); two tiles in flight keep the phases independent.
// Trailing row-groups (when width is not a multiple of 2*svlw) use tile 0 with
// predicated rows.  Both dst and src are indexed by the absolute depth index k:
//   dst[k*width + r] = src[r*src_stride + k],  k in [k0,k1), r in [0,width).
// The symm packers reuse this for the diagonal-split transposed/direct regions
// (a depth sub-range at a depth offset, with a tail-panel width < kSmeMr).
template <typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack_range(float* EIGEN_RESTRICT dst, const float* EIGEN_RESTRICT src,
                                                         Index src_stride, Index k0, Index k1,
                                                         int width) __arm_streaming __arm_inout("za") {
  const svfloat32_t zero = svdup_f32(0.f);
  const svbool_t pg_all = svptrue_b32();
  const int svlw = static_cast<int>(svcntsw());

  for (Index k = k0; k < k1; k += svlw) {
    const int dk = static_cast<int>(sme_min(k1 - k, Index(svlw)));
    const svbool_t pg_d = svwhilelt_b32(uint32_t(0), uint32_t(dk));
    int r0 = 0;
    // Pairs of full row-groups: tiles 0 and 1 in flight.
    for (; r0 + 2 * svlw <= width; r0 += 2 * svlw) {
      for (int r = 0; r < svlw; ++r) {
        svld1_hor_za32(0, uint32_t(r), pg_d, &src[(r0 + r) * src_stride + k]);
        svld1_hor_za32(1, uint32_t(r), pg_d, &src[(r0 + svlw + r) * src_stride + k]);
      }
      for (int c = 0; c < dk; ++c) {
        svst1_f32(pg_all, &dst[(k + c) * width + r0], svread_ver_za32_f32_m(zero, pg_all, 0, uint32_t(c)));
        svst1_f32(pg_all, &dst[(k + c) * width + r0 + svlw], svread_ver_za32_f32_m(zero, pg_all, 1, uint32_t(c)));
      }
    }
    // Trailing row-groups (at most two svlw-wide passes remain, since the pair
    // loop consumed all multiples of 2*svlw): predicate down to the remaining
    // rows.  A single `if` would drop rows when a tail width lands in
    // (svlw, 2*svlw); a loop handles any leftover.
    for (; r0 < width; r0 += svlw) {
      const int rg = sme_min(width - r0, svlw);
      const svbool_t pg_r = svwhilelt_b32(uint32_t(0), uint32_t(rg));
      for (int r = 0; r < rg; ++r) {
        svld1_hor_za32(0, uint32_t(r), pg_d, &src[(r0 + r) * src_stride + k]);
      }
      for (int c = 0; c < dk; ++c) {
        svst1_f32(pg_r, &dst[(k + c) * width + r0], svread_ver_za32_f32_m(zero, pg_r, 0, uint32_t(c)));
      }
    }
  }
}

// Transpose-pack a whole kSmeMr-wide panel over the full depth [0, depth): thin
// wrapper used by the (non-symm) gemm packers.
template <typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack(float* EIGEN_RESTRICT dst, const float* EIGEN_RESTRICT src,
                                                   Index src_stride, Index depth) __arm_streaming __arm_inout("za") {
  static_assert(kSmeMr == kSmeNr, "SME transpose pack assumes square panels");
  sme_transpose_pack_range(dst, src, src_stride, Index(0), depth, kSmeMr);
}

template <typename Index>
static EIGEN_ALWAYS_INLINE void scalar_tail_pack(float* EIGEN_RESTRICT dst_panel, const float* EIGEN_RESTRICT src,
                                                 Index src_stride, Index depth, Index tail) __arm_streaming {
  for (Index k = 0; k < depth; ++k) {
    for (Index i = 0; i < tail; ++i) {
      dst_panel[k * tail + i] = src[i * src_stride + k];
    }
  }
}

// ---------------------------------------------------------------------------
// Generic (mapper-based) packing fallback.
//
// The streaming pack_lhs_*/pack_rhs_* helpers take &lhs(0,0) once and walk it by
// raw pointer + lhs.stride(). That breaks for two DataMapper families:
//   - TensorContractionSubMapper::operator() returns by value, so &lhs(0,0) is
//     address-of-rvalue (a compile error, not just wrong results);
//   - blas_data_mapper with Incr != 1 (inner-strided Maps, e.g. from
//     TriangularSolverMatrix) can't be walked by stride() alone.
// These fall back to the mapper's packet/element interface, emitting the
// identical depth-major panel layout so gebp_kernel can't tell the paths apart.
// ---------------------------------------------------------------------------

// True iff DataMapper exposes .incr() (the blas_data_mapper family); others are
// unit-inner-stride by construction.
template <typename DataMapper, typename EnableIf = void>
struct sme_has_incr : std::false_type {};
template <typename DataMapper>
struct sme_has_incr<DataMapper, void_t<decltype(std::declval<const DataMapper&>().incr())>> : std::true_type {};

template <typename Index, typename DataMapper, std::enable_if_t<sme_has_incr<DataMapper>::value, bool> = true>
EIGEN_ALWAYS_INLINE Index sme_mapper_incr(const DataMapper& m) {
  return static_cast<Index>(m.incr());
}
template <typename Index, typename DataMapper, std::enable_if_t<!sme_has_incr<DataMapper>::value, bool> = true>
EIGEN_ALWAYS_INLINE Index sme_mapper_incr(const DataMapper&) {
  return Index(1);
}

// Whether operator()(i,j) returns an lvalue reference into caller storage (so
// &m(0,0) + stride walking is valid). False for by-value mappers (Tensor's).
template <typename DataMapper, typename Index>
struct sme_mapper_has_direct_access {
  static constexpr bool value = std::is_lvalue_reference<decltype(std::declval<const DataMapper&>()(
      std::declval<Index>(), std::declval<Index>()))>::value;
};

// LHS fallback: pack via the mapper's packet interface, shared by both
// gemm_pack_lhs specializations. Taken by mappers without direct lvalue access
// (TensorContractionSubMapper returns by value) or with a non-unit inner
// stride. Vectorised with NEON packets, exactly like the generic packers drive
// these same mappers. Tensor sub-mappers (the hot path -- tensor contractions
// pack through this on both sides) have contiguous packet loads, but their
// ordinary operator()/loadPacket functions cannot be called from a streaming
// context. Inner-strided ColMajor blas mappers instead require gathers;
// streaming-mode gathers need FEAT_SME_FA64 (absent on e.g. Apple M4), while
// NEON's pgather uses scalar source loads and a contiguous packet store. The
// packet path assumes the mapper's packets advance the first index; that holds
// for ColMajor tensor and blas mappers, but not for RowMajor mappers, whose
// packets run along the storage-inner second index. RowMajor dispatches pass
// vectorise = false and take the scalar element loop.
template <typename Index, typename DataMapper, bool PanelMode>
void sme_pack_lhs_fallback(float* dst_base, const DataMapper& lhs, Index depth, Index rows, Index dst_stride,
                           Index dst_offset, bool vectorise) {
  typedef typename packet_traits<float>::type Packet;
  constexpr Index PacketSize = Index(packet_traits<float>::size);
  constexpr int MR = kSmeMr;

  for (Index i = 0; i < rows; i += MR) {
    const Index w = numext::mini(rows - i, Index(MR));
    float* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * w : dst_base + i * depth;
    const Index peeled_w = vectorise ? (w / PacketSize) * PacketSize : Index(0);
    for (Index k = 0; k < depth; ++k) {
      float* dst_row = dst_panel + k * w;
      Index r = 0;
      for (; r < peeled_w; r += PacketSize) {
        pstoreu(dst_row + r, lhs.template loadPacket<Packet>(i + r, k));
      }
      for (; r < w; ++r) {
        dst_row[r] = lhs(i + r, k);
      }
    }
  }
}

// RHS fallback, mirroring sme_pack_lhs_fallback (including the vectorise
// contract: LinearMapper packets must advance the first (depth) index). The
// packed layout wants consecutive columns contiguous while the mapper's
// packets run along the depth k, so PacketSize columns are loaded as packets
// along k and transposed in-register (the same LinearMapper + ptranspose
// scheme as the generic gemm_pack_rhs).
template <typename Index, typename DataMapper, bool PanelMode>
void sme_pack_rhs_fallback(float* dst_base, const DataMapper& rhs, Index depth, Index cols, Index dst_stride,
                           Index dst_offset, bool vectorise) {
  typedef typename packet_traits<float>::type Packet;
  typedef typename DataMapper::LinearMapper LinearMapper;
  constexpr Index PacketSize = Index(packet_traits<float>::size);
  static_assert(PacketSize == 4, "the transpose micro-block below is written for 4-wide NEON packets");
  constexpr int NR = kSmeNr;
  const Index peeled_depth = (depth / PacketSize) * PacketSize;

  for (Index j = 0; j < cols; j += NR) {
    const Index w = numext::mini(cols - j, Index(NR));
    float* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * w : dst_base + j * depth;
    const Index peeled_w = vectorise ? (w / PacketSize) * PacketSize : Index(0);
    Index c = 0;
    for (; c < peeled_w; c += PacketSize) {
      const LinearMapper dm0 = rhs.getLinearMapper(0, j + c + 0);
      const LinearMapper dm1 = rhs.getLinearMapper(0, j + c + 1);
      const LinearMapper dm2 = rhs.getLinearMapper(0, j + c + 2);
      const LinearMapper dm3 = rhs.getLinearMapper(0, j + c + 3);
      Index k = 0;
      for (; k < peeled_depth; k += PacketSize) {
        PacketBlock<Packet, 4> block;
        block.packet[0] = dm0.template loadPacket<Packet>(k);
        block.packet[1] = dm1.template loadPacket<Packet>(k);
        block.packet[2] = dm2.template loadPacket<Packet>(k);
        block.packet[3] = dm3.template loadPacket<Packet>(k);
        ptranspose(block);
        pstoreu(dst_panel + (k + 0) * w + c, block.packet[0]);
        pstoreu(dst_panel + (k + 1) * w + c, block.packet[1]);
        pstoreu(dst_panel + (k + 2) * w + c, block.packet[2]);
        pstoreu(dst_panel + (k + 3) * w + c, block.packet[3]);
      }
      for (; k < depth; ++k) {
        for (Index p = 0; p < PacketSize; ++p) {
          dst_panel[k * w + c + p] = rhs(k, j + c + p);
        }
      }
    }
    for (; c < w; ++c) {
      for (Index k = 0; k < depth; ++k) {
        dst_panel[k * w + c] = rhs(k, j + c);
      }
    }
  }
}

// Shared dispatch for the four gemm_pack specializations: raw-pointer walk
// when the mapper grants direct unit-inner-stride access, otherwise the
// packet/element fallback. Tag-dispatched so &m(0,0) is only compiled for
// lvalue mappers. UsePacketPath records whether the mapper's packets advance
// the index the fallback needs, independently of its direct-access category.
template <bool UsePacketPath, typename Index, typename DataMapper, typename DirectFn, typename FallbackFn>
EIGEN_ALWAYS_INLINE void sme_dispatch_pack(DirectFn direct, FallbackFn fallback, float* block, const DataMapper& m,
                                           Index depth, Index n, Index stride, Index offset,
                                           std::true_type /* direct access */) {
  if (sme_mapper_incr<Index>(m) == 1) {
    const float* src = (n > 0 && depth > 0) ? &m(0, 0) : nullptr;
    direct(block, src, m.stride(), depth, n, stride, offset);
  } else {
    fallback(block, m, depth, n, stride, offset, UsePacketPath);
  }
}
template <bool UsePacketPath, typename Index, typename DataMapper, typename DirectFn, typename FallbackFn>
EIGEN_ALWAYS_INLINE void sme_dispatch_pack(DirectFn, FallbackFn fallback, float* block, const DataMapper& m,
                                           Index depth, Index n, Index stride, Index offset,
                                           std::false_type /* no direct access */) {
  fallback(block, m, depth, n, stride, offset, UsePacketPath);
}

/*****************************************************************************
 * gebp_traits specialization for SME  (float x float)
 *
 * Overrides mr and nr so that:
 *   - gemm_pack_lhs receives Pack1 = mr, creating uniform LHS panels
 *   - gemm_pack_rhs receives nr, creating uniform RHS panels
 *   - mc is rounded to a multiple of mr, nc to a multiple of nr
 *   - Cache blocking (kc, mc, nc) is recomputed accordingly
 *
 * We provide custom gemm_pack_lhs/gemm_pack_rhs specializations for float,
 * so both ColMajor and RowMajor source matrices produce an identical,
 * simple packed format that the SME kernel consumes.
 *
 * Mixed-scalar products (e.g. MatrixXf * MatrixXcf) also instantiate
 * gemm_pack_lhs<float, ...>, but with Pack1/nr from the generic
 * gebp_traits<float, complex<float>> (mr=6, nr=4) and are consumed by the
 * generic gebp_kernel, not the SME one. So the specializations below pin
 * Pack1/nr_ to kSmeMr/kSmeNr: only the instantiation that feeds the SME
 * gebp_kernel matches; mixed-scalar ones fall through to the generic template.
 * This is load-bearing: it relies on no other float consumer instantiating the
 * packer with mr == kSmeMr (holds today -- generic float traits give mr <= 12).
 * The kernel side is self-checking (the SME gebp_kernel static_asserts
 * mr/nr == kSmeMr/kSmeNr, so a float traits change breaks the build instead of
 * silently mispairing packer and kernel); the packer side is enforced by the
 * static_asserts below for the in-tree mixed-scalar traits (downstream code
 * instantiating the packers with hand-picked mr/nr remains uncovered).
 *****************************************************************************/

template <>
class gebp_traits<float, float, false, false, Architecture::Target, GEBPPacketFull>
    : public gebp_traits<float, float, false, false, Architecture::Target, GEBPPacketHalf> {
 public:
  // The base class provides all the standard typedefs (LhsPacket, etc.)
  // We only override the register-block sizes.
  enum {
    mr = kSmeMr,  // LHS panel width
    nr = kSmeNr   // RHS panel width
  };
};

// The packers do not know the opposite scalar type, so kSmeMr/kSmeNr are
// effectively SME-format tags. Ensure the in-tree mixed-scalar traits cannot
// select an SME packer whose output would be consumed by the generic kernel.
static_assert(int(gebp_traits<float, std::complex<float>>::mr) != kSmeMr,
              "gebp_traits<float, complex<float>>::mr collides with kSmeMr: the SME gemm_pack_lhs would silently "
              "emit SME panel layout for the generic gebp_kernel");
static_assert(int(gebp_traits<std::complex<float>, float>::nr) != kSmeNr,
              "gebp_traits<complex<float>, float>::nr collides with kSmeNr: the SME gemm_pack_rhs would silently "
              "emit SME panel layout for the generic gebp_kernel");

/*****************************************************************************
 * gemm_pack_lhs specialization for SME  (float, ColMajor)
 *
 * Packs the LHS matrix into uniform panels of width mr = kSmeMr.
 * Each depth step k writes exactly MR contiguous floats.
 *****************************************************************************/

// Pack1 is pinned to kSmeMr (rather than left open) so this specialization
// only matches consumers that actually feed the SME gebp_kernel -- see
// "Mixed-scalar products" in the gebp_traits doc comment above.
template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, kSmeMr, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  typedef float Scalar;
  // Conjugate is deliberately ignored: conj is the identity for real scalars,
  // and Conjugate=true instantiations do occur (e.g. the SYMM above-diagonal
  // transposed pack). A complex port of these packers must actually conjugate.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming static void pack_lhs_colmajor(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src,
                                                        Index src_stride, Index depth, Index rows, Index dst_stride,
                                                        Index dst_offset) {
    constexpr int MR = kSmeMr;
    const Index peeled_rows = (rows / MR) * MR;

    // Full panels of width MR, streamed in svlw-wide predicated chunks.
    for (Index i = 0; i < peeled_rows; i += MR) {
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * MR : dst_base + i * depth;
      sve_copy_panel(dst_panel, src + i, src_stride, depth, MR);
    }

    // Tail panel: rows < MR, use predicated SVE.
    if (peeled_rows < rows) {
      const Index tail = rows - peeled_rows;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_rows * dst_stride + dst_offset * tail : dst_base + peeled_rows * depth;
      sve_copy_panel(dst_panel, src + peeled_rows, src_stride, depth, static_cast<int>(tail));
    }
  }

  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    if (PanelMode) {
      eigen_assert(stride >= depth && offset <= stride);
    }
    // Inner-strided ColMajor blas mappers' packets advance the row index, so
    // the fallback may use them.
    sme_dispatch_pack<true>(&pack_lhs_colmajor, &sme_pack_lhs_fallback<Index, DataMapper, PanelMode>, blockA, lhs,
                            depth, rows, stride, offset,
                            bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

// RowMajor LHS packer -- SME in-ZA transpose.
//
// The packed output wants depth-major layout (MR rows contiguous per depth
// step) but the RowMajor source has rows contiguous (strided by depth per
// row).  A natural SVE gather would be slow; instead we use ZA's 2D store
// as a free transpose: load svlw rows as horizontal slices of a ZA.S tile,
// then read vertical slices to produce depth-major output (see
// sme_transpose_pack).
// Pack1 pinned to kSmeMr -- see the ColMajor specialization above.
template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, kSmeMr, Pack2, Packet, RowMajor, Conjugate, PanelMode> {
  typedef float Scalar;
  // See the ColMajor specialization above: Conjugate is ignored, sound only
  // for real scalars.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming __arm_new("za") static void pack_lhs_rowmajor(Scalar* dst_base,
                                                                        const Scalar* EIGEN_RESTRICT src,
                                                                        Index src_stride, Index depth, Index rows,
                                                                        Index dst_stride, Index dst_offset) {
    constexpr int MR = kSmeMr;
    const Index peeled_rows = (rows / MR) * MR;

    for (Index i = 0; i < peeled_rows; i += MR) {
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * MR : dst_base + i * depth;
      sme_transpose_pack(dst_panel, src + i * src_stride, src_stride, depth);
    }

    // Row tail (rows - peeled_rows in [1, MR-1]).  This branch runs at most
    // once per pack_lhs call with < MR rows and would need a partial-ZA-tile
    // dance to vectorise; total copies are < MR * depth per call, which is
    // noise vs the main packer's workload, so scalar is the simple choice.
    if (peeled_rows < rows) {
      const Index tail = rows - peeled_rows;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_rows * dst_stride + dst_offset * tail : dst_base + peeled_rows * depth;
      scalar_tail_pack(dst_panel, src + peeled_rows * src_stride, src_stride, depth, tail);
    }
  }

  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    if (PanelMode) {
      eigen_assert(stride >= depth && offset <= stride);
    }
    // Inner-strided RowMajor blas mappers' packets advance the depth index, not
    // the row index, so the fallback must stay scalar (see
    // sme_pack_lhs_fallback).
    sme_dispatch_pack<false>(&pack_lhs_rowmajor, &sme_pack_lhs_fallback<Index, DataMapper, PanelMode>, blockA, lhs,
                             depth, rows, stride, offset,
                             bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

/*****************************************************************************
 * gemm_pack_rhs specialization for SME  (float, ColMajor) -- SME in-ZA
 * transpose, mirroring the RowMajor LHS packer.
 *
 * Packs the RHS matrix into panels of width nr = kSmeNr.  ColMajor source has
 * columns contiguous; we load NR columns as horizontal ZA slices and then
 * read verticals to produce depth-major packed output.
 *****************************************************************************/

// nr_ is pinned to kSmeNr (rather than left open) so this specialization
// only matches consumers that actually feed the SME gebp_kernel -- see
// "Mixed-scalar products" in the gebp_traits doc comment above.
template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, kSmeNr, ColMajor, Conjugate, PanelMode> {
  typedef float Scalar;
  // See gemm_pack_lhs above: Conjugate is ignored, sound only for real scalars.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming __arm_new("za") static void pack_rhs_colmajor(Scalar* dst_base,
                                                                        const Scalar* EIGEN_RESTRICT src,
                                                                        Index src_stride, Index depth, Index cols,
                                                                        Index dst_stride, Index dst_offset) {
    constexpr int NR = kSmeNr;
    const Index peeled_cols = (cols / NR) * NR;

    for (Index j = 0; j < peeled_cols; j += NR) {
      Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * NR : dst_base + j * depth;
      sme_transpose_pack(dst_panel, src + j * src_stride, src_stride, depth);
    }

    // Col tail (cols - peeled_cols in [1, NR-1]).  Same reasoning as the LHS
    // RowMajor packer's row tail: runs at most once per call, < NR cols, not
    // worth the partial-ZA-tile handling.
    if (peeled_cols < cols) {
      const Index tail = cols - peeled_cols;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_cols * dst_stride + dst_offset * tail : dst_base + peeled_cols * depth;
      scalar_tail_pack(dst_panel, src + peeled_cols * src_stride, src_stride, depth, tail);
    }
  }

  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) {
    if (PanelMode) {
      eigen_assert(stride >= depth && offset <= stride);
    }
    // Inner-strided ColMajor blas mappers' LinearMapper packets advance the
    // depth index, which is what the fallback transposes.
    sme_dispatch_pack<true>(&pack_rhs_colmajor, &sme_pack_rhs_fallback<Index, DataMapper, PanelMode>, blockB, rhs,
                            depth, cols, stride, offset,
                            bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

// RowMajor RHS packer -- streaming SVE copy (mirrors the ColMajor LHS packer).
// Rows are contiguous in the source, so each depth-step is NR contiguous fp32.
// nr_ pinned to kSmeNr -- see the ColMajor specialization above.
template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, kSmeNr, RowMajor, Conjugate, PanelMode> {
  typedef float Scalar;
  // See gemm_pack_lhs above: Conjugate is ignored, sound only for real scalars.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming static void pack_rhs_rowmajor(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src,
                                                        Index src_stride, Index depth, Index cols, Index dst_stride,
                                                        Index dst_offset) {
    constexpr int NR = kSmeNr;
    const Index peeled_cols = (cols / NR) * NR;

    for (Index j = 0; j < peeled_cols; j += NR) {
      Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * NR : dst_base + j * depth;
      sve_copy_panel(dst_panel, src + j, src_stride, depth, NR);
    }

    if (peeled_cols < cols) {
      const Index tail = cols - peeled_cols;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_cols * dst_stride + dst_offset * tail : dst_base + peeled_cols * depth;
      sve_copy_panel(dst_panel, src + peeled_cols, src_stride, depth, static_cast<int>(tail));
    }
  }

  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) {
    if (PanelMode) {
      eigen_assert(stride >= depth && offset <= stride);
    }
    // Inner-strided RowMajor blas mappers' LinearMapper packets advance the
    // column index, not depth, so the fallback must stay scalar (see
    // sme_pack_rhs_fallback).
    sme_dispatch_pack<false>(&pack_rhs_rowmajor, &sme_pack_rhs_fallback<Index, DataMapper, PanelMode>, blockB, rhs,
                             depth, cols, stride, offset,
                             bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

/*****************************************************************************
 * sme_store_za_tile -- Store one ZA.S tile back to C with alpha scaling.
 *
 * `pw` is the row-predicate width for this tile, `cw` the col-predicate width
 * (both <= the runtime svlw).
 *****************************************************************************/

template <int TileId, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_za_tile(float* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col, float alpha,
                                           Index row_start, int pw, Index col_start,
                                           int cw) __arm_streaming __arm_inout("za") {
  const svbool_t pg_m = svwhilelt_b32((uint32_t)0, (uint32_t)pw);
  const svbool_t pg_n = svwhilelt_b32((uint32_t)0, (uint32_t)cw);
  // FMLA and FADD have equal latency/throughput on ARMv9 cores, and
  // multiplying by alpha=1.0 is exact in IEEE-754 so the FMLA form is
  // bit-identical to FADD in that case.  A single unconditional FMLA
  // keeps the store compact and measures no worse (and a few percent
  // better on small matrices, where the branch would otherwise disrupt
  // instruction scheduling).
  const svfloat32_t vzero = svdup_f32(0.f);
  const svfloat32_t valpha = svdup_f32(alpha);

  if (C_stride_row == 1) {
    // Column-major C: extract vertical slices (columns of the ZA tile)
    for (int ci = 0; ci < cw; ++ci) {
      svfloat32_t vres = svread_ver_za32_f32_m(vzero, pg_m, TileId, (uint32_t)ci);
      float* pC = C + row_start + (col_start + ci) * C_stride_col;
      svfloat32_t vc = svld1_f32(pg_m, pC);
      svst1_f32(pg_m, pC, svmla_f32_x(pg_m, vc, vres, valpha));
    }
  } else if (C_stride_col == 1) {
    // Row-major C: extract horizontal slices (rows of the ZA tile)
    for (int ri = 0; ri < pw; ++ri) {
      svfloat32_t vres = svread_hor_za32_f32_m(vzero, pg_n, TileId, (uint32_t)ri);
      float* pC = C + (row_start + ri) * C_stride_row + col_start;
      svfloat32_t vc = svld1_f32(pg_n, pC);
      svst1_f32(pg_n, pC, svmla_f32_x(pg_n, vc, vres, valpha));
    }
  } else {
    // General stride: extract rows to temp buffer, scatter to C.  scratch
    // holds one ZA row; every caller passes cw <= min(svlw, kSmeNr) (a tile
    // never spans more than the logical block), so kSmeNr is a static
    // bound independent of the runtime svlw.
    float scratch[kSmeNr];
    for (int ri = 0; ri < pw; ++ri) {
      svfloat32_t vres = svread_hor_za32_f32_m(vzero, pg_n, TileId, (uint32_t)ri);
      vres = svmul_f32_x(pg_n, vres, valpha);
      svst1_f32(pg_n, scratch, vres);
      for (int ci = 0; ci < cw; ++ci) {
        C[(row_start + ri) * C_stride_row + (col_start + ci) * C_stride_col] += scratch[ci];
      }
    }
  }
}

/*****************************************************************************
 * sme_store_2x2_grid -- store the (up to) 2x2 grid of svlw x svlw ZA tiles.
 *
 * Tile layout:  0 = (row-lo, col-lo)  1 = (row-lo, col-hi)
 *               2 = (row-hi, col-lo)  3 = (row-hi, col-hi)
 * The col-hi tiles (1, 3) are stored only when chi > 0 and the row-hi tiles
 * (2, 3) only when rhi > 0, so a single tile, a 1x2/2x1 pair, or the full grid
 * all route through here.  Runs once per sub-block pass, after a depth loop
 * that dwarfs it, so the branches cost nothing and predict perfectly (the
 * pattern repeats across blocks).
 *****************************************************************************/

template <typename Index>
EIGEN_ALWAYS_INLINE void sme_store_2x2_grid(float* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                            float alpha, Index row_start, int rlo, int rhi, Index col_start, int clo,
                                            int chi) __arm_streaming __arm_inout("za") {
  const int svlw = static_cast<int>(svcntsw());
  sme_store_za_tile<0>(C, C_stride_row, C_stride_col, alpha, row_start, rlo, col_start, clo);
  if (chi > 0) {
    sme_store_za_tile<1>(C, C_stride_row, C_stride_col, alpha, row_start, rlo, col_start + svlw, chi);
  }
  if (rhi > 0) {
    sme_store_za_tile<2>(C, C_stride_row, C_stride_col, alpha, row_start + svlw, rhi, col_start, clo);
    if (chi > 0) {
      sme_store_za_tile<3>(C, C_stride_row, C_stride_col, alpha, row_start + svlw, rhi, col_start + svlw, chi);
    }
  }
}

// One depth step's worth of the exact-match grid: the four FMOPAs that take the
// lo/hi halves of a packed A column and a packed B column and accumulate the
// 2x2 ZA-tile outer product.  `all` is svptrue_b32() because this is only used
// on the exact-match path, where the block fills the grid (pg == svptrue_b32),
// so factoring it out is identical to the inline form.
static EIGEN_ALWAYS_INLINE void outer_product_2x2(svfloat32_t a_lo, svfloat32_t a_hi, svfloat32_t b_lo,
                                                  svfloat32_t b_hi) __arm_streaming __arm_inout("za") {
  const svbool_t all = svptrue_b32();
  svmopa_za32_f32_m(0, all, all, a_lo, b_lo);
  svmopa_za32_f32_m(1, all, all, a_lo, b_hi);
  svmopa_za32_f32_m(2, all, all, a_hi, b_lo);
  svmopa_za32_f32_m(3, all, all, a_hi, b_hi);
}

/*****************************************************************************
 * sme_process -- micro-kernel for one pw x cw output block.
 *
 * Tiles the block into svlw x svlw ZA tiles, processed in passes of up to a 2x2
 * tile grid (the 4-ZA-tile budget): several (2*svlw) x (2*svlw) sub-block passes
 * when the grid is smaller than the block, tiles predicated down to the block
 * width when it is larger.  blA/blB are packed depth-major with depth-strides
 * pw and cw respectively.
 *
 * When the block matches the tile grid exactly (pw == cw == 2 * svlw), the
 * packed rows are also contiguous across depth steps, enabling the
 * hand-scheduled loop below: per 4 unrolled depth steps, 2 svld1_f32_x4 per
 * side (each spanning 2 depth steps) feed 16 FMOPAs -- a 1:1 compute:load
 * ratio at the vector level.  All other geometries use predicated
 * per-depth-step loads.
 *****************************************************************************/

template <typename Index>
EIGEN_ALWAYS_INLINE void sme_process(float* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                     const float* EIGEN_RESTRICT blA, const float* EIGEN_RESTRICT blB, Index depth,
                                     float alpha, Index row_start, int pw, Index col_start,
                                     int cw) __arm_streaming __arm_inout("za") {
  const int svlw = static_cast<int>(svcntsw());

  for (int rt = 0; rt < pw; rt += 2 * svlw) {
    const int rpw = sme_min(pw - rt, 2 * svlw);
    const int rlo = sme_min(rpw, svlw);
    const int rhi = rpw - rlo;  // >= 0; > 0 only when rpw > svlw, in which case rlo == svlw
    const svbool_t pg_rlo = svwhilelt_b32((uint32_t)0, (uint32_t)rlo);
    const svbool_t pg_rhi = svwhilelt_b32((uint32_t)0, (uint32_t)rhi);

    for (int ct = 0; ct < cw; ct += 2 * svlw) {
      const int cpw = sme_min(cw - ct, 2 * svlw);
      const int clo = sme_min(cpw, svlw);
      const int chi = cpw - clo;
      const svbool_t pg_clo = svwhilelt_b32((uint32_t)0, (uint32_t)clo);
      const svbool_t pg_chi = svwhilelt_b32((uint32_t)0, (uint32_t)chi);

      svzero_za();
      if (pw == 2 * svlw && cw == 2 * svlw) {
        // The block is exactly one full-grid patch (single pass, rt == ct ==
        // 0, rlo == rhi == clo == chi == svlw), so a packed row is the
        // patch's slice and rows are contiguous across depth steps: x4 loads
        // each span 2 of them, e.g. va_01 = [d0 lo, d0 hi, d1 lo, d1 hi].
        const svcount_t pn = svptrue_c32();
        const Index depth_4 = (depth / 4) * 4;
        Index k = 0;
        for (; k < depth_4; k += 4) {
          svfloat32x4_t va_01 = svld1_f32_x4(pn, &blA[k * pw]);
          svfloat32x4_t vb_01 = svld1_f32_x4(pn, &blB[k * cw]);

          // d0
          outer_product_2x2(svget4_f32(va_01, 0), svget4_f32(va_01, 1), svget4_f32(vb_01, 0), svget4_f32(vb_01, 1));
          // d1
          outer_product_2x2(svget4_f32(va_01, 2), svget4_f32(va_01, 3), svget4_f32(vb_01, 2), svget4_f32(vb_01, 3));

          svfloat32x4_t va_23 = svld1_f32_x4(pn, &blA[(k + 2) * pw]);
          svfloat32x4_t vb_23 = svld1_f32_x4(pn, &blB[(k + 2) * cw]);

          // d2
          outer_product_2x2(svget4_f32(va_23, 0), svget4_f32(va_23, 1), svget4_f32(vb_23, 0), svget4_f32(vb_23, 1));
          // d3
          outer_product_2x2(svget4_f32(va_23, 2), svget4_f32(va_23, 3), svget4_f32(vb_23, 2), svget4_f32(vb_23, 3));
        }
        // Depth tail: one x2 load per side per step.
        for (; k < depth; ++k) {
          svfloat32x2_t va = svld1_f32_x2(pn, &blA[k * pw]);
          svfloat32x2_t vb = svld1_f32_x2(pn, &blB[k * cw]);
          outer_product_2x2(svget2_f32(va, 0), svget2_f32(va, 1), svget2_f32(vb, 0), svget2_f32(vb, 1));
        }
      } else {
        for (Index k = 0; k < depth; ++k) {
          svfloat32_t a_lo = svld1_f32(pg_rlo, &blA[k * pw + rt]);
          svfloat32_t b_lo = svld1_f32(pg_clo, &blB[k * cw + ct]);
          svmopa_za32_f32_m(0, pg_rlo, pg_clo, a_lo, b_lo);
          svfloat32_t b_hi = svdup_f32(0.f);
          if (chi > 0) {
            b_hi = svld1_f32(pg_chi, &blB[k * cw + ct + svlw]);
            svmopa_za32_f32_m(1, pg_rlo, pg_chi, a_lo, b_hi);
          }
          if (rhi > 0) {
            svfloat32_t a_hi = svld1_f32(pg_rhi, &blA[k * pw + rt + svlw]);
            svmopa_za32_f32_m(2, pg_rhi, pg_clo, a_hi, b_lo);
            if (chi > 0) {
              svmopa_za32_f32_m(3, pg_rhi, pg_chi, a_hi, b_hi);
            }
          }
        }
      }

      // Store the (up to) 2x2 grid of tiles for this sub-block pass.
      sme_store_2x2_grid(C, C_stride_row, C_stride_col, alpha, row_start + rt, rlo, rhi, col_start + ct, clo, chi);
    }
  }
}

template <typename Index>
EIGEN_DONT_INLINE __arm_locally_streaming __arm_new("za") void sme_gebp_impl(
    float* C, Index C_stride_row, Index C_stride_col, const float* blockA, const float* blockB, Index rows, Index depth,
    Index cols, float alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  constexpr int MR = kSmeMr;
  constexpr int NR = kSmeNr;

  // Column-outer, row-inner: keeps blB (one kc × NR panel) hot in L1 while
  // smaller blA tiles stream from L2.  The outer GOTO loop in
  // GeneralMatrixMatrix.h ensures blockA fits in L2 via mc-blocking.  Each
  // packed panel is depth-major with depth-stride equal to its width (MR/NR
  // for full panels, the tail width otherwise), so that width is passed as
  // both the logical block size and the load stride to sme_process; partial
  // blocks are tiled and predicated inside the generic path.
  for (Index j = 0; j < cols; j += NR) {
    const int cw = static_cast<int>(sme_min(cols - j, Index(NR)));
    const float* blB = blockB + j * strideB + offsetB * cw;

    for (Index i = 0; i < rows; i += MR) {
      const int pw = static_cast<int>(sme_min(rows - i, Index(MR)));
      const float* blA = blockA + i * strideA + offsetA * pw;
      sme_process(C, C_stride_row, C_stride_col, blA, blB, depth, alpha, i, pw, j, cw);
    }
  }
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef float Scalar;
  typedef float ResScalar;

  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows,
                                    Index depth, Index cols, ResScalar alpha, Index strideA = -1, Index strideB = -1,
                                    Index offsetA = 0, Index offsetB = 0) {
    static_assert(!ConjugateLhs && !ConjugateRhs, "SME fp32 kernel does not support conjugation");
    static_assert(mr == kSmeMr && nr == kSmeNr, "SME fp32 kernel expects kSmeMr/kSmeNr-sized packed panels");

    if (strideA == -1) strideA = depth;
    if (strideB == -1) strideB = depth;

    if (rows <= 0 || cols <= 0 || depth <= 0) return;

    Scalar* C_base = const_cast<Scalar*>(&res(0, 0));
    const Index C_stride_row = &res(1, 0) - &res(0, 0);
    const Index C_stride_col = &res(0, 1) - &res(0, 0);

    sme_gebp_impl(C_base, C_stride_row, C_stride_col, blockA, blockB, rows, depth, cols, alpha, strideA, strideB,
                  offsetA, offsetB);
  }
};

// ---------------------------------------------------------------------------
// Selfadjoint (SYMM) packers.
//
// product_selfadjoint_matrix packs the selfadjoint operand (stored as one
// triangle) through symm_pack_lhs/symm_pack_rhs, which materialize the full
// matrix as they pack. The generic SYMM packers emit packet-width sub-panels
// for the generic gebp_kernel, whereas the SME kernel expects uniform
// kSmeMr/kSmeNr-wide depth-major panels. These packers perform the same
// triangle mirroring in the SME layout; complex SYMM keeps the generic path.
//
// The packer receives the operand in an orientation where row >= col is the
// stored triangle. It reads that half directly and mirrors the other half
// (conjugation is an identity for float):
//   full(row,col) = (row >= col) ? m(row,col) : m(col,row)
//
// Regions wholly below or above the diagonal use the normal dense copy or
// transpose packers. Only the width-wide part of a panel crossed by the
// diagonal needs special handling: each depth row is split between the stored
// triangle and its mirrored half.
//
// For a panel at offset j (entries j+c, c in [0,w)) and global row k2+k, the
// three depth regions are:
//   transposed k in [0, j-k2)      : k2+k < j+c for all c -> m(j+c, k2+k)
//   straddle   k in [j-k2, j+w-k2) : diagonal crosses      -> per-k split
//   direct     k in [j+w-k2, depth): k2+k > j+c for all c  -> m(k2+k, j+c)
//
// The RHS uses this mapping directly. The LHS packs full(j+r, k), which equals
// full(k, j+r) by symmetry, so it uses the same mapping with k2 == 0 relative
// to its diagonal-anchored base pointer.
// ---------------------------------------------------------------------------

// Streaming packer shared by the LHS (k2 == 0) and RHS symm specializations.
// ColM selects the ColMajor selfadjoint operand.
template <int StorageOrder, typename Index>
EIGEN_DONT_INLINE __arm_locally_streaming __arm_new("za") void sme_symm_pack_panels(float* block,
                                                                                    const float* EIGEN_RESTRICT base,
                                                                                    Index stride, Index depth,
                                                                                    Index outer, Index k2) {
  static_assert(kSmeMr == kSmeNr, "the shared SYMM packer assumes square panels");
  constexpr int PACK = kSmeMr;
  constexpr bool ColM = (StorageOrder == ColMajor);
  const int svlw = static_cast<int>(svcntsw());

  for (Index j = 0; j < outer; j += PACK) {
    const int w = static_cast<int>(sme_min(outer - j, Index(PACK)));
    float* dst = block + j * depth;  // depth-major panel of width w

    // Depth-region boundaries (all clamped to [0, depth]).
    const Index raw_t = j - k2, raw_s = j + Index(w) - k2;
    const Index t_end = raw_t <= 0 ? Index(0) : sme_min(raw_t, depth);  // transposed [0, t_end)
    const Index s_end = raw_s <= 0 ? Index(0) : sme_min(raw_s, depth);  // straddle   [t_end, s_end)
    // direct [s_end, depth)

    // Transposed region: full(k2+k, j+c) = m(j+c, k2+k).
    if (t_end > 0) {
      EIGEN_IF_CONSTEXPR (ColM) {
        sve_copy_panel_range(dst, base + j + k2 * stride, stride, Index(0), t_end, w);
      } else {
        sme_transpose_pack_range(dst, base + j * stride + k2, stride, Index(0), t_end, w);
      }
    }
    // Direct region: full(k2+k, j+c) = m(k2+k, j+c).
    if (s_end < depth) {
      EIGEN_IF_CONSTEXPR (ColM) {
        sme_transpose_pack_range(dst, base + k2 + j * stride, stride, s_end, depth, w);
      } else {
        sve_copy_panel_range(dst, base + k2 * stride + j, stride, s_end, depth, w);
      }
    }
    // Straddle band: the diagonal crosses the panel at c* = (k2+k) - j (in
    // [0, w) for every k in the band), splitting each depth step into a
    // direct head (c < c*: m(k2+k, j+c)) and a mirrored tail (c >= c*:
    // m(j+c, k2+k); at c == c* both name the diagonal element). One side is
    // contiguous in c -- the head for RowMajor, the tail for ColMajor -- and
    // is copied with predicated vectors; the other walks the stride scalar.
    for (Index k = t_end; k < s_end; ++k) {
      const Index row = k2 + k;
      const int cs = static_cast<int>(row - j);
      float* dst_row = dst + k * w;
      EIGEN_IF_CONSTEXPR (ColM) {
        const float* head = base + row + j * stride;  // m(row, j+c): stride-strided
        for (int c = 0; c < cs; ++c, head += stride) dst_row[c] = *head;
        const float* tail = base + j + row * stride;  // m(j+c, row): contiguous
        for (int c = cs; c < w; c += svlw) {
          const svbool_t pred = svwhilelt_b32(uint32_t(c), uint32_t(w));
          svst1_f32(pred, dst_row + c, svld1_f32(pred, tail + c));
        }
      } else {
        const float* head = base + row * stride + j;  // m(row, j+c): contiguous
        for (int c = 0; c < cs; c += svlw) {
          const svbool_t pred = svwhilelt_b32(uint32_t(c), uint32_t(cs));
          svst1_f32(pred, dst_row + c, svld1_f32(pred, head + c));
        }
        const float* tail = base + (j + Index(cs)) * stride + row;  // m(j+c, row): stride-strided
        for (int c = cs; c < w; ++c, tail += stride) dst_row[c] = *tail;
      }
    }
  }
}

// symm_pack_lhs/rhs SME specializations: emit the uniform kSmeMr/kSmeNr panels
// sme_gebp_impl reads. Pack1/nr pinned exactly as gemm_pack_lhs/rhs above.
template <typename Index, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<float, Index, kSmeMr, Pack2_dummy, StorageOrder> {
  typedef float Scalar;
  // Note: generic symm_pack_lhs's "cols" is the depth extent, and the LHS
  // block is diagonal-anchored (base = &lhs(k2,k2)), so its depth offset is 0.
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const Scalar* lhs_, Index lhsStride, Index cols, Index rows) const {
    sme_symm_pack_panels<StorageOrder, Index>(blockA, lhs_, lhsStride, cols, rows, Index(0));
  }
};

template <typename Index, int StorageOrder>
struct symm_pack_rhs<float, Index, kSmeNr, StorageOrder> {
  typedef float Scalar;
  // Note: generic symm_pack_rhs's "rows" is the depth extent (end_k = k2 + rows), not a row count.
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const Scalar* rhs_, Index rhsStride, Index rows, Index cols,
                                    Index k2) const {
    sme_symm_pack_panels<StorageOrder, Index>(blockB, rhs_, rhsStride, rows, cols, k2);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SME_GENERALBLOCKPANELKERNEL_H
