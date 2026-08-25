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
// The micro-kernel is organised around a logical mr x nr output block, packed
// depth-major (mr contiguous scalars per depth step).  Those dimensions are
// compile-time constants: they feed gebp_traits (cache blocking) and the
// packers, none of which can depend on a runtime value.
//
// The *physical* tiling of that block onto ZA tiles, on the other hand, is
// driven by the runtime streaming vector length.  A ZA tile of Scalar is
// svl x svl, where svl is the number of Scalars in a streaming vector
// (svcntsw() for fp32, svcntsd() for fp64).  The block is covered by up to a
// 2x2 grid of svl x svl tiles, iterated in sub-block passes when the grid is
// smaller than the block (and predicated down to it when larger).
//
// fp32 uses the 4 ZA.S tiles, so the 2x2 grid is all of ZA.  fp64 uses ZA.D, of
// which there are 8, and deliberately leaves tiles 4-7 idle: a 2x2 grid loads 2
// packed vectors per side per depth step to feed 4 FMOPAs, i.e. 64 bytes of
// packed panel per FMOPA at either element width, and FMOPA issues at the same
// rate for both.  A 2x4 grid over all eight needs a quarter less panel traffic
// per FMOPA and still measures 0.92-1.00x of the 2x2 on Apple M4, so the wider
// block is not worth its L1 footprint.
//
// This translation unit must be built without -msve-vector-bits (scalable/VLA
// mode); see the guard in ConfigureVectorization.h for the rationale.
// Everything below derives lane counts/predicates from the runtime svl; when a
// block matches the tile grid exactly, the micro-kernel additionally switches
// to a hand-scheduled multi-vector-load loop (see sme_process).
// ---------------------------------------------------------------------------

// The per-element-width half of the ACLE surface: everything the kernel needs
// that is selected by the element type alone rather than by an argument.
// Operations that can be overloaded on their arguments are free functions
// below, and the ones taking a ZA tile number take it as a template parameter
// because the underlying instructions encode it as an immediate.
template <typename Scalar>
struct sme_traits;

template <>
struct sme_traits<float> {
  using Vec = svfloat32_t;
  using Vec2 = svfloat32x2_t;
  using Vec4 = svfloat32x4_t;
  // ZA.S tiles.
  static constexpr int kNumTiles = 4;
  static EIGEN_ALWAYS_INLINE int svl() __arm_streaming_compatible { return static_cast<int>(svcntsw()); }
  template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
  static EIGEN_ALWAYS_INLINE svbool_t whilelt(T begin, T end) __arm_streaming {
    return svwhilelt_b32(begin, end);
  }
  static EIGEN_ALWAYS_INLINE svbool_t ptrue() __arm_streaming { return svptrue_b32(); }
  static EIGEN_ALWAYS_INLINE svcount_t ptrue_c() __arm_streaming { return svptrue_c32(); }
  static EIGEN_ALWAYS_INLINE Vec dup(float x) __arm_streaming { return svdup_f32(x); }
};

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <>
struct sme_traits<double> {
  using Vec = svfloat64_t;
  using Vec2 = svfloat64x2_t;
  using Vec4 = svfloat64x4_t;
  // ZA.D tiles.
  static constexpr int kNumTiles = 8;
  static EIGEN_ALWAYS_INLINE int svl() __arm_streaming_compatible { return static_cast<int>(svcntsd()); }
  template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
  static EIGEN_ALWAYS_INLINE svbool_t whilelt(T begin, T end) __arm_streaming {
    return svwhilelt_b64(begin, end);
  }
  static EIGEN_ALWAYS_INLINE svbool_t ptrue() __arm_streaming { return svptrue_b64(); }
  static EIGEN_ALWAYS_INLINE svcount_t ptrue_c() __arm_streaming { return svptrue_c64(); }
  static EIGEN_ALWAYS_INLINE Vec dup(double x) __arm_streaming { return svdup_f64(x); }
};
#endif

// Contiguous predicated load/store, fused multiply-add and multiply.
static EIGEN_ALWAYS_INLINE svfloat32_t sme_ld1(svbool_t pg, const float* p) __arm_streaming { return svld1_f32(pg, p); }
static EIGEN_ALWAYS_INLINE void sme_st1(svbool_t pg, float* p, svfloat32_t v) __arm_streaming { svst1_f32(pg, p, v); }
static EIGEN_ALWAYS_INLINE svfloat32x2_t sme_ld1_x2(svcount_t pn, const float* p) __arm_streaming {
  return svld1_f32_x2(pn, p);
}
static EIGEN_ALWAYS_INLINE svfloat32x4_t sme_ld1_x4(svcount_t pn, const float* p) __arm_streaming {
  return svld1_f32_x4(pn, p);
}
static EIGEN_ALWAYS_INLINE svfloat32_t sme_mla(svbool_t pg, svfloat32_t acc, svfloat32_t a,
                                               svfloat32_t b) __arm_streaming {
  return svmla_f32_x(pg, acc, a, b);
}
static EIGEN_ALWAYS_INLINE svfloat32_t sme_mul(svbool_t pg, svfloat32_t a, svfloat32_t b) __arm_streaming {
  return svmul_f32_x(pg, a, b);
}
template <int Lane>
static EIGEN_ALWAYS_INLINE svfloat32_t sme_get(svfloat32x2_t v) __arm_streaming {
  return svget2_f32(v, Lane);
}
template <int Lane>
static EIGEN_ALWAYS_INLINE svfloat32_t sme_get(svfloat32x4_t v) __arm_streaming {
  return svget4_f32(v, Lane);
}

// ZA tile access. The tile number is an instruction immediate, hence a template
// parameter; the slice number is a register operand and stays a value.
template <int Tile>
static EIGEN_ALWAYS_INLINE void sme_ld1_hor_za(uint32_t slice, svbool_t pg,
                                               const float* p) __arm_streaming __arm_inout("za") {
  svld1_hor_za32(Tile, slice, pg, p);
}
template <int Tile>
static EIGEN_ALWAYS_INLINE svfloat32_t sme_read_hor_za(svfloat32_t zero, svbool_t pg,
                                                       uint32_t slice) __arm_streaming __arm_inout("za") {
  return svread_hor_za32_f32_m(zero, pg, Tile, slice);
}
template <int Tile>
static EIGEN_ALWAYS_INLINE svfloat32_t sme_read_ver_za(svfloat32_t zero, svbool_t pg,
                                                       uint32_t slice) __arm_streaming __arm_inout("za") {
  return svread_ver_za32_f32_m(zero, pg, Tile, slice);
}
template <int Tile>
static EIGEN_ALWAYS_INLINE void sme_mopa(svbool_t pm, svbool_t pn, svfloat32_t a,
                                         svfloat32_t b) __arm_streaming __arm_inout("za") {
  svmopa_za32_f32_m(Tile, pm, pn, a, b);
}

#ifdef EIGEN_VECTORIZE_SME_F64F64
static EIGEN_ALWAYS_INLINE svfloat64_t sme_ld1(svbool_t pg, const double* p) __arm_streaming {
  return svld1_f64(pg, p);
}
static EIGEN_ALWAYS_INLINE void sme_st1(svbool_t pg, double* p, svfloat64_t v) __arm_streaming { svst1_f64(pg, p, v); }
static EIGEN_ALWAYS_INLINE svfloat64x2_t sme_ld1_x2(svcount_t pn, const double* p) __arm_streaming {
  return svld1_f64_x2(pn, p);
}
static EIGEN_ALWAYS_INLINE svfloat64x4_t sme_ld1_x4(svcount_t pn, const double* p) __arm_streaming {
  return svld1_f64_x4(pn, p);
}
static EIGEN_ALWAYS_INLINE svfloat64_t sme_mla(svbool_t pg, svfloat64_t acc, svfloat64_t a,
                                               svfloat64_t b) __arm_streaming {
  return svmla_f64_x(pg, acc, a, b);
}
static EIGEN_ALWAYS_INLINE svfloat64_t sme_mul(svbool_t pg, svfloat64_t a, svfloat64_t b) __arm_streaming {
  return svmul_f64_x(pg, a, b);
}
template <int Lane>
static EIGEN_ALWAYS_INLINE svfloat64_t sme_get(svfloat64x2_t v) __arm_streaming {
  return svget2_f64(v, Lane);
}
template <int Lane>
static EIGEN_ALWAYS_INLINE svfloat64_t sme_get(svfloat64x4_t v) __arm_streaming {
  return svget4_f64(v, Lane);
}

template <int Tile>
static EIGEN_ALWAYS_INLINE void sme_ld1_hor_za(uint32_t slice, svbool_t pg,
                                               const double* p) __arm_streaming __arm_inout("za") {
  svld1_hor_za64(Tile, slice, pg, p);
}
template <int Tile>
static EIGEN_ALWAYS_INLINE svfloat64_t sme_read_hor_za(svfloat64_t zero, svbool_t pg,
                                                       uint32_t slice) __arm_streaming __arm_inout("za") {
  return svread_hor_za64_f64_m(zero, pg, Tile, slice);
}
template <int Tile>
static EIGEN_ALWAYS_INLINE svfloat64_t sme_read_ver_za(svfloat64_t zero, svbool_t pg,
                                                       uint32_t slice) __arm_streaming __arm_inout("za") {
  return svread_ver_za64_f64_m(zero, pg, Tile, slice);
}
template <int Tile>
static EIGEN_ALWAYS_INLINE void sme_mopa(svbool_t pm, svbool_t pn, svfloat64_t a,
                                         svfloat64_t b) __arm_streaming __arm_inout("za") {
  svmopa_za64_f64_m(Tile, pm, pn, a, b);
}
#endif  // EIGEN_VECTORIZE_SME_F64F64

// Logical micro-kernel block (LHS/RHS panel widths): a full 2x2 ZA-tile grid at
// the 512-bit design point, where a streaming vector holds 64 / sizeof(Scalar)
// scalars.  Other SVLs tile the block at runtime.  If a future SVL ever
// justifies a larger block, this is the only knob -- but don't grow it
// speculatively, a doubled block measures slower at SVL=512.
static constexpr int kSmeDesignVectorBytes = 64;

template <typename Scalar>
struct sme_block {
  static constexpr int mr = 2 * kSmeDesignVectorBytes / int(sizeof(Scalar));
  static constexpr int nr = mr;
};

static constexpr int kSmeMr = sme_block<float>::mr;
static constexpr int kSmeNr = sme_block<float>::nr;
#ifdef EIGEN_VECTORIZE_SME_F64F64
static constexpr int kSmeMrD = sme_block<double>::mr;
static constexpr int kSmeNrD = sme_block<double>::nr;
#endif

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
// and `dst` to the panel base.  Generalised over the runtime svl: the panel is
// covered in svl-wide column chunks, each streamed over the depth sub-range.
// The chunk loop is outermost so each chunk's predicate is computed once instead
// of per depth step (the runtime chunk count keeps the compiler from hoisting it
// on its own).  The symm packers reuse this for the diagonal-split direct/
// transposed regions (a contiguous depth sub-range at a depth offset).
template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel_range(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                                     Index src_stride, Index k0, Index k1, int width) __arm_streaming {
  const int svl = sme_traits<Scalar>::svl();
  for (int off = 0; off < width; off += svl) {
    const svbool_t pred = sme_traits<Scalar>::whilelt(off, width);
    for (Index k = k0; k < k1; ++k) {
      sme_st1(pred, &dst[k * width + off], sme_ld1(pred, &src[k * src_stride + off]));
    }
  }
}

// Copy the full depth [0, depth): thin wrapper used by the (non-symm) gemm
// packers, which always pack a whole panel.
template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                               Index src_stride, Index depth, int width) __arm_streaming {
  sve_copy_panel_range(dst, src, src_stride, Index(0), depth, width);
}

// Transpose-pack `width` source rows into depth-major packed output using ZA's
// 2D store as a free transpose, for the depth sub-range [k0, k1): a svl x svl
// block of source (svl rows x svl depth) is loaded as horizontal ZA slices,
// then read back as vertical slices, which emits it depth-major.  Row-groups of
// svl rows are processed two at a time through ZA tiles 0 and 1: ZA is not
// renamed, so a single tile would stall every load pass on the previous read
// pass (write-after-read); two tiles in flight keep the phases independent.
// Trailing row-groups (when width is not a multiple of 2*svl) use tile 0 with
// predicated rows.  Both dst and src are indexed by the absolute depth index k:
//   dst[k*width + r] = src[r*src_stride + k],  k in [k0,k1), r in [0,width).
// The symm packers reuse this for the diagonal-split transposed/direct regions
// (a depth sub-range at a depth offset, with a tail-panel width < mr).
template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack_range(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                                         Index src_stride, Index k0, Index k1,
                                                         int width) __arm_streaming __arm_inout("za") {
  using Traits = sme_traits<Scalar>;
  const typename Traits::Vec zero = Traits::dup(Scalar(0));
  const svbool_t pg_all = Traits::ptrue();
  const int svl = Traits::svl();

  for (Index k = k0; k < k1; k += svl) {
    const int dk = static_cast<int>(sme_min(k1 - k, Index(svl)));
    const svbool_t pg_d = Traits::whilelt(k, k1);
    int r0 = 0;
    // Pairs of full row-groups: tiles 0 and 1 in flight.
    for (; r0 + 2 * svl <= width; r0 += 2 * svl) {
      for (int r = 0; r < svl; ++r) {
        sme_ld1_hor_za<0>(uint32_t(r), pg_d, &src[(r0 + r) * src_stride + k]);
        sme_ld1_hor_za<1>(uint32_t(r), pg_d, &src[(r0 + svl + r) * src_stride + k]);
      }
      for (int c = 0; c < dk; ++c) {
        sme_st1(pg_all, &dst[(k + c) * width + r0], sme_read_ver_za<0>(zero, pg_all, uint32_t(c)));
        sme_st1(pg_all, &dst[(k + c) * width + r0 + svl], sme_read_ver_za<1>(zero, pg_all, uint32_t(c)));
      }
    }
    // Trailing row-groups (at most two svl-wide passes remain, since the pair
    // loop consumed all multiples of 2*svl): predicate down to the remaining
    // rows.  A single `if` would drop rows when a tail width lands in
    // (svl, 2*svl); a loop handles any leftover.
    for (; r0 < width; r0 += svl) {
      const int rg = sme_min(width - r0, svl);
      const svbool_t pg_r = Traits::whilelt(r0, width);
      for (int r = 0; r < rg; ++r) {
        sme_ld1_hor_za<0>(uint32_t(r), pg_d, &src[(r0 + r) * src_stride + k]);
      }
      for (int c = 0; c < dk; ++c) {
        sme_st1(pg_r, &dst[(k + c) * width + r0], sme_read_ver_za<0>(zero, pg_r, uint32_t(c)));
      }
    }
  }
}

// Transpose-pack a whole `width`-wide panel over the full depth [0, depth):
// thin wrapper used by the (non-symm) gemm packers.
template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                                   Index src_stride, Index depth,
                                                   int width) __arm_streaming __arm_inout("za") {
  sme_transpose_pack_range(dst, src, src_stride, Index(0), depth, width);
}

// Transposing copy for a panel narrower than the pack width:
//   dst_panel[k*tail + i] = src[i*src_stride + k].
// Kept outside the caller's __arm_locally_streaming region: it needs neither SVE
// nor ZA, and streaming mode runs scalar floating-point ~40x slower on Apple M4.
// Outside it the source rows are contiguous in k, so PacketSize of them
// transpose in register as in sme_pack_rhs_fallback; a product with cols < nr is
// packed entirely here.
template <typename Scalar, typename Index>
static void tail_transpose_pack(Scalar* EIGEN_RESTRICT dst_panel, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                Index depth, Index tail) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int PacketSize = int(packet_traits<Scalar>::size);
  const Index peeled_tail = (tail / Index(PacketSize)) * Index(PacketSize);
  const Index peeled_depth = (depth / Index(PacketSize)) * Index(PacketSize);

  Index i = 0;
  for (; i < peeled_tail; i += Index(PacketSize)) {
    Index k = 0;
    for (; k < peeled_depth; k += Index(PacketSize)) {
      PacketBlock<Packet, PacketSize> block;
      for (int p = 0; p < PacketSize; ++p) {
        block.packet[p] = ploadu<Packet>(src + (i + Index(p)) * src_stride + k);
      }
      ptranspose(block);
      for (int p = 0; p < PacketSize; ++p) {
        pstoreu(dst_panel + (k + Index(p)) * tail + i, block.packet[p]);
      }
    }
    for (; k < depth; ++k) {
      for (Index p = 0; p < Index(PacketSize); ++p) dst_panel[k * tail + i + p] = src[(i + p) * src_stride + k];
    }
  }
  for (; i < tail; ++i) {
    for (Index k = 0; k < depth; ++k) dst_panel[k * tail + i] = src[i * src_stride + k];
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
template <typename Scalar, int MR, typename Index, typename DataMapper, bool PanelMode>
void sme_pack_lhs_fallback(Scalar* dst_base, const DataMapper& lhs, Index depth, Index rows, Index dst_stride,
                           Index dst_offset, bool vectorise) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr Index PacketSize = Index(packet_traits<Scalar>::size);

  for (Index i = 0; i < rows; i += MR) {
    const Index w = numext::mini(rows - i, Index(MR));
    Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * w : dst_base + i * depth;
    const Index peeled_w = vectorise ? (w / PacketSize) * PacketSize : Index(0);
    for (Index k = 0; k < depth; ++k) {
      Scalar* dst_row = dst_panel + k * w;
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

// The PacketSize column sub-mappers one packed column group loads from.
template <typename DataMapper, typename Index, std::size_t... Is>
EIGEN_ALWAYS_INLINE std::array<typename DataMapper::LinearMapper, sizeof...(Is)> sme_column_mappers(
    const DataMapper& rhs, Index col, std::index_sequence<Is...>) {
  return {{rhs.getLinearMapper(0, col + Index(Is))...}};
}

// RHS fallback, mirroring sme_pack_lhs_fallback (including the vectorise
// contract: LinearMapper packets must advance the first (depth) index). The
// packed layout wants consecutive columns contiguous while the mapper's
// packets run along the depth k, so PacketSize columns are loaded as packets
// along k and transposed in-register (the same LinearMapper + ptranspose
// scheme as the generic gemm_pack_rhs).
template <typename Scalar, int NR, typename Index, typename DataMapper, bool PanelMode>
void sme_pack_rhs_fallback(Scalar* dst_base, const DataMapper& rhs, Index depth, Index cols, Index dst_stride,
                           Index dst_offset, bool vectorise) {
  using Packet = typename packet_traits<Scalar>::type;
  using LinearMapper = typename DataMapper::LinearMapper;
  constexpr int PacketSize = int(packet_traits<Scalar>::size);
  const Index peeled_depth = (depth / Index(PacketSize)) * Index(PacketSize);

  for (Index j = 0; j < cols; j += NR) {
    const Index w = numext::mini(cols - j, Index(NR));
    Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * w : dst_base + j * depth;
    const Index peeled_w = vectorise ? (w / Index(PacketSize)) * Index(PacketSize) : Index(0);
    Index c = 0;
    for (; c < peeled_w; c += Index(PacketSize)) {
      // Loop-invariant in k, but not hoisted out of the k loop by the compiler
      // for a mapper that returns its sub-mappers by value -- which is the hot
      // path here: tensor contractions pack through TensorContractionSubMapper.
      const std::array<LinearMapper, PacketSize> dm =
          sme_column_mappers(rhs, j + c, std::make_index_sequence<PacketSize>{});
      Index k = 0;
      for (; k < peeled_depth; k += Index(PacketSize)) {
        PacketBlock<Packet, PacketSize> block;
        for (int p = 0; p < PacketSize; ++p) {
          block.packet[p] = dm[p].template loadPacket<Packet>(k);
        }
        ptranspose(block);
        for (int p = 0; p < PacketSize; ++p) {
          pstoreu(dst_panel + (k + Index(p)) * w + c, block.packet[p]);
        }
      }
      for (; k < depth; ++k) {
        for (Index p = 0; p < Index(PacketSize); ++p) {
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
template <bool UsePacketPath, typename Scalar, typename Index, typename DataMapper, typename DirectFn,
          typename FallbackFn>
EIGEN_ALWAYS_INLINE void sme_dispatch_pack(DirectFn direct, FallbackFn fallback, Scalar* block, const DataMapper& m,
                                           Index depth, Index n, Index stride, Index offset,
                                           std::true_type /* direct access */) {
  if (sme_mapper_incr<Index>(m) == 1) {
    const Scalar* src = (n > 0 && depth > 0) ? &m(0, 0) : nullptr;
    direct(block, src, m.stride(), depth, n, stride, offset);
  } else {
    fallback(block, m, depth, n, stride, offset, UsePacketPath);
  }
}
template <bool UsePacketPath, typename Scalar, typename Index, typename DataMapper, typename DirectFn,
          typename FallbackFn>
EIGEN_ALWAYS_INLINE void sme_dispatch_pack(DirectFn, FallbackFn fallback, Scalar* block, const DataMapper& m,
                                           Index depth, Index n, Index stride, Index offset,
                                           std::false_type /* no direct access */) {
  fallback(block, m, depth, n, stride, offset, UsePacketPath);
}

/*****************************************************************************
 * gebp_traits specializations for SME  (float x float, double x double)
 *
 * Override mr and nr so that:
 *   - gemm_pack_lhs receives Pack1 = mr, creating uniform LHS panels
 *   - gemm_pack_rhs receives nr, creating uniform RHS panels
 *   - mc is rounded to a multiple of mr, nc to a multiple of nr
 *   - Cache blocking (kc, mc, nc) is recomputed accordingly
 *
 * We provide custom gemm_pack_lhs/gemm_pack_rhs specializations for both
 * scalars, so both ColMajor and RowMajor source matrices produce an identical,
 * simple packed format that the SME kernel consumes.
 *
 * Mixed-scalar products (e.g. MatrixXf * MatrixXcf) also instantiate
 * gemm_pack_lhs<float, ...>, but with Pack1/nr from the generic
 * gebp_traits<float, complex<float>> (mr=6, nr=4) and are consumed by the
 * generic gebp_kernel, not the SME one. So the specializations below pin
 * Pack1/nr_ to the SME block sizes: only the instantiation that feeds the SME
 * gebp_kernel matches; mixed-scalar ones fall through to the generic template.
 * This is load-bearing: it relies on no other consumer of the same scalar
 * instantiating the packer with mr == the SME block size (holds today --
 * generic float traits give mr <= 12). The kernel side is self-checking (the
 * SME gebp_kernel static_asserts mr/nr against the block sizes, so a traits
 * change breaks the build instead of silently mispairing packer and kernel);
 * the packer side is enforced by the static_asserts below for the in-tree
 * mixed-scalar traits (downstream code instantiating the packers with
 * hand-picked mr/nr remains uncovered).
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

// The packers do not know the opposite scalar type, so the SME block sizes are
// effectively SME-format tags. Ensure the in-tree mixed-scalar traits cannot
// select an SME packer whose output would be consumed by the generic kernel.
static_assert(int(gebp_traits<float, std::complex<float>>::mr) != kSmeMr,
              "gebp_traits<float, complex<float>>::mr collides with kSmeMr: the SME gemm_pack_lhs would silently "
              "emit SME panel layout for the generic gebp_kernel");
static_assert(int(gebp_traits<std::complex<float>, float>::nr) != kSmeNr,
              "gebp_traits<complex<float>, float>::nr collides with kSmeNr: the SME gemm_pack_rhs would silently "
              "emit SME panel layout for the generic gebp_kernel");

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <>
class gebp_traits<double, double, false, false, Architecture::Target, GEBPPacketFull>
    : public gebp_traits<double, double, false, false, Architecture::Target, GEBPPacketHalf> {
 public:
  // As above, only the register-block sizes are overridden.
  static constexpr int mr = kSmeMrD;
  static constexpr int nr = kSmeNrD;
};

static_assert(int(gebp_traits<double, std::complex<double>>::mr) != kSmeMrD,
              "gebp_traits<double, complex<double>>::mr collides with kSmeMrD: the SME gemm_pack_lhs would silently "
              "emit SME panel layout for the generic gebp_kernel");
static_assert(int(gebp_traits<std::complex<double>, double>::nr) != kSmeNrD,
              "gebp_traits<complex<double>, double>::nr collides with kSmeNrD: the SME gemm_pack_rhs would silently "
              "emit SME panel layout for the generic gebp_kernel");
#endif

/*****************************************************************************
 * gemm_pack_lhs for SME  (ColMajor source)
 *
 * Packs the LHS matrix into uniform panels of width mr.
 * Each depth step k writes exactly MR contiguous scalars.
 *****************************************************************************/

template <typename Scalar, int MR, typename Index, typename DataMapper, bool PanelMode>
struct sme_pack_lhs_colmajor {
  // Conjugate is deliberately ignored by the specializations below: conj is the
  // identity for real scalars, and Conjugate=true instantiations do occur (e.g.
  // the SYMM above-diagonal transposed pack). A complex port of these packers
  // must actually conjugate.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                                  Index depth, Index rows, Index dst_stride, Index dst_offset) {
    const Index peeled_rows = (rows / MR) * MR;

    // Full panels of width MR, streamed in svl-wide predicated chunks.
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
    sme_dispatch_pack<true>(&pack_direct, &sme_pack_lhs_fallback<Scalar, MR, Index, DataMapper, PanelMode>, blockA, lhs,
                            depth, rows, stride, offset,
                            bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

// RowMajor LHS packer -- SME in-ZA transpose.
//
// The packed output wants depth-major layout (MR rows contiguous per depth
// step) but the RowMajor source has rows contiguous (strided by depth per
// row).  A natural SVE gather would be slow; instead we use ZA's 2D store
// as a free transpose: load svl rows as horizontal slices of a ZA tile,
// then read vertical slices to produce depth-major output (see
// sme_transpose_pack).
template <typename Scalar, int MR, typename Index, typename DataMapper, bool PanelMode>
struct sme_pack_lhs_rowmajor {
  // See sme_pack_lhs_colmajor: Conjugate is ignored, sound only for real scalars.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming __arm_new("za") static void pack_full_panels(Scalar* dst_base,
                                                                       const Scalar* EIGEN_RESTRICT src,
                                                                       Index src_stride, Index depth, Index peeled_rows,
                                                                       Index dst_stride, Index dst_offset) {
    for (Index i = 0; i < peeled_rows; i += MR) {
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * MR : dst_base + i * depth;
      sme_transpose_pack(dst_panel, src + i * src_stride, src_stride, depth, MR);
    }
  }

  static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride, Index depth, Index rows,
                          Index dst_stride, Index dst_offset) {
    const Index peeled_rows = (rows / MR) * MR;

    if (peeled_rows > 0) {
      pack_full_panels(dst_base, src, src_stride, depth, peeled_rows, dst_stride, dst_offset);
    }

    // Row tail (rows - peeled_rows in [1, MR-1]).  This branch runs at most
    // once per pack_lhs call with < MR rows and would need a partial-ZA-tile
    // dance to vectorise; total copies are < MR * depth per call, which is
    // noise vs the main packer's workload, so scalar is the simple choice --
    // taken outside the streaming region above (see tail_transpose_pack).
    if (peeled_rows < rows) {
      const Index tail = rows - peeled_rows;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_rows * dst_stride + dst_offset * tail : dst_base + peeled_rows * depth;
      tail_transpose_pack(dst_panel, src + peeled_rows * src_stride, src_stride, depth, tail);
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
    sme_dispatch_pack<false>(&pack_direct, &sme_pack_lhs_fallback<Scalar, MR, Index, DataMapper, PanelMode>, blockA,
                             lhs, depth, rows, stride, offset,
                             bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

/*****************************************************************************
 * gemm_pack_rhs for SME  (ColMajor source) -- SME in-ZA transpose, mirroring
 * the RowMajor LHS packer.
 *
 * Packs the RHS matrix into panels of width nr.  ColMajor source has
 * columns contiguous; we load NR columns as horizontal ZA slices and then
 * read verticals to produce depth-major packed output.
 *****************************************************************************/

template <typename Scalar, int NR, typename Index, typename DataMapper, bool PanelMode>
struct sme_pack_rhs_colmajor {
  // See sme_pack_lhs_colmajor: Conjugate is ignored, sound only for real scalars.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming __arm_new("za") static void pack_full_panels(Scalar* dst_base,
                                                                       const Scalar* EIGEN_RESTRICT src,
                                                                       Index src_stride, Index depth, Index peeled_cols,
                                                                       Index dst_stride, Index dst_offset) {
    for (Index j = 0; j < peeled_cols; j += NR) {
      Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * NR : dst_base + j * depth;
      sme_transpose_pack(dst_panel, src + j * src_stride, src_stride, depth, NR);
    }
  }

  static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride, Index depth, Index cols,
                          Index dst_stride, Index dst_offset) {
    const Index peeled_cols = (cols / NR) * NR;

    if (peeled_cols > 0) {
      pack_full_panels(dst_base, src, src_stride, depth, peeled_cols, dst_stride, dst_offset);
    }

    // Col tail (cols - peeled_cols in [1, NR-1]).  Same reasoning as the LHS
    // RowMajor packer's row tail: runs at most once per call, < NR cols, not
    // worth the partial-ZA-tile handling, and taken outside the streaming
    // region above (see tail_transpose_pack).
    if (peeled_cols < cols) {
      const Index tail = cols - peeled_cols;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_cols * dst_stride + dst_offset * tail : dst_base + peeled_cols * depth;
      tail_transpose_pack(dst_panel, src + peeled_cols * src_stride, src_stride, depth, tail);
    }
  }

  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) {
    if (PanelMode) {
      eigen_assert(stride >= depth && offset <= stride);
    }
    // Inner-strided ColMajor blas mappers' LinearMapper packets advance the
    // depth index, which is what the fallback transposes.
    sme_dispatch_pack<true>(&pack_direct, &sme_pack_rhs_fallback<Scalar, NR, Index, DataMapper, PanelMode>, blockB, rhs,
                            depth, cols, stride, offset,
                            bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

// RowMajor RHS packer -- streaming SVE copy (mirrors the ColMajor LHS packer).
// Rows are contiguous in the source, so each depth-step is NR contiguous scalars.
template <typename Scalar, int NR, typename Index, typename DataMapper, bool PanelMode>
struct sme_pack_rhs_rowmajor {
  // See sme_pack_lhs_colmajor: Conjugate is ignored, sound only for real scalars.
  static_assert(!NumTraits<Scalar>::IsComplex, "the SME packers only support real scalars");

  __arm_locally_streaming static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                                  Index depth, Index cols, Index dst_stride, Index dst_offset) {
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
    sme_dispatch_pack<false>(&pack_direct, &sme_pack_rhs_fallback<Scalar, NR, Index, DataMapper, PanelMode>, blockB,
                             rhs, depth, cols, stride, offset,
                             bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

// Pack1/nr_ are pinned to the SME block sizes (rather than left open) so these
// specializations only match consumers that actually feed the SME gebp_kernel
// -- see "Mixed-scalar products" in the gebp_traits doc comment above.
template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, kSmeMr, Pack2, Packet, ColMajor, Conjugate, PanelMode>
    : sme_pack_lhs_colmajor<float, kSmeMr, Index, DataMapper, PanelMode> {};

template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, kSmeMr, Pack2, Packet, RowMajor, Conjugate, PanelMode>
    : sme_pack_lhs_rowmajor<float, kSmeMr, Index, DataMapper, PanelMode> {};

template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, kSmeNr, ColMajor, Conjugate, PanelMode>
    : sme_pack_rhs_colmajor<float, kSmeNr, Index, DataMapper, PanelMode> {};

template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, kSmeNr, RowMajor, Conjugate, PanelMode>
    : sme_pack_rhs_rowmajor<float, kSmeNr, Index, DataMapper, PanelMode> {};

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, kSmeMrD, Pack2, Packet, ColMajor, Conjugate, PanelMode>
    : sme_pack_lhs_colmajor<double, kSmeMrD, Index, DataMapper, PanelMode> {};

template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, kSmeMrD, Pack2, Packet, RowMajor, Conjugate, PanelMode>
    : sme_pack_lhs_rowmajor<double, kSmeMrD, Index, DataMapper, PanelMode> {};

template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<double, Index, DataMapper, kSmeNrD, ColMajor, Conjugate, PanelMode>
    : sme_pack_rhs_colmajor<double, kSmeNrD, Index, DataMapper, PanelMode> {};

template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<double, Index, DataMapper, kSmeNrD, RowMajor, Conjugate, PanelMode>
    : sme_pack_rhs_rowmajor<double, kSmeNrD, Index, DataMapper, PanelMode> {};
#endif

/*****************************************************************************
 * sme_store_za_tile -- Store one ZA tile back to C with alpha scaling.
 *
 * `pw` is the row-predicate width for this tile, `cw` the col-predicate width
 * (both <= the runtime svl).
 *****************************************************************************/

template <typename Scalar, int TileId, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_za_tile(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                           Scalar alpha, Index row_start, int pw, Index col_start,
                                           int cw) __arm_streaming __arm_inout("za") {
  using Traits = sme_traits<Scalar>;
  using Vec = typename Traits::Vec;
  const svbool_t pg_m = Traits::whilelt(0, pw);
  const svbool_t pg_n = Traits::whilelt(0, cw);
  // FMLA and FADD have equal latency/throughput on ARMv9 cores, and
  // multiplying by alpha=1.0 is exact in IEEE-754 so the FMLA form is
  // bit-identical to FADD in that case.  A single unconditional FMLA
  // keeps the store compact and measures no worse (and a few percent
  // better on small matrices, where the branch would otherwise disrupt
  // instruction scheduling).
  const Vec vzero = Traits::dup(Scalar(0));
  const Vec valpha = Traits::dup(alpha);

  // Two C slices are loaded before either is stored: a C line the caller wrote
  // from non-streaming code just before the kernel does not forward across the
  // mode switch on Apple M4, and a serial load/store pays that latency per slice.
  // C = A*B meets the condition on every call, since evalTo zeroes the
  // destination first. SVE vectors are sizeless, hence the spelled-out pair.
  if (C_stride_row == 1) {
    // Column-major C: extract vertical slices (columns of the ZA tile)
    int ci = 0;
    for (; ci + 2 <= cw; ci += 2) {
      Scalar* p0 = C + row_start + (col_start + ci) * C_stride_col;
      Scalar* p1 = p0 + C_stride_col;
      Vec c0 = sme_ld1(pg_m, p0);
      Vec c1 = sme_ld1(pg_m, p1);
      sme_st1(pg_m, p0, sme_mla(pg_m, c0, sme_read_ver_za<TileId>(vzero, pg_m, (uint32_t)ci), valpha));
      sme_st1(pg_m, p1, sme_mla(pg_m, c1, sme_read_ver_za<TileId>(vzero, pg_m, (uint32_t)(ci + 1)), valpha));
    }
    if (ci < cw) {
      Scalar* pC = C + row_start + (col_start + ci) * C_stride_col;
      Vec vc = sme_ld1(pg_m, pC);
      sme_st1(pg_m, pC, sme_mla(pg_m, vc, sme_read_ver_za<TileId>(vzero, pg_m, (uint32_t)ci), valpha));
    }
  } else if (C_stride_col == 1) {
    // Row-major C: extract horizontal slices (rows of the ZA tile)
    int ri = 0;
    for (; ri + 2 <= pw; ri += 2) {
      Scalar* p0 = C + (row_start + ri) * C_stride_row + col_start;
      Scalar* p1 = p0 + C_stride_row;
      Vec c0 = sme_ld1(pg_n, p0);
      Vec c1 = sme_ld1(pg_n, p1);
      sme_st1(pg_n, p0, sme_mla(pg_n, c0, sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)ri), valpha));
      sme_st1(pg_n, p1, sme_mla(pg_n, c1, sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)(ri + 1)), valpha));
    }
    if (ri < pw) {
      Scalar* pC = C + (row_start + ri) * C_stride_row + col_start;
      Vec vc = sme_ld1(pg_n, pC);
      sme_st1(pg_n, pC, sme_mla(pg_n, vc, sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)ri), valpha));
    }
  } else {
    // General stride: extract rows to temp buffer, scatter to C.  scratch
    // holds one ZA row; every caller passes cw <= min(svl, nr) (a tile
    // never spans more than the logical block), so nr is a static
    // bound independent of the runtime svl.
    Scalar scratch[sme_block<Scalar>::nr];
    for (int ri = 0; ri < pw; ++ri) {
      Vec vres = sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)ri);
      vres = sme_mul(pg_n, vres, valpha);
      sme_st1(pg_n, scratch, vres);
      for (int ci = 0; ci < cw; ++ci) {
        C[(row_start + ri) * C_stride_row + (col_start + ci) * C_stride_col] += scratch[ci];
      }
    }
  }
}

/*****************************************************************************
 * sme_store_2x2_grid -- store the (up to) 2x2 grid of svl x svl ZA tiles.
 *
 * Tile layout:  0 = (row-lo, col-lo)  1 = (row-lo, col-hi)
 *               2 = (row-hi, col-lo)  3 = (row-hi, col-hi)
 * The col-hi tiles (1, 3) are stored only when chi > 0 and the row-hi tiles
 * (2, 3) only when rhi > 0, so a single tile, a 1x2/2x1 pair, or the full grid
 * all route through here.  Runs once per sub-block pass, after a depth loop
 * that dwarfs it, so the branches cost nothing and predict perfectly (the
 * pattern repeats across blocks).
 *****************************************************************************/

template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_2x2_grid(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                            Scalar alpha, Index row_start, int rlo, int rhi, Index col_start, int clo,
                                            int chi) __arm_streaming __arm_inout("za") {
  const int svl = sme_traits<Scalar>::svl();
  sme_store_za_tile<Scalar, 0>(C, C_stride_row, C_stride_col, alpha, row_start, rlo, col_start, clo);
  if (chi > 0) {
    sme_store_za_tile<Scalar, 1>(C, C_stride_row, C_stride_col, alpha, row_start, rlo, col_start + svl, chi);
  }
  if (rhi > 0) {
    sme_store_za_tile<Scalar, 2>(C, C_stride_row, C_stride_col, alpha, row_start + svl, rhi, col_start, clo);
    if (chi > 0) {
      sme_store_za_tile<Scalar, 3>(C, C_stride_row, C_stride_col, alpha, row_start + svl, rhi, col_start + svl, chi);
    }
  }
}

// One depth step's worth of the exact-match grid: the four FMOPAs that take the
// lo/hi halves of a packed A column and a packed B column and accumulate the
// 2x2 ZA-tile outer product.  `all` is the all-true predicate because this is
// only used on the exact-match path, where the block fills the grid, so
// factoring it out is identical to the inline form.
template <typename Scalar>
static EIGEN_ALWAYS_INLINE void outer_product_2x2(
    typename sme_traits<Scalar>::Vec a_lo, typename sme_traits<Scalar>::Vec a_hi, typename sme_traits<Scalar>::Vec b_lo,
    typename sme_traits<Scalar>::Vec b_hi) __arm_streaming __arm_inout("za") {
  const svbool_t all = sme_traits<Scalar>::ptrue();
  sme_mopa<0>(all, all, a_lo, b_lo);
  sme_mopa<1>(all, all, a_lo, b_hi);
  sme_mopa<2>(all, all, a_hi, b_lo);
  sme_mopa<3>(all, all, a_hi, b_hi);
}

/*****************************************************************************
 * sme_process -- micro-kernel for one pw x cw output block.
 *
 * Tiles the block into svl x svl ZA tiles, processed in passes of up to a 2x2
 * tile grid: several (2*svl) x (2*svl) sub-block passes when the grid is
 * smaller than the block, tiles predicated down to the block width when it is
 * larger.  blA/blB are packed depth-major with depth-strides pw and cw
 * respectively.
 *
 * When the block matches the tile grid exactly (pw == cw == 2 * svl), the
 * packed rows are also contiguous across depth steps, enabling the
 * hand-scheduled loop below: per 4 unrolled depth steps, 2 x4 loads per
 * side (each spanning 2 depth steps) feed 16 FMOPAs -- a 1:1 compute:load
 * ratio at the vector level.  All other geometries use predicated
 * per-depth-step loads.
 *****************************************************************************/

template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                     const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blB, Index depth,
                                     Scalar alpha, Index row_start, int pw, Index col_start,
                                     int cw) __arm_streaming __arm_inout("za") {
  using Traits = sme_traits<Scalar>;
  using Vec = typename Traits::Vec;
  const int svl = Traits::svl();

  for (int rt = 0; rt < pw; rt += 2 * svl) {
    const int rpw = sme_min(pw - rt, 2 * svl);
    const int rlo = sme_min(rpw, svl);
    const int rhi = rpw - rlo;  // >= 0; > 0 only when rpw > svl, in which case rlo == svl
    const svbool_t pg_rlo = Traits::whilelt(rt, pw);
    const svbool_t pg_rhi = Traits::whilelt(rt + svl, pw);

    for (int ct = 0; ct < cw; ct += 2 * svl) {
      const int cpw = sme_min(cw - ct, 2 * svl);
      const int clo = sme_min(cpw, svl);
      const int chi = cpw - clo;
      const svbool_t pg_clo = Traits::whilelt(ct, cw);
      const svbool_t pg_chi = Traits::whilelt(ct + svl, cw);

      svzero_za();
      if (pw == 2 * svl && cw == 2 * svl) {
        // The block is exactly one full-grid patch (single pass, rt == ct ==
        // 0, rlo == rhi == clo == chi == svl), so a packed row is the
        // patch's slice and rows are contiguous across depth steps: x4 loads
        // each span 2 of them, e.g. va_01 = [d0 lo, d0 hi, d1 lo, d1 hi].
        const svcount_t pn = Traits::ptrue_c();
        const Index depth_4 = (depth / 4) * 4;
        Index k = 0;
        for (; k < depth_4; k += 4) {
          typename Traits::Vec4 va_01 = sme_ld1_x4(pn, &blA[k * pw]);
          typename Traits::Vec4 vb_01 = sme_ld1_x4(pn, &blB[k * cw]);

          // d0
          outer_product_2x2<Scalar>(sme_get<0>(va_01), sme_get<1>(va_01), sme_get<0>(vb_01), sme_get<1>(vb_01));
          // d1
          outer_product_2x2<Scalar>(sme_get<2>(va_01), sme_get<3>(va_01), sme_get<2>(vb_01), sme_get<3>(vb_01));

          typename Traits::Vec4 va_23 = sme_ld1_x4(pn, &blA[(k + 2) * pw]);
          typename Traits::Vec4 vb_23 = sme_ld1_x4(pn, &blB[(k + 2) * cw]);

          // d2
          outer_product_2x2<Scalar>(sme_get<0>(va_23), sme_get<1>(va_23), sme_get<0>(vb_23), sme_get<1>(vb_23));
          // d3
          outer_product_2x2<Scalar>(sme_get<2>(va_23), sme_get<3>(va_23), sme_get<2>(vb_23), sme_get<3>(vb_23));
        }
        // Depth tail: one x2 load per side per step.
        for (; k < depth; ++k) {
          typename Traits::Vec2 va = sme_ld1_x2(pn, &blA[k * pw]);
          typename Traits::Vec2 vb = sme_ld1_x2(pn, &blB[k * cw]);
          outer_product_2x2<Scalar>(sme_get<0>(va), sme_get<1>(va), sme_get<0>(vb), sme_get<1>(vb));
        }
      } else {
        for (Index k = 0; k < depth; ++k) {
          Vec a_lo = sme_ld1(pg_rlo, &blA[k * pw + rt]);
          Vec b_lo = sme_ld1(pg_clo, &blB[k * cw + ct]);

          Vec a_hi =
              sme_ld1(pg_rhi, (const Scalar* EIGEN_RESTRICT)(uintptr_t(blA) + (k * pw + rt + svl) * sizeof(Scalar)));
          Vec b_hi =
              sme_ld1(pg_chi, (const Scalar* EIGEN_RESTRICT)(uintptr_t(blB) + (k * cw + ct + svl) * sizeof(Scalar)));

          sme_mopa<0>(pg_rlo, pg_clo, a_lo, b_lo);
          if (svptest_any(pg_chi, pg_chi))
            sme_mopa<1>(pg_rlo, pg_chi, a_lo, b_hi);
          if (svptest_any(pg_rhi, pg_rhi)) {
            sme_mopa<2>(pg_rhi, pg_clo, a_hi, b_lo);
            if (svptest_any(pg_chi, pg_chi))
              sme_mopa<3>(pg_rhi, pg_chi, a_hi, b_hi);
          }
        }
      }

      // Store the (up to) 2x2 grid of tiles for this sub-block pass.
      sme_store_2x2_grid(C, C_stride_row, C_stride_col, alpha, row_start + rt, rlo, rhi, col_start + ct, clo, chi);
    }
  }
}

template <typename Scalar, typename Index>
EIGEN_DONT_INLINE __arm_locally_streaming __arm_new("za") void sme_gebp_impl(
    Scalar* C, Index C_stride_row, Index C_stride_col, const Scalar* blockA, const Scalar* blockB, Index rows,
    Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  constexpr int MR = sme_block<Scalar>::mr;
  constexpr int NR = sme_block<Scalar>::nr;

  // Column-outer, row-inner: keeps blB (one kc × NR panel) hot in L1 while
  // smaller blA tiles stream from L2.  The outer GOTO loop in
  // GeneralMatrixMatrix.h ensures blockA fits in L2 via mc-blocking.  Each
  // packed panel is depth-major with depth-stride equal to its width (MR/NR
  // for full panels, the tail width otherwise), so that width is passed as
  // both the logical block size and the load stride to sme_process; partial
  // blocks are tiled and predicated inside the generic path.
  for (Index j = 0; j < cols; j += NR) {
    const int cw = static_cast<int>(sme_min(cols - j, Index(NR)));
    const Scalar* blB = blockB + j * strideB + offsetB * cw;

    for (Index i = 0; i < rows; i += MR) {
      const int pw = static_cast<int>(sme_min(rows - i, Index(MR)));
      const Scalar* blA = blockA + i * strideA + offsetA * pw;
      sme_process(C, C_stride_row, C_stride_col, blA, blB, depth, alpha, i, pw, j, cw);
    }
  }
}

template <typename Scalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct sme_gebp_kernel {
  using ResScalar = Scalar;

  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows,
                                    Index depth, Index cols, ResScalar alpha, Index strideA = -1, Index strideB = -1,
                                    Index offsetA = 0, Index offsetB = 0) {
    static_assert(!ConjugateLhs && !ConjugateRhs, "the SME kernel does not support conjugation");
    static_assert(mr == sme_block<Scalar>::mr && nr == sme_block<Scalar>::nr,
                  "the SME kernel expects packed panels of the SME block width");

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

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
    : sme_gebp_kernel<float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {};

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<double, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
    : sme_gebp_kernel<double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {};
#endif

// sme_has_gebp_kernel (products/GeneralBlockPanelKernel.h) drives the cache
// blocking and the GEMM loop order, and is declared before this header. A pair
// listed there but not specialized here would be packed and blocked for SME and
// then handed to the generic kernel.
static_assert(sme_has_gebp_kernel<float, float>::value, "the SME float kernel is not advertised to the GEMM driver");
#ifdef EIGEN_VECTORIZE_SME_F64F64
static_assert(sme_has_gebp_kernel<double, double>::value, "the SME double kernel is not advertised to the GEMM driver");
#else
static_assert(!sme_has_gebp_kernel<double, double>::value,
              "double is advertised to the GEMM driver without FEAT_SME_F64F64 to implement it");
#endif

// ---------------------------------------------------------------------------
// Selfadjoint (SYMM) packers.
//
// product_selfadjoint_matrix packs the selfadjoint operand (stored as one
// triangle) through symm_pack_lhs/symm_pack_rhs, which materialize the full
// matrix as they pack. The generic SYMM packers emit packet-width sub-panels
// for the generic gebp_kernel, whereas the SME kernel expects uniform
// mr/nr-wide depth-major panels. These packers perform the same
// triangle mirroring in the SME layout; complex SYMM keeps the generic path.
//
// The packer receives the operand in an orientation where row >= col is the
// stored triangle. It reads that half directly and mirrors the other half
// (conjugation is an identity for real scalars):
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
// Depth-region boundaries for the panel at outer offset `j`, all clamped to
// [0, depth]: the diagonal splits it into a transposed head [0, t_end), a
// straddle band [t_end, s_end) and a direct tail [s_end, depth).
template <typename Index>
static EIGEN_ALWAYS_INLINE void sme_symm_panel_regions(Index j, int w, Index depth, Index k2, Index& t_end,
                                                       Index& s_end) __arm_streaming_compatible {
  const Index raw_t = j - k2, raw_s = j + Index(w) - k2;
  t_end = raw_t <= 0 ? Index(0) : sme_min(raw_t, depth);
  s_end = raw_s <= 0 ? Index(0) : sme_min(raw_s, depth);
}

// The two dense regions of every panel, which are ordinary copies or ZA
// transposes of the stored triangle. ColM selects the ColMajor operand.
template <typename Scalar, int StorageOrder, typename Index>
EIGEN_DONT_INLINE __arm_locally_streaming __arm_new("za") void sme_symm_pack_dense_regions(
    Scalar* block, const Scalar* EIGEN_RESTRICT base, Index stride, Index depth, Index outer, Index k2) {
  constexpr int PACK = sme_block<Scalar>::mr;
  constexpr bool ColM = (StorageOrder == ColMajor);

  for (Index j = 0; j < outer; j += PACK) {
    const int w = static_cast<int>(sme_min(outer - j, Index(PACK)));
    Scalar* dst = block + j * depth;  // depth-major panel of width w
    Index t_end, s_end;
    sme_symm_panel_regions(j, w, depth, k2, t_end, s_end);

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
  }
}

// The diagonal band of every panel: the diagonal crosses at c* = (k2+k) - j
// (in [0, w) throughout the band), so each depth step splits into a direct head
// (c < c*: m(k2+k, j+c)) and a mirrored tail (c >= c*: m(j+c, k2+k); at c == c*
// both name the diagonal element).
//
// Kept out of the streaming region above for the reason tail_transpose_pack gives,
// at the cost of a second pass over the panels: it is scalar floating-point,
// and fusing it made the float SYMM packers 2-11x slower.
template <typename Scalar, int StorageOrder, typename Index>
EIGEN_DONT_INLINE void sme_symm_pack_straddle(Scalar* block, const Scalar* EIGEN_RESTRICT base, Index stride,
                                              Index depth, Index outer, Index k2) {
  constexpr int PACK = sme_block<Scalar>::mr;
  constexpr bool ColM = (StorageOrder == ColMajor);

  for (Index j = 0; j < outer; j += PACK) {
    const int w = static_cast<int>(numext::mini(outer - j, Index(PACK)));
    Scalar* dst = block + j * depth;
    Index t_end, s_end;
    sme_symm_panel_regions(j, w, depth, k2, t_end, s_end);

    for (Index k = t_end; k < s_end; ++k) {
      const Index row = k2 + k;
      const int cs = static_cast<int>(row - j);
      Scalar* dst_row = dst + k * w;
      EIGEN_IF_CONSTEXPR (ColM) {
        const Scalar* head = base + row + j * stride;  // m(row, j+c): stride-strided
        for (int c = 0; c < cs; ++c, head += stride) dst_row[c] = *head;
        const Scalar* tail = base + j + row * stride;  // m(j+c, row): contiguous
        for (int c = cs; c < w; ++c) dst_row[c] = tail[c];
      } else {
        const Scalar* head = base + row * stride + j;  // m(row, j+c): contiguous
        for (int c = 0; c < cs; ++c) dst_row[c] = head[c];
        const Scalar* tail = base + (j + Index(cs)) * stride + row;  // m(j+c, row): stride-strided
        for (int c = cs; c < w; ++c, tail += stride) dst_row[c] = *tail;
      }
    }
  }
}

// Packer shared by the LHS (k2 == 0) and RHS symm specializations.
template <typename Scalar, int StorageOrder, typename Index>
EIGEN_DONT_INLINE void sme_symm_pack_panels(Scalar* block, const Scalar* EIGEN_RESTRICT base, Index stride, Index depth,
                                            Index outer, Index k2) {
  static_assert(sme_block<Scalar>::mr == sme_block<Scalar>::nr, "the shared SYMM packer assumes square panels");
  sme_symm_pack_dense_regions<Scalar, StorageOrder, Index>(block, base, stride, depth, outer, k2);
  sme_symm_pack_straddle<Scalar, StorageOrder, Index>(block, base, stride, depth, outer, k2);
}

// symm_pack_lhs/rhs SME specializations: emit the uniform mr/nr panels
// sme_gebp_impl reads. Pack1/nr pinned exactly as gemm_pack_lhs/rhs above.
template <typename Scalar, int StorageOrder, typename Index>
struct sme_symm_pack_lhs {
  // Note: generic symm_pack_lhs's "cols" is the depth extent, and the LHS
  // block is diagonal-anchored (base = &lhs(k2,k2)), so its depth offset is 0.
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const Scalar* lhs_, Index lhsStride, Index cols, Index rows) const {
    sme_symm_pack_panels<Scalar, StorageOrder, Index>(blockA, lhs_, lhsStride, cols, rows, Index(0));
  }
};

template <typename Scalar, int StorageOrder, typename Index>
struct sme_symm_pack_rhs {
  // Note: generic symm_pack_rhs's "rows" is the depth extent (end_k = k2 + rows), not a row count.
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const Scalar* rhs_, Index rhsStride, Index rows, Index cols,
                                    Index k2) const {
    sme_symm_pack_panels<Scalar, StorageOrder, Index>(blockB, rhs_, rhsStride, rows, cols, k2);
  }
};

template <typename Index, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<float, Index, kSmeMr, Pack2_dummy, StorageOrder> : sme_symm_pack_lhs<float, StorageOrder, Index> {
};

template <typename Index, int StorageOrder>
struct symm_pack_rhs<float, Index, kSmeNr, StorageOrder> : sme_symm_pack_rhs<float, StorageOrder, Index> {};

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <typename Index, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<double, Index, kSmeMrD, Pack2_dummy, StorageOrder>
    : sme_symm_pack_lhs<double, StorageOrder, Index> {};

template <typename Index, int StorageOrder>
struct symm_pack_rhs<double, Index, kSmeNrD, StorageOrder> : sme_symm_pack_rhs<double, StorageOrder, Index> {};
#endif

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SME_GENERALBLOCKPANELKERNEL_H
