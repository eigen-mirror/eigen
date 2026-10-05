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
// A complex accumulator is a *pair* of tiles holding its real and imaginary
// halves, since FMOPA only takes real operands, so complex<float> gets a 1x2
// grid of pairs out of ZA.S and complex<double> a 2x2 grid out of ZA.D.  The
// packed panels are correspondingly split -- one depth step holds the real
// parts of the panel width then the imaginary ones -- so the four real outer
// products a complex one expands to reuse both operands and the panel traffic
// per FMOPA halves relative to the real kernels.
//
// This translation unit must be built without -msve-vector-bits (scalable/VLA
// mode); see the guard in ConfigureVectorization.h for the rationale.
// Everything below derives lane counts/predicates from the runtime svl; when a
// block matches the tile grid exactly, the micro-kernel additionally switches
// to a hand-scheduled multi-vector-load loop (see sme_process).
// ---------------------------------------------------------------------------

// The streaming vector width the block sizes are chosen at: 512 bits, where a
// vector holds 64 / sizeof(element) elements.  Other SVLs tile the block at
// runtime.  If a future SVL ever justifies a larger block, this is the only
// knob -- but don't grow it speculatively, a doubled block measures slower at
// SVL=512.
static constexpr int kSmeDesignVectorBytes = 64;

// Logical micro-kernel block (LHS/RHS panel widths), as a grid of ZA tiles each
// svl x svl elements of the ZA element width -- Scalar itself for a real
// scalar, its real part for a complex one, whose halves accumulate into
// separate tiles -- a full 2x2 grid for real scalars.
// A complex accumulator takes a tile pair, so complex<float> gets half the grid
// cells of float; the grid stays two cells wide, which measures 1.2-1.9x a
// two-cell-tall one below 128 on Apple M4 and matches it above.
template <typename Scalar>
struct sme_block {
  static constexpr int kGridRows = 2;
  static constexpr int kGridCols = 2;
  static constexpr int mr = kGridRows * kSmeDesignVectorBytes / int(sizeof(Scalar));
  static constexpr int nr = kGridCols * kSmeDesignVectorBytes / int(sizeof(Scalar));
};

template <typename RealScalar>
struct sme_block<std::complex<RealScalar>> {
  static constexpr int kGridCols = 2;
  static constexpr int kGridRows = sme_tile_count<RealScalar>::value / (2 * kGridCols);
  static constexpr int mr = kGridRows * kSmeDesignVectorBytes / int(sizeof(RealScalar));
  static constexpr int nr = kGridCols * kSmeDesignVectorBytes / int(sizeof(RealScalar));
};

static constexpr int kSmeMr = sme_block<float>::mr;
static constexpr int kSmeNr = sme_block<float>::nr;
static constexpr int kSmeMrC = sme_block<std::complex<float>>::mr;
static constexpr int kSmeNrC = sme_block<std::complex<float>>::nr;
#ifdef EIGEN_VECTORIZE_SME_F64F64
static constexpr int kSmeMrD = sme_block<double>::mr;
static constexpr int kSmeNrD = sme_block<double>::nr;
static constexpr int kSmeMrCD = sme_block<std::complex<double>>::mr;
static constexpr int kSmeNrCD = sme_block<std::complex<double>>::nr;
#endif

// ---------------------------------------------------------------------------
// Packed panel layout and the primitives that produce it.
//
// A packed panel is depth-major.  For a real element type one depth step holds
// `width` contiguous scalars; for a complex one it holds the `width` real parts
// followed by the `width` imaginary parts, so the micro-kernel feeds a ZA tile
// pair from two contiguous vector loads and never deinterleaves inside the
// depth loop.  Either way a width x depth panel occupies width * depth Scalars,
// which is what the GEMM driver allocates.
//
// `Conjugate` negates the imaginary half.  Conjugation is the identity on real
// scalars, so the real overloads ignore it -- and Conjugate=true instantiations
// do reach them, from the SYMM above-diagonal transposed pack.
// ---------------------------------------------------------------------------

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
template <bool Conjugate, typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel_range(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                                     Index src_stride, Index k0, Index k1, int width) __arm_streaming {
  using Traits = sme_packet_traits<Scalar>;
  const int svl = Traits::size();
  if (width == 2 * svl) {
    // Full panel: one two-vector load and store per depth step, four steps in flight.
    const svcount_t pn = Traits::ptrue_c();
    Index k = k0;
    for (; k + 4 <= k1; k += 4) {
      const auto v0 = ploadu_x2(pn, &src[k * src_stride]);
      const auto v1 = ploadu_x2(pn, &src[(k + 1) * src_stride]);
      const auto v2 = ploadu_x2(pn, &src[(k + 2) * src_stride]);
      const auto v3 = ploadu_x2(pn, &src[(k + 3) * src_stride]);
      pstoreu_x2(pn, &dst[k * width], v0);
      pstoreu_x2(pn, &dst[(k + 1) * width], v1);
      pstoreu_x2(pn, &dst[(k + 2) * width], v2);
      pstoreu_x2(pn, &dst[(k + 3) * width], v3);
    }
    for (; k < k1; ++k) pstoreu_x2(pn, &dst[k * width], ploadu_x2(pn, &src[k * src_stride]));
    return;
  }
  for (int off = 0; off < width; off += svl) {
    const svbool_t pred = sme_packet_traits<Scalar>::whilelt(off, width);
    for (Index k = k0; k < k1; ++k) {
      pstoreu(pred, &dst[k * width + off], ploadu(pred, &src[k * src_stride + off]));
    }
  }
}

// Four full 2*svl-wide panels from one 8*svl-wide source strip (panel p at dst + p * panel_stride): two four-vector
// loads per depth step, and the core prefetches the strip into L2 four steps ahead.
template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel_quad(Scalar* EIGEN_RESTRICT dst, Index panel_stride,
                                                    const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                                    Index depth) __arm_streaming {
  using Traits = sme_packet_traits<Scalar>;
  const int w = 2 * Traits::size();
  const svcount_t pn = Traits::ptrue_c();
  const int strip_bytes = 4 * w * int(sizeof(Scalar));
  for (Index k = 0; k < depth; ++k) {
    const Scalar* c = src + k * src_stride;
    if (k + 4 < depth)
      for (int l = 0; l < strip_bytes; l += 128)
        __builtin_prefetch(reinterpret_cast<const char*>(c + 4 * src_stride) + l, 0, 2);
    const auto a = ploadu_x4(pn, c), b = ploadu_x4(pn, c + 2 * w);
    pstoreu_x2(pn, dst + k * w, pcreate(pget<0>(a), pget<1>(a)));
    pstoreu_x2(pn, dst + panel_stride + k * w, pcreate(pget<2>(a), pget<3>(a)));
    pstoreu_x2(pn, dst + 2 * panel_stride + k * w, pcreate(pget<0>(b), pget<1>(b)));
    pstoreu_x2(pn, dst + 3 * panel_stride + k * w, pcreate(pget<2>(b), pget<3>(b)));
  }
}

// Packs the leading groups of four full MR-wide panels with sve_copy_panel_quad; returns the rows it covered.
// Complex panels keep the per-panel path (their packed layout splits real and imaginary halves).
template <bool PanelMode, typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE Index sme_pack_quad_panels(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src,
                                                      Index src_stride, Index depth, Index rows, Index mr,
                                                      Index dst_stride, Index dst_offset) __arm_streaming {
  if (mr != 2 * sme_packet_traits<Scalar>::size()) return 0;
  const Index panel_stride = PanelMode ? mr * dst_stride : mr * depth;
  Index i = 0;
  for (; i + 4 * mr <= rows; i += 4 * mr) {
    Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * mr : dst_base + i * depth;
    sve_copy_panel_quad(dst_panel, panel_stride, src + i, src_stride, depth);
  }
  return i;
}
template <bool PanelMode, typename RealScalar, typename Index>
static EIGEN_ALWAYS_INLINE Index sme_pack_quad_panels(std::complex<RealScalar>*, const std::complex<RealScalar>*, Index,
                                                      Index, Index, Index, Index, Index) __arm_streaming {
  return 0;
}

// Complex overload: UZP1/UZP2 split the source's interleaved pairs into the two
// halves of the packed depth step.  A chunk of w complex elements spans 2*w
// interleaved reals, hence a pair of source predicates; the upper one is empty
// whenever 2*w fits in one vector.
template <bool Conjugate, typename RealScalar, typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel_range(std::complex<RealScalar>* EIGEN_RESTRICT dst,
                                                     const std::complex<RealScalar>* EIGEN_RESTRICT src,
                                                     Index src_stride, Index k0, Index k1, int width) __arm_streaming {
  using Traits = sme_packet_traits<RealScalar>;
  using Vec = typename Traits::type;
  RealScalar* EIGEN_RESTRICT rdst = reinterpret_cast<RealScalar*>(dst);
  const RealScalar* EIGEN_RESTRICT rsrc = reinterpret_cast<const RealScalar*>(src);
  const int svl = Traits::size();
  const Index step = Index(2 * width);
  for (int off = 0; off < width; off += svl) {
    const int w = width - off;
    const int lanes = 2 * w;
    const svbool_t pg_w = Traits::whilelt(off, width);
    const svbool_t pg_lo = Traits::whilelt(0, lanes);
    const svbool_t pg_hi = Traits::whilelt(svl, lanes);
    for (Index k = k0; k < k1; ++k) {
      const RealScalar* p = rsrc + Index(2) * (k * src_stride + Index(off));
      const Vec v_lo = ploadu(pg_lo, p);
      // pg_hi is all-false when 2*w fits in one vector, and an inactive lane
      // makes no memory access -- but p + svl may still be past the source, so
      // the address is formed through sme_offset.
      const Vec v_hi = ploadu(pg_hi, sme_offset(p, svl));
      Vec im = puzp2(v_lo, v_hi);
      EIGEN_IF_CONSTEXPR (Conjugate) {
        im = pnegate(pg_w, im);
      }
      pstoreu(pg_w, &rdst[k * step + Index(off)], puzp1(v_lo, v_hi));
      pstoreu(pg_w, &rdst[k * step + Index(width + off)], im);
    }
  }
}

// Copy the full depth [0, depth): thin wrapper used by the (non-symm) gemm
// packers, which always pack a whole panel.
template <bool Conjugate, typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                               Index src_stride, Index depth, int width) __arm_streaming {
  sve_copy_panel_range<Conjugate>(dst, src, src_stride, Index(0), depth, width);
}

// Full 2*svl-wide panel through all four tiles, four rows or depth steps per ZA move, source prefetched into L2
// two steps ahead; returns the first depth index of the tail (< 2*svl) it leaves to the caller.
template <typename RealScalar, typename Index>
static EIGEN_ALWAYS_INLINE Index sme_transpose_pack_pair(RealScalar* EIGEN_RESTRICT dst,
                                                         const RealScalar* EIGEN_RESTRICT src, Index src_stride,
                                                         Index k0, Index k1) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<RealScalar>;
  const int svl = Traits::size(), w = 2 * svl;
  const svcount_t pn = Traits::ptrue_c();
  Index k = k0;
  for (; k + w <= k1; k += w) {
    const bool pf = k + 2 * w <= k1;
    for (int r = 0; r < svl; r += 4) {
      const RealScalar* p = src + Index(r) * src_stride + k;
      const RealScalar* q = p + Index(svl) * src_stride;
      if (pf)
        for (int u = 0; u < 4; ++u) {
          __builtin_prefetch(p + u * src_stride + 2 * w, 0, 2);
          __builtin_prefetch(q + u * src_stride + 2 * w, 0, 2);
        }
      const auto a0 = ploadu_x2(pn, p), a1 = ploadu_x2(pn, p + src_stride), a2 = ploadu_x2(pn, p + 2 * src_stride),
                 a3 = ploadu_x2(pn, p + 3 * src_stride);
      sme_write_hor_za_vg4<0>(uint32_t(r), pget<0>(a0), pget<0>(a1), pget<0>(a2), pget<0>(a3));
      sme_write_hor_za_vg4<1>(uint32_t(r), pget<1>(a0), pget<1>(a1), pget<1>(a2), pget<1>(a3));
      const auto b0 = ploadu_x2(pn, q), b1 = ploadu_x2(pn, q + src_stride), b2 = ploadu_x2(pn, q + 2 * src_stride),
                 b3 = ploadu_x2(pn, q + 3 * src_stride);
      sme_write_hor_za_vg4<2>(uint32_t(r), pget<0>(b0), pget<0>(b1), pget<0>(b2), pget<0>(b3));
      sme_write_hor_za_vg4<3>(uint32_t(r), pget<1>(b0), pget<1>(b1), pget<1>(b2), pget<1>(b3));
    }
    for (int c = 0; c < svl; c += 4) {
      const auto t0 = sme_read_ver_za_vg4<0, RealScalar>(uint32_t(c)),
                 t2 = sme_read_ver_za_vg4<2, RealScalar>(uint32_t(c));
      const auto t1 = sme_read_ver_za_vg4<1, RealScalar>(uint32_t(c)),
                 t3 = sme_read_ver_za_vg4<3, RealScalar>(uint32_t(c));
      RealScalar* d0 = dst + (k + c) * w;
      RealScalar* d1 = d0 + Index(svl) * w;
      pstoreu_x2(pn, d0, pcreate(pget<0>(t0), pget<0>(t2)));
      pstoreu_x2(pn, d0 + w, pcreate(pget<1>(t0), pget<1>(t2)));
      pstoreu_x2(pn, d0 + 2 * w, pcreate(pget<2>(t0), pget<2>(t2)));
      pstoreu_x2(pn, d0 + 3 * w, pcreate(pget<3>(t0), pget<3>(t2)));
      pstoreu_x2(pn, d1, pcreate(pget<0>(t1), pget<0>(t3)));
      pstoreu_x2(pn, d1 + w, pcreate(pget<1>(t1), pget<1>(t3)));
      pstoreu_x2(pn, d1 + 2 * w, pcreate(pget<2>(t1), pget<2>(t3)));
      pstoreu_x2(pn, d1 + 3 * w, pcreate(pget<3>(t1), pget<3>(t3)));
    }
  }
  return k;
}

// A row of a partial panel, or zeros past its last row (those slices are never stored).
template <typename RealScalar>
static EIGEN_ALWAYS_INLINE typename sme_packet_traits<RealScalar>::type_x2 sme_row_or_zero(
    const RealScalar* p, bool valid, svcount_t pn, typename sme_packet_traits<RealScalar>::type zero) __arm_streaming {
  return valid ? ploadu_x2(pn, p) : pcreate(zero, zero);
}

// Partial panel (width < 2*svl) through the tiles as sme_transpose_pack_pair does, rows past the width as zeros and
// predicated stores of `width` scalars per depth step; returns the first depth index of the tail it leaves.
template <typename RealScalar, typename Index>
static EIGEN_ALWAYS_INLINE Index sme_transpose_pack_partial(RealScalar* EIGEN_RESTRICT dst,
                                                            const RealScalar* EIGEN_RESTRICT src, Index src_stride,
                                                            Index k0, Index k1,
                                                            int width) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<RealScalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size(), w2 = 2 * svl;
  const svcount_t pn = Traits::ptrue_c();
  const Vec zero = pset1<Vec>(RealScalar(0));
  const int rows_lo = sme_min(width, svl), rows_hi = width - rows_lo;
  const svbool_t pg_lo = Traits::whilelt(0, rows_lo), pg_hi = Traits::whilelt(0, rows_hi);
  const svbool_t pg_two = Traits::whilelt(0, 2 * width);
  Index k = k0;
  for (; k + w2 <= k1; k += w2) {
    for (int r = 0; r < rows_lo; r += 4) {
      const RealScalar* q = src + Index(r) * src_stride + k;
      const auto a0 = sme_row_or_zero<RealScalar>(q, r < rows_lo, pn, zero);
      const auto a1 = sme_row_or_zero<RealScalar>(q + src_stride, r + 1 < rows_lo, pn, zero);
      const auto a2 = sme_row_or_zero<RealScalar>(q + 2 * src_stride, r + 2 < rows_lo, pn, zero);
      const auto a3 = sme_row_or_zero<RealScalar>(q + 3 * src_stride, r + 3 < rows_lo, pn, zero);
      sme_write_hor_za_vg4<0>(uint32_t(r), pget<0>(a0), pget<0>(a1), pget<0>(a2), pget<0>(a3));
      sme_write_hor_za_vg4<1>(uint32_t(r), pget<1>(a0), pget<1>(a1), pget<1>(a2), pget<1>(a3));
    }
    for (int r = 0; r < rows_hi; r += 4) {
      const RealScalar* q = src + Index(svl + r) * src_stride + k;
      const auto a0 = sme_row_or_zero<RealScalar>(q, r < rows_hi, pn, zero);
      const auto a1 = sme_row_or_zero<RealScalar>(q + src_stride, r + 1 < rows_hi, pn, zero);
      const auto a2 = sme_row_or_zero<RealScalar>(q + 2 * src_stride, r + 2 < rows_hi, pn, zero);
      const auto a3 = sme_row_or_zero<RealScalar>(q + 3 * src_stride, r + 3 < rows_hi, pn, zero);
      sme_write_hor_za_vg4<2>(uint32_t(r), pget<0>(a0), pget<0>(a1), pget<0>(a2), pget<0>(a3));
      sme_write_hor_za_vg4<3>(uint32_t(r), pget<1>(a0), pget<1>(a1), pget<1>(a2), pget<1>(a3));
    }
    if (2 * width <= svl) {
      // Two depth steps per store: half as many stores, and full lines at width svl/2.
      for (int c = 0; c < svl; c += 4) {
        const auto t0 = sme_read_ver_za_vg4<0, RealScalar>(uint32_t(c)),
                   t1 = sme_read_ver_za_vg4<1, RealScalar>(uint32_t(c));
        RealScalar* d0 = dst + (k + c) * width;
        RealScalar* d1 = d0 + Index(svl) * width;
        pstoreu(pg_two, d0, psplice(pg_lo, pget<0>(t0), pget<1>(t0)));
        pstoreu(pg_two, d0 + 2 * width, psplice(pg_lo, pget<2>(t0), pget<3>(t0)));
        pstoreu(pg_two, d1, psplice(pg_lo, pget<0>(t1), pget<1>(t1)));
        pstoreu(pg_two, d1 + 2 * width, psplice(pg_lo, pget<2>(t1), pget<3>(t1)));
      }
      continue;
    }
    for (int c = 0; c < svl; c += 4) {
      const auto t0 = sme_read_ver_za_vg4<0, RealScalar>(uint32_t(c)),
                 t1 = sme_read_ver_za_vg4<1, RealScalar>(uint32_t(c));
      RealScalar* d0 = dst + (k + c) * width;
      RealScalar* d1 = d0 + Index(svl) * width;
      pstoreu(pg_lo, d0, pget<0>(t0));
      pstoreu(pg_lo, d0 + width, pget<1>(t0));
      pstoreu(pg_lo, d0 + 2 * width, pget<2>(t0));
      pstoreu(pg_lo, d0 + 3 * width, pget<3>(t0));
      pstoreu(pg_lo, d1, pget<0>(t1));
      pstoreu(pg_lo, d1 + width, pget<1>(t1));
      pstoreu(pg_lo, d1 + 2 * width, pget<2>(t1));
      pstoreu(pg_lo, d1 + 3 * width, pget<3>(t1));
      if (rows_hi > 0) {
        const auto t2 = sme_read_ver_za_vg4<2, RealScalar>(uint32_t(c)),
                   t3 = sme_read_ver_za_vg4<3, RealScalar>(uint32_t(c));
        pstoreu(pg_hi, d0 + svl, pget<0>(t2));
        pstoreu(pg_hi, d0 + width + svl, pget<1>(t2));
        pstoreu(pg_hi, d0 + 2 * width + svl, pget<2>(t2));
        pstoreu(pg_hi, d0 + 3 * width + svl, pget<3>(t2));
        pstoreu(pg_hi, d1 + svl, pget<0>(t3));
        pstoreu(pg_hi, d1 + width + svl, pget<1>(t3));
        pstoreu(pg_hi, d1 + 2 * width + svl, pget<2>(t3));
        pstoreu(pg_hi, d1 + 3 * width + svl, pget<3>(t3));
      }
    }
  }
  return k;
}

// Transpose-pack `width` source rows into depth-major packed output using ZA's
// 2D store as a free transpose, for the depth sub-range [k0, k1): a svl x svl
// block of source (svl rows x svl depth) is loaded as horizontal ZA slices,
// then read back as vertical slices, which emits it depth-major.  Row-groups of
// svl rows are processed two at a time through ZA tiles 0 and 1: ZA is not
// renamed, so a single tile would stall every load pass on the previous read
// pass (write-after-read); two tiles in flight keep the phases independent.
// Trailing row-groups (when width is not a multiple of 2*svl) use tile 0 with
// predicated rows -- which is also what a panel narrower than 2*svl gets in
// full: complex<float> has mr = svl at SVL=512, so its LHS panels take the
// single-tile path and do not get the write-after-read overlap described above.
// Widening the gate would need the pairing to run over depth instead of rows.
// Both dst and src are indexed by the absolute depth index k:
//   dst[k*width + r] = src[r*src_stride + k],  k in [k0,k1), r in [0,width).
// The symm packers reuse this for the diagonal-split transposed/direct regions
// (a depth sub-range at a depth offset, with a tail-panel width < mr).
//
// NegateOddRows negates every odd output depth row.  That is how the complex
// overload below conjugates: in the real view of a complex panel those rows are
// exactly the imaginary halves.
template <bool NegateOddRows, typename RealScalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack_real(RealScalar* EIGEN_RESTRICT dst,
                                                        const RealScalar* EIGEN_RESTRICT src, Index src_stride,
                                                        Index k0, Index k1,
                                                        int width) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<RealScalar>;
  using Vec = typename Traits::type;
  const Vec zero = pset1<Vec>(RealScalar(0));
  const svbool_t pg_all = Traits::ptrue();
  const int svl = Traits::size();

  Index k = k0;
  EIGEN_IF_CONSTEXPR (!NegateOddRows) {
    // Short ranges keep the single-tile path: the four-slice moves only pay off over several fills, and need a tile
    // of at least four slices.
    if (svl >= 4 && width == 2 * svl && k1 - k0 >= Index(8 * svl))
      k = sme_transpose_pack_pair(dst, src, src_stride, k0, k1);
    else if (svl >= 4 && width < 2 * svl && k1 - k0 >= Index(8 * svl))
      k = sme_transpose_pack_partial(dst, src, src_stride, k0, k1, width);
  }
  for (; k < k1; k += svl) {
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
        Vec v0 = sme_read_ver_za<0>(zero, pg_all, uint32_t(c));
        Vec v1 = sme_read_ver_za<1>(zero, pg_all, uint32_t(c));
        EIGEN_IF_CONSTEXPR (NegateOddRows) {
          if (((k + Index(c)) & Index(1)) != Index(0)) {
            v0 = pnegate(pg_all, v0);
            v1 = pnegate(pg_all, v1);
          }
        }
        pstoreu(pg_all, &dst[(k + c) * width + r0], v0);
        pstoreu(pg_all, &dst[(k + c) * width + r0 + svl], v1);
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
        Vec v0 = sme_read_ver_za<0>(zero, pg_r, uint32_t(c));
        EIGEN_IF_CONSTEXPR (NegateOddRows) {
          if (((k + Index(c)) & Index(1)) != Index(0)) v0 = pnegate(pg_r, v0);
        }
        pstoreu(pg_r, &dst[(k + c) * width + r0], v0);
      }
    }
  }
}

template <bool Conjugate, typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack_range(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                                         Index src_stride, Index k0, Index k1,
                                                         int width) __arm_streaming __arm_inout("za") {
  sme_transpose_pack_real<false>(dst, src, src_stride, k0, k1, width);
}

// Complex overload.  A ColMajor (RowMajor) complex operand is a ColMajor
// (RowMajor) real one of twice the depth and twice the stride, and transposing
// that real view already emits the split layout: real-view depth 2k lands at
// packed offset k*(2*width) and depth 2k+1 at k*(2*width) + width, the real and
// imaginary halves of packed depth step k.
template <bool Conjugate, typename RealScalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack_range(std::complex<RealScalar>* EIGEN_RESTRICT dst,
                                                         const std::complex<RealScalar>* EIGEN_RESTRICT src,
                                                         Index src_stride, Index k0, Index k1,
                                                         int width) __arm_streaming __arm_inout("za") {
  sme_transpose_pack_real<Conjugate>(reinterpret_cast<RealScalar*>(dst), reinterpret_cast<const RealScalar*>(src),
                                     Index(2) * src_stride, Index(2) * k0, Index(2) * k1, width);
}

// Transpose-pack a whole `width`-wide panel over the full depth [0, depth):
// thin wrapper used by the (non-symm) gemm packers.
template <bool Conjugate, typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                                   Index src_stride, Index depth,
                                                   int width) __arm_streaming __arm_inout("za") {
  sme_transpose_pack_range<Conjugate>(dst, src, src_stride, Index(0), depth, width);
}

// Transposing copy for a panel narrower than the pack width:
//   dst_panel[k*tail + i] = src[i*src_stride + k].
// Kept outside the caller's __arm_locally_streaming region: it needs neither SVE
// nor ZA, and streaming mode runs scalar floating-point ~40x slower on Apple M4.
// Outside it the source rows are contiguous in k, so PacketSize of them
// transpose in register as in sme_pack_rhs_fallback; a product with cols < nr is
// packed entirely here.  NegateOddRows is as in sme_transpose_pack_real.
template <bool NegateOddRows, typename RealScalar, typename Index>
static void tail_transpose_pack_real(RealScalar* EIGEN_RESTRICT dst_panel, const RealScalar* EIGEN_RESTRICT src,
                                     Index src_stride, Index depth, Index tail) {
  using Packet = typename packet_traits<RealScalar>::type;
  constexpr int PacketSize = int(packet_traits<RealScalar>::size);
  const Index peeled_tail = (tail / Index(PacketSize)) * Index(PacketSize);
  const Index peeled_depth = numext::round_down(depth, Index(PacketSize));

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
        Packet row = block.packet[p];
        EIGEN_IF_CONSTEXPR (NegateOddRows) {
          if (((k + Index(p)) & Index(1)) != Index(0)) row = pnegate(row);
        }
        pstoreu(dst_panel + (k + Index(p)) * tail + i, row);
      }
    }
    for (; k < depth; ++k) {
      const bool negate = NegateOddRows && ((k & Index(1)) != Index(0));
      for (Index p = 0; p < Index(PacketSize); ++p) {
        const RealScalar v = src[(i + p) * src_stride + k];
        dst_panel[k * tail + i + p] = negate ? -v : v;
      }
    }
  }
  for (; i < tail; ++i) {
    for (Index k = 0; k < depth; ++k) {
      const bool negate = NegateOddRows && ((k & Index(1)) != Index(0));
      const RealScalar v = src[i * src_stride + k];
      dst_panel[k * tail + i] = negate ? -v : v;
    }
  }
}

template <bool Conjugate, typename Scalar, typename Index>
static void tail_transpose_pack(Scalar* EIGEN_RESTRICT dst_panel, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                Index depth, Index tail) {
  tail_transpose_pack_real<false>(dst_panel, src, src_stride, depth, tail);
}

template <bool Conjugate, typename RealScalar, typename Index>
static void tail_transpose_pack(std::complex<RealScalar>* EIGEN_RESTRICT dst_panel,
                                const std::complex<RealScalar>* EIGEN_RESTRICT src, Index src_stride, Index depth,
                                Index tail) {
  tail_transpose_pack_real<Conjugate>(reinterpret_cast<RealScalar*>(dst_panel),
                                      reinterpret_cast<const RealScalar*>(src), Index(2) * src_stride, Index(2) * depth,
                                      tail);
}

// Tail panel of a deep block through the ZA transposer (sme_transpose_pack_partial); shallow ones and those of at
// most 4 columns keep the NEON tail_transpose_pack, which is faster there.
template <bool Conjugate, typename Scalar, typename Index>
__arm_locally_streaming __arm_new("za") static void sme_tail_pack_streaming(Scalar* dst_panel, const Scalar* src,
                                                                            Index src_stride, Index depth, int tail) {
  sme_transpose_pack<Conjugate>(dst_panel, src, src_stride, depth, tail);
}
template <bool Conjugate, typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void tail_pack(Scalar* dst_panel, const Scalar* src, Index src_stride, Index depth,
                                          Index tail) {
  EIGEN_IF_CONSTEXPR (!NumTraits<Scalar>::IsComplex) {
    if (tail > 4 && depth >= Index(4 * sme_block<Scalar>::nr)) {
      sme_tail_pack_streaming<Conjugate>(dst_panel, src, src_stride, depth, static_cast<int>(tail));
      return;
    }
  }
  tail_transpose_pack<Conjugate>(dst_panel, src, src_stride, depth, tail);
}

// De-interleaving load and interleaving store of PS complex values.
static EIGEN_ALWAYS_INLINE void sme_neon_ld2(const float* p, Packet4f& re, Packet4f& im) {
  const float32x4x2_t v = vld2q_f32(p);
  re = v.val[0];
  im = v.val[1];
}
static EIGEN_ALWAYS_INLINE void sme_neon_st2(float* p, Packet4f re, Packet4f im) {
  float32x4x2_t v;
  v.val[0] = re;
  v.val[1] = im;
  vst2q_f32(p, v);
}
static EIGEN_ALWAYS_INLINE void sme_neon_ld2(const double* p, Packet2d& re, Packet2d& im) {
  const float64x2x2_t v = vld2q_f64(p);
  re = v.val[0];
  im = v.val[1];
}
static EIGEN_ALWAYS_INLINE void sme_neon_st2(double* p, Packet2d re, Packet2d im) {
  float64x2x2_t v;
  v.val[0] = re;
  v.val[1] = im;
  vst2q_f64(p, v);
}

// NEON copy packers for the direct-access, unit-stride case: dst[k*w + r] =
// src[r + k*src_stride], w contiguous scalars per depth step; complex panels
// split into the depth step's real then imaginary halves, conjugated if asked.
template <bool Conjugate, typename Scalar, typename Index>
static void neon_copy_panel(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src, Index src_stride, Index depth,
                            Index w) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr Index PS = Index(packet_traits<Scalar>::size);
  const Index peeled4 = numext::round_down(w, 4 * PS);
  const Index peeled = numext::round_down(w, PS);
  for (Index k = 0; k < depth; ++k) {
    const Scalar* s = src + k * src_stride;
    Scalar* d = dst + k * w;
    Index r = 0;
    for (; r < peeled4; r += 4 * PS) {
      const Packet p0 = ploadu<Packet>(s + r), p1 = ploadu<Packet>(s + r + PS);
      const Packet p2 = ploadu<Packet>(s + r + 2 * PS), p3 = ploadu<Packet>(s + r + 3 * PS);
      pstoreu(d + r, p0);
      pstoreu(d + r + PS, p1);
      pstoreu(d + r + 2 * PS, p2);
      pstoreu(d + r + 3 * PS, p3);
    }
    for (; r < peeled; r += PS) pstoreu(d + r, ploadu<Packet>(s + r));
    for (; r < w; ++r) d[r] = s[r];
  }
}
template <bool Conjugate, typename RealScalar, typename Index>
static void neon_copy_panel(std::complex<RealScalar>* EIGEN_RESTRICT dst,
                            const std::complex<RealScalar>* EIGEN_RESTRICT src, Index src_stride, Index depth,
                            Index w) {
  using Packet = typename packet_traits<RealScalar>::type;
  constexpr Index PS = Index(packet_traits<RealScalar>::size);
  RealScalar* rd = reinterpret_cast<RealScalar*>(dst);
  const RealScalar* rs = reinterpret_cast<const RealScalar*>(src);
  const Index peeled = numext::round_down(w, PS);
  for (Index k = 0; k < depth; ++k) {
    const RealScalar* s = rs + k * 2 * src_stride;
    RealScalar* d = rd + k * 2 * w;
    Index r = 0;
    for (; r < peeled; r += PS) {
      Packet re, im;
      sme_neon_ld2(s + 2 * r, re, im);
      pstoreu(d + r, re);
      pstoreu(d + w + r, Conjugate ? pnegate(im) : im);
    }
    for (; r < w; ++r) {
      d[r] = s[2 * r];
      d[w + r] = Conjugate ? -s[2 * r + 1] : s[2 * r + 1];
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

// Store one element of a packed depth step of width w.  Real scalars land at
// dst_step[r]; complex ones split into the step's real and imaginary halves,
// which is the same base pointer reinterpreted, since a complex depth step of
// width w spans 2*w reals.
template <bool Conjugate, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_pack_store(Scalar* dst_step, Index w, Index r, const Scalar& v) {
  EIGEN_UNUSED_VARIABLE(w);
  dst_step[r] = conj_if<Conjugate>()(v);
}
template <bool Conjugate, typename RealScalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_pack_store(std::complex<RealScalar>* dst_step, Index w, Index r,
                                        const std::complex<RealScalar>& v) {
  const std::complex<RealScalar> cv = conj_if<Conjugate>()(v);
  RealScalar* p = reinterpret_cast<RealScalar*>(dst_step);
  p[r] = numext::real(cv);
  p[w + r] = numext::imag(cv);
}

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
// vectorise = false and take the scalar element loop. Complex scalars always
// take it too: a complex packet store would emit the interleaved layout, not the
// split one the kernel reads.
template <typename Scalar, int MR, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
void sme_pack_lhs_fallback(Scalar* dst_base, const DataMapper& lhs, Index depth, Index rows, Index dst_stride,
                           Index dst_offset, bool vectorise) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr Index PacketSize = Index(packet_traits<Scalar>::size);
  constexpr bool HasPacketPath = !NumTraits<Scalar>::IsComplex;

  for (Index i = 0; i < rows; i += MR) {
    const Index w = numext::mini(rows - i, Index(MR));
    Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * w : dst_base + i * depth;
    const Index peeled_w = (vectorise && HasPacketPath) ? numext::round_down(w, Index(PacketSize)) : Index(0);
    for (Index k = 0; k < depth; ++k) {
      Scalar* dst_step = dst_panel + k * w;
      Index r = 0;
      for (; r < peeled_w; r += PacketSize) {
        pstoreu(dst_step + r, lhs.template loadPacket<Packet>(i + r, k));
      }
      for (; r < w; ++r) {
        sme_pack_store<Conjugate>(dst_step, w, r, lhs(i + r, k));
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
template <typename Scalar, int NR, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
void sme_pack_rhs_fallback(Scalar* dst_base, const DataMapper& rhs, Index depth, Index cols, Index dst_stride,
                           Index dst_offset, bool vectorise) {
  using Packet = typename packet_traits<Scalar>::type;
  using LinearMapper = typename DataMapper::LinearMapper;
  constexpr int PacketSize = int(packet_traits<Scalar>::size);
  constexpr bool HasPacketPath = !NumTraits<Scalar>::IsComplex;
  const Index peeled_depth = (depth / Index(PacketSize)) * Index(PacketSize);

  for (Index j = 0; j < cols; j += NR) {
    const Index w = numext::mini(cols - j, Index(NR));
    Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * w : dst_base + j * depth;
    const Index peeled_w = (vectorise && HasPacketPath) ? numext::round_down(w, Index(PacketSize)) : Index(0);
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
          sme_pack_store<Conjugate>(dst_panel + k * w, w, c + p, rhs(k, j + c + p));
        }
      }
    }
    for (; c < w; ++c) {
      for (Index k = 0; k < depth; ++k) {
        sme_pack_store<Conjugate>(dst_panel + k * w, w, c, rhs(k, j + c));
      }
    }
  }
}

// NEON dispatch invariant, constants fitted on Apple M4 (SVL 512). Packers: panel depth (the stride in
// panel mode) <= sme_neon_max_depth && width <= sme_neon_max_panel. Kernel: both sides pass the packer
// test with strideA / strideB as the panel depths && (max(rows, cols) <= w(depth) || min <= thin_dim).
// So a NEON kernel reads NEON-packed panels, and ZA over NEON-packed panels (~40 ns/KB) is bounded by
// sme_neon_max_panel; the exception is SYRK, which packs B at the full size but runs 32x32 diagonal blocks.
#ifndef EIGEN_SME_NEON_MAX_DEPTH
template <typename Scalar>
struct sme_neon_max_depth : std::integral_constant<int, 16> {};
template <>
struct sme_neon_max_depth<float> : std::integral_constant<int, 24> {};
template <>
struct sme_neon_max_depth<std::complex<double>> : std::integral_constant<int, 8> {};
#else
template <typename Scalar>
struct sme_neon_max_depth : std::integral_constant<int, EIGEN_SME_NEON_MAX_DEPTH> {};
#endif
#ifndef EIGEN_SME_NEON_MAX_WIDTH
template <typename Scalar>
struct sme_neon_max_width : std::integral_constant<int, 48> {};
template <>
struct sme_neon_max_width<double> : std::integral_constant<int, 40> {};
template <>
struct sme_neon_max_width<std::complex<float>> : std::integral_constant<int, 16> {};
template <>
struct sme_neon_max_width<std::complex<double>> : std::integral_constant<int, 24> {};
#else
template <typename Scalar>
struct sme_neon_max_width : std::integral_constant<int, EIGEN_SME_NEON_MAX_WIDTH> {};
#endif
// Up to sme_neon_shallow_depth the NEON range widens to sme_neon_shallow_width:
// the blocked decompositions update sub-blocks of that depth in place, at
// offsets the ZA slice stores handle poorly.
#ifndef EIGEN_SME_NEON_SHALLOW_DEPTH
template <typename Scalar>
struct sme_neon_shallow_depth : std::integral_constant<int, 8> {};
#else
template <typename Scalar>
struct sme_neon_shallow_depth : std::integral_constant<int, EIGEN_SME_NEON_SHALLOW_DEPTH> {};
#endif
#ifndef EIGEN_SME_NEON_SHALLOW_WIDTH
template <typename Scalar>
struct sme_neon_shallow_width : std::integral_constant<int, 96> {};
template <>
struct sme_neon_shallow_width<double> : std::integral_constant<int, 64> {};
template <>
struct sme_neon_shallow_width<std::complex<float>> : std::integral_constant<int, 32> {};
template <>
struct sme_neon_shallow_width<std::complex<double>> : std::integral_constant<int, 24> {};
#else
template <typename Scalar>
struct sme_neon_shallow_width : std::integral_constant<int, EIGEN_SME_NEON_SHALLOW_WIDTH> {};
#endif
// No panel wider than this is packed with NEON at any depth: the triangular solvers slice a tall
// panel into depth-8 pieces for ZA. The packers cannot mirror the kernel's width rule, or a thin
// block's wide side would be streaming-packed and read by NEON, the costly direction; the price is
// a shallow panel in (w, this] read by ZA, 1.4x the streaming-packed time at 128x128x16 float.
#ifndef EIGEN_SME_NEON_MAX_PANEL
template <typename Scalar>
struct sme_neon_max_panel : std::integral_constant<int, 256> {};
template <typename RealScalar>
struct sme_neon_max_panel<std::complex<RealScalar>> : std::integral_constant<int, 128> {};
#else
template <typename Scalar>
struct sme_neon_max_panel : std::integral_constant<int, EIGEN_SME_NEON_MAX_PANEL> {};
#endif
// A block this narrow on one side runs on NEON whatever its other side: the
// blocked decompositions update long strips of that width in place.
#ifndef EIGEN_SME_NEON_THIN_DIM
template <typename Scalar>
struct sme_neon_thin_dim : std::integral_constant<int, 32> {};
template <>
struct sme_neon_thin_dim<std::complex<float>> : std::integral_constant<int, 8> {};
template <>
struct sme_neon_thin_dim<std::complex<double>> : std::integral_constant<int, 4> {};
#else
template <typename Scalar>
struct sme_neon_thin_dim : std::integral_constant<int, EIGEN_SME_NEON_THIN_DIM> {};
#endif

// `width` is the panel's rows (LHS) or cols (RHS).
template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_pack_with_neon(Index depth, Index width) {
#if defined(EIGEN_SME_NO_NEON_SMALL_BLOCKS)
  EIGEN_UNUSED_VARIABLE(depth);
  EIGEN_UNUSED_VARIABLE(width);
  return false;
#elif defined(EIGEN_SME_FORCE_NEON_SMALL_BLOCKS)
  EIGEN_UNUSED_VARIABLE(depth);
  EIGEN_UNUSED_VARIABLE(width);
  return true;
#else
  return depth <= Index(sme_neon_max_depth<Scalar>::value) && width <= Index(sme_neon_max_panel<Scalar>::value);
#endif
}

// strideA / strideB are the packed panels' depths, which exceed `depth` when
// the caller packed them in panel mode and runs the kernel on a slice.
template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_kernel_with_neon(Index rows, Index cols, Index depth, Index strideA, Index strideB) {
#if defined(EIGEN_SME_NO_NEON_SMALL_BLOCKS) || defined(EIGEN_SME_FORCE_NEON_SMALL_BLOCKS)
  EIGEN_UNUSED_VARIABLE(depth);
  return sme_pack_with_neon<Scalar>(strideA, rows) && sme_pack_with_neon<Scalar>(strideB, cols);
#else
  const Index wide = numext::maxi(rows, cols);
  const Index w = depth <= Index(sme_neon_shallow_depth<Scalar>::value) ? Index(sme_neon_shallow_width<Scalar>::value)
                                                                        : Index(sme_neon_max_width<Scalar>::value);
  return sme_pack_with_neon<Scalar>(strideA, rows) && sme_pack_with_neon<Scalar>(strideB, cols) &&
         (wide <= w || numext::mini(rows, cols) <= Index(sme_neon_thin_dim<Scalar>::value));
#endif
}

// Shared dispatch for the four gemm_pack specializations: raw-pointer walk
// when the mapper grants direct unit-inner-stride access, otherwise the
// packet/element fallback. Tag-dispatched so &m(0,0) is only compiled for
// lvalue mappers. UsePacketPath records whether the mapper's packets advance
// the index the fallback needs, independently of its direct-access category.
// In panel mode the call packs one depth slice of a panel `stride` deep that the
// kernel consumes whole (the triangular solvers and TRMM), so the mode decision
// uses the panel's depth, not the slice's.
template <bool UsePacketPath, bool PanelMode, typename Scalar, typename Index, typename DataMapper, typename DirectFn,
          typename NeonFn, typename FallbackFn>
EIGEN_ALWAYS_INLINE void sme_dispatch_pack(DirectFn direct, NeonFn neon, FallbackFn fallback, Scalar* block,
                                           const DataMapper& m, Index depth, Index n, Index stride, Index offset,
                                           std::true_type /* direct access */) {
  if (sme_mapper_incr<Index>(m) == 1) {
    const Scalar* src = (n > 0 && depth > 0) ? &m(0, 0) : nullptr;
    if (sme_pack_with_neon<Scalar>(PanelMode ? stride : depth, n)) {
      neon(block, src, m.stride(), depth, n, stride, offset);
    } else {
      sme_fpsr_guard fpsr;
      direct(block, src, m.stride(), depth, n, stride, offset);
    }
  } else {
    fallback(block, m, depth, n, stride, offset, UsePacketPath);
  }
}
template <bool UsePacketPath, bool PanelMode, typename Scalar, typename Index, typename DataMapper, typename DirectFn,
          typename NeonFn, typename FallbackFn>
EIGEN_ALWAYS_INLINE void sme_dispatch_pack(DirectFn, NeonFn, FallbackFn fallback, Scalar* block, const DataMapper& m,
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

// Complex block sizes, as above, but left open over the conjugation flags. A
// complex operand really does reach the kernel conjugated -- from an adjoint or
// conjugate product -- and the generic complex traits keep mr/nr independent of
// that, so pinning <false, false> here would hand a conjugated instantiation the
// generic block sizes while its kernel expects the SME ones.
template <bool ConjLhs_, bool ConjRhs_>
class gebp_traits<std::complex<float>, std::complex<float>, ConjLhs_, ConjRhs_, Architecture::Target, GEBPPacketFull>
    : public gebp_traits<std::complex<float>, std::complex<float>, ConjLhs_, ConjRhs_, Architecture::Target,
                         GEBPPacketHalf> {
 public:
  static constexpr int mr = kSmeMrC;
  static constexpr int nr = kSmeNrC;
};

// The mixed-scalar guard, with the roles of the two operands swapped relative
// to the real case: gemm_pack_lhs is instantiated with the LHS scalar and
// Traits::mr, gemm_pack_rhs with the RHS scalar and Traits::nr.
static_assert(int(gebp_traits<std::complex<float>, float>::mr) != kSmeMrC,
              "gebp_traits<complex<float>, float>::mr collides with kSmeMrC: the SME gemm_pack_lhs would silently "
              "emit SME panel layout for the generic gebp_kernel");
static_assert(int(gebp_traits<float, std::complex<float>>::nr) != kSmeNrC,
              "gebp_traits<float, complex<float>>::nr collides with kSmeNrC: the SME gemm_pack_rhs would silently "
              "emit SME panel layout for the generic gebp_kernel");

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <bool ConjLhs_, bool ConjRhs_>
class gebp_traits<std::complex<double>, std::complex<double>, ConjLhs_, ConjRhs_, Architecture::Target, GEBPPacketFull>
    : public gebp_traits<std::complex<double>, std::complex<double>, ConjLhs_, ConjRhs_, Architecture::Target,
                         GEBPPacketHalf> {
 public:
  static constexpr int mr = kSmeMrCD;
  static constexpr int nr = kSmeNrCD;
};

static_assert(int(gebp_traits<std::complex<double>, double>::mr) != kSmeMrCD,
              "gebp_traits<complex<double>, double>::mr collides with kSmeMrCD: the SME gemm_pack_lhs would silently "
              "emit SME panel layout for the generic gebp_kernel");
static_assert(int(gebp_traits<double, std::complex<double>>::nr) != kSmeNrCD,
              "gebp_traits<double, complex<double>>::nr collides with kSmeNrCD: the SME gemm_pack_rhs would silently "
              "emit SME panel layout for the generic gebp_kernel");
#endif

/*****************************************************************************
 * gemm_pack_lhs for SME  (ColMajor source)
 *
 * Packs the LHS matrix into uniform panels of width mr.
 * Each depth step k writes exactly MR contiguous scalars.
 *****************************************************************************/

template <typename Scalar, int MR, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct sme_pack_lhs_colmajor {
  // Non-streaming NEON copy of every panel, for the shallow blocks the NEON
  // kernel consumes.
  static EIGEN_ALWAYS_INLINE void pack_neon(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                            Index depth, Index rows, Index dst_stride, Index dst_offset) {
    for (Index i = 0; i < rows; i += MR) {
      const Index w = numext::mini(Index(MR), rows - i);
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * w : dst_base + i * depth;
      neon_copy_panel<Conjugate>(dst_panel, src + i, src_stride, depth, w);
    }
  }

  // EIGEN_DONT_INLINE: GCC (14.1 through trunk) may inline a __arm_locally_streaming function into a
  // non-streaming caller and drop its mode switch. The other streaming entry points are __arm_new("za"),
  // which GCC does not inline.
  __arm_locally_streaming EIGEN_DONT_INLINE static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src,
                                                                    Index src_stride, Index depth, Index rows,
                                                                    Index dst_stride, Index dst_offset) {
    const Index peeled_rows = (rows / MR) * MR;
    const Index i0 = sme_pack_quad_panels<PanelMode>(dst_base, src, src_stride, depth, peeled_rows, Index(MR),
                                                     dst_stride, dst_offset);

    // Full panels of width MR, streamed in svl-wide predicated chunks.
    for (Index i = i0; i < peeled_rows; i += MR) {
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * MR : dst_base + i * depth;
      sve_copy_panel<Conjugate>(dst_panel, src + i, src_stride, depth, MR);
    }

    // Tail panel: rows < MR, use predicated SVE.
    if (peeled_rows < rows) {
      const Index tail = rows - peeled_rows;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_rows * dst_stride + dst_offset * tail : dst_base + peeled_rows * depth;
      sve_copy_panel<Conjugate>(dst_panel, src + peeled_rows, src_stride, depth, static_cast<int>(tail));
    }
  }

  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                                    Index offset = 0) {
    if (PanelMode) {
      eigen_assert(stride >= depth && offset <= stride);
    }
    // Inner-strided ColMajor blas mappers' packets advance the row index, so
    // the fallback may use them.
    sme_dispatch_pack<true, PanelMode>(
        &pack_direct, &pack_neon, &sme_pack_lhs_fallback<Scalar, MR, Index, DataMapper, Conjugate, PanelMode>, blockA,
        lhs, depth, rows, stride, offset, bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
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
template <typename Scalar, int MR, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct sme_pack_lhs_rowmajor {
  // Non-streaming NEON copy of every panel, for the shallow blocks the NEON
  // kernel consumes.
  static EIGEN_ALWAYS_INLINE void pack_neon(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                            Index depth, Index rows, Index dst_stride, Index dst_offset) {
    for (Index i = 0; i < rows; i += MR) {
      const Index w = numext::mini(Index(MR), rows - i);
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * w : dst_base + i * depth;
      tail_transpose_pack<Conjugate>(dst_panel, src + i * src_stride, src_stride, depth, w);
    }
  }

  __arm_locally_streaming __arm_new("za") static void pack_full_panels(Scalar* dst_base,
                                                                       const Scalar* EIGEN_RESTRICT src,
                                                                       Index src_stride, Index depth, Index peeled_rows,
                                                                       Index dst_stride, Index dst_offset) {
    for (Index i = 0; i < peeled_rows; i += MR) {
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * MR : dst_base + i * depth;
      sme_transpose_pack<Conjugate>(dst_panel, src + i * src_stride, src_stride, depth, MR);
    }
  }

  static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride, Index depth, Index rows,
                          Index dst_stride, Index dst_offset) {
    const Index peeled_rows = numext::round_down(rows, MR);

    if (peeled_rows > 0) {
      pack_full_panels(dst_base, src, src_stride, depth, peeled_rows, dst_stride, dst_offset);
    }

    // Row tail (rows - peeled_rows in [1, MR-1]), at most once per call: see tail_pack.
    if (peeled_rows < rows) {
      const Index tail = rows - peeled_rows;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_rows * dst_stride + dst_offset * tail : dst_base + peeled_rows * depth;
      tail_pack<Conjugate>(dst_panel, src + peeled_rows * src_stride, src_stride, depth, tail);
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
    sme_dispatch_pack<false, PanelMode>(
        &pack_direct, &pack_neon, &sme_pack_lhs_fallback<Scalar, MR, Index, DataMapper, Conjugate, PanelMode>, blockA,
        lhs, depth, rows, stride, offset, bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
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

template <typename Scalar, int NR, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct sme_pack_rhs_colmajor {
  // Non-streaming NEON copy of every panel, for the shallow blocks the NEON
  // kernel consumes.
  static EIGEN_ALWAYS_INLINE void pack_neon(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                            Index depth, Index cols, Index dst_stride, Index dst_offset) {
    for (Index i = 0; i < cols; i += NR) {
      const Index w = numext::mini(Index(NR), cols - i);
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * w : dst_base + i * depth;
      tail_transpose_pack<Conjugate>(dst_panel, src + i * src_stride, src_stride, depth, w);
    }
  }

  __arm_locally_streaming __arm_new("za") static void pack_full_panels(Scalar* dst_base,
                                                                       const Scalar* EIGEN_RESTRICT src,
                                                                       Index src_stride, Index depth, Index peeled_cols,
                                                                       Index dst_stride, Index dst_offset) {
    for (Index j = 0; j < peeled_cols; j += NR) {
      Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * NR : dst_base + j * depth;
      sme_transpose_pack<Conjugate>(dst_panel, src + j * src_stride, src_stride, depth, NR);
    }
  }

  static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride, Index depth, Index cols,
                          Index dst_stride, Index dst_offset) {
    const Index peeled_cols = numext::round_down(cols, NR);

    if (peeled_cols > 0) {
      pack_full_panels(dst_base, src, src_stride, depth, peeled_cols, dst_stride, dst_offset);
    }

    // Col tail (cols - peeled_cols in [1, NR-1]), at most once per call: see tail_pack.
    if (peeled_cols < cols) {
      const Index tail = cols - peeled_cols;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_cols * dst_stride + dst_offset * tail : dst_base + peeled_cols * depth;
      tail_pack<Conjugate>(dst_panel, src + peeled_cols * src_stride, src_stride, depth, tail);
    }
  }

  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) {
    if (PanelMode) {
      eigen_assert(stride >= depth && offset <= stride);
    }
    // Inner-strided ColMajor blas mappers' LinearMapper packets advance the
    // depth index, which is what the fallback transposes.
    sme_dispatch_pack<true, PanelMode>(
        &pack_direct, &pack_neon, &sme_pack_rhs_fallback<Scalar, NR, Index, DataMapper, Conjugate, PanelMode>, blockB,
        rhs, depth, cols, stride, offset, bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

// RowMajor RHS packer -- streaming SVE copy (mirrors the ColMajor LHS packer).
// Rows are contiguous in the source, so each depth-step is NR contiguous scalars.
template <typename Scalar, int NR, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct sme_pack_rhs_rowmajor {
  // Non-streaming NEON copy of every panel, for the shallow blocks the NEON
  // kernel consumes.
  static EIGEN_ALWAYS_INLINE void pack_neon(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src, Index src_stride,
                                            Index depth, Index cols, Index dst_stride, Index dst_offset) {
    for (Index i = 0; i < cols; i += NR) {
      const Index w = numext::mini(Index(NR), cols - i);
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * w : dst_base + i * depth;
      neon_copy_panel<Conjugate>(dst_panel, src + i, src_stride, depth, w);
    }
  }

  // EIGEN_DONT_INLINE: as in sme_pack_lhs_colmajor::pack_direct.
  __arm_locally_streaming EIGEN_DONT_INLINE static void pack_direct(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src,
                                                                    Index src_stride, Index depth, Index cols,
                                                                    Index dst_stride, Index dst_offset) {
    const Index peeled_cols = (cols / NR) * NR;

    for (Index j = 0; j < peeled_cols; j += NR) {
      Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * NR : dst_base + j * depth;
      sve_copy_panel<Conjugate>(dst_panel, src + j, src_stride, depth, NR);
    }

    if (peeled_cols < cols) {
      const Index tail = cols - peeled_cols;
      Scalar* dst_panel =
          PanelMode ? dst_base + peeled_cols * dst_stride + dst_offset * tail : dst_base + peeled_cols * depth;
      sve_copy_panel<Conjugate>(dst_panel, src + peeled_cols, src_stride, depth, static_cast<int>(tail));
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
    sme_dispatch_pack<false, PanelMode>(
        &pack_direct, &pack_neon, &sme_pack_rhs_fallback<Scalar, NR, Index, DataMapper, Conjugate, PanelMode>, blockB,
        rhs, depth, cols, stride, offset, bool_constant<sme_mapper_has_direct_access<DataMapper, Index>::value>{});
  }
};

// Pack1/nr_ are pinned to the SME block sizes (rather than left open) so these
// specializations only match consumers that actually feed the SME gebp_kernel
// -- see "Mixed-scalar products" in the gebp_traits doc comment above.
#define EIGEN_SME_DECLARE_GEMM_PACKERS(SCALAR, MR, NR)                                                       \
  template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode> \
  struct gemm_pack_lhs<SCALAR, Index, DataMapper, MR, Pack2, Packet, ColMajor, Conjugate, PanelMode>         \
      : sme_pack_lhs_colmajor<SCALAR, MR, Index, DataMapper, Conjugate, PanelMode> {};                       \
                                                                                                             \
  template <typename Index, typename DataMapper, int Pack2, typename Packet, bool Conjugate, bool PanelMode> \
  struct gemm_pack_lhs<SCALAR, Index, DataMapper, MR, Pack2, Packet, RowMajor, Conjugate, PanelMode>         \
      : sme_pack_lhs_rowmajor<SCALAR, MR, Index, DataMapper, Conjugate, PanelMode> {};                       \
                                                                                                             \
  template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>                             \
  struct gemm_pack_rhs<SCALAR, Index, DataMapper, NR, ColMajor, Conjugate, PanelMode>                        \
      : sme_pack_rhs_colmajor<SCALAR, NR, Index, DataMapper, Conjugate, PanelMode> {};                       \
                                                                                                             \
  template <typename Index, typename DataMapper, bool Conjugate, bool PanelMode>                             \
  struct gemm_pack_rhs<SCALAR, Index, DataMapper, NR, RowMajor, Conjugate, PanelMode>                        \
      : sme_pack_rhs_rowmajor<SCALAR, NR, Index, DataMapper, Conjugate, PanelMode> {};

EIGEN_SME_DECLARE_GEMM_PACKERS(float, kSmeMr, kSmeNr)
EIGEN_SME_DECLARE_GEMM_PACKERS(std::complex<float>, kSmeMrC, kSmeNrC)
#ifdef EIGEN_VECTORIZE_SME_F64F64
EIGEN_SME_DECLARE_GEMM_PACKERS(double, kSmeMrD, kSmeNrD)
EIGEN_SME_DECLARE_GEMM_PACKERS(std::complex<double>, kSmeMrCD, kSmeNrCD)
#endif

#undef EIGEN_SME_DECLARE_GEMM_PACKERS

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
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const svbool_t pg_m = Traits::whilelt(0, pw);
  const svbool_t pg_n = Traits::whilelt(0, cw);
  // FMLA and FADD have equal latency/throughput on ARMv9 cores, and
  // multiplying by alpha=1.0 is exact in IEEE-754 so the FMLA form is
  // bit-identical to FADD in that case.  A single unconditional FMLA
  // keeps the store compact and measures no worse (and a few percent
  // better on small matrices, where the branch would otherwise disrupt
  // instruction scheduling).
  const Vec vzero = pset1<Vec>(Scalar(0));
  const Vec valpha = pset1<Vec>(alpha);

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
      Vec c0 = ploadu(pg_m, p0);
      Vec c1 = ploadu(pg_m, p1);
      pstoreu(pg_m, p0, pmadd(pg_m, sme_read_ver_za<TileId>(vzero, pg_m, (uint32_t)ci), valpha, c0));
      pstoreu(pg_m, p1, pmadd(pg_m, sme_read_ver_za<TileId>(vzero, pg_m, (uint32_t)(ci + 1)), valpha, c1));
    }
    if (ci < cw) {
      Scalar* pC = C + row_start + (col_start + ci) * C_stride_col;
      Vec vc = ploadu(pg_m, pC);
      pstoreu(pg_m, pC, pmadd(pg_m, sme_read_ver_za<TileId>(vzero, pg_m, (uint32_t)ci), valpha, vc));
    }
  } else if (C_stride_col == 1) {
    // Row-major C: extract horizontal slices (rows of the ZA tile)
    int ri = 0;
    for (; ri + 2 <= pw; ri += 2) {
      Scalar* p0 = C + (row_start + ri) * C_stride_row + col_start;
      Scalar* p1 = p0 + C_stride_row;
      Vec c0 = ploadu(pg_n, p0);
      Vec c1 = ploadu(pg_n, p1);
      pstoreu(pg_n, p0, pmadd(pg_n, sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)ri), valpha, c0));
      pstoreu(pg_n, p1, pmadd(pg_n, sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)(ri + 1)), valpha, c1));
    }
    if (ri < pw) {
      Scalar* pC = C + (row_start + ri) * C_stride_row + col_start;
      Vec vc = ploadu(pg_n, pC);
      pstoreu(pg_n, pC, pmadd(pg_n, sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)ri), valpha, vc));
    }
  } else {
    // General stride: extract rows to temp buffer, scatter to C.  scratch
    // holds one ZA row; every caller passes cw <= min(svl, nr) (a tile
    // never spans more than the logical block), so nr is a static
    // bound independent of the runtime svl.
    Scalar scratch[sme_block<Scalar>::nr];
    for (int ri = 0; ri < pw; ++ri) {
      Vec vres = sme_read_hor_za<TileId>(vzero, pg_n, (uint32_t)ri);
      vres = pmul(pg_n, vres, valpha);
      pstoreu(pg_n, scratch, vres);
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

// Full 2*svl x 2*svl block into column-major C: four columns per vertical four-slice read of the two tiles that
// share them, each column loaded and stored whole with two-vector accesses, all four loaded before any is stored.
template <int TileLo, int TileHi, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_tile_pair_colmajor(Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha,
                                                      int svl) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<Scalar>;
  const svcount_t pn = Traits::ptrue_c();
  const svbool_t pg = Traits::ptrue();
  const typename Traits::type valpha = pset1<typename Traits::type>(alpha);
  for (int c = 0; c < svl; c += 4) {
    const auto lo = sme_read_ver_za_vg4<TileLo, Scalar>(uint32_t(c));
    const auto hi = sme_read_ver_za_vg4<TileHi, Scalar>(uint32_t(c));
    Scalar* p = C + Index(c) * ldc;
    const auto c0 = ploadu_x2(pn, p), c1 = ploadu_x2(pn, p + ldc), c2 = ploadu_x2(pn, p + 2 * ldc),
               c3 = ploadu_x2(pn, p + 3 * ldc);
    pstoreu_x2(pn, p,
               pcreate(pmadd(pg, pget<0>(lo), valpha, pget<0>(c0)), pmadd(pg, pget<0>(hi), valpha, pget<1>(c0))));
    pstoreu_x2(pn, p + ldc,
               pcreate(pmadd(pg, pget<1>(lo), valpha, pget<0>(c1)), pmadd(pg, pget<1>(hi), valpha, pget<1>(c1))));
    pstoreu_x2(pn, p + 2 * ldc,
               pcreate(pmadd(pg, pget<2>(lo), valpha, pget<0>(c2)), pmadd(pg, pget<2>(hi), valpha, pget<1>(c2))));
    pstoreu_x2(pn, p + 3 * ldc,
               pcreate(pmadd(pg, pget<3>(lo), valpha, pget<0>(c3)), pmadd(pg, pget<3>(hi), valpha, pget<1>(c3))));
  }
}

template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_2x2_grid(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                            Scalar alpha, Index row_start, int rlo, int rhi, Index col_start, int clo,
                                            int chi) __arm_streaming __arm_inout("za") {
  const int svl = sme_packet_traits<Scalar>::size();
  if (svl >= 4 && C_stride_row == 1 && rlo == svl && rhi == svl && clo == svl && chi == svl) {
    Scalar* c0 = C + row_start + col_start * C_stride_col;
    sme_store_tile_pair_colmajor<0, 2>(c0, C_stride_col, alpha, svl);
    sme_store_tile_pair_colmajor<1, 3>(c0 + Index(svl) * C_stride_col, C_stride_col, alpha, svl);
    return;
  }
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
    typename sme_packet_traits<Scalar>::type a_lo, typename sme_packet_traits<Scalar>::type a_hi,
    typename sme_packet_traits<Scalar>::type b_lo,
    typename sme_packet_traits<Scalar>::type b_hi) __arm_streaming __arm_inout("za") {
  const svbool_t all = sme_packet_traits<Scalar>::ptrue();
  sme_mopa<0>(all, all, a_lo, b_lo);
  sme_mopa<1>(all, all, a_lo, b_hi);
  sme_mopa<2>(all, all, a_hi, b_lo);
  sme_mopa<3>(all, all, a_hi, b_hi);
}

/*****************************************************************************
 * Complex accumulator: a pair of ZA tiles holding the real and imaginary
 * halves of one grid cell.
 *
 * With a = ar + i*sa*ai and b = br + i*sb*bi, where sa is -1 when the LHS is
 * conjugated and sb likewise for the RHS,
 *
 *   re(a*b) = ar*br - (sa*sb) * ai*bi,     im(a*b) = sb * ar*bi + sa * ai*br,
 *
 * so all four real outer products differ only in whether they accumulate
 * (FMOPA) or subtract (FMOPS) -- a compile-time choice, with no work in the
 * depth loop and no separate conjugating packer.
 *
 * Slices come back out through sme_read_slice, which applies the complex
 * alpha and interleaves the halves with ZIP1/ZIP2 into the two vectors that
 * cover one slice's worth of contiguous complex results.  The kernel that
 * folds a real alpha keeps them deinterleaved instead (see
 * sme_accumulate_pair_real_alpha).
 *
 * A ZA tile number is an instruction immediate, so cells outside the tile grid
 * (complex<float> has two tile pairs, hence a single grid row) are dropped by
 * the InGrid specialization rather than by a runtime guard, which would still
 * have to name an in-range tile.
 *****************************************************************************/

// One slice of a tile pair, re-interleaved into the two vectors that cover its
// complex results: `lo` the first half, `hi` the second.  ScaleByAlpha applies
// the complex alpha, four predicated FP ops the caller skips when alpha is 1
// (see sme_store_za_pair) -- streaming-mode FP is de-rated enough on Apple M4
// that those four cost about as much as the rest of the slice.
template <typename RealScalar, int TileRe, int TileIm, bool Vertical, bool ScaleByAlpha>
EIGEN_ALWAYS_INLINE void sme_read_slice(
    svbool_t pg, typename sme_packet_traits<RealScalar>::type valpha_re,
    typename sme_packet_traits<RealScalar>::type valpha_im, typename sme_packet_traits<RealScalar>::type vzero,
    uint32_t slice, typename sme_packet_traits<RealScalar>::type& lo,
    typename sme_packet_traits<RealScalar>::type& hi) __arm_streaming __arm_inout("za") {
  using Vec = typename sme_packet_traits<RealScalar>::type;
  Vec re, im;
  EIGEN_IF_CONSTEXPR (Vertical) {
    re = sme_read_ver_za<TileRe>(vzero, pg, slice);
    im = sme_read_ver_za<TileIm>(vzero, pg, slice);
  } else {
    re = sme_read_hor_za<TileRe>(vzero, pg, slice);
    im = sme_read_hor_za<TileIm>(vzero, pg, slice);
  }
  EIGEN_IF_CONSTEXPR (ScaleByAlpha) {
    const Vec out_re = pnmadd(pg, im, valpha_im, pmul(pg, re, valpha_re));
    const Vec out_im = pmadd(pg, re, valpha_im, pmul(pg, im, valpha_re));
    lo = pzip1(out_re, out_im);
    hi = pzip2(out_re, out_im);
  } else {
    lo = pzip1(re, im);
    hi = pzip2(re, im);
  }
}

// Accumulate a tile pair's `slices` slices into C along its contiguous axis,
// `step` reals apart.  `lanes` is twice a slice's complex count; when it fits
// one vector the high half's predicate is empty, so its load and store are
// no-ops even though the destination has nothing at p + svl to point at.
template <typename RealScalar, int TileRe, int TileIm, bool Vertical, bool ScaleByAlpha, typename Index>
EIGEN_ALWAYS_INLINE void sme_accumulate_pair_impl(
    RealScalar* EIGEN_RESTRICT p, Index step, int slices, int lanes, svbool_t pg,
    typename sme_packet_traits<RealScalar>::type valpha_re, typename sme_packet_traits<RealScalar>::type valpha_im,
    typename sme_packet_traits<RealScalar>::type vzero) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<RealScalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size();
  const svbool_t pl0 = Traits::whilelt(0, lanes);
  const svbool_t pl1 = Traits::whilelt(svl, lanes);
  for (int s = 0; s < slices; ++s, p += step) {
    Vec lo, hi;
    sme_read_slice<RealScalar, TileRe, TileIm, Vertical, ScaleByAlpha>(pg, valpha_re, valpha_im, vzero, uint32_t(s), lo,
                                                                       hi);
    pstoreu(pl0, p, padd(pl0, ploadu(pl0, p), lo));
    // pl1 is all-false when one vector covers the slice, and an inactive lane
    // neither reads nor writes -- so this needs no `lanes > svl` guard, only an
    // address the destination is allowed to form.
    RealScalar* EIGEN_RESTRICT phi = sme_offset(p, Index(svl));
    pstoreu(pl1, phi, padd(pl1, ploadu(pl1, phi), hi));
  }
}

// A real alpha on slices spanning two vectors, in the FoldRealAlpha kernel:
// LD2/ST2 keep C deinterleaved with one predicate lane per complex result, so
// each half is one FMA, c + alpha*acc, as in the real-scalar store.  A single
// FMA rounds once, so it overflows only when the exact result does.  This
// covers C.noalias() -= A*B, whose alpha is -1.
//
// `limit`, the slices left in the C block from the first one, bounds the
// prefetch to the block (unbounded, it measured 0.45-0.5x on 32^3 and 64^3).
// The prefetch runs 8 slices ahead of the C load that heads each FMA; without
// it the fold ran at 0.94x of that at 1024^3.  The barrier keeps the prefetch
// address setup inside its branch.
//
// A complex alpha keeps sme_accumulate_pair_impl, which forms alpha*acc before
// adding C.  Folding C into either of its two FMAs, e.g. (c_re + re*ar) - im*ai,
// saves two ops per slice but overflows when C and the first product exceed
// the range before the second cancels them (!3139 review: acc = (h,h),
// alpha = (1,1), C = (3h,0)).  On Apple M4, LD2/ST2 measured no gain with
// unchanged arithmetic and 0.68-0.75x on one-vector slices, and FCMLA 0.70x
// (#3129).
template <typename RealScalar, int TileRe, int TileIm, bool Vertical, typename Index>
EIGEN_ALWAYS_INLINE void sme_accumulate_pair_real_alpha(
    RealScalar* EIGEN_RESTRICT p, Index step, int slices, Index limit, svbool_t pg,
    typename sme_packet_traits<RealScalar>::type valpha,
    typename sme_packet_traits<RealScalar>::type vzero) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<RealScalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size();
  constexpr int kPrefetchSlices = 8;
  for (int s = 0; s < slices; ++s, p += step) {
    if (s + kPrefetchSlices < limit) {
      RealScalar* pf = p;
      EIGEN_OPTIMIZATION_BARRIER(pf)
      pf = sme_offset(pf, Index(kPrefetchSlices) * step);
      __builtin_prefetch(pf, 1, 3);
      __builtin_prefetch(sme_offset(pf, Index(svl)), 1, 3);
    }
    Vec re, im;
    EIGEN_IF_CONSTEXPR (Vertical) {
      re = sme_read_ver_za<TileRe>(vzero, pg, uint32_t(s));
      im = sme_read_ver_za<TileIm>(vzero, pg, uint32_t(s));
    } else {
      re = sme_read_hor_za<TileRe>(vzero, pg, uint32_t(s));
      im = sme_read_hor_za<TileIm>(vzero, pg, uint32_t(s));
    }
    const typename Traits::type_x2 c = pld2(pg, p);
    pst2(pg, p, pcreate(pmadd(pg, re, valpha, pget<0>(c)), pmadd(pg, im, valpha, pget<1>(c))));
  }
}

// FoldRealAlpha selects the kernel instantiation.  The folding kernel only
// sees a real alpha other than 1 (see sme_gebp_dispatch); the other kernel
// keeps exactly the unscaled and complex-alpha paths, so the fold's code never
// reaches products that cannot use it.
template <typename RealScalar, int TileRe, int TileIm, bool Vertical, bool FoldRealAlpha, typename Index>
EIGEN_ALWAYS_INLINE void sme_accumulate_pair(
    bool scale_by_alpha, RealScalar* EIGEN_RESTRICT p, Index step, int slices, Index limit, int lanes, svbool_t pg,
    typename sme_packet_traits<RealScalar>::type valpha_re, typename sme_packet_traits<RealScalar>::type valpha_im,
    typename sme_packet_traits<RealScalar>::type vzero) __arm_streaming __arm_inout("za") {
  if (FoldRealAlpha && lanes > sme_packet_traits<RealScalar>::size()) {
    sme_accumulate_pair_real_alpha<RealScalar, TileRe, TileIm, Vertical>(p, step, slices, limit, pg, valpha_re, vzero);
  } else if (scale_by_alpha) {
    sme_accumulate_pair_impl<RealScalar, TileRe, TileIm, Vertical, true>(p, step, slices, lanes, pg, valpha_re,
                                                                         valpha_im, vzero);
  } else {
    sme_accumulate_pair_impl<RealScalar, TileRe, TileIm, Vertical, false>(p, step, slices, lanes, pg, valpha_re,
                                                                          valpha_im, vzero);
  }
}

// Store one complex tile pair back to C.  `pw` is the row-predicate width for
// this cell and `cw` the column one, both <= the runtime svl.
template <typename RealScalar, int TileRe, int TileIm, bool FoldRealAlpha, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_za_pair(std::complex<RealScalar>* EIGEN_RESTRICT C, Index C_stride_row,
                                           Index C_stride_col, std::complex<RealScalar> alpha, Index row_start, int pw,
                                           Index col_start, int cw, Index rows,
                                           Index cols) __arm_streaming __arm_inout("za") {
  using Scalar = std::complex<RealScalar>;
  using Traits = sme_packet_traits<RealScalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size();
  const svbool_t pg_m = Traits::whilelt(0, pw);
  const svbool_t pg_n = Traits::whilelt(0, cw);
  const Vec vzero = pset1<Vec>(RealScalar(0));
  // std::complex's accessors are ordinary functions, which clang cannot inline
  // into a streaming context; the resulting mode switch would sit in this loop.
  // A complex is layout-compatible with its two-element real array, so read the
  // parts through that view instead.
  const RealScalar* alpha_parts = reinterpret_cast<const RealScalar*>(&alpha);
  const Vec valpha_re = pset1<Vec>(alpha_parts[0]);
  const Vec valpha_im = pset1<Vec>(alpha_parts[1]);
  // Scaling by 1 + 0i is exact, so skipping it is bit-identical -- and it is by
  // far the common case, since a plain product carries alpha = 1.
  // With FoldRealAlpha, sme_gebp_dispatch has already established that alpha
  // is real and not 1.
  const bool scale = !(alpha_parts[0] == RealScalar(1) && alpha_parts[1] == RealScalar(0));
  RealScalar* EIGEN_RESTRICT rC = reinterpret_cast<RealScalar*>(C);

  if (C_stride_row == 1) {
    // Column-major C: vertical slices are the tile pair's columns, and one
    // slice is pw contiguous complex results, i.e. 2*pw reals.
    RealScalar* p = rC + Index(2) * (row_start + col_start * C_stride_col);
    sme_accumulate_pair<RealScalar, TileRe, TileIm, true, FoldRealAlpha>(
        scale, p, Index(2) * C_stride_col, cw, cols - col_start, 2 * pw, pg_m, valpha_re, valpha_im, vzero);
  } else if (C_stride_col == 1) {
    // Row-major C: horizontal slices are the tile pair's rows.
    RealScalar* p = rC + Index(2) * (row_start * C_stride_row + col_start);
    sme_accumulate_pair<RealScalar, TileRe, TileIm, false, FoldRealAlpha>(
        scale, p, Index(2) * C_stride_row, pw, rows - row_start, 2 * cw, pg_n, valpha_re, valpha_im, vzero);
  } else {
    // General stride: interleave a row into a temp buffer, scatter to C.  Every
    // caller passes cw <= min(svl, nr), so nr is a static bound on the buffer,
    // independent of the runtime svl.  This path is scalar anyway, so it always
    // takes the scaling form.
    Scalar scratch[sme_block<Scalar>::nr];
    RealScalar* rscratch = reinterpret_cast<RealScalar*>(scratch);
    const int lanes = 2 * cw;
    const svbool_t pl0 = Traits::whilelt(0, lanes);
    const svbool_t pl1 = Traits::whilelt(svl, lanes);
    for (int ri = 0; ri < pw; ++ri) {
      Vec lo, hi;
      sme_read_slice<RealScalar, TileRe, TileIm, false, true>(pg_n, valpha_re, valpha_im, vzero, uint32_t(ri), lo, hi);
      pstoreu(pl0, rscratch, lo);
      // scratch is nr complex, i.e. 2*nr >= 2*svl reals, so rscratch + svl is
      // always in bounds; pl1 is all-false when one vector already covers cw.
      pstoreu(pl1, rscratch + svl, hi);
      for (int ci = 0; ci < cw; ++ci) {
        C[(row_start + ri) * C_stride_row + (col_start + ci) * C_stride_col] += scratch[ci];
      }
    }
  }
}

// Grid cell (R, C) of a complex block: its tile pair, the four signed outer
// products that feed it, and its store.
template <typename Scalar, int R, int C, bool ConjLhs, bool ConjRhs,
          bool InGrid = (R < sme_block<Scalar>::kGridRows && C < sme_block<Scalar>::kGridCols)>
struct sme_complex_cell {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Vec = typename sme_packet_traits<RealScalar>::type;
  static constexpr int kTileRe = 2 * (R * sme_block<Scalar>::kGridCols + C);
  static constexpr int kTileIm = kTileRe + 1;

  static EIGEN_ALWAYS_INLINE void accumulate(svbool_t pm, svbool_t pn, Vec a_re, Vec a_im, Vec b_re,
                                             Vec b_im) __arm_streaming __arm_inout("za") {
    sme_mopa_signed<kTileRe, false>(pm, pn, a_re, b_re);
    sme_mopa_signed<kTileRe, ConjLhs == ConjRhs>(pm, pn, a_im, b_im);
    sme_mopa_signed<kTileIm, ConjRhs>(pm, pn, a_re, b_im);
    sme_mopa_signed<kTileIm, ConjLhs>(pm, pn, a_im, b_re);
  }

  template <bool FoldRealAlpha, typename Index>
  static EIGEN_ALWAYS_INLINE void store(Scalar* EIGEN_RESTRICT dst, Index C_stride_row, Index C_stride_col,
                                        Scalar alpha, Index row_start, int pw, Index col_start, int cw, Index rows,
                                        Index cols) __arm_streaming __arm_inout("za") {
    sme_store_za_pair<RealScalar, kTileRe, kTileIm, FoldRealAlpha>(dst, C_stride_row, C_stride_col, alpha, row_start,
                                                                   pw, col_start, cw, rows, cols);
  }
};

template <typename Scalar, int R, int C, bool ConjLhs, bool ConjRhs>
struct sme_complex_cell<Scalar, R, C, ConjLhs, ConjRhs, false> {
  using Vec = typename sme_packet_traits<typename NumTraits<Scalar>::Real>::type;
  static EIGEN_ALWAYS_INLINE void accumulate(svbool_t, svbool_t, Vec, Vec, Vec, Vec) __arm_streaming __arm_inout("za") {
  }
  template <bool FoldRealAlpha, typename Index>
  static EIGEN_ALWAYS_INLINE void store(Scalar*, Index, Index, Scalar, Index, int, Index, int, Index,
                                        Index) __arm_streaming __arm_inout("za") {}
};

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

template <bool ConjLhs, bool ConjRhs, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                     const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blB, Index depth,
                                     Scalar alpha, Index row_start, int pw, Index col_start, int cw,
                                     Index a_step) __arm_streaming __arm_inout("za") {
  // Conjugation is the identity on real scalars, so this overload ignores it.
  // a_step is the distance between A's depth steps: pw for a packed panel, the
  // column stride for a ColMajor source read in place.
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size();

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
        if (a_step == Index(pw)) {
          for (; k < depth_4; k += 4) {
            typename Traits::type_x4 va_01 = ploadu_x4(pn, &blA[k * pw]);
            typename Traits::type_x4 vb_01 = ploadu_x4(pn, &blB[k * cw]);

            // d0
            outer_product_2x2<Scalar>(pget<0>(va_01), pget<1>(va_01), pget<0>(vb_01), pget<1>(vb_01));
            // d1
            outer_product_2x2<Scalar>(pget<2>(va_01), pget<3>(va_01), pget<2>(vb_01), pget<3>(vb_01));

            typename Traits::type_x4 va_23 = ploadu_x4(pn, &blA[(k + 2) * pw]);
            typename Traits::type_x4 vb_23 = ploadu_x4(pn, &blB[(k + 2) * cw]);

            // d2
            outer_product_2x2<Scalar>(pget<0>(va_23), pget<1>(va_23), pget<0>(vb_23), pget<1>(vb_23));
            // d3
            outer_product_2x2<Scalar>(pget<2>(va_23), pget<3>(va_23), pget<2>(vb_23), pget<3>(vb_23));
          }
        } else {
          // A read in place: its depth steps are a_step apart, one x2 load each.
          for (; k < depth_4; k += 4) {
            typename Traits::type_x2 va_0 = ploadu_x2(pn, &blA[k * a_step]);
            typename Traits::type_x2 va_1 = ploadu_x2(pn, &blA[(k + 1) * a_step]);
            typename Traits::type_x4 vb_01 = ploadu_x4(pn, &blB[k * cw]);
            outer_product_2x2<Scalar>(pget<0>(va_0), pget<1>(va_0), pget<0>(vb_01), pget<1>(vb_01));
            outer_product_2x2<Scalar>(pget<0>(va_1), pget<1>(va_1), pget<2>(vb_01), pget<3>(vb_01));
            typename Traits::type_x2 va_2 = ploadu_x2(pn, &blA[(k + 2) * a_step]);
            typename Traits::type_x2 va_3 = ploadu_x2(pn, &blA[(k + 3) * a_step]);
            typename Traits::type_x4 vb_23 = ploadu_x4(pn, &blB[(k + 2) * cw]);
            outer_product_2x2<Scalar>(pget<0>(va_2), pget<1>(va_2), pget<0>(vb_23), pget<1>(vb_23));
            outer_product_2x2<Scalar>(pget<0>(va_3), pget<1>(va_3), pget<2>(vb_23), pget<3>(vb_23));
          }
        }
        // Depth tail: one x2 load per side per step.
        for (; k < depth; ++k) {
          typename Traits::type_x2 va = ploadu_x2(pn, &blA[k * a_step]);
          typename Traits::type_x2 vb = ploadu_x2(pn, &blB[k * cw]);
          outer_product_2x2<Scalar>(pget<0>(va), pget<1>(va), pget<0>(vb), pget<1>(vb));
        }
      } else {
        for (Index k = 0; k < depth; ++k) {
          Vec a_lo = ploadu(pg_rlo, &blA[k * a_step + rt]);
          Vec b_lo = ploadu(pg_clo, &blB[k * cw + ct]);

          Vec a_hi = ploadu(pg_rhi, sme_offset(blA, k * a_step + rt + svl));
          Vec b_hi = ploadu(pg_chi, sme_offset(blB, k * cw + ct + svl));

          sme_mopa<0>(pg_rlo, pg_clo, a_lo, b_lo);
          if (svptest_any(pg_chi, pg_chi)) sme_mopa<1>(pg_rlo, pg_chi, a_lo, b_hi);
          if (svptest_any(pg_rhi, pg_rhi)) {
            sme_mopa<2>(pg_rhi, pg_clo, a_hi, b_lo);
            if (svptest_any(pg_chi, pg_chi)) sme_mopa<3>(pg_rhi, pg_chi, a_hi, b_hi);
          }
        }
      }

      // Store the (up to) 2x2 grid of tiles for this sub-block pass.
      sme_store_2x2_grid(C, C_stride_row, C_stride_col, alpha, row_start + rt, rlo, rhi, col_start + ct, clo, chi);
    }
  }
}

/*****************************************************************************
 * sme_store_complex_grid -- store the (up to) 2x2 grid of tile pairs, exactly
 * as sme_store_2x2_grid does for single tiles.  Cells outside the grid are
 * dropped at compile time by sme_complex_cell, so a narrower grid simply never
 * reaches them (its hi widths are structurally zero).
 *****************************************************************************/
template <typename Scalar, bool ConjLhs, bool ConjRhs, bool FoldRealAlpha, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_complex_grid(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                                Scalar alpha, Index row_start, int rlo, int rhi, Index col_start,
                                                int clo, int chi, Index rows,
                                                Index cols) __arm_streaming __arm_inout("za") {
  const int svl = sme_packet_traits<typename NumTraits<Scalar>::Real>::size();
  sme_complex_cell<Scalar, 0, 0, ConjLhs, ConjRhs>::template store<FoldRealAlpha>(
      C, C_stride_row, C_stride_col, alpha, row_start, rlo, col_start, clo, rows, cols);
  if (chi > 0) {
    sme_complex_cell<Scalar, 0, 1, ConjLhs, ConjRhs>::template store<FoldRealAlpha>(
        C, C_stride_row, C_stride_col, alpha, row_start, rlo, col_start + svl, chi, rows, cols);
  }
  if (rhi > 0) {
    sme_complex_cell<Scalar, 1, 0, ConjLhs, ConjRhs>::template store<FoldRealAlpha>(
        C, C_stride_row, C_stride_col, alpha, row_start + svl, rhi, col_start, clo, rows, cols);
    if (chi > 0) {
      sme_complex_cell<Scalar, 1, 1, ConjLhs, ConjRhs>::template store<FoldRealAlpha>(
          C, C_stride_row, C_stride_col, alpha, row_start + svl, rhi, col_start + svl, chi, rows, cols);
    }
  }
}

/*****************************************************************************
 * sme_process, complex overload -- same sub-block structure over a grid of
 * complex accumulators, each a pair of ZA tiles (see sme_complex_cell).
 *
 * The packed panels are read through their real view: one depth step is `pw`
 * (`cw`) reals followed by as many imaginary ones, so a cell's four operands
 * are four contiguous predicated loads at a fixed offset apart, and the four
 * outer products they feed reuse all of them.
 *****************************************************************************/
template <bool ConjLhs, bool ConjRhs, bool FoldRealAlpha, typename RealScalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process(std::complex<RealScalar>* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                     const std::complex<RealScalar>* EIGEN_RESTRICT blA,
                                     const std::complex<RealScalar>* EIGEN_RESTRICT blB, Index depth,
                                     std::complex<RealScalar> alpha, Index row_start, int pw, Index col_start, int cw,
                                     Index lhs_step, Index rows, Index cols) __arm_streaming __arm_inout("za") {
  // Complex panels are always packed (split real/imaginary halves): lhs_step is pw.
  EIGEN_UNUSED_VARIABLE(lhs_step);
  using Scalar = std::complex<RealScalar>;
  using Traits = sme_packet_traits<RealScalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size();
  constexpr int GridRows = sme_block<Scalar>::kGridRows;
  constexpr int GridCols = sme_block<Scalar>::kGridCols;
  using Cell00 = sme_complex_cell<Scalar, 0, 0, ConjLhs, ConjRhs>;
  using Cell01 = sme_complex_cell<Scalar, 0, 1, ConjLhs, ConjRhs>;
  using Cell10 = sme_complex_cell<Scalar, 1, 0, ConjLhs, ConjRhs>;
  using Cell11 = sme_complex_cell<Scalar, 1, 1, ConjLhs, ConjRhs>;

  const RealScalar* EIGEN_RESTRICT rA = reinterpret_cast<const RealScalar*>(blA);
  const RealScalar* EIGEN_RESTRICT rB = reinterpret_cast<const RealScalar*>(blB);
  const Index a_step = Index(2 * pw);
  const Index b_step = Index(2 * cw);

  for (int rt = 0; rt < pw; rt += GridRows * svl) {
    const int rpw = sme_min(pw - rt, GridRows * svl);
    const int r0 = sme_min(rpw, svl);
    const int r1 = rpw - r0;
    const svbool_t pg_r0 = Traits::whilelt(0, rpw);
    const svbool_t pg_r1 = Traits::whilelt(svl, rpw);

    for (int ct = 0; ct < cw; ct += GridCols * svl) {
      const int cpw = sme_min(cw - ct, GridCols * svl);
      const int c0 = sme_min(cpw, svl);
      const int c1 = cpw - c0;
      const svbool_t pg_c0 = Traits::whilelt(0, cpw);
      const svbool_t pg_c1 = Traits::whilelt(svl, cpw);

      svzero_za();
      for (Index k = 0; k < depth; ++k) {
        const RealScalar* pa = rA + k * a_step + Index(rt);
        const RealScalar* pb = rB + k * b_step + Index(ct);
        // The second-tile loads are unconditional: their predicates are empty
        // when the grid is a single tile wide or tall, so they touch no memory.
        // Grouping them lets the compiler issue them in parallel, and gating the
        // outer products on svptest_any keeps the depth loop unswitched without
        // materialising the r1/c1 counts here.
        const Vec a0_re = ploadu(pg_r0, pa);
        const Vec a0_im = ploadu(pg_r0, pa + pw);
        const Vec b0_re = ploadu(pg_c0, pb);
        const Vec b0_im = ploadu(pg_c0, pb + cw);
        const Vec a1_re = ploadu(pg_r1, sme_offset(pa, Index(svl)));
        const Vec a1_im = ploadu(pg_r1, sme_offset(pa, Index(pw) + Index(svl)));
        const Vec b1_re = ploadu(pg_c1, sme_offset(pb, Index(svl)));
        const Vec b1_im = ploadu(pg_c1, sme_offset(pb, Index(cw) + Index(svl)));
        Cell00::accumulate(pg_r0, pg_c0, a0_re, a0_im, b0_re, b0_im);
        if (svptest_any(pg_c1, pg_c1)) {
          Cell01::accumulate(pg_r0, pg_c1, a0_re, a0_im, b1_re, b1_im);
        }
        if (svptest_any(pg_r1, pg_r1)) {
          Cell10::accumulate(pg_r1, pg_c0, a1_re, a1_im, b0_re, b0_im);
          if (svptest_any(pg_c1, pg_c1)) {
            Cell11::accumulate(pg_r1, pg_c1, a1_re, a1_im, b1_re, b1_im);
          }
        }
      }

      sme_store_complex_grid<Scalar, ConjLhs, ConjRhs, FoldRealAlpha>(
          C, C_stride_row, C_stride_col, alpha, row_start + rt, r0, r1, col_start + ct, c0, c1, rows, cols);
    }
  }
}

// A RHS panel of at most svl columns fills only one tile column of the 2 x 2 grid, so the four tiles are stacked along
// M instead: a full LHS panel (tiles 0 and 1) and the next one (tiles 2 and 3, pw1 rows) against the same B vector.
template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_narrow(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                            const Scalar* EIGEN_RESTRICT blA0, Index step0,
                                            const Scalar* EIGEN_RESTRICT blA1, Index step1, int pw1,
                                            const Scalar* EIGEN_RESTRICT blB, int cw, Index depth, Scalar alpha,
                                            Index row_start, Index col_start) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size();
  const svcount_t pc = Traits::ptrue_c();
  const svbool_t all = Traits::ptrue();
  const svbool_t pn = Traits::whilelt(0, cw);
  const svbool_t p1lo = Traits::whilelt(0, pw1);
  const svbool_t p1hi = Traits::whilelt(svl, pw1);
  svzero_za();
  for (Index k = 0; k < depth; ++k) {
    const auto a0 = ploadu_x2(pc, blA0 + k * step0);
    const Vec a1lo = ploadu(p1lo, blA1 + k * step1);
    const Vec a1hi = ploadu(p1hi, sme_offset(blA1, k * step1 + svl));
    const Vec b = ploadu(pn, blB + k * cw);
    sme_mopa<0>(all, pn, pget<0>(a0), b);
    sme_mopa<1>(all, pn, pget<1>(a0), b);
    sme_mopa<2>(p1lo, pn, a1lo, b);
    sme_mopa<3>(p1hi, pn, a1hi, b);
  }
  sme_store_za_tile<Scalar, 0>(C, C_stride_row, C_stride_col, alpha, row_start, svl, col_start, cw);
  sme_store_za_tile<Scalar, 1>(C, C_stride_row, C_stride_col, alpha, row_start + svl, svl, col_start, cw);
  sme_store_za_tile<Scalar, 2>(C, C_stride_row, C_stride_col, alpha, row_start + 2 * svl, sme_min(pw1, svl), col_start,
                               cw);
  if (pw1 > svl)
    sme_store_za_tile<Scalar, 3>(C, C_stride_row, C_stride_col, alpha, row_start + 3 * svl, pw1 - svl, col_start, cw);
}

// Tile Dst += tile Src, four columns per step.
template <int Dst, int Src, typename Scalar>
EIGEN_ALWAYS_INLINE void sme_fold_tile() __arm_streaming __arm_inout("za") {
  const svbool_t all = sme_packet_traits<Scalar>::ptrue();
  for (int c = 0; c < sme_packet_traits<Scalar>::size(); c += 4) {
    const auto d = sme_read_ver_za_vg4<Dst, Scalar>(uint32_t(c)), s = sme_read_ver_za_vg4<Src, Scalar>(uint32_t(c));
    sme_write_ver_za_vg4<Dst>(uint32_t(c),
                              pcreate(padd(all, pget<0>(d), pget<0>(s)), padd(all, pget<1>(d), pget<1>(s)),
                                      padd(all, pget<2>(d), pget<2>(s)), padd(all, pget<3>(d), pget<3>(s))));
  }
}

// A block that fills one or two tiles (pw or cw <= svl): FMOPAs into one tile wait on each other, so consecutive
// depth steps go to the spare tiles and the partial sums are folded before the store.
template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_split(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                           const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blB,
                                           Index depth, Scalar alpha, Index row_start, int pw, Index col_start, int cw,
                                           Index a_step) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const int svl = Traits::size();
  const svbool_t pr0 = Traits::whilelt(0, pw), pr1 = Traits::whilelt(svl, pw);
  const svbool_t pc0 = Traits::whilelt(0, cw), pc1 = Traits::whilelt(svl, cw);
  svzero_za();
  Index k = 0;
  if (pw <= svl && cw <= svl) {
    for (; k + 4 <= depth; k += 4) {
      const Vec a0 = ploadu(pr0, blA + k * a_step), a1 = ploadu(pr0, blA + (k + 1) * a_step);
      const Vec a2 = ploadu(pr0, blA + (k + 2) * a_step), a3 = ploadu(pr0, blA + (k + 3) * a_step);
      const Vec b0 = ploadu(pc0, blB + k * cw), b1 = ploadu(pc0, blB + (k + 1) * cw);
      const Vec b2 = ploadu(pc0, blB + (k + 2) * cw), b3 = ploadu(pc0, blB + (k + 3) * cw);
      sme_mopa<0>(pr0, pc0, a0, b0);
      sme_mopa<1>(pr0, pc0, a1, b1);
      sme_mopa<2>(pr0, pc0, a2, b2);
      sme_mopa<3>(pr0, pc0, a3, b3);
    }
    for (; k < depth; ++k) sme_mopa<0>(pr0, pc0, ploadu(pr0, blA + k * a_step), ploadu(pc0, blB + k * cw));
    sme_fold_tile<0, 1, Scalar>();
    sme_fold_tile<2, 3, Scalar>();
    sme_fold_tile<0, 2, Scalar>();
    sme_store_za_tile<Scalar, 0>(C, C_stride_row, C_stride_col, alpha, row_start, pw, col_start, cw);
  } else if (pw <= svl) {
    // Tiles 0 and 1 take the even depth steps, 2 and 3 the odd ones.
    for (; k + 2 <= depth; k += 2) {
      const Vec a0 = ploadu(pr0, blA + k * a_step), a1 = ploadu(pr0, blA + (k + 1) * a_step);
      const Vec b0 = ploadu(pc0, blB + k * cw), b0h = ploadu(pc1, blB + k * cw + svl);
      const Vec b1 = ploadu(pc0, blB + (k + 1) * cw), b1h = ploadu(pc1, blB + (k + 1) * cw + svl);
      sme_mopa<0>(pr0, pc0, a0, b0);
      sme_mopa<1>(pr0, pc1, a0, b0h);
      sme_mopa<2>(pr0, pc0, a1, b1);
      sme_mopa<3>(pr0, pc1, a1, b1h);
    }
    if (k < depth) {
      const Vec a0 = ploadu(pr0, blA + k * a_step);
      sme_mopa<0>(pr0, pc0, a0, ploadu(pc0, blB + k * cw));
      sme_mopa<1>(pr0, pc1, a0, ploadu(pc1, blB + k * cw + svl));
    }
    sme_fold_tile<0, 2, Scalar>();
    sme_fold_tile<1, 3, Scalar>();
    sme_store_2x2_grid(C, C_stride_row, C_stride_col, alpha, row_start, pw, 0, col_start, svl, cw - svl);
  } else {
    // cw <= svl < pw: tiles 0 and 2 take the even depth steps, 1 and 3 the odd ones.
    for (; k + 2 <= depth; k += 2) {
      const Vec a0 = ploadu(pr0, blA + k * a_step), a0h = ploadu(pr1, blA + k * a_step + svl);
      const Vec a1 = ploadu(pr0, blA + (k + 1) * a_step), a1h = ploadu(pr1, blA + (k + 1) * a_step + svl);
      const Vec b0 = ploadu(pc0, blB + k * cw), b1 = ploadu(pc0, blB + (k + 1) * cw);
      sme_mopa<0>(pr0, pc0, a0, b0);
      sme_mopa<2>(pr1, pc0, a0h, b0);
      sme_mopa<1>(pr0, pc0, a1, b1);
      sme_mopa<3>(pr1, pc0, a1h, b1);
    }
    if (k < depth) {
      const Vec b0 = ploadu(pc0, blB + k * cw);
      sme_mopa<0>(pr0, pc0, ploadu(pr0, blA + k * a_step), b0);
      sme_mopa<2>(pr1, pc0, ploadu(pr1, blA + k * a_step + svl), b0);
    }
    sme_fold_tile<0, 1, Scalar>();
    sme_fold_tile<2, 3, Scalar>();
    sme_store_2x2_grid(C, C_stride_row, C_stride_col, alpha, row_start, svl, pw - svl, col_start, cw, 0);
  }
}

// One pw x cw block: sme_process_split when it fills at most two tiles of the 2 x 2 grid; complex blocks keep
// sme_process. `split_ok` (sme_split_ok) holds once per call, outside the block loops.
template <bool ConjLhs, bool ConjRhs, bool FoldRealAlpha, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_block(Scalar* C, Index rs, Index cs, const Scalar* blA, const Scalar* blB,
                                           Index depth, Scalar alpha, Index row_start, int pw, Index col_start, int cw,
                                           Index a_step, bool split_ok, Index,
                                           Index) __arm_streaming __arm_inout("za") {
  const int svl = sme_packet_traits<Scalar>::size();
  if (split_ok && (pw <= svl || cw <= svl))
    sme_process_split(C, rs, cs, blA, blB, depth, alpha, row_start, pw, col_start, cw, a_step);
  else
    sme_process<ConjLhs, ConjRhs>(C, rs, cs, blA, blB, depth, alpha, row_start, pw, col_start, cw, a_step);
}
template <bool ConjLhs, bool ConjRhs, bool FoldRealAlpha, typename RealScalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_block(std::complex<RealScalar>* C, Index rs, Index cs,
                                           const std::complex<RealScalar>* blA, const std::complex<RealScalar>* blB,
                                           Index depth, std::complex<RealScalar> alpha, Index row_start, int pw,
                                           Index col_start, int cw, Index a_step, bool, Index rows,
                                           Index cols) __arm_streaming __arm_inout("za") {
  sme_process<ConjLhs, ConjRhs, FoldRealAlpha>(C, rs, cs, blA, blB, depth, alpha, row_start, pw, col_start, cw, a_step,
                                               rows, cols);
}

// Whether the vector length gives the tile shapes the block kernels assume: the depth-split kernel covers one 2 x 2
// grid of blocks up to MR x NR with four-slice tile folds, and the narrow kernel stacks two LHS panels of 2*svl rows.
template <typename Scalar>
EIGEN_ALWAYS_INLINE bool sme_split_ok() __arm_streaming {
  const int svl = sme_packet_traits<typename NumTraits<Scalar>::Real>::size();
  return svl >= 4 && sme_block<Scalar>::mr <= 2 * svl && sme_block<Scalar>::nr <= 2 * svl;
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE bool sme_narrow_ok() __arm_streaming {
  return sme_block<Scalar>::mr == 2 * sme_packet_traits<typename NumTraits<Scalar>::Real>::size();
}

// sme_process_narrow for real scalars; complex panels keep the 2 x 2 path.
template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_narrow_dispatch(Scalar* C, Index rs, Index cs, const Scalar* blA0, Index step0,
                                                     const Scalar* blA1, Index step1, int pw1, const Scalar* blB,
                                                     int cw, Index depth, Scalar alpha, Index row_start,
                                                     Index col_start) __arm_streaming __arm_inout("za") {
  sme_process_narrow(C, rs, cs, blA0, step0, blA1, step1, pw1, blB, cw, depth, alpha, row_start, col_start);
}
template <typename RealScalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_narrow_dispatch(std::complex<RealScalar>*, Index, Index,
                                                     const std::complex<RealScalar>*, Index,
                                                     const std::complex<RealScalar>*, Index, int,
                                                     const std::complex<RealScalar>*, int, Index,
                                                     std::complex<RealScalar>, Index,
                                                     Index) __arm_streaming __arm_inout("za") {}

// Core-side prefetch into L2 of the column-major C block the next sme_process call reads and writes.
template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_prefetch_next_c(const Scalar* C, Index C_stride_row, Index C_stride_col, Index i,
                                                    Index j, Index rows, Index cols, int mr,
                                                    int nr) __arm_streaming_compatible {
  if (C_stride_row != 1 || j >= cols) return;
  const Index h = sme_min(rows - i, Index(mr)), w = sme_min(cols - j, Index(nr));
  const Index bytes = h * Index(sizeof(Scalar));
  for (Index c = 0; c < w; ++c)
    for (Index b = 0; b < bytes; b += 128)
      __builtin_prefetch(reinterpret_cast<const char*>(C + i + (j + c) * C_stride_col) + b, 1, 2);
}

template <typename Scalar, bool ConjLhs, bool ConjRhs, bool FoldRealAlpha = false, typename Index>
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
  // both the logical block size and the load stride to the block kernels.
  // The narrow kernel stacks two LHS panels of exactly 2*svl rows.
  // A small C block (256 KB or less) gains nothing from the prefetch and pays its instructions on every call.
  const bool prefetch_c = rows * cols * Index(sizeof(Scalar)) > Index(256 * 1024);
  const bool split_ok = sme_split_ok<Scalar>(), narrow_ok = sme_narrow_ok<Scalar>();
  for (Index j = 0; j < cols; j += NR) {
    const int cw = static_cast<int>(sme_min(cols - j, Index(NR)));
    const Scalar* blB = blockB + j * strideB + offsetB * cw;

    Index i = 0;
    EIGEN_IF_CONSTEXPR (!NumTraits<Scalar>::IsComplex) {
      if (narrow_ok && cw <= sme_packet_traits<typename NumTraits<Scalar>::Real>::size())
        for (; i + MR < rows; i += 2 * MR) {
          const int pw1 = static_cast<int>(sme_min(rows - i - MR, Index(MR)));
          sme_process_narrow_dispatch(C, C_stride_row, C_stride_col, blockA + i * strideA + offsetA * MR, Index(MR),
                                      blockA + (i + MR) * strideA + offsetA * pw1, Index(pw1), pw1, blB, cw, depth,
                                      alpha, i, j);
        }
    }
    for (; i < rows; i += MR) {
      const int pw = static_cast<int>(sme_min(rows - i, Index(MR)));
      const Scalar* blA = blockA + i * strideA + offsetA * pw;
      if (prefetch_c)
        sme_prefetch_next_c(C, C_stride_row, C_stride_col, i + MR < rows ? i + MR : Index(0),
                            i + MR < rows ? j : j + NR, rows, cols, MR, NR);
      sme_process_block<ConjLhs, ConjRhs, FoldRealAlpha>(C, C_stride_row, C_stride_col, blA, blB, depth, alpha, i, pw,
                                                         j, cw, Index(pw), split_ok, rows, cols);
    }
  }
}

// Picks the sme_gebp_impl instantiation.  A complex product with a real alpha
// other than 1, e.g. the -1 of C.noalias() -= A*B, gets the kernel that folds
// C into one FMA per component (see sme_accumulate_pair_real_alpha); every
// other product gets the kernel without that path.  Real scalars have one.
template <typename Scalar, bool ConjLhs, bool ConjRhs, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct sme_gebp_dispatch {
  template <typename Index>
  static void run(Scalar* C, Index C_stride_row, Index C_stride_col, const Scalar* blockA, const Scalar* blockB,
                  Index rows, Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA,
                  Index offsetB) {
    sme_gebp_impl<Scalar, ConjLhs, ConjRhs, false>(C, C_stride_row, C_stride_col, blockA, blockB, rows, depth, cols,
                                                   alpha, strideA, strideB, offsetA, offsetB);
  }
};

template <typename Scalar, bool ConjLhs, bool ConjRhs>
struct sme_gebp_dispatch<Scalar, ConjLhs, ConjRhs, true> {
  template <typename Index>
  static void run(Scalar* C, Index C_stride_row, Index C_stride_col, const Scalar* blockA, const Scalar* blockB,
                  Index rows, Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA,
                  Index offsetB) {
    using RealScalar = typename NumTraits<Scalar>::Real;
    if (numext::imag(alpha) == RealScalar(0) && numext::real(alpha) != RealScalar(1)) {
      sme_gebp_impl<Scalar, ConjLhs, ConjRhs, true>(C, C_stride_row, C_stride_col, blockA, blockB, rows, depth, cols,
                                                    alpha, strideA, strideB, offsetA, offsetB);
    } else {
      sme_gebp_impl<Scalar, ConjLhs, ConjRhs, false>(C, C_stride_row, C_stride_col, blockA, blockB, rows, depth, cols,
                                                     alpha, strideA, strideB, offsetA, offsetB);
    }
  }
};

// gebp with the LHS read from a ColMajor source: rows i..i+pw of column k are
// at lhs + i + k * lda, the packed layout with a_step = lda. Real scalars only.
template <typename Scalar, typename Index>
EIGEN_DONT_INLINE __arm_locally_streaming __arm_new("za") void sme_gebp_impl_direct_lhs(
    Scalar* C, Index C_stride_row, Index C_stride_col, const Scalar* lhs, Index lda, const Scalar* blockB, Index rows,
    Index depth, Index cols, Scalar alpha, Index strideB, Index offsetB) {
  constexpr int MR = sme_block<Scalar>::mr;
  constexpr int NR = sme_block<Scalar>::nr;
  const bool split_ok = sme_split_ok<Scalar>(), narrow_ok = sme_narrow_ok<Scalar>();
  for (Index j = 0; j < cols; j += NR) {
    const int cw = static_cast<int>(sme_min(cols - j, Index(NR)));
    const Scalar* blB = blockB + j * strideB + offsetB * cw;
    Index i = 0;
    if (narrow_ok && cw <= sme_packet_traits<Scalar>::size())
      for (; i + MR < rows; i += 2 * MR)
        sme_process_narrow(C, C_stride_row, C_stride_col, lhs + i, lda, lhs + i + MR, lda,
                           static_cast<int>(sme_min(rows - i - MR, Index(MR))), blB, cw, depth, alpha, i, j);
    for (; i < rows; i += MR) {
      const int pw = static_cast<int>(sme_min(rows - i, Index(MR)));
      sme_process_block<false, false, false>(C, C_stride_row, C_stride_col, lhs + i, blB, depth, alpha, i, pw, j, cw,
                                             lda, split_ok, rows, cols);
    }
  }
}

// In-place ColMajor LHS: deep enough for the ZA kernel, a stride that is not a 4 KB multiple (such columns map to one
// L1 set) and a block of at most 4 MB, past which the strided re-reads lose to the packed panel.
#ifndef EIGEN_SME_DIRECT_LHS_MAX_STRIDE_BYTES
#define EIGEN_SME_DIRECT_LHS_MAX_STRIDE_BYTES 16384
#endif
#ifndef EIGEN_SME_DIRECT_LHS_MAX_BLOCK_BYTES
#define EIGEN_SME_DIRECT_LHS_MAX_BLOCK_BYTES (4 << 20)
#endif
#ifndef EIGEN_SME_DIRECT_LHS_MAX_SPAN_BYTES
#define EIGEN_SME_DIRECT_LHS_MAX_SPAN_BYTES (16 << 20)
#endif
template <typename Scalar, typename Index>
bool sme_direct_lhs_ok(Index lhsStride, Index rows, Index depth, Index cols) {
#ifdef EIGEN_SME_FORCE_NEON_SMALL_BLOCKS
  EIGEN_UNUSED_VARIABLE(lhsStride);
  EIGEN_UNUSED_VARIABLE(rows);
  EIGEN_UNUSED_VARIABLE(depth);
  EIGEN_UNUSED_VARIABLE(cols);
  return false;
#else
  if (NumTraits<Scalar>::IsComplex || depth <= Index(sme_neon_max_depth<Scalar>::value)) return false;
  // A RHS of one panel reads each LHS element once, so packing it only adds a copy: always for the narrow kernel,
  // and for the 2 x 2 one while a panel spans few enough pages.
  const std::size_t stride_bytes = std::size_t(lhsStride) * sizeof(Scalar);
  if (cols <= Index(sme_block<Scalar>::nr)) {
    const std::size_t span_limit = std::size_t(EIGEN_SME_DIRECT_LHS_MAX_SPAN_BYTES);
    if (span_limit == 0) return false;
    return cols <= Index(sme_block<Scalar>::nr / 2) || std::size_t(depth) * stride_bytes <= span_limit;
  }
  return stride_bytes % 4096 != 0 && stride_bytes <= std::size_t(EIGEN_SME_DIRECT_LHS_MAX_STRIDE_BYTES) &&
         std::size_t(rows) * std::size_t(depth) * sizeof(Scalar) <= std::size_t(EIGEN_SME_DIRECT_LHS_MAX_BLOCK_BYTES);
#endif
}

// NEON path for small blocks: a shallow block pays the ZA enable and a dependent FMOPA chain per
// tile for little work, so its panels are consumed with NEON in the SME layout (one depth step is
// pw contiguous scalars, complex ones as pw reals then pw imaginaries).
// The NCol columns of one depth step, loaded once; each column is then an
// immediate lane of a fused multiply-add (nmadd: acc - a * b[lane]).
template <typename Scalar, int NCol>
struct sme_neon_cols;
template <>
struct sme_neon_cols<float, 4> {
  using Vec = float32x4_t;
  static EIGEN_ALWAYS_INLINE Vec load(const float* b) { return vld1q_f32(b); }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f madd(Packet4f acc, Packet4f a, Vec b) {
    return vfmaq_laneq_f32(acc, a, b, C);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f nmadd(Packet4f acc, Packet4f a, Vec b) {
    return vfmsq_laneq_f32(acc, a, b, C);
  }
};
template <>
struct sme_neon_cols<float, 2> {
  using Vec = float32x2_t;
  static EIGEN_ALWAYS_INLINE Vec load(const float* b) { return vld1_f32(b); }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f madd(Packet4f acc, Packet4f a, Vec b) {
    return vfmaq_lane_f32(acc, a, b, C);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f nmadd(Packet4f acc, Packet4f a, Vec b) {
    return vfmsq_lane_f32(acc, a, b, C);
  }
};
template <>
struct sme_neon_cols<float, 1> {
  using Vec = float32x4_t;
  static EIGEN_ALWAYS_INLINE Vec load(const float* b) { return vld1q_dup_f32(b); }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f madd(Packet4f acc, Packet4f a, Vec b) {
    return vfmaq_f32(acc, a, b);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f nmadd(Packet4f acc, Packet4f a, Vec b) {
    return vfmsq_f32(acc, a, b);
  }
};
template <>
struct sme_neon_cols<double, 4> {
  struct Vec {
    float64x2_t lo, hi;
  };
  static EIGEN_ALWAYS_INLINE Vec load(const double* b) { return {vld1q_f64(b), vld1q_f64(b + 2)}; }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d madd(Packet2d acc, Packet2d a, Vec b) {
    return vfmaq_laneq_f64(acc, a, C < 2 ? b.lo : b.hi, C & 1);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d nmadd(Packet2d acc, Packet2d a, Vec b) {
    return vfmsq_laneq_f64(acc, a, C < 2 ? b.lo : b.hi, C & 1);
  }
};
template <>
struct sme_neon_cols<double, 2> {
  using Vec = float64x2_t;
  static EIGEN_ALWAYS_INLINE Vec load(const double* b) { return vld1q_f64(b); }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d madd(Packet2d acc, Packet2d a, Vec b) {
    return vfmaq_laneq_f64(acc, a, b, C);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d nmadd(Packet2d acc, Packet2d a, Vec b) {
    return vfmsq_laneq_f64(acc, a, b, C);
  }
};
template <>
struct sme_neon_cols<double, 1> {
  using Vec = float64x2_t;
  static EIGEN_ALWAYS_INLINE Vec load(const double* b) { return vld1q_dup_f64(b); }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d madd(Packet2d acc, Packet2d a, Vec b) {
    return vfmaq_f64(acc, a, b);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d nmadd(Packet2d acc, Packet2d a, Vec b) {
    return vfmsq_f64(acc, a, b);
  }
};

template <>
struct sme_neon_cols<float, 8> {
  struct Vec {
    float32x4_t lo, hi;
  };
  static EIGEN_ALWAYS_INLINE Vec load(const float* b) { return {vld1q_f32(b), vld1q_f32(b + 4)}; }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f madd(Packet4f acc, Packet4f a, Vec b) {
    return vfmaq_laneq_f32(acc, a, C < 4 ? b.lo : b.hi, C & 3);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet4f nmadd(Packet4f acc, Packet4f a, Vec b) {
    return vfmsq_laneq_f32(acc, a, C < 4 ? b.lo : b.hi, C & 3);
  }
};
template <>
struct sme_neon_cols<double, 8> {
  struct Vec {
    float64x2_t v[4];
  };
  static EIGEN_ALWAYS_INLINE Vec load(const double* b) {
    return {{vld1q_f64(b), vld1q_f64(b + 2), vld1q_f64(b + 4), vld1q_f64(b + 6)}};
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d madd(Packet2d acc, Packet2d a, Vec b) {
    return vfmaq_laneq_f64(acc, a, b.v[C >> 1], C & 1);
  }
  template <int C>
  static EIGEN_ALWAYS_INLINE Packet2d nmadd(Packet2d acc, Packet2d a, Vec b) {
    return vfmsq_laneq_f64(acc, a, b.v[C >> 1], C & 1);
  }
};

// Compile-time loop over the columns of a tile (lane numbers are immediates).
template <int C, int NCol>
struct sme_neon_col_loop {
  template <typename Cols, typename Packet, int NPack>
  static EIGEN_ALWAYS_INLINE void real(Packet (&acc)[NPack][NCol], const Packet (&av)[NPack], typename Cols::Vec b) {
    for (int p = 0; p < NPack; ++p) acc[p][C] = Cols::template madd<C>(acc[p][C], av[p], b);
    sme_neon_col_loop<C + 1, NCol>::template real<Cols, Packet, NPack>(acc, av, b);
  }
  template <bool ConjLhs, bool ConjRhs, typename Cols, typename Packet, int NPack>
  static EIGEN_ALWAYS_INLINE void cplx(Packet (&acc_re)[NPack][NCol], Packet (&acc_im)[NPack][NCol],
                                       const Packet (&are)[NPack], const Packet (&aim)[NPack], typename Cols::Vec bre,
                                       typename Cols::Vec bim) {
    for (int p = 0; p < NPack; ++p) {
      acc_re[p][C] = Cols::template madd<C>(acc_re[p][C], are[p], bre);
      acc_re[p][C] = (ConjLhs != ConjRhs) ? Cols::template madd<C>(acc_re[p][C], aim[p], bim)
                                          : Cols::template nmadd<C>(acc_re[p][C], aim[p], bim);
      acc_im[p][C] = ConjLhs ? Cols::template nmadd<C>(acc_im[p][C], aim[p], bre)
                             : Cols::template madd<C>(acc_im[p][C], aim[p], bre);
      acc_im[p][C] = ConjRhs ? Cols::template nmadd<C>(acc_im[p][C], are[p], bim)
                             : Cols::template madd<C>(acc_im[p][C], are[p], bim);
    }
    sme_neon_col_loop<C + 1, NCol>::template cplx<ConjLhs, ConjRhs, Cols, Packet, NPack>(acc_re, acc_im, are, aim, bre,
                                                                                         bim);
  }
};
template <int NCol>
struct sme_neon_col_loop<NCol, NCol> {
  template <typename Cols, typename Packet, int NPack>
  static EIGEN_ALWAYS_INLINE void real(Packet (&)[NPack][NCol], const Packet (&)[NPack], typename Cols::Vec) {}
  template <bool ConjLhs, bool ConjRhs, typename Cols, typename Packet, int NPack>
  static EIGEN_ALWAYS_INLINE void cplx(Packet (&)[NPack][NCol], Packet (&)[NPack][NCol], const Packet (&)[NPack],
                                       const Packet (&)[NPack], typename Cols::Vec, typename Cols::Vec) {}
};

// One (NPack * PacketSize) x NCol micro-tile of a real block; NPack == 0 is a
// single scalar row. Pointers arrive offset to the tile's first row and column.
template <typename Scalar, typename Index, int NPack, int NCol>
struct sme_neon_tile {
  using Packet = typename packet_traits<Scalar>::type;
  using Cols = sme_neon_cols<Scalar, NCol>;
  static constexpr int PS = packet_traits<Scalar>::size;
  // Narrow tiles split the depth over KU accumulator chains to hide FMA latency.
  static constexpr int KU = (NPack * NCol >= 4) ? 1 : 4 / (NPack * NCol);
  static EIGEN_ALWAYS_INLINE void step(Packet (&acc)[NPack][NCol], const Scalar* a, const Scalar* b) {
    Packet av[NPack];
    for (int p = 0; p < NPack; ++p) av[p] = ploadu<Packet>(a + p * PS);
    sme_neon_col_loop<0, NCol>::template real<Cols, Packet, NPack>(acc, av, Cols::load(b));
  }
  static EIGEN_ALWAYS_INLINE void run(Scalar* C, Index rs, Index cs, const Scalar* blA, const Scalar* blB, Index depth,
                                      Scalar alpha, Index pw, Index cw) {
    Packet acc[KU][NPack][NCol];
    for (int u = 0; u < KU; ++u)
      for (int p = 0; p < NPack; ++p)
        for (int c = 0; c < NCol; ++c) acc[u][p][c] = pset1<Packet>(Scalar(0));
    Index k = 0;
    for (; k + KU <= depth; k += KU) {
      for (int u = 0; u < KU; ++u) step(acc[u], blA + (k + u) * pw, blB + (k + u) * cw);
    }
    for (; k < depth; ++k) step(acc[0], blA + k * pw, blB + k * cw);
    for (int u = 1; u < KU; ++u)
      for (int p = 0; p < NPack; ++p)
        for (int c = 0; c < NCol; ++c) acc[0][p][c] = padd(acc[0][p][c], acc[u][p][c]);
    const Packet valpha = pset1<Packet>(alpha);
    for (int c = 0; c < NCol; ++c) {
      for (int p = 0; p < NPack; ++p) {
        Scalar* pc = C + Index(p * PS) * rs + Index(c) * cs;
        if (rs == 1) {
          pstoreu(pc, pmadd(acc[0][p][c], valpha, ploadu<Packet>(pc)));
        } else {
          pscatter<Scalar, Packet>(pc, pmadd(acc[0][p][c], valpha, pgather<Scalar, Packet>(pc, rs)), rs);
        }
      }
    }
  }
};

template <typename Scalar, typename Index, int NCol>
struct sme_neon_tile<Scalar, Index, 0, NCol> {
  static EIGEN_ALWAYS_INLINE void run(Scalar* C, Index rs, Index cs, const Scalar* blA, const Scalar* blB, Index depth,
                                      Scalar alpha, Index pw, Index cw) {
    EIGEN_UNUSED_VARIABLE(rs);
    constexpr int KU = (NCol >= 4) ? 1 : 4 / NCol;
    Scalar acc[KU][NCol];
    for (int u = 0; u < KU; ++u)
      for (int c = 0; c < NCol; ++c) acc[u][c] = Scalar(0);
    Index k = 0;
    for (; k + KU <= depth; k += KU) {
      for (int u = 0; u < KU; ++u) {
        const Scalar a = blA[(k + u) * pw];
        const Scalar* b = blB + (k + u) * cw;
        for (int c = 0; c < NCol; ++c) acc[u][c] += a * b[c];
      }
    }
    for (; k < depth; ++k) {
      const Scalar a = blA[k * pw];
      const Scalar* b = blB + k * cw;
      for (int c = 0; c < NCol; ++c) acc[0][c] += a * b[c];
    }
    for (int u = 1; u < KU; ++u)
      for (int c = 0; c < NCol; ++c) acc[0][c] += acc[u][c];
    for (int c = 0; c < NCol; ++c) C[Index(c) * cs] += alpha * acc[0][c];
  }
};

// Complex counterpart over the real view of the panels. Conjugation is folded
// into the signs of the four real products.
template <typename RealScalar, typename Index, bool ConjLhs, bool ConjRhs, int NPack, int NCol>
struct sme_neon_ctile {
  using Scalar = std::complex<RealScalar>;
  using Packet = typename packet_traits<RealScalar>::type;
  using Cols = sme_neon_cols<RealScalar, NCol>;
  static constexpr int PS = packet_traits<RealScalar>::size;
  static EIGEN_ALWAYS_INLINE void run(Scalar* C, Index rs, Index cs, const RealScalar* rA, const RealScalar* rB,
                                      Index depth, Scalar alpha, Index pw, Index cw) {
    Packet acc_re[NPack][NCol], acc_im[NPack][NCol];
    for (int p = 0; p < NPack; ++p) {
      for (int c = 0; c < NCol; ++c) {
        acc_re[p][c] = pset1<Packet>(RealScalar(0));
        acc_im[p][c] = pset1<Packet>(RealScalar(0));
      }
    }
    for (Index k = 0; k < depth; ++k) {
      const RealScalar* a = rA + k * 2 * pw;
      const RealScalar* b = rB + k * 2 * cw;
      Packet are[NPack], aim[NPack];
      for (int p = 0; p < NPack; ++p) {
        are[p] = ploadu<Packet>(a + p * PS);
        aim[p] = ploadu<Packet>(a + pw + p * PS);
      }
      sme_neon_col_loop<0, NCol>::template cplx<ConjLhs, ConjRhs, Cols, Packet, NPack>(
          acc_re, acc_im, are, aim, Cols::load(b), Cols::load(b + cw));
    }
    const RealScalar ar = numext::real(alpha), ai = numext::imag(alpha);
    const Packet valpha_re = pset1<Packet>(ar), valpha_im = pset1<Packet>(ai);
    for (int c = 0; c < NCol; ++c) {
      for (int p = 0; p < NPack; ++p) {
        RealScalar* pc = reinterpret_cast<RealScalar*>(C + Index(p * PS) * rs + Index(c) * cs);
        if (rs == 1) {
          // Unit row stride: the column is PS interleaved (re, im) pairs.
          Packet cre, cim;
          sme_neon_ld2(pc, cre, cim);
          cre = pnmadd(valpha_im, acc_im[p][c], pmadd(valpha_re, acc_re[p][c], cre));
          cim = pmadd(valpha_im, acc_re[p][c], pmadd(valpha_re, acc_im[p][c], cim));
          sme_neon_st2(pc, cre, cim);
        } else {
          RealScalar re[PS], im[PS];
          pstoreu(re, acc_re[p][c]);
          pstoreu(im, acc_im[p][c]);
          for (int l = 0; l < PS; ++l) {
            RealScalar* pl = pc + Index(l) * 2 * rs;
            pl[0] += ar * re[l] - ai * im[l];
            pl[1] += ar * im[l] + ai * re[l];
          }
        }
      }
    }
  }
};

template <typename RealScalar, typename Index, bool ConjLhs, bool ConjRhs, int NCol>
struct sme_neon_ctile<RealScalar, Index, ConjLhs, ConjRhs, 0, NCol> {
  using Scalar = std::complex<RealScalar>;
  static EIGEN_ALWAYS_INLINE void run(Scalar* C, Index rs, Index cs, const RealScalar* rA, const RealScalar* rB,
                                      Index depth, Scalar alpha, Index pw, Index cw) {
    EIGEN_UNUSED_VARIABLE(rs);
    constexpr int KU = (NCol >= 4) ? 1 : 4 / NCol;
    RealScalar acc_re[KU][NCol], acc_im[KU][NCol];
    for (int u = 0; u < KU; ++u)
      for (int c = 0; c < NCol; ++c) acc_re[u][c] = acc_im[u][c] = RealScalar(0);
    for (Index k = 0; k < depth; ++k) {
      const int u = int(k % KU);
      const RealScalar* a = rA + k * 2 * pw;
      const RealScalar* b = rB + k * 2 * cw;
      const RealScalar are = a[0], aim = ConjLhs ? -a[pw] : a[pw];
      for (int c = 0; c < NCol; ++c) {
        const RealScalar bre = b[c], bim = ConjRhs ? -b[cw + c] : b[cw + c];
        acc_re[u][c] += are * bre - aim * bim;
        acc_im[u][c] += are * bim + aim * bre;
      }
    }
    for (int u = 1; u < KU; ++u)
      for (int c = 0; c < NCol; ++c) {
        acc_re[0][c] += acc_re[u][c];
        acc_im[0][c] += acc_im[u][c];
      }
    const RealScalar ar = numext::real(alpha), ai = numext::imag(alpha);
    for (int c = 0; c < NCol; ++c) {
      RealScalar* pc = reinterpret_cast<RealScalar*>(C + Index(c) * cs);
      pc[0] += ar * acc_re[0][c] - ai * acc_im[0][c];
      pc[1] += ar * acc_im[0][c] + ai * acc_re[0][c];
    }
  }
};

// Row loop of one column group: 3, 2 and 1 packets of rows, then scalar rows.
template <bool ConjLhs, bool ConjRhs, int NCol, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_neon_column_group(Scalar* C, Index rs, Index cs, const Scalar* blA, const Scalar* blB,
                                               Index depth, Scalar alpha, Index pw, Index cw) {
  constexpr Index PS = Index(packet_traits<Scalar>::size);
  Index r = 0;
  for (; r + 3 * PS <= pw; r += 3 * PS)
    sme_neon_tile<Scalar, Index, 3, NCol>::run(C + r * rs, rs, cs, blA + r, blB, depth, alpha, pw, cw);
  for (; r + 2 * PS <= pw; r += 2 * PS)
    sme_neon_tile<Scalar, Index, 2, NCol>::run(C + r * rs, rs, cs, blA + r, blB, depth, alpha, pw, cw);
  for (; r + PS <= pw; r += PS)
    sme_neon_tile<Scalar, Index, 1, NCol>::run(C + r * rs, rs, cs, blA + r, blB, depth, alpha, pw, cw);
  for (; r < pw; ++r)
    sme_neon_tile<Scalar, Index, 0, NCol>::run(C + r * rs, rs, cs, blA + r, blB, depth, alpha, pw, cw);
}

template <bool ConjLhs, bool ConjRhs, int NCol, typename RealScalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_neon_column_group(std::complex<RealScalar>* C, Index rs, Index cs, const RealScalar* rA,
                                               const RealScalar* rB, Index depth, std::complex<RealScalar> alpha,
                                               Index pw, Index cw) {
  // Two accumulators per packet, so the ladder stops at two packets of rows.
  constexpr Index PS = Index(packet_traits<RealScalar>::size);
  Index r = 0;
  for (; r + 2 * PS <= pw; r += 2 * PS)
    sme_neon_ctile<RealScalar, Index, ConjLhs, ConjRhs, 2, NCol>::run(C + r * rs, rs, cs, rA + r, rB, depth, alpha, pw,
                                                                      cw);
  for (; r + PS <= pw; r += PS)
    sme_neon_ctile<RealScalar, Index, ConjLhs, ConjRhs, 1, NCol>::run(C + r * rs, rs, cs, rA + r, rB, depth, alpha, pw,
                                                                      cw);
  for (; r < pw; ++r)
    sme_neon_ctile<RealScalar, Index, ConjLhs, ConjRhs, 0, NCol>::run(C + r * rs, rs, cs, rA + r, rB, depth, alpha, pw,
                                                                      cw);
}

// One pw x cw block: columns in groups of eight, four, two, then single columns.
template <bool ConjLhs, bool ConjRhs, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_neon_block(Scalar* C, Index rs, Index cs, const Scalar* blA, const Scalar* blB,
                                        Index depth, Scalar alpha, Index pw, Index cw) {
  Index c = 0;
  for (; c + 8 <= cw; c += 8)
    sme_neon_column_group<ConjLhs, ConjRhs, 8>(C + c * cs, rs, cs, blA, blB + c, depth, alpha, pw, cw);
  for (; c + 4 <= cw; c += 4)
    sme_neon_column_group<ConjLhs, ConjRhs, 4>(C + c * cs, rs, cs, blA, blB + c, depth, alpha, pw, cw);
  for (; c + 2 <= cw; c += 2)
    sme_neon_column_group<ConjLhs, ConjRhs, 2>(C + c * cs, rs, cs, blA, blB + c, depth, alpha, pw, cw);
  for (; c < cw; ++c)
    sme_neon_column_group<ConjLhs, ConjRhs, 1>(C + c * cs, rs, cs, blA, blB + c, depth, alpha, pw, cw);
}

template <bool ConjLhs, bool ConjRhs, typename RealScalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_neon_block(std::complex<RealScalar>* C, Index rs, Index cs,
                                        const std::complex<RealScalar>* blA, const std::complex<RealScalar>* blB,
                                        Index depth, std::complex<RealScalar> alpha, Index pw, Index cw) {
  const RealScalar* rA = reinterpret_cast<const RealScalar*>(blA);
  const RealScalar* rB = reinterpret_cast<const RealScalar*>(blB);
  Index c = 0;
  for (; c + 4 <= cw; c += 4)
    sme_neon_column_group<ConjLhs, ConjRhs, 4>(C + c * cs, rs, cs, rA, rB + c, depth, alpha, pw, cw);
  for (; c + 2 <= cw; c += 2)
    sme_neon_column_group<ConjLhs, ConjRhs, 2>(C + c * cs, rs, cs, rA, rB + c, depth, alpha, pw, cw);
  for (; c < cw; ++c) sme_neon_column_group<ConjLhs, ConjRhs, 1>(C + c * cs, rs, cs, rA, rB + c, depth, alpha, pw, cw);
}

// Same panel walk as sme_gebp_impl, outside any streaming region.
template <typename Scalar, bool ConjLhs, bool ConjRhs, typename Index>
EIGEN_ALWAYS_INLINE void sme_gebp_neon(Scalar* C, Index C_stride_row, Index C_stride_col, const Scalar* blockA,
                                       const Scalar* blockB, Index rows, Index depth, Index cols, Scalar alpha,
                                       Index strideA, Index strideB, Index offsetA, Index offsetB) {
  constexpr Index MR = Index(sme_block<Scalar>::mr);
  constexpr Index NR = Index(sme_block<Scalar>::nr);
  for (Index j = 0; j < cols; j += NR) {
    const Index cw = numext::mini(cols - j, NR);
    const Scalar* blB = blockB + j * strideB + offsetB * cw;
    for (Index i = 0; i < rows; i += MR) {
      const Index pw = numext::mini(rows - i, MR);
      const Scalar* blA = blockA + i * strideA + offsetA * pw;
      sme_neon_block<ConjLhs, ConjRhs>(C + i * C_stride_row + j * C_stride_col, C_stride_row, C_stride_col, blA, blB,
                                       depth, alpha, pw, cw);
    }
  }
}

template <typename Scalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct sme_gebp_kernel {
  using ResScalar = Scalar;

  EIGEN_DONT_INLINE void operator()(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows,
                                    Index depth, Index cols, ResScalar alpha, Index strideA = -1, Index strideB = -1,
                                    Index offsetA = 0, Index offsetB = 0) {
    // Real scalars never reach the kernel conjugated (conj_helper folds it into
    // the identity long before), so the real path stays free of the flags.
    static_assert(NumTraits<Scalar>::IsComplex || (!ConjugateLhs && !ConjugateRhs),
                  "the SME kernel does not support conjugation of real scalars");
    static_assert(mr == sme_block<Scalar>::mr && nr == sme_block<Scalar>::nr,
                  "the SME kernel expects packed panels of the SME block width");

    if (strideA == -1) strideA = depth;
    if (strideB == -1) strideB = depth;

    if (rows <= 0 || cols <= 0 || depth <= 0) return;

    Scalar* C_base = const_cast<Scalar*>(&res(0, 0));
    const Index C_stride_row = &res(1, 0) - &res(0, 0);
    const Index C_stride_col = &res(0, 1) - &res(0, 0);

    if (sme_kernel_with_neon<Scalar>(rows, cols, depth, strideA, strideB)) {
      sme_gebp_neon<Scalar, ConjugateLhs, ConjugateRhs>(C_base, C_stride_row, C_stride_col, blockA, blockB, rows, depth,
                                                        cols, alpha, strideA, strideB, offsetA, offsetB);
      return;
    }

    sme_fpsr_guard fpsr;
    sme_gebp_dispatch<Scalar, ConjugateLhs, ConjugateRhs>::run(C_base, C_stride_row, C_stride_col, blockA, blockB, rows,
                                                               depth, cols, alpha, strideA, strideB, offsetA, offsetB);
  }

  // The LHS block read from its ColMajor source (see sme_direct_lhs_ok); only
  // instantiated for real scalars.
  EIGEN_DONT_INLINE void run_direct_lhs(const DataMapper& res, const Scalar* lhs, Index lhsStride, const Scalar* blockB,
                                        Index rows, Index depth, Index cols, ResScalar alpha, Index strideB = -1,
                                        Index offsetB = 0) {
    if (strideB == -1) strideB = depth;
    if (rows <= 0 || cols <= 0 || depth <= 0) return;
    Scalar* C_base = const_cast<Scalar*>(&res(0, 0));
    const Index C_stride_row = &res(1, 0) - &res(0, 0);
    const Index C_stride_col = &res(0, 1) - &res(0, 0);
    sme_fpsr_guard fpsr;
    sme_gebp_impl_direct_lhs<Scalar, Index>(C_base, C_stride_row, C_stride_col, lhs, lhsStride, blockB, rows, depth,
                                            cols, alpha, strideB, offsetB);
  }
};

#define EIGEN_SME_DECLARE_GEBP_KERNEL(SCALAR)                                                          \
  template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs> \
  struct gebp_kernel<SCALAR, SCALAR, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>            \
      : sme_gebp_kernel<SCALAR, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {};

EIGEN_SME_DECLARE_GEBP_KERNEL(float)
EIGEN_SME_DECLARE_GEBP_KERNEL(std::complex<float>)
#ifdef EIGEN_VECTORIZE_SME_F64F64
EIGEN_SME_DECLARE_GEBP_KERNEL(double)
EIGEN_SME_DECLARE_GEBP_KERNEL(std::complex<double>)
#endif

#undef EIGEN_SME_DECLARE_GEBP_KERNEL

// sme_has_gebp_kernel (products/GeneralBlockPanelKernel.h) drives the cache
// blocking and the GEMM loop order, and is declared before this header. A pair
// listed there but not specialized here would be packed and blocked for SME and
// then handed to the generic kernel.
static_assert(sme_has_gebp_kernel<float, float>::value, "the SME float kernel is not advertised to the GEMM driver");
static_assert(sme_has_gebp_kernel<std::complex<float>, std::complex<float>>::value,
              "the SME complex<float> kernel is not advertised to the GEMM driver");
#ifdef EIGEN_VECTORIZE_SME_F64F64
static_assert(sme_has_gebp_kernel<double, double>::value, "the SME double kernel is not advertised to the GEMM driver");
static_assert(sme_has_gebp_kernel<std::complex<double>, std::complex<double>>::value,
              "the SME complex<double> kernel is not advertised to the GEMM driver");
#else
static_assert(!sme_has_gebp_kernel<double, double>::value,
              "double is advertised to the GEMM driver without FEAT_SME_F64F64 to implement it");
static_assert(!sme_has_gebp_kernel<std::complex<double>, std::complex<double>>::value,
              "complex<double> is advertised to the GEMM driver without FEAT_SME_F64F64 to implement it");
#endif

// ---------------------------------------------------------------------------
// Selfadjoint (SYMM) packers.
//
// product_selfadjoint_matrix packs the selfadjoint operand (stored as one
// triangle) through symm_pack_lhs/symm_pack_rhs, which materialize the full
// matrix as they pack. The generic SYMM packers emit packet-width sub-panels
// for the generic gebp_kernel, whereas the SME kernel expects uniform
// mr/nr-wide depth-major panels. These packers perform the same
// triangle mirroring in the SME layout.
//
// The packer receives the operand in an orientation where row >= col is the
// stored triangle. It reads that half directly and mirrors the other half:
//   full(row,col) = (row >= col) ? m(row,col) : conj(m(col,row))
// and a selfadjoint view defines the diagonal's imaginary part as zero, so
// full(k,k) is real(m(k,k)). Both reduce to the identity on real scalars.
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
// The RHS packs full(k2+k, j+c) and uses this mapping directly, so its mirrored
// half is the transposed region. The LHS packs full(j+r, k), which is the
// conjugate of full(k, j+r), so it reuses the same mapping with k2 == 0
// relative to its diagonal-anchored base pointer but conjugates the opposite
// regions -- the direct one and the straddle band's head. IsLhs selects which.
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
template <typename Scalar, int StorageOrder, bool IsLhs, typename Index>
EIGEN_DONT_INLINE __arm_locally_streaming __arm_new("za") void sme_symm_pack_dense_regions(
    Scalar* block, const Scalar* EIGEN_RESTRICT base, Index stride, Index depth, Index outer, Index k2) {
  constexpr int PACK = IsLhs ? sme_block<Scalar>::mr : sme_block<Scalar>::nr;
  constexpr bool ColM = (StorageOrder == ColMajor);
  // The transposed region is the RHS's mirrored half and the direct one the
  // LHS's, so exactly one of the two is conjugated (see above).
  constexpr bool ConjTransposed = !IsLhs;
  constexpr bool ConjDirect = IsLhs;

  for (Index j = 0; j < outer; j += PACK) {
    const int w = static_cast<int>(sme_min(outer - j, Index(PACK)));
    Scalar* dst = block + j * depth;  // depth-major panel of width w
    Index t_end, s_end;
    sme_symm_panel_regions(j, w, depth, k2, t_end, s_end);

    // Transposed region: full(k2+k, j+c) = m(j+c, k2+k).
    if (t_end > 0) {
      EIGEN_IF_CONSTEXPR (ColM) {
        sve_copy_panel_range<ConjTransposed>(dst, base + j + k2 * stride, stride, Index(0), t_end, w);
      } else {
        sme_transpose_pack_range<ConjTransposed>(dst, base + j * stride + k2, stride, Index(0), t_end, w);
      }
    }
    // Direct region: full(k2+k, j+c) = m(k2+k, j+c).
    if (s_end < depth) {
      EIGEN_IF_CONSTEXPR (ColM) {
        sme_transpose_pack_range<ConjDirect>(dst, base + k2 + j * stride, stride, s_end, depth, w);
      } else {
        sve_copy_panel_range<ConjDirect>(dst, base + k2 * stride + j, stride, s_end, depth, w);
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
template <typename Scalar, int StorageOrder, bool IsLhs, typename Index>
EIGEN_DONT_INLINE void sme_symm_pack_straddle(Scalar* block, const Scalar* EIGEN_RESTRICT base, Index stride,
                                              Index depth, Index outer, Index k2) {
  constexpr int PACK = IsLhs ? sme_block<Scalar>::mr : sme_block<Scalar>::nr;
  constexpr bool ColM = (StorageOrder == ColMajor);
  // The band's tail is the RHS's mirrored half and its head the LHS's, exactly
  // as the dense regions above.
  constexpr bool ConjHead = IsLhs;
  constexpr bool ConjTail = !IsLhs;

  for (Index j = 0; j < outer; j += PACK) {
    const int w = static_cast<int>(numext::mini(outer - j, Index(PACK)));
    Scalar* dst = block + j * depth;
    Index t_end, s_end;
    sme_symm_panel_regions(j, w, depth, k2, t_end, s_end);

    for (Index k = t_end; k < s_end; ++k) {
      const Index row = k2 + k;
      const int cs = static_cast<int>(row - j);
      Scalar* dst_row = dst + k * w;
      const Index wi = Index(w);
      EIGEN_IF_CONSTEXPR (ColM) {
        const Scalar* head = base + row + j * stride;  // m(row, j+c): stride-strided
        for (int c = 0; c < cs; ++c, head += stride) sme_pack_store<ConjHead>(dst_row, wi, Index(c), *head);
        const Scalar* tail = base + j + row * stride;  // m(j+c, row): contiguous
        int c = cs;
        // A selfadjoint view defines the diagonal's imaginary part as zero;
        // for a real scalar that is already true, so the peel folds away.
        EIGEN_IF_CONSTEXPR (NumTraits<Scalar>::IsComplex) {
          sme_pack_store<false>(dst_row, wi, Index(cs), Scalar(numext::real(tail[cs])));
          ++c;
        }
        for (; c < w; ++c) sme_pack_store<ConjTail>(dst_row, wi, Index(c), tail[c]);
      } else {
        const Scalar* head = base + row * stride + j;  // m(row, j+c): contiguous
        for (int c = 0; c < cs; ++c) sme_pack_store<ConjHead>(dst_row, wi, Index(c), head[c]);
        const Scalar* tail = base + (j + Index(cs)) * stride + row;  // m(j+c, row): stride-strided
        int c = cs;
        EIGEN_IF_CONSTEXPR (NumTraits<Scalar>::IsComplex) {
          sme_pack_store<false>(dst_row, wi, Index(cs), Scalar(numext::real(*tail)));
          tail += stride;
          ++c;
        }
        for (; c < w; ++c, tail += stride) sme_pack_store<ConjTail>(dst_row, wi, Index(c), *tail);
      }
    }
  }
}

// Packer shared by the LHS (k2 == 0) and RHS symm specializations.
template <typename Scalar, int StorageOrder, bool IsLhs, typename Index>
EIGEN_DONT_INLINE void sme_symm_pack_panels(Scalar* block, const Scalar* EIGEN_RESTRICT base, Index stride, Index depth,
                                            Index outer, Index k2) {
  {
    sme_fpsr_guard fpsr;
    sme_symm_pack_dense_regions<Scalar, StorageOrder, IsLhs, Index>(block, base, stride, depth, outer, k2);
  }
  sme_symm_pack_straddle<Scalar, StorageOrder, IsLhs, Index>(block, base, stride, depth, outer, k2);
}

// symm_pack_lhs/rhs SME specializations: emit the uniform mr/nr panels
// sme_gebp_impl reads. Pack1/nr pinned exactly as gemm_pack_lhs/rhs above.
template <typename Scalar, int StorageOrder, typename Index>
struct sme_symm_pack_lhs {
  // Note: generic symm_pack_lhs's "cols" is the depth extent, and the LHS
  // block is diagonal-anchored (base = &lhs(k2,k2)), so its depth offset is 0.
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const Scalar* lhs_, Index lhsStride, Index cols, Index rows) const {
    sme_symm_pack_panels<Scalar, StorageOrder, true, Index>(blockA, lhs_, lhsStride, cols, rows, Index(0));
  }
};

template <typename Scalar, int StorageOrder, typename Index>
struct sme_symm_pack_rhs {
  // Note: generic symm_pack_rhs's "rows" is the depth extent (end_k = k2 + rows), not a row count.
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const Scalar* rhs_, Index rhsStride, Index rows, Index cols,
                                    Index k2) const {
    sme_symm_pack_panels<Scalar, StorageOrder, false, Index>(blockB, rhs_, rhsStride, rows, cols, k2);
  }
};

#define EIGEN_SME_DECLARE_SYMM_PACKERS(SCALAR, MR, NR)               \
  template <typename Index, int Pack2_dummy, int StorageOrder>       \
  struct symm_pack_lhs<SCALAR, Index, MR, Pack2_dummy, StorageOrder> \
      : sme_symm_pack_lhs<SCALAR, StorageOrder, Index> {};           \
                                                                     \
  template <typename Index, int StorageOrder>                        \
  struct symm_pack_rhs<SCALAR, Index, NR, StorageOrder> : sme_symm_pack_rhs<SCALAR, StorageOrder, Index> {};

EIGEN_SME_DECLARE_SYMM_PACKERS(float, kSmeMr, kSmeNr)
EIGEN_SME_DECLARE_SYMM_PACKERS(std::complex<float>, kSmeMrC, kSmeNrC)
#ifdef EIGEN_VECTORIZE_SME_F64F64
EIGEN_SME_DECLARE_SYMM_PACKERS(double, kSmeMrD, kSmeNrD)
EIGEN_SME_DECLARE_SYMM_PACKERS(std::complex<double>, kSmeMrCD, kSmeNrCD)
#endif

#undef EIGEN_SME_DECLARE_SYMM_PACKERS

// Tiny results (at most 2 * PS x 8): a NEON outer-product kernel with the whole result in registers, called before
// any blocking or packing. Both the coeff-based product (one dependent FMA chain per result) and the packed paths
// are several times slower there.
template <typename Scalar>
struct sme_tiny_neon;
template <>
struct sme_tiny_neon<float> {
  using V = float32x4_t;
  static constexpr int PS = 4;
  static EIGEN_ALWAYS_INLINE V ld(const float* p) { return vld1q_f32(p); }
  static EIGEN_ALWAYS_INLINE V zero() { return vdupq_n_f32(0.f); }
  // The first n (< PS) scalars at p, zeros after them.
  static EIGEN_ALWAYS_INLINE V ld_part(const float* p, int n) {
    if (n <= 0) return zero();
    if (n == 1) return vld1q_lane_f32(p, zero(), 0);
    const V v = vcombine_f32(vld1_f32(p), vdup_n_f32(0.f));
    return n == 2 ? v : vld1q_lane_f32(p + 2, v, 2);
  }
  // A full load permuted by byte indices; indices of 16 or more give zeros.
  static EIGEN_ALWAYS_INLINE V ld_tbl(const float* p, uint8x16_t idx) {
    return vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(vld1q_f32(p)), idx));
  }
  template <int L>
  static EIGEN_ALWAYS_INLINE V fma_lane(V c, V a, V b) {
    return vfmaq_laneq_f32(c, a, b, L);
  }
  static EIGEN_ALWAYS_INLINE V fma_n(V c, V a, float b) { return vfmaq_n_f32(c, a, b); }
  static EIGEN_ALWAYS_INLINE void st(float* p, V v) { vst1q_f32(p, v); }
  static EIGEN_ALWAYS_INLINE V add(V a, V b) { return vaddq_f32(a, b); }
  // Rows in, columns out.
  static EIGEN_ALWAYS_INLINE void transpose(V (&r)[PS]) {
    const float64x2_t t0 = vreinterpretq_f64_f32(vtrn1q_f32(r[0], r[1]));
    const float64x2_t t1 = vreinterpretq_f64_f32(vtrn2q_f32(r[0], r[1]));
    const float64x2_t t2 = vreinterpretq_f64_f32(vtrn1q_f32(r[2], r[3]));
    const float64x2_t t3 = vreinterpretq_f64_f32(vtrn2q_f32(r[2], r[3]));
    r[0] = vreinterpretq_f32_f64(vtrn1q_f64(t0, t2));
    r[1] = vreinterpretq_f32_f64(vtrn1q_f64(t1, t3));
    r[2] = vreinterpretq_f32_f64(vtrn2q_f64(t0, t2));
    r[3] = vreinterpretq_f32_f64(vtrn2q_f64(t1, t3));
  }
};
template <>
struct sme_tiny_neon<double> {
  using V = float64x2_t;
  static constexpr int PS = 2;
  static EIGEN_ALWAYS_INLINE V ld(const double* p) { return vld1q_f64(p); }
  static EIGEN_ALWAYS_INLINE V zero() { return vdupq_n_f64(0.); }
  static EIGEN_ALWAYS_INLINE V ld_part(const double* p, int n) { return n > 0 ? vld1q_lane_f64(p, zero(), 0) : zero(); }
  static EIGEN_ALWAYS_INLINE V ld_tbl(const double* p, uint8x16_t idx) {
    return vreinterpretq_f64_u8(vqtbl1q_u8(vreinterpretq_u8_f64(vld1q_f64(p)), idx));
  }
  template <int L>
  static EIGEN_ALWAYS_INLINE V fma_lane(V c, V a, V b) {
    return vfmaq_laneq_f64(c, a, b, L);
  }
  static EIGEN_ALWAYS_INLINE V fma_n(V c, V a, double b) { return vfmaq_n_f64(c, a, b); }
  static EIGEN_ALWAYS_INLINE void st(double* p, V v) { vst1q_f64(p, v); }
  static EIGEN_ALWAYS_INLINE V add(V a, V b) { return vaddq_f64(a, b); }
  static EIGEN_ALWAYS_INLINE void transpose(V (&r)[PS]) {
    const V t0 = vtrn1q_f64(r[0], r[1]);
    r[1] = vtrn2q_f64(r[0], r[1]);
    r[0] = t0;
  }
};

// One depth chunk of FMAs; the lane index must be a constant, hence the recursion over it. Depth step u of a chunk
// accumulates into set u % S: four FMA pipes of latency four need about 16 independent chains.
template <typename T, int RV, int NC, int S>
struct sme_tiny_step {
  using V = typename T::V;
  static constexpr int PS = T::PS;
  // ColMajor B: b[j] holds depth steps k .. k + PS - 1 of column j; lane U is step k + U.
  template <int U>
  static EIGEN_ALWAYS_INLINE void col(V (&acc)[S][NC][RV], const V (&a)[PS][RV], const V (&b)[NC],
                                      std::integral_constant<int, U>) {
    EIGEN_UNROLL_LOOP
    for (int j = 0; j < NC; ++j) {
      EIGEN_UNROLL_LOOP
      for (int r = 0; r < RV; ++r) acc[U % S][j][r] = T::template fma_lane<U>(acc[U % S][j][r], a[U][r], b[j]);
    }
    col(acc, a, b, std::integral_constant<int, U + 1>());
  }
  static EIGEN_ALWAYS_INLINE void col(V (&)[S][NC][RV], const V (&)[PS][RV], const V (&)[NC],
                                      std::integral_constant<int, PS>) {}
  // RowMajor B: b[c] holds columns c * PS .. of one depth step.
  template <int J>
  static EIGEN_ALWAYS_INLINE void row(V (&acc)[NC][RV], const V (&a)[RV], const V (&b)[NC / PS],
                                      std::integral_constant<int, J>) {
    EIGEN_UNROLL_LOOP
    for (int r = 0; r < RV; ++r) acc[J][r] = T::template fma_lane<J % PS>(acc[J][r], a[r], b[J / PS]);
    row(acc, a, b, std::integral_constant<int, J + 1>());
  }
  static EIGEN_ALWAYS_INLINE void row(V (&)[NC][RV], const V (&)[RV], const V (&)[NC / PS],
                                      std::integral_constant<int, NC>) {}
};

// C (m x n, m <= RV * PS, n <= NC) += alpha * A * B, reading only A's m rows and B's n columns: rows and columns
// past the result repeat the last valid one or load as zeros, and their sums are dropped.
template <typename Scalar, int RV, int NC, int LhsOrder, int RhsOrder, typename Index>
EIGEN_DONT_INLINE void sme_tiny_gemm_kernel(Index m, Index n, Index depth, const Scalar* A, Index lda, const Scalar* B,
                                            Index ldb, Scalar* C, Index incr, Index ldc, Scalar alpha) {
  using T = sme_tiny_neon<Scalar>;
  using V = typename T::V;
  constexpr int PS = T::PS, MR = RV * PS, S = 16 / (RV * NC) < PS ? 16 / (RV * NC) : PS;
  using Step = sme_tiny_step<T, RV, NC, S>;
  V acc[S][NC][RV];
  EIGEN_UNROLL_LOOP
  for (int t = 0; t < S; ++t) {
    EIGEN_UNROLL_LOOP
    for (int j = 0; j < NC; ++j) {
      EIGEN_UNROLL_LOOP
      for (int r = 0; r < RV; ++r) acc[t][j][r] = T::zero();
    }
  }
  const Scalar* a_row[MR];
  for (int i = 0; i < MR; ++i) a_row[i] = A + numext::mini(Index(i), m - 1) * lda;
  const Scalar* b_col[NC];
  for (int j = 0; j < NC; ++j) b_col[j] = B + numext::mini(Index(j), n - 1) * ldb;
  // A column (B row) vector r holds cnt valid scalars. Once m (n) >= PS, each is one full load ending at its last
  // valid scalar, shifted down by a byte table with zeros after; below that, a partial load.
  int a_cnt[RV], b_cnt[NC / PS];
  Index a_off[RV], b_off[NC / PS];
  uint8x16_t a_idx[RV], b_idx[NC / PS];
  const auto shift_table = [](int cnt) {
    EIGEN_ALIGN16 uint8_t t[16];
    for (int b = 0; b < 16; ++b)
      t[b] = b < cnt * int(sizeof(Scalar)) ? uint8_t(b + (PS - cnt) * int(sizeof(Scalar))) : 0xff;
    return vld1q_u8(t);
  };
  for (int r = 0; r < RV; ++r) {
    a_cnt[r] = int(numext::mini(Index(PS), m - r * PS));
    a_off[r] = r * PS + a_cnt[r] - PS;
    a_idx[r] = shift_table(a_cnt[r]);
  }
  for (int c = 0; c < NC / PS; ++c) {
    b_cnt[c] = int(numext::maxi(Index(0), numext::mini(Index(PS), n - c * PS)));
    b_off[c] = numext::maxi(Index(0), Index(c * PS + b_cnt[c] - PS));
    b_idx[c] = shift_table(b_cnt[c]);
  }
  const bool a_full = m >= PS, b_full = n >= PS;
  const Index kmain = (depth / PS) * PS;
  for (Index k = 0; k < kmain; k += PS) {
    V a[PS][RV];
    EIGEN_IF_CONSTEXPR (LhsOrder == ColMajor) {
      EIGEN_UNROLL_LOOP
      for (int u = 0; u < PS; ++u) {
        EIGEN_UNROLL_LOOP
        for (int r = 0; r < RV; ++r)
          a[u][r] = a_cnt[r] == PS ? T::ld(A + (k + u) * lda + r * PS)
                    : a_full       ? T::ld_tbl(A + (k + u) * lda + a_off[r], a_idx[r])
                                   : T::ld_part(A + (k + u) * lda + r * PS, a_cnt[r]);
      }
    } else {
      EIGEN_UNROLL_LOOP
      for (int r = 0; r < RV; ++r) {
        V blk[PS];
        EIGEN_UNROLL_LOOP
        for (int q = 0; q < PS; ++q) blk[q] = T::ld(a_row[r * PS + q] + k);
        T::transpose(blk);
        EIGEN_UNROLL_LOOP
        for (int u = 0; u < PS; ++u) a[u][r] = blk[u];
      }
    }
    EIGEN_IF_CONSTEXPR (RhsOrder == ColMajor) {
      V b[NC];
      EIGEN_UNROLL_LOOP
      for (int j = 0; j < NC; ++j) b[j] = T::ld(b_col[j] + k);
      Step::col(acc, a, b, std::integral_constant<int, 0>());
    } else {
      EIGEN_UNROLL_LOOP
      for (int u = 0; u < PS; ++u) {
        V b[NC / PS];
        EIGEN_UNROLL_LOOP
        for (int c = 0; c < NC / PS; ++c)
          b[c] = b_cnt[c] == PS  ? T::ld(B + (k + u) * ldb + c * PS)
                 : b_cnt[c] == 0 ? T::zero()
                 : b_full        ? T::ld_tbl(B + (k + u) * ldb + b_off[c], b_idx[c])
                                 : T::ld_part(B + (k + u) * ldb + c * PS, b_cnt[c]);
        Step::row(acc[u % S], a[u], b, std::integral_constant<int, 0>());
      }
    }
  }
  for (Index k = kmain; k < depth; ++k) {
    EIGEN_ALIGN16 Scalar at[MR];
    for (int i = 0; i < MR; ++i)
      at[i] = LhsOrder == ColMajor ? A[k * lda + numext::mini(Index(i), m - 1)] : a_row[i][k];
    V a[RV];
    EIGEN_UNROLL_LOOP
    for (int r = 0; r < RV; ++r) a[r] = T::ld(at + r * PS);
    EIGEN_UNROLL_LOOP
    for (int j = 0; j < NC; ++j) {
      const Scalar b = RhsOrder == ColMajor ? b_col[j][k] : B[k * ldb + numext::mini(Index(j), n - 1)];
      EIGEN_UNROLL_LOOP
      for (int r = 0; r < RV; ++r) acc[0][j][r] = T::fma_n(acc[0][j][r], a[r], b);
    }
  }
  EIGEN_UNROLL_LOOP
  for (int t = 1; t < S; ++t) {
    EIGEN_UNROLL_LOOP
    for (int j = 0; j < NC; ++j) {
      EIGEN_UNROLL_LOOP
      for (int r = 0; r < RV; ++r) acc[0][j][r] = T::add(acc[0][j][r], acc[t][j][r]);
    }
  }
  EIGEN_ALIGN16 Scalar out[NC][MR];
  EIGEN_UNROLL_LOOP
  for (int j = 0; j < NC; ++j) {
    EIGEN_UNROLL_LOOP
    for (int r = 0; r < RV; ++r) T::st(&out[j][r * PS], acc[0][j][r]);
  }
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < m; ++i) C[i * incr + j * ldc] += alpha * out[j][i];
}

template <typename Scalar, int LhsOrder, int RhsOrder, typename Index>
bool sme_tiny_gemm(Index rows, Index cols, Index depth, const Scalar* lhs, Index lhsStride, const Scalar* rhs,
                   Index rhsStride, Scalar* res, Index resIncr, Index resStride, Scalar alpha) {
  constexpr int PS = sme_tiny_neon<Scalar>::PS;
  if (!sme_tiny_gemm_wins<Scalar>(rows, cols, depth)) return false;
  const bool one_vec = rows <= PS, four_cols = cols <= 4;
  if (one_vec && four_cols)
    sme_tiny_gemm_kernel<Scalar, 1, 4, LhsOrder, RhsOrder>(rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res,
                                                           resIncr, resStride, alpha);
  else if (one_vec)
    sme_tiny_gemm_kernel<Scalar, 1, 8, LhsOrder, RhsOrder>(rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res,
                                                           resIncr, resStride, alpha);
  else if (four_cols)
    sme_tiny_gemm_kernel<Scalar, 2, 4, LhsOrder, RhsOrder>(rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res,
                                                           resIncr, resStride, alpha);
  else
    sme_tiny_gemm_kernel<Scalar, 2, 8, LhsOrder, RhsOrder>(rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res,
                                                           resIncr, resStride, alpha);
  return true;
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SME_GENERALBLOCKPANELKERNEL_H
