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

// Streaming vector length in floats, derived from the compile-time SVE VL.
// EIGEN_ARM64_SVE_VL is set in ConfigureVectorization.h from __ARM_FEATURE_SVE_BITS.
static constexpr int kSmeVlFloats = EIGEN_ARM64_SVE_VL / 32;

// Micro-kernel dimensions (2x2 ZA tile grid).
//   mr = 2 * SVL32 = 32  (LHS panel width, loaded as 2 VL-wide rows)
//   nr = 2 * SVL32 = 32  (RHS panel width, loaded as 2 VL-wide cols)
//   The 4 ZA tiles are arranged as a 2x2 grid covering a 32x32 output block:
//     ZA0 = A_lo x B_lo  (rows  0-15, cols  0-15)
//     ZA1 = A_lo x B_hi  (rows  0-15, cols 16-31)
//     ZA2 = A_hi x B_lo  (rows 16-31, cols  0-15)
//     ZA3 = A_hi x B_hi  (rows 16-31, cols 16-31)
// Per depth step: 2 LHS + 2 RHS VL-wide loads drive 4 FMOPAs -- a 1:1
// compute:load ratio, and each of A_lo/A_hi/B_lo/B_hi is reused across 2
// FMOPAs.
static constexpr int kSmeVl = kSmeVlFloats;  // 16 -- one VL's worth of floats
static constexpr int kSmeMr = 2 * kSmeVl;    // 32 -- LHS panel width
static constexpr int kSmeNr = 2 * kSmeVl;    // 32 -- RHS panel width

template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sve_copy_panel(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                               Index src_stride, Index depth, int width) __arm_streaming {
  const int lo_w = width > kSmeVl ? kSmeVl : width;
  const int hi_w = width > kSmeVl ? width - kSmeVl : 0;
  const svbool_t pred_lo = svwhilelt_b32(uint32_t(0), uint32_t(lo_w));
  const svbool_t pred_hi = svwhilelt_b32(uint32_t(0), uint32_t(hi_w));
  for (Index k = 0; k < depth; ++k) {
    svst1_f32(pred_lo, &dst[k * width], svld1_f32(pred_lo, &src[k * src_stride]));
    if (hi_w > 0) {
      svst1_f32(pred_hi, &dst[k * width + kSmeVl], svld1_f32(pred_hi, &src[k * src_stride + kSmeVl]));
    }
  }
}

template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void sme_transpose_pack_32(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src,
                                                      Index src_stride, Index depth) __arm_streaming __arm_inout("za") {
  static_assert(kSmeMr == kSmeNr, "SME transpose pack assumes square 32-wide panels");
  constexpr int PACK = kSmeMr;
  constexpr int VL = kSmeVl;
  const Index depth_vl = (depth / VL) * VL;
  const svbool_t pg_all = svptrue_b32();
  const svfloat32_t zero = svdup_f32(0.f);

  for (Index k = 0; k < depth_vl; k += VL) {
    for (uint32_t r = 0; r < VL; ++r) {
      svld1_hor_za32(0, r, pg_all, &src[(r + 0) * src_stride + k]);
      svld1_hor_za32(1, r, pg_all, &src[(r + VL) * src_stride + k]);
    }
    for (uint32_t c = 0; c < VL; ++c) {
      svst1_f32(pg_all, &dst[(k + c) * PACK + 0], svread_ver_za32_f32_m(zero, pg_all, 0, c));
      svst1_f32(pg_all, &dst[(k + c) * PACK + VL], svread_ver_za32_f32_m(zero, pg_all, 1, c));
    }
  }
  if (depth_vl < depth) {
    const int dtail = static_cast<int>(depth - depth_vl);
    const svbool_t pg_tail = svwhilelt_b32(uint32_t(0), uint32_t(dtail));
    for (uint32_t r = 0; r < VL; ++r) {
      svld1_hor_za32(0, r, pg_tail, &src[(r + 0) * src_stride + depth_vl]);
      svld1_hor_za32(1, r, pg_tail, &src[(r + VL) * src_stride + depth_vl]);
    }
    for (int c = 0; c < dtail; ++c) {
      svst1_f32(pg_all, &dst[(depth_vl + c) * PACK + 0], svread_ver_za32_f32_m(zero, pg_all, 0, uint32_t(c)));
      svst1_f32(pg_all, &dst[(depth_vl + c) * PACK + VL], svread_ver_za32_f32_m(zero, pg_all, 1, uint32_t(c)));
    }
  }
}

template <typename Scalar, typename Index>
static EIGEN_ALWAYS_INLINE void scalar_tail_pack(Scalar* EIGEN_RESTRICT dst_panel, const Scalar* EIGEN_RESTRICT src,
                                                 Index src_stride, Index depth, Index tail) __arm_streaming {
  for (Index k = 0; k < depth; ++k) {
    for (Index i = 0; i < tail; ++i) {
      dst_panel[k * tail + i] = src[i * src_stride + k];
    }
  }
}

/*****************************************************************************
 * gebp_traits specialization for SME  (float x float)
 *
 * Overrides mr and nr so that:
 *   - gemm_pack_lhs receives Pack1 = mr = 32, creating uniform LHS panels
 *   - gemm_pack_rhs receives nr = 32, creating uniform RHS panels
 *   - mc is rounded to a multiple of 32, nc to a multiple of 32
 *   - Cache blocking (kc, mc, nc) is recomputed accordingly
 *
 * We provide custom gemm_pack_lhs/gemm_pack_rhs specializations for float,
 * so both ColMajor and RowMajor source matrices produce an identical,
 * simple packed format that the SME kernel consumes.
 *****************************************************************************/

template <>
class gebp_traits<float, float, false, false, Architecture::Target, GEBPPacketFull>
    : public gebp_traits<float, float, false, false, Architecture::Target, GEBPPacketHalf> {
 public:
  // The base class provides all the standard typedefs (LhsPacket, etc.)
  // We only override the register-block sizes.
  enum {
    mr = kSmeMr,  // 32 -- LHS panel width
    nr = kSmeNr   // 32 -- RHS panel width
  };
};

/*****************************************************************************
 * gemm_pack_lhs specialization for SME  (float, ColMajor)
 *
 * Packs the LHS matrix into uniform panels of width = mr = 32.
 * Each depth step k writes exactly MR contiguous floats (2 full SVE registers).
 *****************************************************************************/

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  typedef float Scalar;

  __arm_locally_streaming static void pack_lhs_colmajor(Scalar* dst_base, const Scalar* EIGEN_RESTRICT src,
                                                        Index src_stride, Index depth, Index rows, Index dst_stride,
                                                        Index dst_offset) {
    constexpr int MR = kSmeMr;  // 32
    const Index peeled_rows = (rows / MR) * MR;

    // Full panels of width MR — two VL-wide SVE load+stores per depth step.
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
    const Scalar* src = (rows > 0 && depth > 0) ? &lhs(0, 0) : nullptr;
    pack_lhs_colmajor(blockA, src, lhs.stride(), depth, rows, stride, offset);
  }
};

// RowMajor LHS packer -- SME in-ZA transpose.
//
// The packed output wants depth-major layout (32 rows contiguous per depth
// step) but the RowMajor source has rows contiguous (strided by depth per
// row).  A natural SVE gather would be slow; instead we use ZA's 2D store
// as a free transpose: load 16 rows as horizontal slices of a ZA.S tile,
// then read vertical slices to produce depth-major output.  At MR=32 we
// use two tiles per VL-depth iteration (rows 0-15 in tile 0, rows 16-31
// in tile 1).
template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode> {
  typedef float Scalar;

  __arm_locally_streaming __arm_new("za") static void pack_lhs_rowmajor(Scalar* dst_base,
                                                                        const Scalar* EIGEN_RESTRICT src,
                                                                        Index src_stride, Index depth, Index rows,
                                                                        Index dst_stride, Index dst_offset) {
    constexpr int MR = kSmeMr;
    const Index peeled_rows = (rows / MR) * MR;

    for (Index i = 0; i < peeled_rows; i += MR) {
      Scalar* dst_panel = PanelMode ? dst_base + i * dst_stride + dst_offset * MR : dst_base + i * depth;
      sme_transpose_pack_32(dst_panel, src + i * src_stride, src_stride, depth);
    }

    // Row tail (rows - peeled_rows in [1, MR-1]).  This branch runs at most
    // once per pack_lhs call with < MR = 32 rows and would need a partial-ZA-
    // tile dance to vectorise; total copies are < 32 * depth per call, which
    // is noise vs the main packer's workload, so scalar is the simple choice.
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
    const Scalar* src = (rows > 0 && depth > 0) ? &lhs(0, 0) : nullptr;
    pack_lhs_rowmajor(blockA, src, lhs.stride(), depth, rows, stride, offset);
  }
};

/*****************************************************************************
 * gemm_pack_rhs specialization for SME  (float, ColMajor) -- SME in-ZA
 * transpose, mirroring the RowMajor LHS packer.
 *
 * Packs the RHS matrix into panels of width = nr = 32.  ColMajor source has
 * columns contiguous; we load 32 columns as horizontal ZA slices and then
 * read verticals to produce depth-major packed output.
 *****************************************************************************/

template <typename Index, typename DataMapper, int nr_, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr_, ColMajor, Conjugate, PanelMode> {
  typedef float Scalar;

  __arm_locally_streaming __arm_new("za") static void pack_rhs_colmajor(Scalar* dst_base,
                                                                        const Scalar* EIGEN_RESTRICT src,
                                                                        Index src_stride, Index depth, Index cols,
                                                                        Index dst_stride, Index dst_offset) {
    constexpr int NR = kSmeNr;
    const Index peeled_cols = (cols / NR) * NR;

    for (Index j = 0; j < peeled_cols; j += NR) {
      Scalar* dst_panel = PanelMode ? dst_base + j * dst_stride + dst_offset * NR : dst_base + j * depth;
      sme_transpose_pack_32(dst_panel, src + j * src_stride, src_stride, depth);
    }

    // Col tail (cols - peeled_cols in [1, NR-1]).  Same reasoning as the LHS
    // RowMajor packer's row tail: runs at most once per call, < NR = 32 cols,
    // not worth the partial-ZA-tile handling.
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
    const Scalar* src = (cols > 0 && depth > 0) ? &rhs(0, 0) : nullptr;
    pack_rhs_colmajor(blockB, src, rhs.stride(), depth, cols, stride, offset);
  }
};

// RowMajor RHS packer -- streaming SVE copy (mirrors the ColMajor LHS packer).
// Rows are contiguous in the source, so each depth-step is NR=32 contiguous
// fp32 = 2 VL-wide slices.
template <typename Index, typename DataMapper, int nr_, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr_, RowMajor, Conjugate, PanelMode> {
  typedef float Scalar;

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
    const Scalar* src = (cols > 0 && depth > 0) ? &rhs(0, 0) : nullptr;
    pack_rhs_rowmajor(blockB, src, rhs.stride(), depth, cols, stride, offset);
  }
};

/*****************************************************************************
 * sme_store_za_tile -- Store one ZA.S tile back to C with alpha scaling.
 *
 * `pw` is the row-predicate width for this tile, `cw` the col-predicate width.
 *****************************************************************************/

template <int TileId, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_store_za_tile(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                           Scalar alpha, Index row_start, int pw, Index col_start,
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
      Scalar* pC = C + row_start + (col_start + ci) * C_stride_col;
      svfloat32_t vc = svld1_f32(pg_m, pC);
      svst1_f32(pg_m, pC, svmla_f32_x(pg_m, vc, vres, valpha));
    }
  } else if (C_stride_col == 1) {
    // Row-major C: extract horizontal slices (rows of the ZA tile)
    for (int ri = 0; ri < pw; ++ri) {
      svfloat32_t vres = svread_hor_za32_f32_m(vzero, pg_n, TileId, (uint32_t)ri);
      Scalar* pC = C + (row_start + ri) * C_stride_row + col_start;
      svfloat32_t vc = svld1_f32(pg_n, pC);
      svst1_f32(pg_n, pC, svmla_f32_x(pg_n, vc, vres, valpha));
    }
  } else {
    // General stride: extract rows to temp buffer, scatter to C
    Scalar scratch[kSmeVl];
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
 * sme_process_2x2 -- 2x2 tile micro-kernel (pw x NR output)
 *
 * Performance-critical inner kernel.  Per 4 unrolled depth steps, performs
 *    2 svld1_f32_x4 for LHS  (8 VL vectors = 4 depth × 2 rows)
 *    2 svld1_f32_x4 for RHS  (8 VL vectors = 4 depth × 2 cols)
 *   16 svmopa                (4 depth × 2x2 tile grid)
 * Total: 4 load instructions + 16 FMOPAs per 4 depth steps -- a 1:1
 * compute:load ratio at the vector level.
 *
 * `pw` is the actual LHS panel width (== MR for full panels, < MR for the
 * tail).  Full path uses the compile-time constant MR as the LHS stride.
 *****************************************************************************/

template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_2x2(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                         const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blB,
                                         Index depth, Scalar alpha, Index row_start, int pw,
                                         Index col_start) __arm_streaming __arm_inout("za") {
  constexpr int MR = kSmeMr;  // 32 (full LHS panel width)
  constexpr int NR = kSmeNr;  // 32 (full RHS panel width)
  constexpr int VL = kSmeVl;  // 16

  const int pw_lo = pw > VL ? VL : pw;
  const int pw_hi = pw > VL ? pw - VL : 0;
  const svbool_t pg_lo = svwhilelt_b32((uint32_t)0, (uint32_t)pw_lo);
  const svbool_t pg_hi = svwhilelt_b32((uint32_t)0, (uint32_t)pw_hi);
  const svbool_t pg_all = svptrue_b32();
  const svcount_t pn_all = svptrue_c32();

  svzero_za();

  // ---- 4x unrolled depth loop ----
  if (pw == MR) {
    const Index depth_4 = (depth / 4) * 4;
    Index k = 0;
    for (; k < depth_4; k += 4) {
      // Load 4 depth steps × 2 LHS halves = 8 VL vectors as two x4 loads.
      //   va_01 = [d0 lo, d0 hi, d1 lo, d1 hi]
      //   va_23 = [d2 lo, d2 hi, d3 lo, d3 hi]
      svfloat32x4_t va_01 = svld1_f32_x4(pn_all, &blA[k * MR]);
      svfloat32x4_t vb_01 = svld1_f32_x4(pn_all, &blB[k * NR]);

      // d0
      svfloat32_t a0_lo = svget4_f32(va_01, 0), a0_hi = svget4_f32(va_01, 1);
      svfloat32_t b0_lo = svget4_f32(vb_01, 0), b0_hi = svget4_f32(vb_01, 1);
      svmopa_za32_f32_m(0, pg_all, pg_all, a0_lo, b0_lo);
      svmopa_za32_f32_m(1, pg_all, pg_all, a0_lo, b0_hi);
      svmopa_za32_f32_m(2, pg_all, pg_all, a0_hi, b0_lo);
      svmopa_za32_f32_m(3, pg_all, pg_all, a0_hi, b0_hi);

      // d1
      svfloat32_t a1_lo = svget4_f32(va_01, 2), a1_hi = svget4_f32(va_01, 3);
      svfloat32_t b1_lo = svget4_f32(vb_01, 2), b1_hi = svget4_f32(vb_01, 3);
      svmopa_za32_f32_m(0, pg_all, pg_all, a1_lo, b1_lo);
      svmopa_za32_f32_m(1, pg_all, pg_all, a1_lo, b1_hi);
      svmopa_za32_f32_m(2, pg_all, pg_all, a1_hi, b1_lo);
      svmopa_za32_f32_m(3, pg_all, pg_all, a1_hi, b1_hi);

      svfloat32x4_t va_23 = svld1_f32_x4(pn_all, &blA[(k + 2) * MR]);
      svfloat32x4_t vb_23 = svld1_f32_x4(pn_all, &blB[(k + 2) * NR]);

      // d2
      svfloat32_t a2_lo = svget4_f32(va_23, 0), a2_hi = svget4_f32(va_23, 1);
      svfloat32_t b2_lo = svget4_f32(vb_23, 0), b2_hi = svget4_f32(vb_23, 1);
      svmopa_za32_f32_m(0, pg_all, pg_all, a2_lo, b2_lo);
      svmopa_za32_f32_m(1, pg_all, pg_all, a2_lo, b2_hi);
      svmopa_za32_f32_m(2, pg_all, pg_all, a2_hi, b2_lo);
      svmopa_za32_f32_m(3, pg_all, pg_all, a2_hi, b2_hi);

      // d3
      svfloat32_t a3_lo = svget4_f32(va_23, 2), a3_hi = svget4_f32(va_23, 3);
      svfloat32_t b3_lo = svget4_f32(vb_23, 2), b3_hi = svget4_f32(vb_23, 3);
      svmopa_za32_f32_m(0, pg_all, pg_all, a3_lo, b3_lo);
      svmopa_za32_f32_m(1, pg_all, pg_all, a3_lo, b3_hi);
      svmopa_za32_f32_m(2, pg_all, pg_all, a3_hi, b3_lo);
      svmopa_za32_f32_m(3, pg_all, pg_all, a3_hi, b3_hi);
    }
    for (; k < depth; ++k) {
      svfloat32_t a_lo = svld1_f32(pg_all, &blA[k * MR]);
      svfloat32_t a_hi = svld1_f32(pg_all, &blA[k * MR + VL]);
      svfloat32_t b_lo = svld1_f32(pg_all, &blB[k * NR]);
      svfloat32_t b_hi = svld1_f32(pg_all, &blB[k * NR + VL]);
      svmopa_za32_f32_m(0, pg_all, pg_all, a_lo, b_lo);
      svmopa_za32_f32_m(1, pg_all, pg_all, a_lo, b_hi);
      svmopa_za32_f32_m(2, pg_all, pg_all, a_hi, b_lo);
      svmopa_za32_f32_m(3, pg_all, pg_all, a_hi, b_hi);
    }
  } else {
    // ---- Tail LHS panel (pw < MR, stride = pw) ----
    // Predicate the mopa rows; RHS is still full NR wide.
    for (Index k = 0; k < depth; ++k) {
      svfloat32_t a_lo = svld1_f32(pg_lo, &blA[k * pw]);
      svfloat32_t b_lo = svld1_f32(pg_all, &blB[k * NR]);
      svfloat32_t b_hi = svld1_f32(pg_all, &blB[k * NR + VL]);
      svmopa_za32_f32_m(0, pg_lo, pg_all, a_lo, b_lo);
      svmopa_za32_f32_m(1, pg_lo, pg_all, a_lo, b_hi);
      if (pw_hi > 0) {
        svfloat32_t a_hi = svld1_f32(pg_hi, &blA[k * pw + VL]);
        svmopa_za32_f32_m(2, pg_hi, pg_all, a_hi, b_lo);
        svmopa_za32_f32_m(3, pg_hi, pg_all, a_hi, b_hi);
      }
    }
  }

  // Store ZA tiles back to C, split into the 2x2 grid.
  // ZA0: rows [0..pw_lo), cols [0..VL)
  // ZA1: rows [0..pw_lo), cols [VL..NR)
  // ZA2: rows [VL..VL+pw_hi), cols [0..VL)
  // ZA3: rows [VL..VL+pw_hi), cols [VL..NR)
  sme_store_za_tile<0>(C, C_stride_row, C_stride_col, alpha, row_start, pw_lo, col_start, VL);
  sme_store_za_tile<1>(C, C_stride_row, C_stride_col, alpha, row_start, pw_lo, col_start + VL, VL);
  if (pw_hi > 0) {
    sme_store_za_tile<2>(C, C_stride_row, C_stride_col, alpha, row_start + VL, pw_hi, col_start, VL);
    sme_store_za_tile<3>(C, C_stride_row, C_stride_col, alpha, row_start + VL, pw_hi, col_start + VL, VL);
  }
}

/*****************************************************************************
 * sme_process_microblock -- Handle a single partial tile of size (pw × cw).
 *
 * Used for the tail column region when 0 < cw < NR.  One ZA tile per call.
 *****************************************************************************/

template <int TileId, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_process_microblock(Scalar* EIGEN_RESTRICT C, Index C_stride_row, Index C_stride_col,
                                                const Scalar* EIGEN_RESTRICT blA, const Scalar* EIGEN_RESTRICT blB,
                                                int blA_stride, int blB_stride, Index depth, Scalar alpha,
                                                Index row_start, int pw, Index col_start,
                                                int cw) __arm_streaming __arm_inout("za") {
  const svbool_t pg_m = svwhilelt_b32((uint32_t)0, (uint32_t)pw);
  const svbool_t pg_n = svwhilelt_b32((uint32_t)0, (uint32_t)cw);

  // svzero_mask_za's mask selects ZA.D byte-tiles (8 bits, one per ZA.D tile).
  // An fp32 ZA.S tile is composed of two ZA.D byte-tiles at indices
  // {TileId, TileId + 4}, so the mask 0x11 << TileId = (1 << TileId) |
  // (1 << (TileId + 4)) zeroes exactly this ZA.S tile.
  svzero_mask_za(0x11 << TileId);

  for (Index k = 0; k < depth; ++k) {
    svfloat32_t va = svld1_f32(pg_m, &blA[k * blA_stride]);
    svfloat32_t vb = svld1_f32(pg_n, &blB[k * blB_stride]);
    svmopa_za32_f32_m(TileId, pg_m, pg_n, va, vb);
  }

  sme_store_za_tile<TileId>(C, C_stride_row, C_stride_col, alpha, row_start, pw, col_start, cw);
}

template <typename Scalar, typename Index>
EIGEN_DONT_INLINE __arm_locally_streaming __arm_new("za") void sme_gebp_impl(
    Scalar* C, Index C_stride_row, Index C_stride_col, const Scalar* blockA, const Scalar* blockB, Index rows,
    Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  static_assert(Eigen::internal::is_same<Scalar, float>::value, "SME fp32 kernel only supports float");

  constexpr int MR = kSmeMr;  // 32
  constexpr int NR = kSmeNr;  // 32
  constexpr int VL = kSmeVl;  // 16

  const Index peeled_cols = (cols / NR) * NR;
  const Index peeled_rows = (rows / MR) * MR;
  const int tail_pw = static_cast<int>(rows - peeled_rows);

  // Column-outer, row-inner: keeps blB (one kc × NR panel) hot in L1 while
  // smaller blA tiles stream from L2.  The outer GOTO loop in
  // GeneralMatrixMatrix.h ensures blockA fits in L2 via mc-blocking.
  for (Index j = 0; j < peeled_cols; j += NR) {
    const Scalar* blB = &blockB[j * strideB + offsetB * NR];

    for (Index i = 0; i < peeled_rows; i += MR) {
      const Scalar* blA = blockA + i * strideA + offsetA * MR;
      sme_process_2x2(C, C_stride_row, C_stride_col, blA, blB, depth, alpha, i, MR, j);
    }

    if (tail_pw > 0) {
      const Scalar* blA = blockA + peeled_rows * strideA + offsetA * tail_pw;
      sme_process_2x2(C, C_stride_row, C_stride_col, blA, blB, depth, alpha, peeled_rows, tail_pw, j);
    }
  }

  // Tail columns (cols - peeled_cols in [1, NR - 1]).
  if (peeled_cols < cols) {
    const int tail_cols = static_cast<int>(cols - peeled_cols);
    const Scalar* blB_tail = &blockB[peeled_cols * strideB + offsetB * tail_cols];
    const int lo_cw = tail_cols > VL ? VL : tail_cols;
    const int hi_cw = tail_cols > VL ? tail_cols - VL : 0;

    const Index num_panels = peeled_rows + (tail_pw > 0 ? MR : 0);
    for (Index i = 0; i < num_panels; i += MR) {
      const int pw = (i < peeled_rows) ? MR : tail_pw;
      const int pw_lo = pw > VL ? VL : pw;
      const int pw_hi = pw > VL ? pw - VL : 0;
      const Scalar* blA = blockA + i * strideA + offsetA * pw;

      // lo col slice (width lo_cw, rows row_start .. row_start+pw_lo)
      sme_process_microblock<0>(C, C_stride_row, C_stride_col, blA, blB_tail, pw, tail_cols, depth, alpha, i, pw_lo,
                                peeled_cols, lo_cw);
      // hi col slice (width hi_cw), if present
      if (hi_cw > 0) {
        sme_process_microblock<1>(C, C_stride_row, C_stride_col, blA, blB_tail + VL, pw, tail_cols, depth, alpha, i,
                                  pw_lo, peeled_cols + VL, hi_cw);
      }
      if (pw_hi > 0) {
        sme_process_microblock<2>(C, C_stride_row, C_stride_col, blA + VL, blB_tail, pw, tail_cols, depth, alpha,
                                  i + VL, pw_hi, peeled_cols, lo_cw);
        if (hi_cw > 0) {
          sme_process_microblock<3>(C, C_stride_row, C_stride_col, blA + VL, blB_tail + VL, pw, tail_cols, depth, alpha,
                                    i + VL, pw_hi, peeled_cols + VL, hi_cw);
        }
      }
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

    if (strideA == -1) strideA = depth;
    if (strideB == -1) strideB = depth;

    if (rows <= 0 || cols <= 0 || depth <= 0) return;

    Scalar* C_base = const_cast<Scalar*>(&res(0, 0));
    const Index C_stride_row = &res(1, 0) - &res(0, 0);
    const Index C_stride_col = &res(0, 1) - &res(0, 0);

    sme_gebp_impl<Scalar, Index>(C_base, C_stride_row, C_stride_col, blockA, blockB, rows, depth, cols, alpha, strideA,
                                 strideB, offsetA, offsetB);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SME_GENERALBLOCKPANELKERNEL_H
